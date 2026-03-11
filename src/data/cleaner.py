"""
LLM-based data cleaning for product catalog.

Fixes product-level issues found in the cleansing report:
  - Inconsistent / whitespace-padded product names
  - Mixed-case and near-duplicate product groups
  - Empty product groups
  - Same sku_code with conflicting names across data sources

Two-pass DeepSeek strategy:
  Pass 1 — Build a canonical product group taxonomy from the raw group list.
  Pass 2 — Clean each product (normalize name, assign canonical group) in batches,
            always providing the canonical group list so the LLM stays consistent.

Incremental runs:
  Cleaned results are cached in data/cleaned/product_catalog.json.
  On subsequent runs only NEW or CHANGED products (and groups) are sent to the API.
  Use force=True to ignore the cache and re-process everything.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data.schema import COL_ARTICLE, COL_PRODUCT_GROUP, COL_SKU_CODE, COL_SKU_NAME
from src.utils.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CATALOG_CACHE_FILE = "product_catalog.json"
GROUPS_CACHE_FILE = "canonical_groups.json"
GROUP_MAPPING_CACHE_FILE = "group_mapping.json"  # raw group name -> canonical group name
BATCH_SIZE = 50                  # products per DeepSeek request
MAX_RETRIES = 3
RETRY_DELAY = 2.0                # seconds between retries


# ---------------------------------------------------------------------------
# Catalog extraction
# ---------------------------------------------------------------------------

def extract_product_catalog(
    sales_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    products_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a deduplicated product catalog from all raw sources.

    Priority: sales (richest metadata) > products > stock.
    Returns a DataFrame with columns: sku_code, sku_name, product_group, article.
    """
    frames = []

    if not sales_df.empty:
        cols = [c for c in [COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP, COL_ARTICLE] if c in sales_df.columns]
        frames.append(sales_df[cols].copy())

    if not products_df.empty:
        cols = [c for c in [COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP] if c in products_df.columns]
        frames.append(products_df[cols].copy())

    if not stock_df.empty:
        cols = [c for c in [COL_SKU_CODE, COL_SKU_NAME] if c in stock_df.columns]
        frames.append(stock_df[cols].copy())

    if not frames:
        return pd.DataFrame(columns=[COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP, COL_ARTICLE])

    combined = pd.concat(frames, ignore_index=True)

    # Ensure all expected columns exist
    for col in [COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP, COL_ARTICLE]:
        if col not in combined.columns:
            combined[col] = ""

    combined[COL_SKU_CODE] = combined[COL_SKU_CODE].astype(str).str.strip()
    combined = combined[combined[COL_SKU_CODE].str.len() > 0].copy()

    # Deduplicate: keep first occurrence per sku_code (sales rows appear first)
    catalog = combined.drop_duplicates(subset=[COL_SKU_CODE], keep="first").reset_index(drop=True)
    return catalog[[COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP, COL_ARTICLE]]


# ---------------------------------------------------------------------------
# Trivial code-level fixes (no LLM)
# ---------------------------------------------------------------------------

def fix_trivial_issues(catalog: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from all text fields. No LLM involved."""
    df = catalog.copy()
    for col in [COL_SKU_CODE, COL_SKU_NAME, COL_PRODUCT_GROUP, COL_ARTICLE]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].replace("nan", "")
    return df


# ---------------------------------------------------------------------------
# Catalog cache management
# ---------------------------------------------------------------------------

def load_cached_catalog(cleaned_dir: Path) -> dict[str, dict]:
    """
    Load the cleaned product catalog cache.
    Returns dict: sku_code -> {sku_name, product_group, article}.
    Empty dict if no cache exists.
    """
    path = cleaned_dir / CATALOG_CACHE_FILE
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_catalog_cache(catalog: dict[str, dict], cleaned_dir: Path) -> None:
    """Persist the full cleaned catalog to data/cleaned/product_catalog.json."""
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    path = cleaned_dir / CATALOG_CACHE_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    logger.info("Saved catalog cache: %d products → %s", len(catalog), path)


def load_cached_groups(cleaned_dir: Path) -> list[str]:
    """Load the canonical product group list. Empty list if not cached."""
    path = cleaned_dir / GROUPS_CACHE_FILE
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_groups_cache(groups: list[str], cleaned_dir: Path) -> None:
    """Persist the canonical group list to data/cleaned/canonical_groups.json."""
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    path = cleaned_dir / GROUPS_CACHE_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(groups, f, ensure_ascii=False, indent=2)
    logger.info("Saved canonical groups cache: %d groups → %s", len(groups), path)


def load_cached_group_mapping(cleaned_dir: Path) -> dict[str, str]:
    """Load the raw->canonical group mapping cache. Empty dict if not found."""
    path = cleaned_dir / GROUP_MAPPING_CACHE_FILE
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_group_mapping_cache(mapping: dict[str, str], cleaned_dir: Path) -> None:
    """Persist raw->canonical group mapping to data/cleaned/group_mapping.json."""
    cleaned_dir.mkdir(parents=True, exist_ok=True)
    path = cleaned_dir / GROUP_MAPPING_CACHE_FILE
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
    logger.info("Saved group mapping cache: %d entries → %s", len(mapping), path)


# ---------------------------------------------------------------------------
# Diff: find new / changed products
# ---------------------------------------------------------------------------

def find_new_products(
    current_catalog: pd.DataFrame,
    cached_catalog: dict[str, dict],
) -> pd.DataFrame:
    """
    Return rows from current_catalog whose sku_code is not in the cache,
    OR whose sku_name or product_group changed since last cleaning.
    """
    if not cached_catalog:
        return current_catalog.copy()

    new_rows = []
    for _, row in current_catalog.iterrows():
        code = str(row[COL_SKU_CODE]).strip()
        if code not in cached_catalog:
            new_rows.append(row)
            continue
        cached = cached_catalog[code]
        # Re-process if the raw name or group has changed (LLM needs new input)
        raw_name = str(row.get(COL_SKU_NAME, "")).strip()
        raw_group = str(row.get(COL_PRODUCT_GROUP, "")).strip()
        if raw_name != cached.get("raw_sku_name", "") or raw_group != cached.get("raw_product_group", ""):
            new_rows.append(row)

    if not new_rows:
        return pd.DataFrame(columns=current_catalog.columns)
    return pd.DataFrame(new_rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# DeepSeek API helpers
# ---------------------------------------------------------------------------

def _make_client(api_key: str, base_url: str, model: str):
    """Return an OpenAI-compatible client for the given base_url and model."""
    from openai import OpenAI
    return OpenAI(
        api_key=api_key,
        base_url=base_url.rstrip("/"),
    ), model


def _call_llm(
    client,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> str:
    """Call DeepSeek with retry logic. Returns the raw response text."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            logger.warning("DeepSeek attempt %d/%d failed: %s", attempt, MAX_RETRIES, exc)
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY * attempt)
    raise RuntimeError(f"DeepSeek API failed after {MAX_RETRIES} attempts")


def _extract_json(raw: str) -> Any:
    """Extract JSON from LLM response, stripping markdown fences if present."""
    raw = raw.strip()
    # Strip ```json ... ``` fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Pass 1: Clean product group taxonomy
# ---------------------------------------------------------------------------

def clean_groups_with_llm(
    raw_groups: list[str],
    cached_groups: list[str],
    cached_group_mapping: dict[str, str],
    api_key: str,
    base_url: str,
    model: str,
) -> tuple[list[str], dict[str, str]]:
    """
    Pass 1: Send raw product groups to DeepSeek to produce a canonical taxonomy.

    Diffs against cached_group_mapping (raw -> canonical) so the LLM is only
    called for raw group names that have never been processed before.

    Returns:
        canonical_groups: full merged list of clean group names
        group_mapping: raw group name -> canonical group name (full, merged with cache)
    """
    # A raw group is "new" if it has never appeared in the raw->canonical mapping cache
    new_raw = [g for g in raw_groups if g not in cached_group_mapping and g.strip()]

    if not new_raw:
        logger.info(
            "No new product groups — reusing cached mapping (%d groups, %d raw entries)",
            len(cached_groups), len(cached_group_mapping),
        )
        # Return cached state; any raw group not in mapping falls back to stripped value
        full_mapping = dict(cached_group_mapping)
        for g in raw_groups:
            if g not in full_mapping:
                full_mapping[g] = g.strip()
        return list(cached_groups), full_mapping

    logger.info(
        "Pass 1: cleaning %d new product groups (cached: %d)...",
        len(new_raw), len(cached_groups),
    )

    system_prompt = (
        "You are a data quality expert for a Russian bakery / grocery retail company. "
        "Your task is to clean a list of product group names. "
        "Respond ONLY with valid JSON — no markdown, no explanation."
    )

    user_prompt = (
        f"Existing canonical product groups (already clean, do NOT modify):\n"
        f"{json.dumps(cached_groups, ensure_ascii=False)}\n\n"
        f"New raw product groups to process:\n"
        f"{json.dumps(new_raw, ensure_ascii=False)}\n\n"
        "Instructions:\n"
        "1. For each new raw group, decide: merge into an existing canonical group, "
        "   create a new clean canonical group, or discard if clearly invalid (e.g. 'БРАК' = waste/defects).\n"
        "2. Normalize casing (Title Case for Russian), fix leading/trailing spaces, merge near-duplicates "
        "   ('Прочие' / 'Прочие товары' / 'Товары' → one name, e.g. 'Прочие товары').\n"
        "3. Output a JSON object with two keys:\n"
        "   - 'canonical_groups': array of ALL canonical group names "
        "     (existing ones unchanged + any new ones you created)\n"
        "   - 'mapping': object mapping each input raw group to its canonical group name\n"
        "   Example: {\"canonical_groups\": [\"Хлеб\", \"Торты\"], "
        "\"mapping\": {\"ТОРТЫ\": \"Торты\", \"хлеб \": \"Хлеб\"}}"
    )

    client, _model = _make_client(api_key, base_url, model)
    raw_response = _call_llm(client, _model, system_prompt, user_prompt)
    try:
        parsed = _extract_json(raw_response)
        new_canonical = parsed.get("canonical_groups", cached_groups)
        new_mapping: dict[str, str] = parsed.get("mapping", {})
    except Exception as exc:
        logger.error("Failed to parse Pass 1 response: %s\nRaw: %s", exc, raw_response[:500])
        # Fallback: keep existing canonical list, map new groups to themselves
        new_canonical = list(cached_groups) + [g.strip().title() for g in new_raw]
        new_mapping = {g: g.strip() for g in new_raw}

    # Ensure cached groups are still in canonical (LLM must not drop them)
    for g in cached_groups:
        if g not in new_canonical:
            new_canonical.append(g)

    # Merge new LLM mapping on top of cached mapping
    full_mapping = dict(cached_group_mapping)
    full_mapping.update(new_mapping)

    # Fill any raw groups still missing (should not happen, but safe fallback)
    for g in raw_groups:
        if g not in full_mapping:
            full_mapping[g] = g.strip()

    logger.info("Pass 1 complete: %d canonical groups, %d total raw mappings", len(new_canonical), len(full_mapping))
    return new_canonical, full_mapping


# ---------------------------------------------------------------------------
# Pass 2: Clean products in batches
# ---------------------------------------------------------------------------

def clean_products_with_llm(
    products: pd.DataFrame,
    canonical_groups: list[str],
    api_key: str,
    base_url: str,
    model: str,
) -> list[dict]:
    """
    Pass 2: Clean product names and assign canonical groups via DeepSeek.

    Sends products in batches of BATCH_SIZE. Each batch includes the full
    canonical group list so the LLM stays consistent across batches.

    Returns a list of dicts: {sku_code, sku_name, product_group, article}
    """
    if products.empty:
        return []

    client, model_name = _make_client(api_key, base_url, model)

    system_prompt = (
        "You are a data quality expert for a Russian bakery / grocery retail company. "
        "Clean product records. Respond ONLY with valid JSON — no markdown, no explanation."
    )

    groups_json = json.dumps(canonical_groups, ensure_ascii=False)

    results: list[dict] = []
    rows = products.to_dict(orient="records")
    total_batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(0, len(rows), BATCH_SIZE):
        batch = rows[batch_idx: batch_idx + BATCH_SIZE]
        batch_num = batch_idx // BATCH_SIZE + 1
        logger.info("Pass 2: batch %d/%d (%d products)...", batch_num, total_batches, len(batch))

        # Simplify the batch: only send fields the LLM needs
        batch_input = [
            {
                "sku_code": str(r.get(COL_SKU_CODE, "")),
                "sku_name": str(r.get(COL_SKU_NAME, "")),
                "product_group": str(r.get(COL_PRODUCT_GROUP, "")),
                "article": str(r.get(COL_ARTICLE, "")),
            }
            for r in batch
        ]

        user_prompt = (
            f"Canonical product group list (use ONLY these, or propose a new one with justification):\n"
            f"{groups_json}\n\n"
            f"Product records to clean:\n"
            f"{json.dumps(batch_input, ensure_ascii=False)}\n\n"
            "Instructions for each product:\n"
            "1. sku_name: normalize formatting — strip extra spaces, fix obvious typos, "
            "   standardize weight notation (always use '0,300 кг' format with comma, "
            "   e.g. '0.3 кг' → '0,300 кг', '300гр' → '300 г'). Keep the canonical form "
            "   when the same sku_code has multiple conflicting names.\n"
            "2. product_group: assign one group from the canonical list. "
            "   If none fits, propose a new group name and add it to the list.\n"
            "3. article: keep as-is unless it is clearly wrong (e.g. empty string is fine).\n"
            "4. sku_code: do NOT modify.\n"
            "Return a JSON object with key 'products': array of cleaned records "
            "with the same fields: sku_code, sku_name, product_group, article."
        )

        raw_response = _call_llm(client, model_name, system_prompt, user_prompt)
        try:
            parsed = _extract_json(raw_response)
            batch_results = parsed.get("products", [])
            if not isinstance(batch_results, list):
                raise ValueError("'products' is not a list")
            results.extend(batch_results)
        except Exception as exc:
            logger.error("Failed to parse Pass 2 batch %d response: %s\nRaw: %s", batch_num, exc, raw_response[:500])
            # Fallback: return batch unchanged
            results.extend(batch_input)

    logger.info("Pass 2 complete: %d products cleaned", len(results))
    return results


# ---------------------------------------------------------------------------
# Merge and apply
# ---------------------------------------------------------------------------

def merge_catalogs(
    cached_catalog: dict[str, dict],
    new_cleaned: list[dict],
    raw_catalog: pd.DataFrame,
) -> dict[str, dict]:
    """
    Merge newly cleaned products into the full catalog cache.

    Each entry stores both the clean fields AND the original raw values
    (raw_sku_name, raw_product_group) so incremental diff can detect changes.
    """
    updated = dict(cached_catalog)

    # Build a raw lookup for recording original values
    raw_lookup: dict[str, dict] = {}
    for _, row in raw_catalog.iterrows():
        code = str(row[COL_SKU_CODE]).strip()
        raw_lookup[code] = {
            "raw_sku_name": str(row.get(COL_SKU_NAME, "")).strip(),
            "raw_product_group": str(row.get(COL_PRODUCT_GROUP, "")).strip(),
        }

    for item in new_cleaned:
        code = str(item.get("sku_code", "")).strip()
        if not code:
            continue
        raw = raw_lookup.get(code, {})
        updated[code] = {
            "sku_name": str(item.get("sku_name", code)),
            "product_group": str(item.get("product_group", "")),
            "article": str(item.get("article", "")),
            "raw_sku_name": raw.get("raw_sku_name", ""),
            "raw_product_group": raw.get("raw_product_group", ""),
        }

    return updated


def apply_cleaning(raw_df: pd.DataFrame, clean_catalog: dict[str, dict]) -> pd.DataFrame:
    """
    Apply the cleaned product catalog to a raw DataFrame.

    Remaps sku_name and product_group by sku_code. Other columns are unchanged.
    """
    if raw_df.empty or not clean_catalog:
        return raw_df

    df = raw_df.copy()
    codes = df[COL_SKU_CODE].astype(str).str.strip()

    if COL_SKU_NAME in df.columns:
        df[COL_SKU_NAME] = codes.map(
            lambda c: clean_catalog[c]["sku_name"] if c in clean_catalog else df.loc[df[COL_SKU_CODE].astype(str).str.strip() == c, COL_SKU_NAME].iloc[0] if (df[COL_SKU_CODE].astype(str).str.strip() == c).any() else c
        )

    if COL_PRODUCT_GROUP in df.columns:
        df[COL_PRODUCT_GROUP] = codes.map(
            lambda c: clean_catalog[c]["product_group"] if c in clean_catalog else df.loc[df[COL_SKU_CODE].astype(str).str.strip() == c, COL_PRODUCT_GROUP].iloc[0] if (df[COL_SKU_CODE].astype(str).str.strip() == c).any() else ""
        )

    return df


def _apply_cleaning_fast(raw_df: pd.DataFrame, clean_catalog: dict[str, dict]) -> pd.DataFrame:
    """
    Vectorized version of apply_cleaning using map() for performance.

    Applies: sku_code (strip), sku_name, product_group, article from catalog.
    """
    if raw_df.empty or not clean_catalog:
        return raw_df

    df = raw_df.copy()
    code_series = df[COL_SKU_CODE].astype(str).str.strip()

    # Strip sku_code in output (raw often has trailing spaces)
    if COL_SKU_CODE in df.columns:
        df[COL_SKU_CODE] = code_series

    if COL_SKU_NAME in df.columns:
        name_map = {c: v["sku_name"] for c, v in clean_catalog.items()}
        df[COL_SKU_NAME] = code_series.map(name_map).fillna(df[COL_SKU_NAME])

    if COL_PRODUCT_GROUP in df.columns:
        group_map = {c: v["product_group"] for c, v in clean_catalog.items()}
        df[COL_PRODUCT_GROUP] = code_series.map(group_map).fillna(df[COL_PRODUCT_GROUP])

    if COL_ARTICLE in df.columns:
        article_map = {c: (v.get("article") or "") for c, v in clean_catalog.items()}
        mapped = code_series.map(article_map)
        raw_stripped = df[COL_ARTICLE].astype(str).str.strip().replace("nan", "")
        df[COL_ARTICLE] = mapped.where(mapped.notna(), raw_stripped).fillna("")

    return df


# ---------------------------------------------------------------------------
# Save cleaned output
# ---------------------------------------------------------------------------

def save_cleaned(
    data: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """
    Write cleaned DataFrames to data/cleaned/ as parquet files.

    data: dict with keys like 'sales', 'stock', 'losses', 'products'.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        path = output_dir / f"{name}.parquet"
        df.to_parquet(path, index=False)
        logger.info("Saved cleaned %s: %d rows → %s", name, len(df), path)


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------

def run_cleaning(
    sales_df: pd.DataFrame,
    stock_df: pd.DataFrame,
    losses_df: pd.DataFrame,
    products_df: pd.DataFrame,
    cleaned_dir: Path,
    api_key: str,
    base_url: str,
    model: str,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """
    Full cleaning pipeline. Returns dict of cleaned DataFrames.

    Steps:
      1. Extract product catalog from all raw sources
      2. Fix trivial whitespace issues in code
      3. If not force: load cached catalog and diff to find new/changed products
      4. Pass 1: clean product groups (incremental or full)
      5. Pass 2: clean new/changed products in batches with canonical group list
      6. Merge into cached catalog and save cache
      7. Apply cleaned catalog to all raw DataFrames
      8. Save cleaned parquet files to cleaned_dir
    """
    # Step 1-2: extract and fix trivials
    catalog = extract_product_catalog(sales_df, stock_df, products_df)
    catalog = fix_trivial_issues(catalog)
    logger.info("Product catalog: %d unique SKUs extracted from raw data", len(catalog))

    # Step 3: load cache and diff
    cached_catalog = {} if force else load_cached_catalog(cleaned_dir)
    cached_groups = [] if force else load_cached_groups(cleaned_dir)
    cached_group_mapping = {} if force else load_cached_group_mapping(cleaned_dir)

    new_products = find_new_products(catalog, cached_catalog)
    logger.info(
        "Products to process: %d new/changed (total catalog: %d, cached: %d)",
        len(new_products), len(catalog), len(cached_catalog),
    )

    # Step 4: Pass 1 — clean groups (incremental: only unseen raw group names hit the LLM)
    raw_groups = [
        g for g in catalog[COL_PRODUCT_GROUP].unique().tolist()
        if g and g.strip()
    ]
    canonical_groups, group_mapping = clean_groups_with_llm(
        raw_groups, cached_groups, cached_group_mapping, api_key, base_url, model
    )
    save_groups_cache(canonical_groups, cleaned_dir)
    save_group_mapping_cache(group_mapping, cleaned_dir)

    # Apply group mapping to new_products before sending to LLM
    if not new_products.empty and COL_PRODUCT_GROUP in new_products.columns:
        new_products = new_products.copy()
        new_products[COL_PRODUCT_GROUP] = new_products[COL_PRODUCT_GROUP].map(
            lambda g: group_mapping.get(g, g)
        )

    # Step 5: Pass 2 — clean products
    if not new_products.empty:
        cleaned_items = clean_products_with_llm(new_products, canonical_groups, api_key, base_url, model)
    else:
        cleaned_items = []
        logger.info("No new products to clean — using cached catalog as-is")

    # Step 6: merge into cache
    full_catalog = merge_catalogs(cached_catalog, cleaned_items, catalog)
    save_catalog_cache(full_catalog, cleaned_dir)

    # Step 7: apply to all raw DataFrames
    cleaned_sales = _apply_cleaning_fast(sales_df, full_catalog)
    cleaned_stock = _apply_cleaning_fast(stock_df, full_catalog)
    cleaned_losses = _apply_cleaning_fast(losses_df, full_catalog)
    cleaned_products = _apply_cleaning_fast(products_df, full_catalog)

    # Step 8: save
    cleaned_data = {
        "sales": cleaned_sales,
        "stock": cleaned_stock,
        "losses": cleaned_losses,
        "products": cleaned_products,
    }
    save_cleaned(cleaned_data, cleaned_dir)

    n_merged = len(full_catalog) - len(cached_catalog) if not force else len(full_catalog)
    logger.info(
        "Cleaning complete: %d products in catalog (%d newly cleaned)",
        len(full_catalog), len(cleaned_items),
    )

    return cleaned_data
