"""
Hybrid product matching: fuzzy (B) + embeddings (D).

Finds SKUs that refer to the same product despite different names or codes.
Output: sku_code -> canonical_sku_code mapping for merging in the dataset.

Flow:
  1. Normalize + lemmatize product names
  2. Fuzzy pass: within product_group, find pairs with similarity > fuzzy_threshold
  3. Embedding pass: for unmatched, embed names and find nearest neighbors
  4. Build connected components; each component gets one canonical_sku (lowest code)
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd

from src.data.schema import COL_PRODUCT_GROUP, COL_SKU_CODE, COL_SKU_NAME
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Optional deps — embedding pass skipped if not installed
try:
    from rapidfuzz import fuzz
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    import pymorphy2
    PYMORPHY_AVAILABLE = True
except ImportError:
    PYMORPHY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    from sklearn.neighbors import NearestNeighbors
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace, remove extra punctuation."""
    if not s or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,.]\s*", " ", s)  # normalize comma/dot spacing
    return s


def _lemmatize_ru(text: str) -> str:
    """Lemmatize Russian text. Falls back to normalized text if pymorphy2 missing."""
    if not PYMORPHY_AVAILABLE:
        return _normalize(text)
    try:
        morph = pymorphy2.MorphAnalyzer()
        words = text.split()
        lemmas = []
        for w in words:
            p = morph.parse(w)[0]
            lemmas.append(p.normal_form)
        return " ".join(lemmas)
    except Exception:
        return _normalize(text)


def _fuzzy_similarity(a: str, b: str) -> float:
    """Token-set ratio: robust to word order. Returns 0-100."""
    if not RAPIDFUZZ_AVAILABLE:
        return 0.0
    return fuzz.token_set_ratio(a, b)


def _fuzzy_match_pass(
    products: pd.DataFrame,
    norm_col: str,
    fuzzy_threshold: float,
) -> list[tuple[str, str]]:
    """
    Find pairs (sku_a, sku_b) with fuzzy similarity >= threshold.
    Uses blocking by product_group to avoid O(n²) over full catalog.
    """
    if not RAPIDFUZZ_AVAILABLE:
        logger.warning("rapidfuzz not installed; skipping fuzzy pass")
        return []

    pairs: list[tuple[str, str]] = []
    for grp_name, grp in products.groupby(COL_PRODUCT_GROUP, dropna=False):
        grp = grp.dropna(subset=[COL_SKU_CODE, norm_col])
        codes = grp[COL_SKU_CODE].astype(str).str.strip().unique().tolist()
        texts = grp.set_index(COL_SKU_CODE)[norm_col].to_dict()

        for i, ca in enumerate(codes):
            for cb in codes[i + 1 :]:
                ta = texts.get(ca, "")
                tb = texts.get(cb, "")
                if not ta or not tb:
                    continue
                sim = _fuzzy_similarity(ta, tb)
                if sim >= fuzzy_threshold:
                    pairs.append((ca, cb))

    return pairs


def _embed_and_match(
    products: pd.DataFrame,
    norm_col: str,
    embed_threshold: float,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> list[tuple[str, str]]:
    """
    Embed product names, find nearest neighbors, return pairs above threshold.
    Only runs for products not already matched in fuzzy pass.
    """
    if not EMBEDDINGS_AVAILABLE:
        logger.warning("sentence-transformers or sklearn not installed; skipping embedding pass")
        return []

    df = products.dropna(subset=[COL_SKU_CODE, norm_col]).drop_duplicates(COL_SKU_CODE)
    if len(df) < 2:
        return []

    codes = df[COL_SKU_CODE].astype(str).str.strip().tolist()
    texts = df[norm_col].fillna("").tolist()

    logger.info("Loading embedding model %s (first run may download)...", model_name)
    model = SentenceTransformer(model_name)
    embs = model.encode(texts, show_progress_bar=False)

    nn = NearestNeighbors(n_neighbors=6, metric="cosine")
    nn.fit(embs)
    dists, idxs = nn.kneighbors(embs)

    pairs: list[tuple[str, str]] = []
    for i, (code_a, neighbors) in enumerate(zip(codes, idxs)):
        for k in range(1, len(neighbors)):  # skip self (k=0)
            j = neighbors[k]
            if j >= len(codes):
                continue
            code_b = codes[j]
            if code_a >= code_b:
                continue  # avoid duplicate (a,b) and (b,a)
            sim = 1.0 - dists[i, k]
            if sim >= embed_threshold:
                pairs.append((code_a, code_b))

    return pairs


def _build_canonical_mapping(pairs: list[tuple[str, str]], all_codes: set[str]) -> dict[str, str]:
    """
    From match pairs, build connected components. Canonical = lexicographically
    smallest code in each component. Returns sku_code -> canonical_sku_code.
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = min(ra, rb)
            parent[rb] = parent[ra]

    for a, b in pairs:
        union(a, b)

    # Ensure every code has a canonical (itself if never matched)
    result: dict[str, str] = {}
    for c in all_codes:
        result[c] = find(c)
    return result


def run_product_matching(
    products_df: pd.DataFrame,
    fuzzy_threshold: float = 92.0,
    embed_threshold: float = 0.92,
    use_embeddings: bool = True,
    product_group_col: str = COL_PRODUCT_GROUP,
    sku_code_col: str = COL_SKU_CODE,
    sku_name_col: str = COL_SKU_NAME,
) -> dict[str, str]:
    """
    Run hybrid product matching.

    Args:
        products_df: DataFrame with sku_code, sku_name, product_group.
                     One row per product (deduplicated).
        fuzzy_threshold: Min token_set_ratio (0-100) for fuzzy match.
        embed_threshold: Min cosine similarity for embedding match.
        use_embeddings: If False, skip embedding pass (B-only).

    Returns:
        sku_code -> canonical_sku_code. Unmatched SKUs map to themselves.
    """
    df = products_df.copy()
    df[sku_code_col] = df[sku_code_col].astype(str).str.strip()
    df[sku_name_col] = df[sku_name_col].fillna("").astype(str).str.strip()
    df[product_group_col] = df[product_group_col].fillna("Unknown").astype(str).str.strip()

    # Normalize + lemmatize for matching
    df["_norm"] = df[sku_name_col].apply(lambda x: _lemmatize_ru(x))
    df = df[df["_norm"].str.len() > 1].copy()

    all_codes = set(df[sku_code_col].unique())

    # Pass 1: Fuzzy
    fuzzy_pairs = _fuzzy_match_pass(df, "_norm", fuzzy_threshold)
    logger.info("Fuzzy pass: %d pairs above threshold %.0f", len(fuzzy_pairs), fuzzy_threshold)

    # Pass 2: Embeddings (only for products not in any fuzzy pair)
    matched_codes = set()
    for a, b in fuzzy_pairs:
        matched_codes.add(a)
        matched_codes.add(b)

    embed_pairs: list[tuple[str, str]] = []
    if use_embeddings and EMBEDDINGS_AVAILABLE:
        # Run embedding on all; we'll merge with fuzzy pairs
        embed_pairs = _embed_and_match(df, "_norm", embed_threshold)
        logger.info("Embedding pass: %d pairs above threshold %.2f", len(embed_pairs), embed_threshold)

    all_pairs = fuzzy_pairs + embed_pairs
    mapping = _build_canonical_mapping(all_pairs, all_codes)

    n_canonical = len(set(mapping.values()))
    n_merged = len(all_codes) - n_canonical
    logger.info("Canonical SKUs: %d (merged %d duplicates)", n_canonical, n_merged)

    return mapping


def load_mapping(path: Path) -> dict[str, str]:
    """Load sku_code -> canonical_sku_code from JSON."""
    import json
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_mapping(mapping: dict[str, str], path: Path) -> None:
    """Save mapping to JSON."""
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)
