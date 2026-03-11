"""
Product matching via sentence-transformer embeddings (default) or Weaviate vector DB.

Finds SKUs that refer to the same product despite different names or codes.
Output: sku_code -> canonical_sku_code mapping for merging in the dataset.

Flow (default — embedding mode):
  1. Normalize + lemmatize product names
  2. Embed names with paraphrase-multilingual-MiniLM-L12-v2 and find nearest neighbors
  3. Build connected components; each component gets one canonical_sku (lowest code)

Flow (Weaviate mode — requires docker compose up -d):
  Same steps 1 and 3, but embedding + neighbor search is delegated to Weaviate
  with the text2vec-transformers module. Enable via use_weaviate=True.
"""

from __future__ import annotations

import re
import uuid as uuid_module
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd

from src.data.schema import COL_PRODUCT_GROUP, COL_SKU_CODE, COL_SKU_NAME
from src.utils.logging import get_logger

logger = get_logger(__name__)

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

COLLECTION_NAME = "ProductMatching"


def _normalize(s: str) -> str:
    """Lowercase, strip, collapse whitespace, remove extra punctuation."""
    if not s or not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[,.]\s*", " ", s)
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


def _embed_and_match(
    products: pd.DataFrame,
    norm_col: str,
    embed_threshold: float,
    model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
) -> list[tuple[str, str]]:
    """
    Embed product names in-memory, find nearest neighbors, return pairs above threshold.
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

    nn = NearestNeighbors(n_neighbors=min(6, len(codes)), metric="cosine")
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


# ---------------------------------------------------------------------------
# Weaviate path (disabled by default; enabled via use_weaviate=True)
# ---------------------------------------------------------------------------

def _connect_weaviate(weaviate_url: str):
    """Connect to Weaviate using v4 client. Returns connected WeaviateClient."""
    import weaviate
    from weaviate.connect import ConnectionParams

    parsed = urlparse(weaviate_url)
    host = parsed.hostname or "localhost"
    http_port = parsed.port or 8080
    secure = parsed.scheme == "https"

    client = weaviate.WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host=host,
            http_port=http_port,
            http_secure=secure,
            grpc_host=host,
            grpc_port=50051,
            grpc_secure=secure,
        )
    )
    client.connect()
    return client


def _weaviate_match(
    products: pd.DataFrame,
    norm_col: str,
    embed_threshold: float,
    weaviate_url: str,
) -> list[tuple[str, str]]:
    """
    Insert products into Weaviate, then find nearest-neighbor pairs above threshold.

    Weaviate's text2vec-transformers module generates embeddings automatically
    on insert using the configured paraphrase-multilingual-MiniLM-L12-v2 model.
    Cosine distance threshold = 1 - embed_threshold.
    """
    import weaviate
    from weaviate.classes.config import Configure, Property, DataType
    from weaviate.classes.query import MetadataQuery

    df = products.dropna(subset=[COL_SKU_CODE, norm_col]).drop_duplicates(COL_SKU_CODE)
    if len(df) < 2:
        return []

    distance_threshold = 1.0 - embed_threshold

    client = _connect_weaviate(weaviate_url)
    try:
        if client.collections.exists(COLLECTION_NAME):
            client.collections.delete(COLLECTION_NAME)

        client.collections.create(
            COLLECTION_NAME,
            vectorizer_config=Configure.Vectorizer.text2vec_transformers(
                vectorize_collection_name=False,
            ),
            properties=[
                Property(
                    name="sku_code",
                    data_type=DataType.TEXT,
                    skip_vectorization=True,
                    vectorize_property_name=False,
                ),
                Property(
                    name="name_norm",
                    data_type=DataType.TEXT,
                    vectorize_property_name=False,
                ),
            ],
        )
        logger.info("Weaviate collection '%s' created.", COLLECTION_NAME)

        collection = client.collections.get(COLLECTION_NAME)

        code_to_uuid: dict[str, uuid_module.UUID] = {}
        records = df[[COL_SKU_CODE, norm_col]].copy()
        records[COL_SKU_CODE] = records[COL_SKU_CODE].astype(str).str.strip()

        logger.info("Inserting %d products into Weaviate...", len(records))
        with collection.batch.fixed_size(batch_size=100) as batch:
            for _, row in records.iterrows():
                code = row[COL_SKU_CODE]
                obj_uuid = uuid_module.uuid5(uuid_module.NAMESPACE_DNS, code)
                code_to_uuid[code] = obj_uuid
                batch.add_object(
                    properties={
                        "sku_code": code,
                        "name_norm": str(row[norm_col]),
                    },
                    uuid=obj_uuid,
                )

        failed = collection.batch.failed_objects
        if failed:
            logger.warning("%d objects failed to insert: %s", len(failed), failed[0])

        logger.info(
            "Inserted %d products. Querying nearest neighbors (distance <= %.3f)...",
            len(code_to_uuid),
            distance_threshold,
        )

        pairs: list[tuple[str, str]] = []
        for code_a, uuid_a in code_to_uuid.items():
            response = collection.query.near_object(
                near_object=uuid_a,
                limit=7,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["sku_code"],
            )
            for obj in response.objects:
                if obj.uuid == uuid_a:
                    continue
                if obj.metadata.distance is None:
                    continue
                if obj.metadata.distance > distance_threshold:
                    continue
                code_b = str(obj.properties.get("sku_code", ""))
                if not code_b:
                    continue
                pair = (min(code_a, code_b), max(code_a, code_b))
                if pair not in pairs:
                    pairs.append(pair)

        return pairs

    finally:
        client.close()


# ---------------------------------------------------------------------------
# Union-Find canonical mapping
# ---------------------------------------------------------------------------

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

    result: dict[str, str] = {}
    for c in all_codes:
        result[c] = find(c)
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_product_matching(
    products_df: pd.DataFrame,
    embed_threshold: float = 0.92,
    use_weaviate: bool = False,
    weaviate_url: str = "http://localhost:8080",
    product_group_col: str = COL_PRODUCT_GROUP,
    sku_code_col: str = COL_SKU_CODE,
    sku_name_col: str = COL_SKU_NAME,
) -> dict[str, str]:
    """
    Run product matching via embeddings (default) or Weaviate (use_weaviate=True).

    Args:
        products_df: DataFrame with sku_code, sku_name, product_group.
                     One row per product (deduplicated).
        embed_threshold: Min cosine similarity (0-1) to consider a match.
        use_weaviate: If True, delegate embedding + search to Weaviate (requires
                      docker compose up -d). Default: False (in-memory embeddings).
        weaviate_url: HTTP URL of the Weaviate instance (only used when use_weaviate=True).

    Returns:
        sku_code -> canonical_sku_code. Unmatched SKUs map to themselves.
    """
    df = products_df.copy()
    df[sku_code_col] = df[sku_code_col].astype(str).str.strip()
    df[sku_name_col] = df[sku_name_col].fillna("").astype(str).str.strip()
    df[product_group_col] = df[product_group_col].fillna("Unknown").astype(str).str.strip()

    df["_norm"] = df[sku_name_col].apply(_lemmatize_ru)
    df = df[df["_norm"].str.len() > 1].copy()

    all_codes = set(df[sku_code_col].unique())

    if use_weaviate:
        pairs = _weaviate_match(df, "_norm", embed_threshold, weaviate_url)
        logger.info("Weaviate pass: %d pairs above threshold %.2f", len(pairs), embed_threshold)
    else:
        pairs = _embed_and_match(df, "_norm", embed_threshold)
        logger.info("Embedding pass: %d pairs above threshold %.2f", len(pairs), embed_threshold)

    mapping = _build_canonical_mapping(pairs, all_codes)

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
