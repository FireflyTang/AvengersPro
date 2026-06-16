"""
Script 1 — Embedding generation.

Reads the training data (and, if present, the inference query file), extracts
every query, and embeds them concurrently into the SQLite cache. Long inputs are
NOT truncated here — the embedding endpoint handles that server-side.

Run:  python pipeline/generate_embeddings.py
All parameters come from pipeline/config.py (no command-line arguments).
"""
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm

try:  # works both as `python pipeline/generate_embeddings.py` and `-m pipeline...`
    from . import config
    from .embedding_cache import EmbeddingCache
except ImportError:
    import config
    from embedding_cache import EmbeddingCache


def _read_queries(path: str) -> list[str]:
    """Extract the 'query' field from every line of a JSONL file (if it exists)."""
    if not os.path.exists(path):
        return []
    queries: list[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            q = item.get("query")
            if isinstance(q, str) and q.strip():
                queries.append(q)
    return queries


def main() -> None:
    embedder = EmbeddingCache(
        base_url=config.EMBEDDING_BASE_URL,
        api_key=config.EMBEDDING_API_KEY,
        model_name=config.EMBEDDING_MODEL,
        cache_dir=config.CACHE_DIR,
        max_retries=config.EMBEDDING_MAX_RETRIES,
        initial_delay=config.EMBEDDING_RETRY_INITIAL_DELAY,
        raw_log_path=config.EMBEDDING_RAW_LOG_PATH,
        proxy=config.EMBEDDING_PROXY,
    )

    # Collect unique queries from train + (optional) inference files.
    queries = _read_queries(config.TRAIN_INPUT_PATH) + _read_queries(config.INFER_INPUT_PATH)
    unique = list(dict.fromkeys(queries))  # de-dup, preserve order
    if not unique:
        raise RuntimeError(
            f"No queries found in {config.TRAIN_INPUT_PATH} or {config.INFER_INPUT_PATH}"
        )

    already = sum(1 for q in unique if embedder.has(q))
    to_fetch = len(unique) - already
    print(f"Model: {config.EMBEDDING_MODEL}")
    print(f"Unique queries: {len(unique)} | cached: {already} | to fetch: {to_fetch}")
    print(f"Concurrency: {config.EMBEDDING_MAX_WORKERS} workers, "
          f"max_retries={config.EMBEDDING_MAX_RETRIES}")

    start = time.time()
    with ThreadPoolExecutor(max_workers=config.EMBEDDING_MAX_WORKERS) as executor:
        futures = {executor.submit(embedder.get, q): q for q in unique}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Embedding"):
            future.result()  # propagate errors
    elapsed = time.time() - start

    print(f"\nDone. Embedded {len(unique)} queries in {elapsed:.1f}s.")
    print(f"SQLite cache: {os.path.join(config.CACHE_DIR, 'embeddings.db')}")
    if config.EMBEDDING_RAW_LOG_PATH:
        print(f"Raw request/response log: {config.EMBEDDING_RAW_LOG_PATH}")


if __name__ == "__main__":
    main()
