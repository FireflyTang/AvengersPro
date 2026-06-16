"""
Script 3 — Online inference.

Exposes a thread-safe `Router` class that, given a query, returns the predicted
correctness (accuracy) for EVERY model — it does not pick a single "best" model.
Also runs as a script for batch debugging over a JSONL file.

Programmatic use:
    from pipeline.inference import Router
    router = Router()                  # loads artifacts once
    scores = router.predict("some query")   # {model: predicted_accuracy}, sorted desc

Batch debug:
    python pipeline/inference.py       # reads INFER_INPUT_PATH -> INFER_OUTPUT_PATH

Thread safety: after construction, `predict` only reads immutable arrays loaded at
init and performs stateless NumPy math; it never mutates `self`. EmbeddingCache uses
a thread-safe OpenAI client and per-operation SQLite connections (check_same_thread=
False, WAL), so concurrent `predict` calls from multiple threads are safe.
"""
import json
import os
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np

try:
    from . import config
    from ._common import predict_scores
    from .embedding_cache import EmbeddingCache
except ImportError:
    import config
    from _common import predict_scores
    from embedding_cache import EmbeddingCache


class Router:
    """Loads trained artifacts and scores every model for a query (thread-safe)."""

    def __init__(self, model_dir: str = None):
        model_dir = model_dir or config.MODEL_DIR
        path = Path(model_dir)
        if not path.exists():
            raise FileNotFoundError(f"Model dir not found: {path}. Run train.py first.")

        self.normalizer = joblib.load(path / "normalizer.joblib")
        self.centers = np.load(path / "cluster_centers.npy")

        with open(path / "cluster_rankings.json", "r", encoding="utf-8") as f:
            rankings = json.load(f)
        # cluster ids are stored as strings -> cast back to int
        self.cluster_scores: Dict[int, Dict[str, float]] = {
            int(cid): data["scores"] for cid, data in rankings.items()
        }

        with open(path / "metadata.json", "r", encoding="utf-8") as f:
            metadata = json.load(f)
        self.available_models: List[str] = metadata["available_models"]
        self.top_k = metadata.get("top_k", config.TOP_K)
        self.beta = metadata.get("beta", config.BETA)
        embedding_model = metadata.get("embedding_model", config.EMBEDDING_MODEL)

        self.embedder = EmbeddingCache(
            base_url=config.EMBEDDING_BASE_URL,
            api_key=config.EMBEDDING_API_KEY,
            model_name=embedding_model,
            cache_dir=config.CACHE_DIR,
            max_retries=config.EMBEDDING_MAX_RETRIES,
            initial_delay=config.EMBEDDING_RETRY_INITIAL_DELAY,
            raw_log_path=config.EMBEDDING_RAW_LOG_PATH,
            proxy=config.EMBEDDING_PROXY,
        )

    def predict(self, query: str) -> Dict[str, float]:
        """Return {model: predicted_accuracy} for one query, sorted descending."""
        return self.predict_batch([query])[0]

    def predict_batch(self, queries: List[str]) -> List[Dict[str, float]]:
        """Vectorized prediction for a list of queries."""
        embeddings = np.array([self.embedder.get(q) for q in queries], dtype=np.float64)
        norm = self.normalizer.transform(embeddings)
        return predict_scores(
            norm, self.centers, self.cluster_scores,
            self.available_models, self.top_k, self.beta,
        )


def main() -> None:
    if not os.path.exists(config.INFER_INPUT_PATH):
        raise FileNotFoundError(f"Inference input not found: {config.INFER_INPUT_PATH}")

    queries: List[str] = []
    with open(config.INFER_INPUT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            q = json.loads(line).get("query")
            if isinstance(q, str) and q.strip():
                queries.append(q)

    router = Router()
    preds = router.predict_batch(queries)

    out_path = Path(config.INFER_OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for q, scores in zip(queries, preds):
            f.write(json.dumps({"query": q, "predicted_accuracy": scores}, ensure_ascii=False) + "\n")

    # Console debug: print every model's predicted accuracy per query.
    for q, scores in zip(queries, preds):
        print(f"\nQuery: {q[:80]}")
        for m, s in scores.items():
            print(f"  {m:30s}: {s:.4f}")

    print(f"\nWrote {len(queries)} predictions to: {out_path}")


if __name__ == "__main__":
    main()
