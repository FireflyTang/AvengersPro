"""
Script 2 — Avengers router training.

Loads training data + cached embeddings, fits K-means, computes a per-cluster
mean-accuracy table for every model, optionally evaluates on a held-out split,
and saves the model artifacts for inference.

Run:  python pipeline/train.py
All parameters come from pipeline/config.py (no command-line arguments).
"""
import json
import os
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import Normalizer
from tqdm import tqdm

try:
    from . import config
    from ._common import load_dataset, split_by_dataset, predict_scores
    from .embedding_cache import EmbeddingCache
except ImportError:
    import config
    from _common import load_dataset, split_by_dataset, predict_scores
    from embedding_cache import EmbeddingCache


def _embed(embedder: EmbeddingCache, queries: List[str]) -> np.ndarray:
    """Fetch embeddings (cache-first) for all queries, concurrently."""
    results: List = [None] * len(queries)
    with ThreadPoolExecutor(max_workers=config.EMBEDDING_MAX_WORKERS) as executor:
        futures = {executor.submit(embedder.get, q): i for i, q in enumerate(queries)}
        for future in tqdm(futures, total=len(futures), desc="Loading embeddings"):
            results[futures[future]] = future.result()
    return np.array(results, dtype=np.float64)


def _compute_cluster_scores(
    labels: np.ndarray, records: List[Dict[str, float]], available_models: List[str]
) -> Dict[int, Dict[str, float]]:
    """Per-cluster mean accuracy for each model: cluster_scores[cluster][model]."""
    by_cluster: Dict[int, List[Dict[str, float]]] = defaultdict(list)
    for i, c in enumerate(labels):
        by_cluster[int(c)].append(records[i])

    cluster_scores: Dict[int, Dict[str, float]] = {}
    for cluster_id, recs in by_cluster.items():
        means = {}
        for m in available_models:
            vals = [r[m] for r in recs if m in r]
            means[m] = float(np.mean(vals)) if vals else 0.0
        cluster_scores[cluster_id] = means
    return cluster_scores


def _evaluate(test_items, norm_test, centers, cluster_scores, available_models):
    """Route each test query to its argmax-predicted model; report quality."""
    preds = predict_scores(
        norm_test, centers, cluster_scores, available_models, config.TOP_K, config.BETA
    )

    per_dataset_correct: Dict[str, float] = defaultdict(float)
    per_dataset_total: Dict[str, int] = defaultdict(int)
    selection = Counter()
    abs_errors: List[float] = []
    regrets: List[float] = []

    for item, pred in zip(test_items, preds):
        best_model = next(iter(pred))  # highest predicted accuracy
        true = item["records"]
        ds = item["dataset"]

        per_dataset_correct[ds] += true.get(best_model, 0.0)
        per_dataset_total[ds] += 1
        selection[best_model] += 1

        # prediction quality: |predicted - true| averaged over models present
        for m, p in pred.items():
            if m in true:
                abs_errors.append(abs(p - true[m]))
        regrets.append(max(true.values()) - true.get(best_model, 0.0))

    dataset_acc = {
        ds: per_dataset_correct[ds] / per_dataset_total[ds]
        for ds in per_dataset_total
    }
    overall = sum(dataset_acc.values()) / len(dataset_acc) if dataset_acc else 0.0

    print("\n" + "=" * 60)
    print("EVALUATION (held-out)")
    print("=" * 60)
    print(f"Overall routing accuracy (avg over datasets): {overall:.4f}")
    print("\nPer-dataset routing accuracy:")
    for ds in sorted(dataset_acc):
        print(f"  {ds:20s}: {dataset_acc[ds]:.4f}  ({per_dataset_total[ds]} queries)")
    print("\nRouting choice distribution (argmax model):")
    total_sel = sum(selection.values())
    for m, c in selection.most_common():
        print(f"  {m:30s}: {c:4d} ({c / total_sel * 100:5.1f}%)")
    print(f"\nPrediction MAE (predicted vs true accuracy): {np.mean(abs_errors):.4f}")
    print(f"Mean routing regret (best-possible − routed): {np.mean(regrets):.4f}")


def _save_artifacts(model_dir, normalizer, centers, cluster_scores, available_models):
    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)

    joblib.dump(normalizer, out / "normalizer.joblib")
    np.save(out / "cluster_centers.npy", centers)

    # cluster ids stored as strings; ranking is models sorted by mean accuracy desc
    rankings = {}
    for cluster_id, scores in cluster_scores.items():
        ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        rankings[str(cluster_id)] = {
            "scores": dict(ordered),
            "ranking": [m for m, _ in ordered],
        }
    with open(out / "cluster_rankings.json", "w", encoding="utf-8") as f:
        json.dump(rankings, f, indent=2, ensure_ascii=False)

    metadata = {
        "available_models": available_models,
        "embedding_model": config.EMBEDDING_MODEL,
        "n_clusters": config.N_CLUSTERS,
        "top_k": config.TOP_K,
        "beta": config.BETA,
        "seed": config.SEED,
        "timestamp": datetime.now().isoformat(),
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nArtifacts saved to: {out}")
    print("  normalizer.joblib, cluster_centers.npy, cluster_rankings.json, metadata.json")


def _print_training_summary(labels, cluster_scores, available_models, records):
    counts = Counter(int(c) for c in labels)
    print("\n" + "=" * 60)
    print(f"TRAINING SUMMARY (n_clusters={config.N_CLUSTERS})")
    print("=" * 60)
    print("Cluster size distribution & top model:")
    for cluster_id in sorted(cluster_scores):
        scores = cluster_scores[cluster_id]
        top_model, top_acc = max(scores.items(), key=lambda x: x[1])
        print(f"  cluster {cluster_id:3d}: {counts[cluster_id]:4d} samples | "
              f"top: {top_model} ({top_acc:.3f})")

    print("\nOverall mean accuracy per model (train set):")
    overall = {
        m: float(np.mean([r[m] for r in records if m in r])) for m in available_models
    }
    for m, a in sorted(overall.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m:30s}: {a:.4f}")


def main() -> None:
    items, available_models = load_dataset(config.TRAIN_INPUT_PATH, config.EXCLUDED_MODELS)
    print(f"Loaded {len(items)} items | {len(available_models)} models: {available_models}")

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

    # Decide split: TRAIN_RATIO >= 1.0 -> train on everything, no evaluation.
    do_eval = config.TRAIN_RATIO < 1.0
    if do_eval:
        train_items, test_items = split_by_dataset(items, config.TRAIN_RATIO, config.SEED)
        print(f"Split: {len(train_items)} train / {len(test_items)} test")
    else:
        train_items, test_items = items, []
        print("TRAIN_RATIO >= 1.0 -> training on all data (no evaluation)")

    train_emb = _embed(embedder, [it["query"] for it in train_items])
    normalizer = Normalizer(norm="l2")
    norm_train = normalizer.fit_transform(train_emb)

    print(f"Fitting K-means (k={config.N_CLUSTERS}) on {norm_train.shape} embeddings...")
    kmeans = KMeans(n_clusters=config.N_CLUSTERS, random_state=config.SEED, n_init=10)
    labels = kmeans.fit_predict(norm_train)
    centers = kmeans.cluster_centers_

    train_records = [it["records"] for it in train_items]
    cluster_scores = _compute_cluster_scores(labels, train_records, available_models)
    _print_training_summary(labels, cluster_scores, available_models, train_records)

    if do_eval and test_items:
        test_emb = _embed(embedder, [it["query"] for it in test_items])
        norm_test = normalizer.transform(test_emb)
        _evaluate(test_items, norm_test, centers, cluster_scores, available_models)

    _save_artifacts(config.MODEL_DIR, normalizer, centers, cluster_scores, available_models)


if __name__ == "__main__":
    main()
