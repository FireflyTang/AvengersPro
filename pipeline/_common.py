"""
Shared helpers for the Avengers routing pipeline (data loading, dataset split,
and the prediction-aggregation math). Kept inside the package so the pipeline
stays self-contained.
"""
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np


def normalize_record_value(value) -> float:
    """Coerce a per-model record into a float in the spirit of the original loader.

    bool -> 1.0/0.0, None -> 0.0, int/float -> float. Raises on anything else.
    """
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"record value must be numeric/bool/null, got {type(value)}")


def load_dataset(path: str, excluded_models: List[str]) -> Tuple[List[Dict], List[str]]:
    """Load a JSONL dataset of {query, records, dataset?, index?}.

    Returns (items, available_models). `available_models` is taken from the first
    valid item's records minus `excluded_models`. Items missing any available
    model are skipped (consistent with the original behaviour).
    """
    excluded = set(excluded_models or [])
    raw: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if "query" not in item or "records" not in item:
                raise ValueError(f"line {line_num}: missing 'query' or 'records'")
            if not isinstance(item["query"], str) or not item["query"].strip():
                raise ValueError(f"line {line_num}: 'query' must be a non-empty string")
            if not isinstance(item["records"], dict) or not item["records"]:
                raise ValueError(f"line {line_num}: 'records' must be a non-empty dict")
            records = {m: normalize_record_value(v) for m, v in item["records"].items()}
            raw.append(
                {
                    "query": item["query"],
                    "records": records,
                    "dataset": item.get("dataset", "default"),
                    "index": item.get("index", -1),
                }
            )

    if not raw:
        raise RuntimeError(f"no valid data found in {path}")

    available_models = [m for m in raw[0]["records"].keys() if m not in excluded]
    if not available_models:
        raise RuntimeError("no models left after applying EXCLUDED_MODELS")

    items: List[Dict] = []
    for item in raw:
        missing = set(available_models) - set(item["records"].keys())
        if missing:
            continue  # skip rows that don't cover every available model
        items.append(item)

    return items, available_models


def split_by_dataset(
    items: List[Dict], train_ratio: float, seed: int
) -> Tuple[List[Dict], List[Dict]]:
    """Per-dataset shuffle + split into (train, test). Mirrors the original
    balanced split without depending on the `datasets` library."""
    by_ds: Dict[str, List[Dict]] = defaultdict(list)
    for item in items:
        by_ds[item["dataset"]].append(item)

    rng = random.Random(seed)
    train: List[Dict] = []
    test: List[Dict] = []
    for ds in sorted(by_ds.keys()):
        group = sorted(by_ds[ds], key=lambda x: x["index"])
        rng.shuffle(group)
        n_train = int(round(len(group) * train_ratio))
        # keep at least one sample on each side when the group is splittable
        n_train = min(max(n_train, 1), len(group) - 1) if len(group) > 1 else len(group)
        train.extend(group[:n_train])
        test.extend(group[n_train:])
    return train, test


def predict_scores(
    norm_embeddings: np.ndarray,
    centers: np.ndarray,
    cluster_scores: Dict[int, Dict[str, float]],
    available_models: List[str],
    top_k: int,
    beta: float,
) -> List[Dict[str, float]]:
    """Aggregate per-cluster mean accuracies into a per-model predicted accuracy.

    For each (L2-normalized) query embedding: take the TOP_K nearest clusters by
    cosine distance, weight them by softmax(-beta * distance), and compute
    pred_acc[model] = sum_k prob_k * cluster_scores[k][model].

    Returns one dict per row, sorted by score descending.
    """
    distances = 1.0 - norm_embeddings @ centers.T  # (N, C)
    n_clusters = centers.shape[0]
    k = min(top_k, n_clusters)

    results: List[Dict[str, float]] = []
    for i in range(norm_embeddings.shape[0]):
        d = distances[i]
        closest = np.argsort(d)[:k]
        logits = -beta * d[closest]
        probs = np.exp(logits - logits.max())
        probs /= probs.sum()

        scores = {m: 0.0 for m in available_models}
        for cluster_idx, prob in zip(closest, probs):
            cs = cluster_scores.get(int(cluster_idx))
            if not cs:
                continue
            for m in available_models:
                scores[m] += float(prob) * cs.get(m, 0.0)

        results.append(dict(sorted(scores.items(), key=lambda x: x[1], reverse=True)))
    return results
