# Avengers Routing Pipeline

A self-contained, three-stage refactor of the Avengers router for the
**accuracy-only** use case (cost is ignored). Training data is
`{query, 每个模型答对的概率/正确率}`; inference outputs the **predicted
correctness of every model** for a query (it does not pick a single best model).

All parameters live in `config.py` — there are no command-line arguments.
`pipeline/` does not import anything from the repository's legacy files.

## Stages

| Stage | Script | What it does |
|-------|--------|--------------|
| 1. Embedding | `generate_embeddings.py` | Embeds every query into the SQLite cache (concurrent, retrying, with raw request/response logging). Endpoint auto-truncates long inputs. |
| 2. Training | `train.py` | L2-normalize → K-means → per-cluster mean-accuracy table per model; optional held-out evaluation; saves artifacts to `MODEL_DIR`. |
| 3. Inference | `inference.py` | Thread-safe `Router` class returning `{model: 预测正确率}`; also a `__main__` batch-debug mode. |

## Quick start

1. Edit `pipeline/config.py`:
   - `EMBEDDING_BASE_URL` / `EMBEDDING_API_KEY` / `EMBEDDING_MODEL` — your OpenAI-compatible endpoint.
   - `TRAIN_INPUT_PATH`, `N_CLUSTERS`, `TRAIN_RATIO`, `TOP_K`, `BETA`, etc.
2. Run the stages:
   ```bash
   python pipeline/generate_embeddings.py   # populate .cache/embeddings.db
   python pipeline/train.py                 # writes models/router/
   python pipeline/inference.py             # reads data/queries.jsonl -> results/routing.jsonl
   ```
3. Or use the router programmatically (thread-safe):
   ```python
   from pipeline.inference import Router
   router = Router()
   scores = router.predict("your query")    # {model: predicted_accuracy}, sorted desc
   ```

## Input formats

Training (`TRAIN_INPUT_PATH`, JSONL) — `usages`/cost is ignored:
```json
{"query": "...", "records": {"model_A": 0.95, "model_B": 0.40}, "dataset": "math", "index": 0}
```
Inference (`INFER_INPUT_PATH`, JSONL):
```json
{"query": "..."}
```

## Key parameters (`config.py`)

| Param | Meaning |
|-------|---------|
| `N_CLUSTERS` | K-means cluster count |
| `TRAIN_RATIO` | `<1.0` → split & evaluate; `>=1.0` → train on all, skip eval |
| `TOP_K` | nearest clusters aggregated at inference |
| `BETA` | softmax temperature over cluster distances (no effect when `TOP_K=1`) |
| `EMBEDDING_MAX_WORKERS` | embedding concurrency |
| `EMBEDDING_MAX_RETRIES` / `EMBEDDING_RETRY_INITIAL_DELAY` | retry policy (exponential backoff) |
| `EMBEDDING_RAW_LOG_PATH` | JSONL audit log of raw request/response (`None` to disable) |
| `EMBEDDING_PROXY` | optional HTTP/HTTPS proxy URL for the embedding endpoint (`None` = no proxy; also honors `HTTP_PROXY`/`HTTPS_PROXY` env) |
| `EMBEDDING_VERIFY_SSL` | verify the endpoint's HTTPS certificate; set `False` for self-signed/internal endpoints |

## Predicted correctness

`pred_acc[model] = Σ over TOP_K nearest clusters of softmax(-BETA·distance) ×
(that model's mean accuracy in the cluster)`. The output is an interpretable
`[0,1]` correctness estimate per model, not a rank-based preference score.
