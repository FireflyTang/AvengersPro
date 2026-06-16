"""
Shared configuration for the Avengers routing pipeline.

This is the ONLY place to change parameters. All three scripts
(generate_embeddings.py, train.py, inference.py) import from here.
There is no command-line argument parsing anywhere in the pipeline.
"""
import os

# --- Embedding service (user-provided, OpenAI-compatible, server auto-truncates) ---
# The endpoint is expected to be OpenAI-compatible (exposes /v1/embeddings) and
# to handle truncation of long inputs on the server side. We do NOT truncate here.
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:8000/v1")
EMBEDDING_API_KEY = os.getenv("EMBEDDING_API_KEY", "dummy-key")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "your-embedding-model")

EMBEDDING_MAX_WORKERS = 4              # concurrency (drives ThreadPoolExecutor)
EMBEDDING_MAX_RETRIES = 5              # automatic retries on API error
EMBEDDING_RETRY_INITIAL_DELAY = 1.0   # initial backoff seconds (exponential x2)

# Raw request/response audit log. Each real API call (cache miss) appends one
# JSONL line with the original request and the full original response.
# Set to None to disable. NOTE: raw responses contain full vectors -> large file.
EMBEDDING_RAW_LOG_PATH = ".cache/embedding_raw_log.jsonl"

CACHE_DIR = ".cache"                   # SQLite embedding cache directory

# --- Data paths ---
TRAIN_INPUT_PATH = "data/train.jsonl"      # {query, records:{model:acc}, dataset?, index?}
INFER_INPUT_PATH = "data/queries.jsonl"    # {query}  (batch debugging)
INFER_OUTPUT_PATH = "results/routing.jsonl"
MODEL_DIR = "models/router"                # training artifacts out / inference load dir

# --- Clustering & routing parameters ---
N_CLUSTERS = 32
SEED = 42
TRAIN_RATIO = 0.7    # <1.0 -> split & evaluate; >=1.0 -> train on all, skip evaluation
TOP_K = 3            # number of nearest clusters to aggregate at inference
BETA = 9.0           # softmax temperature over cluster distances (no effect when TOP_K=1)

# Models to exclude from ranking (optional). Anything listed here is dropped from
# the available-model set before training and never scored at inference.
EXCLUDED_MODELS = []
