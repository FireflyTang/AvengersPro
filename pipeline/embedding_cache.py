"""
EmbeddingCache (pipeline-local copy).

Thin wrapper around an OpenAI-compatible embedding endpoint with:
  - SQLite (WAL) on-disk caching keyed by md5(text) + model
  - exponential-backoff retry (configurable count / initial delay)
  - optional raw request/response audit logging to a JSONL file

This is a self-contained copy so that `pipeline/` does not depend on any
file outside the package. Truncation is intentionally NOT done here -- the
embedding endpoint is expected to truncate long inputs on the server side.
"""
from typing import List, Optional
from pathlib import Path
import sqlite3
import hashlib
import json
import os
import time
import threading
from datetime import datetime

from loguru import logger
from openai import OpenAI, RateLimitError, APIError

__all__ = ["EmbeddingCache"]


class EmbeddingCache:
    def __init__(
        self,
        base_url: str = "http://api.openai.com/v1",
        api_key: str = "sk-placeholder",
        model_name: str = "text-embedding-3-large",
        cache_dir: str | os.PathLike = ".cache",
        max_retries: int = 5,
        initial_delay: float = 1.0,
        raw_log_path: Optional[str | os.PathLike] = None,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_delay = initial_delay

        self._client = OpenAI(base_url=base_url, api_key=api_key)

        cache_path = Path(cache_dir)
        cache_path.mkdir(parents=True, exist_ok=True)
        self.db_path = cache_path / "embeddings.db"
        self._init_db()

        # Raw request/response audit log (optional).
        self.raw_log_path = Path(raw_log_path) if raw_log_path else None
        if self.raw_log_path is not None:
            self.raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_lock = threading.Lock()

    # ------------------------------------------------------------------
    # public helpers
    # ------------------------------------------------------------------
    def get(self, text: str) -> List[float]:
        """Return the embedding for *text*, fetching from cache or remote."""
        text_hash = hashlib.md5(text.encode()).hexdigest()

        # 1. try cache
        row = self._select(text_hash)
        if row is not None:
            return row

        # 2. call the embedding endpoint with exponential-backoff retry
        delay = self.initial_delay
        for attempt in range(self.max_retries):
            try:
                request = {"input": text, "model": self.model_name}
                rsp = self._client.embeddings.create(**request)
                self._log_raw(request, rsp)
                emb: List[float] = rsp.data[0].embedding  # type: ignore[index]
                self._insert(text_hash, text, emb)
                return emb
            except RateLimitError:
                logger.warning(
                    f"Rate limited (attempt {attempt + 1}/{self.max_retries}). Retry in {delay:.1f}s"
                )
            except APIError as e:
                logger.warning(
                    f"API error (attempt {attempt + 1}/{self.max_retries}): {e}. Retry in {delay:.1f}s"
                )
            except Exception as e:
                logger.error(f"Unexpected error — abort: {e}")
                raise

            time.sleep(delay)
            delay *= 2

        raise RuntimeError(f"Failed to get embedding after {self.max_retries} retries.")

    def has(self, text: str) -> bool:
        """Return True if *text* is already cached (no API call)."""
        return self._select(hashlib.md5(text.encode()).hexdigest()) is not None

    # ------------------------------------------------------------------
    # raw audit logging
    # ------------------------------------------------------------------
    def _log_raw(self, request: dict, response) -> None:
        """Append the raw request and raw response of a real API call to JSONL."""
        if self.raw_log_path is None:
            return
        try:
            response_dump = response.model_dump()
        except Exception:
            response_dump = str(response)
        entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model_name,
            "request": request,
            "response": response_dump,
        }
        line = json.dumps(entry, ensure_ascii=False)
        with self._log_lock:
            with open(self.raw_log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # ------------------------------------------------------------------
    # private db helpers
    # ------------------------------------------------------------------
    def _init_db(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS embeddings (
                    text_hash TEXT,
                    model     TEXT,
                    embedding TEXT NOT NULL,
                    text      TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY(text_hash, model)
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON embeddings(model);")

    def _select(self, text_hash: str) -> List[float] | None:
        with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
            row = conn.execute(
                "SELECT embedding FROM embeddings WHERE text_hash=? AND model=?",
                (text_hash, self.model_name),
            ).fetchone()
            if row:
                return json.loads(row[0])
            return None

    def _insert(self, text_hash: str, text: str, embedding: List[float]) -> None:
        with sqlite3.connect(self.db_path, timeout=30, check_same_thread=False) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO embeddings (text_hash, model, embedding, text) VALUES (?,?,?,?)",
                (text_hash, self.model_name, json.dumps(embedding), text),
            )
