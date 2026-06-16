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
        proxy: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.initial_delay = initial_delay

        # Optional HTTP/HTTPS proxy. When given, route the OpenAI client through a
        # custom httpx client; otherwise let openai use its default (which still
        # honours HTTP_PROXY/HTTPS_PROXY env vars).
        http_client = self._make_http_client(proxy) if proxy else None
        if http_client is not None:
            self._client = OpenAI(base_url=base_url, api_key=api_key, http_client=http_client)
        else:
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
        text_preview = text[:200].replace("\n", " ")
        delay = self.initial_delay
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                request = {"input": text, "model": self.model_name}
                rsp = self._client.embeddings.create(**request)
                self._log_raw(request, rsp)
                emb: List[float] = rsp.data[0].embedding  # type: ignore[index]
                self._insert(text_hash, text, emb)
                return emb
            except (RateLimitError, APIError) as e:
                last_error = e
                logger.warning(
                    f"Embedding API error (attempt {attempt + 1}/{self.max_retries}), "
                    f"retry in {delay:.1f}s | {self._describe_error(e)} | "
                    f"model={self.model_name} | text[:200]={text_preview!r}"
                )
            except Exception as e:
                last_error = e
                logger.exception(
                    f"Embedding unexpected error — abort | {self._describe_error(e)} | "
                    f"model={self.model_name} | text[:200]={text_preview!r}"
                )
                raise

            time.sleep(delay)
            delay *= 2

        logger.error(
            f"Embedding failed after {self.max_retries} retries | "
            f"last error: {self._describe_error(last_error)} | "
            f"model={self.model_name} | text[:200]={text_preview!r}"
        )
        raise RuntimeError(
            f"Failed to get embedding after {self.max_retries} retries: "
            f"{self._describe_error(last_error)}"
        ) from last_error

    @staticmethod
    def _describe_error(error: Optional[Exception]) -> str:
        """Build a detailed, single-line description of an embedding API error,
        pulling status code / request id / response body when available."""
        if error is None:
            return "unknown error"
        parts = [f"type={type(error).__name__}", f"msg={error}"]
        for attr in ("status_code", "code", "request_id"):
            value = getattr(error, attr, None)
            if value is not None:
                parts.append(f"{attr}={value}")
        body = getattr(error, "body", None)
        if body is not None:
            parts.append(f"body={str(body)[:500]}")
        response = getattr(error, "response", None)
        if response is not None:
            text = getattr(response, "text", None)
            if text:
                parts.append(f"response={str(text)[:500]}")
        # Network errors (e.g. openai.APIConnectionError "Connection error.") carry
        # no HTTP body; the real reason is the wrapped httpx/socket exception in the
        # __cause__/__context__ chain — walk it so the actual cause is visible.
        cause = error.__cause__ or error.__context__
        depth = 0
        seen: set = set()
        while cause is not None and depth < 5:
            entry = f"caused_by={type(cause).__name__}: {cause}"
            if entry not in seen:  # skip httpx->httpcore->socket duplicates
                seen.add(entry)
                parts.append(entry)
            cause = cause.__cause__ or cause.__context__
            depth += 1
        return " ".join(parts)

    def has(self, text: str) -> bool:
        """Return True if *text* is already cached (no API call)."""
        return self._select(hashlib.md5(text.encode()).hexdigest()) is not None

    @staticmethod
    def _make_http_client(proxy: str):
        """Build an httpx client routed through *proxy*, tolerant to httpx versions
        (httpx>=0.26 uses `proxy=`, older uses `proxies=`)."""
        import httpx
        try:
            return httpx.Client(proxy=proxy)
        except TypeError:
            return httpx.Client(proxies=proxy)

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
