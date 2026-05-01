from __future__ import annotations

import hashlib
import os
import sqlite3
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv


MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 1024
TEMPERATURE = 0
CACHE_PATH = Path(__file__).resolve().parent / ".cache" / "llm.sqlite"


class LLMUnavailable(RuntimeError):
    pass


def _cache_key(stage: str, system_prompt: str, user_message: str, model: str) -> str:
    payload = "\n\n".join([stage, model, system_prompt, user_message])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _connect() -> sqlite3.Connection:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(CACHE_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS llm_cache (
            cache_key TEXT PRIMARY KEY,
            stage TEXT NOT NULL,
            model TEXT NOT NULL,
            response TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    return conn


def has_api_key() -> bool:
    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def call_claude(
    stage: str,
    system_prompt: str,
    user_message: str,
    *,
    model: str = MODEL,
    max_tokens: int = MAX_TOKENS,
    temperature: int = TEMPERATURE,
    use_cache: bool = True,
) -> str:
    key = _cache_key(stage, system_prompt, user_message, model)
    if use_cache:
        with _connect() as conn:
            row = conn.execute("SELECT response FROM llm_cache WHERE cache_key = ?", (key,)).fetchone()
            if row:
                return str(row[0])

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise LLMUnavailable("ANTHROPIC_API_KEY is not set")

    try:
        import anthropic
    except ImportError as exc:
        raise LLMUnavailable("anthropic package is not installed") from exc

    client = anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text.strip()

    if use_cache:
        with _connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO llm_cache (cache_key, stage, model, response, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (key, stage, model, text, datetime.now().astimezone().isoformat()),
            )
    return text
