from __future__ import annotations

import os
from pathlib import Path

from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def load_dotenv_near_repo() -> None:
    """Load .env from the repo cwd and its parent without overwriting existing env vars."""
    candidates = [Path.cwd() / ".env", Path.cwd().parent / ".env"]
    seen: set[Path] = set()
    for path in candidates:
        path = path.resolve()
        if path in seen or not path.exists():
            continue
        seen.add(path)
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def normalize_model_for_provider(model: str, provider: str) -> str:
    if provider == "gemini" and model.startswith("google/"):
        return model.split("/", 1)[1]
    return model


def build_openai_client(
    model: str,
    *,
    provider: str = "auto",
) -> tuple[OpenAI, str, str] | None:
    """Return (client, model_name, provider_name) or None if no usable credentials exist."""
    load_dotenv_near_repo()
    wants_gemini = "gemini" in model.lower()

    if provider not in {"auto", "gemini", "openrouter"}:
        raise ValueError(f"Unsupported provider '{provider}'")

    if provider in {"auto", "gemini"} and wants_gemini:
        api_key = os.environ.get("GEMINI_API_KEY", "")
        if api_key:
            return (
                OpenAI(api_key=api_key, base_url=GEMINI_OPENAI_BASE_URL),
                normalize_model_for_provider(model, "gemini"),
                "gemini",
            )
        if provider == "gemini":
            raise RuntimeError("GEMINI_API_KEY is not set")

    if provider in {"auto", "openrouter"}:
        api_key = os.environ.get("OPENROUTER_API_KEY", "")
        if api_key:
            return (
                OpenAI(api_key=api_key, base_url=OPENROUTER_BASE_URL),
                normalize_model_for_provider(model, "openrouter"),
                "openrouter",
            )
        if provider == "openrouter":
            raise RuntimeError("OPENROUTER_API_KEY is not set")

    return None
