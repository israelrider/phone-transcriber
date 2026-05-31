from __future__ import annotations

import os
from typing import Any

import httpx


def chat_completion(
    *,
    host: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_s: float = 300.0,
) -> str:
    url = f"{host.rstrip('/')}/api/chat"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }
    with httpx.Client(timeout=timeout_s) as client:
        r = client.post(url, headers=headers, json=body)
        r.raise_for_status()
        data = r.json()

    msg = data.get("message") or {}
    content = msg.get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected response shape: {data!r}")
    return content


def get_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    return key


def get_host() -> str:
    v = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    if not v:
        raise EnvironmentError("OPENAI_BASE_URL is not set.")
    return v.rstrip("/")


def get_model() -> str:
    v = (os.environ.get("OPENAI_MODEL") or "").strip()
    if not v:
        raise EnvironmentError("OPENAI_MODEL is not set.")
    return v
