"""
Thin wrapper around the OpenAI-compatible chat completions API.

Supports any provider that exposes the OpenAI API shape (OpenAI, Together,
Ollama, LM Studio, vLLM, etc.) via base_url + api_key.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from openai import OpenAI


@dataclass
class LLMConfig:
    """Configuration for one LLM endpoint."""
    model: str
    base_url: str | None = None
    api_key: str | None = None
    temperature: float = 0.7
    max_tokens: int = 1024

    def __post_init__(self) -> None:
        if self.api_key is None:
            self.api_key = os.environ.get("OPENAI_API_KEY", "no-key")
        if self.base_url is None:
            self.base_url = os.environ.get("OPENAI_BASE_URL")


# ── pre-baked configs (users can override via env or constructor) ─────

def fast_model() -> LLMConfig:
    """Cheap/fast model for reflexes (unconscious supervisor sub-agents)."""
    return LLMConfig(
        model=os.environ.get("AGENTIC_FAST_MODEL", "gpt-4.1-mini"),
        temperature=0.3,
        max_tokens=256,
    )


def strong_model() -> LLMConfig:
    """Powerful model for consciousness (conscious supervisor)."""
    return LLMConfig(
        model=os.environ.get("AGENTIC_STRONG_MODEL", "gpt-4.1"),
        temperature=0.7,
        max_tokens=2048,
    )


# ── call helper ──────────────────────────────────────────────────────

def chat(
    config: LLMConfig,
    system: str,
    user: str,
    history: list[dict] | None = None,
) -> str:
    """Single-turn (or multi-turn with history) chat completion."""
    client = OpenAI(
        api_key=config.api_key,
        base_url=config.base_url,
    )
    messages: list[dict] = [{"role": "system", "content": system}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user})

    resp = client.chat.completions.create(
        model=config.model,
        messages=messages,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    return resp.choices[0].message.content or ""


def chat_json(
    config: LLMConfig,
    system: str,
    user: str,
) -> dict:
    """Chat completion expecting a JSON response. Returns parsed dict."""
    raw = chat(config, system, user)
    # Strip markdown code fences if present
    text = raw.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last lines (fences)
        lines = [l for l in lines if not l.strip().startswith("```")]
        text = "\n".join(lines)
    return json.loads(text)
