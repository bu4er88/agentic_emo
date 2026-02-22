"""
Reflex prompt templates for the unconscious supervisor.

Each reflex is a small, cheap sub-agent that:
  1. Receives a stimulus.
  2. Decides in one shot whether the reflex fires.
  3. Returns which emotion(s) to spike and what motor response to emit.

Prompt text is loaded from .md files next to this module.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def _load(name: str) -> str:
    return (_DIR / name).read_text().strip()


REFLEX_PROMPTS: dict[str, dict] = {
    "fight_or_flight": {
        "system": _load("reflex_fight_or_flight.md"),
        "emotions": ["fear", "anger"],
    },
    "startle": {
        "system": _load("reflex_startle.md"),
        "emotions": ["surprise", "fear"],
    },
    "disgust_withdrawal": {
        "system": _load("reflex_disgust_withdrawal.md"),
        "emotions": ["disgust"],
    },
    "social_bonding": {
        "system": _load("reflex_social_bonding.md"),
        "emotions": ["joy", "trust"],
    },
    "curiosity_orienting": {
        "system": _load("reflex_curiosity_orienting.md"),
        "emotions": ["anticipation", "surprise"],
    },
}

SUPERVISOR_UNCONSCIOUS_SYSTEM = _load("supervisor_unconscious.md")
