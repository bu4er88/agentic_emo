"""
Sex-specific personality profiles.

These are the *base* system prompts for the conscious supervisor.
Prompt text is loaded from .md files next to this module.
"""

from __future__ import annotations

from pathlib import Path

_DIR = Path(__file__).parent


def _load(name: str) -> str:
    return (_DIR / name).read_text().strip()


MALE_PROFILE = {
    "sex": "male",
    "base_system": _load("male_base_system.md"),
    "thinking_style": _load("male_thinking_style.md"),
    "social_style": _load("male_social_style.md"),
}

FEMALE_PROFILE = {
    "sex": "female",
    "base_system": _load("female_base_system.md"),
    "thinking_style": _load("female_thinking_style.md"),
    "social_style": _load("female_social_style.md"),
}


def get_profile(sex: str) -> dict:
    """Return the profile dict for the given sex."""
    if sex.lower() in ("m", "male"):
        return MALE_PROFILE
    if sex.lower() in ("f", "female"):
        return FEMALE_PROFILE
    raise ValueError(f"Unknown sex: {sex!r}. Use 'male' or 'female'.")
