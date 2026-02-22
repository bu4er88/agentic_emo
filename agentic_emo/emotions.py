"""
Emotion Engine — dynamic prompt variables that shape the mind.

Each emotion is a small prompt fragment with an intensity (0.0–1.0).
Emotions decay naturally over time and can be spiked (by reflexes) or
gently shifted (by consciousness).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ── emotion definitions ──────────────────────────────────────────────

EMOTION_PROMPTS: dict[str, str] = {
    "joy": (
        "You feel a warm surge of happiness. The world seems brighter, "
        "problems feel solvable, and you're inclined to be generous and open."
    ),
    "fear": (
        "A cold grip of fear tightens around you. You are hyper-alert, "
        "scanning for threats, and your thoughts race toward escape or safety."
    ),
    "anger": (
        "Hot anger rises in you. You feel your boundaries have been violated "
        "and you want to assert yourself forcefully, perhaps aggressively."
    ),
    "sadness": (
        "A heavy weight of sadness settles over you. Energy drains away, "
        "the world feels muted, and you turn inward seeking comfort or meaning."
    ),
    "surprise": (
        "A jolt of surprise seizes you. Your attention snaps to the unexpected "
        "stimulus and all other thoughts pause momentarily."
    ),
    "disgust": (
        "A wave of revulsion washes over you. You want to withdraw, reject, "
        "or push away whatever triggered this feeling."
    ),
    "trust": (
        "A calm sense of trust and safety fills you. You feel connected, "
        "willing to cooperate, and open to vulnerability."
    ),
    "anticipation": (
        "Eager anticipation buzzes through you. You lean forward mentally, "
        "planning and predicting, hungry for what comes next."
    ),
}

# How fast emotions decay per second (half-life style).
# Higher = faster decay. Reflexive spikes decay fast; conscious shifts slow.
DEFAULT_DECAY_RATE = 0.05  # lose ~5 % intensity per second


@dataclass
class EmotionState:
    """A single emotion with intensity and timing metadata."""

    name: str
    intensity: float = 0.0          # 0.0 = absent … 1.0 = overwhelming
    decay_rate: float = DEFAULT_DECAY_RATE
    last_updated: float = field(default_factory=time.time)

    def tick(self) -> None:
        """Apply time-based decay."""
        now = time.time()
        elapsed = now - self.last_updated
        self.intensity *= max(0.0, 1.0 - self.decay_rate * elapsed)
        if self.intensity < 0.01:
            self.intensity = 0.0
        self.last_updated = now

    def spike(self, amount: float, decay_rate: float | None = None) -> None:
        """Short, powerful burst (used by reflexes)."""
        self.intensity = min(1.0, self.intensity + amount)
        if decay_rate is not None:
            self.decay_rate = decay_rate
        else:
            self.decay_rate = 0.15  # fast decay for reflex spikes
        self.last_updated = time.time()

    def shift(self, amount: float, decay_rate: float | None = None) -> None:
        """Gentle, long-lasting adjustment (used by consciousness)."""
        self.intensity = max(0.0, min(1.0, self.intensity + amount))
        if decay_rate is not None:
            self.decay_rate = decay_rate
        else:
            self.decay_rate = 0.01  # slow decay for conscious shifts
        self.last_updated = time.time()


class EmotionEngine:
    """Manages the full emotional state and renders it as a prompt fragment."""

    def __init__(self) -> None:
        self.emotions: dict[str, EmotionState] = {
            name: EmotionState(name=name) for name in EMOTION_PROMPTS
        }

    # ── mutators ─────────────────────────────────────────────────────

    def spike(self, emotion: str, amount: float, decay_rate: float | None = None) -> None:
        if emotion in self.emotions:
            self.emotions[emotion].spike(amount, decay_rate)

    def shift(self, emotion: str, amount: float, decay_rate: float | None = None) -> None:
        if emotion in self.emotions:
            self.emotions[emotion].shift(amount, decay_rate)

    def tick_all(self) -> None:
        for e in self.emotions.values():
            e.tick()

    # ── prompt rendering ─────────────────────────────────────────────

    def render_prompt(self, threshold: float = 0.05) -> str:
        """Build the dynamic emotional context injected into Supervisor-2."""
        self.tick_all()
        active = sorted(
            [(e.name, e.intensity) for e in self.emotions.values() if e.intensity >= threshold],
            key=lambda x: -x[1],
        )
        if not active:
            return "You are emotionally calm and neutral right now."

        lines = ["Your current emotional state:"]
        for name, intensity in active:
            label = _intensity_label(intensity)
            lines.append(f"- {name.upper()} ({label}, intensity {intensity:.0%}): {EMOTION_PROMPTS[name]}")
        return "\n".join(lines)

    def snapshot(self) -> dict[str, float]:
        self.tick_all()
        return {n: round(e.intensity, 3) for n, e in self.emotions.items() if e.intensity > 0}

    def __repr__(self) -> str:
        snap = self.snapshot()
        if not snap:
            return "EmotionEngine(calm)"
        parts = ", ".join(f"{k}={v}" for k, v in snap.items())
        return f"EmotionEngine({parts})"


def _intensity_label(v: float) -> str:
    if v >= 0.8:
        return "overwhelming"
    if v >= 0.5:
        return "strong"
    if v >= 0.25:
        return "moderate"
    return "faint"
