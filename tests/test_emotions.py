"""Tests for the Emotion Engine."""

import time
from unittest.mock import patch

from agentic_emo.emotions import (
    EMOTION_PROMPTS,
    DEFAULT_DECAY_RATE,
    EmotionState,
    EmotionEngine,
    _intensity_label,
)


# ── EmotionState unit tests ─────────────────────────────────────────


class TestEmotionState:
    def test_initial_state(self):
        e = EmotionState(name="joy")
        assert e.name == "joy"
        assert e.intensity == 0.0
        assert e.decay_rate == DEFAULT_DECAY_RATE

    def test_spike_increases_intensity(self):
        e = EmotionState(name="fear")
        e.spike(0.7)
        assert e.intensity == 0.7
        # Default spike decay rate is fast
        assert e.decay_rate == 0.15

    def test_spike_custom_decay_rate(self):
        e = EmotionState(name="fear")
        e.spike(0.5, decay_rate=0.3)
        assert e.intensity == 0.5
        assert e.decay_rate == 0.3

    def test_spike_clamps_at_1(self):
        e = EmotionState(name="anger")
        e.spike(0.6)
        e.spike(0.6)
        assert e.intensity == 1.0

    def test_shift_increases_intensity(self):
        e = EmotionState(name="joy")
        e.shift(0.3)
        assert e.intensity == 0.3
        # Default shift decay rate is slow
        assert e.decay_rate == 0.01

    def test_shift_custom_decay_rate(self):
        e = EmotionState(name="joy")
        e.shift(0.3, decay_rate=0.02)
        assert e.decay_rate == 0.02

    def test_shift_can_decrease(self):
        e = EmotionState(name="anger")
        e.spike(0.8)
        e.shift(-0.3)
        assert abs(e.intensity - 0.5) < 0.01

    def test_shift_clamps_at_0(self):
        e = EmotionState(name="sadness")
        e.shift(-0.5)
        assert e.intensity == 0.0

    def test_shift_clamps_at_1(self):
        e = EmotionState(name="joy")
        e.shift(1.5)
        assert e.intensity == 1.0

    def test_tick_decays_intensity(self):
        e = EmotionState(name="fear")
        e.spike(1.0, decay_rate=1.0)  # Extremely fast decay for testing
        # Simulate 1 second passing
        e.last_updated -= 1.0
        e.tick()
        # With decay_rate=1.0 and 1 second, intensity *= max(0, 1 - 1.0*1) = 0
        assert e.intensity == 0.0

    def test_tick_partial_decay(self):
        e = EmotionState(name="fear")
        e.spike(1.0, decay_rate=0.5)
        e.last_updated -= 0.5  # 0.5 seconds ago
        e.tick()
        # intensity *= max(0, 1 - 0.5*0.5) = 0.75
        assert abs(e.intensity - 0.75) < 0.01

    def test_tick_zeroes_out_tiny_values(self):
        e = EmotionState(name="surprise")
        e.intensity = 0.005
        e.last_updated = time.time()
        e.tick()
        assert e.intensity == 0.0


# ── EmotionEngine unit tests ────────────────────────────────────────


class TestEmotionEngine:
    def test_initial_state_all_zero(self):
        engine = EmotionEngine()
        snap = engine.snapshot()
        assert snap == {}

    def test_all_emotions_present(self):
        engine = EmotionEngine()
        assert set(engine.emotions.keys()) == set(EMOTION_PROMPTS.keys())

    def test_spike_updates_emotion(self):
        engine = EmotionEngine()
        engine.spike("joy", 0.5)
        snap = engine.snapshot()
        assert "joy" in snap
        assert snap["joy"] > 0.4

    def test_spike_unknown_emotion_is_noop(self):
        engine = EmotionEngine()
        engine.spike("nonexistent", 0.5)  # Should not raise
        snap = engine.snapshot()
        assert "nonexistent" not in snap

    def test_shift_updates_emotion(self):
        engine = EmotionEngine()
        engine.shift("trust", 0.3)
        snap = engine.snapshot()
        assert "trust" in snap
        assert snap["trust"] > 0.2

    def test_shift_unknown_emotion_is_noop(self):
        engine = EmotionEngine()
        engine.shift("nonexistent", 0.3)
        snap = engine.snapshot()
        assert "nonexistent" not in snap

    def test_render_prompt_when_calm(self):
        engine = EmotionEngine()
        prompt = engine.render_prompt()
        assert "calm" in prompt.lower()

    def test_render_prompt_with_active_emotions(self):
        engine = EmotionEngine()
        engine.spike("fear", 0.8)
        engine.spike("anger", 0.3)
        prompt = engine.render_prompt()
        assert "FEAR" in prompt
        assert "ANGER" in prompt
        assert "current emotional state" in prompt.lower()

    def test_render_prompt_respects_threshold(self):
        engine = EmotionEngine()
        engine.spike("joy", 0.03)  # Below default threshold of 0.05
        prompt = engine.render_prompt(threshold=0.05)
        assert "JOY" not in prompt

    def test_render_prompt_sorted_by_intensity(self):
        engine = EmotionEngine()
        engine.spike("sadness", 0.3)
        engine.spike("fear", 0.9)
        prompt = engine.render_prompt()
        fear_pos = prompt.index("FEAR")
        sadness_pos = prompt.index("SADNESS")
        assert fear_pos < sadness_pos  # Fear should come first (higher intensity)

    def test_tick_all(self):
        engine = EmotionEngine()
        engine.spike("joy", 0.5, decay_rate=10.0)  # Very fast decay
        for e in engine.emotions.values():
            e.last_updated -= 1.0  # 1 second ago
        engine.tick_all()
        snap = engine.snapshot()
        assert "joy" not in snap  # Should have decayed to 0

    def test_repr_calm(self):
        engine = EmotionEngine()
        assert repr(engine) == "EmotionEngine(calm)"

    def test_repr_with_emotions(self):
        engine = EmotionEngine()
        engine.spike("joy", 0.5)
        r = repr(engine)
        assert "joy=" in r


# ── _intensity_label tests ──────────────────────────────────────────


class TestIntensityLabel:
    def test_overwhelming(self):
        assert _intensity_label(0.9) == "overwhelming"
        assert _intensity_label(0.8) == "overwhelming"

    def test_strong(self):
        assert _intensity_label(0.5) == "strong"
        assert _intensity_label(0.79) == "strong"

    def test_moderate(self):
        assert _intensity_label(0.25) == "moderate"
        assert _intensity_label(0.49) == "moderate"

    def test_faint(self):
        assert _intensity_label(0.1) == "faint"
        assert _intensity_label(0.24) == "faint"
