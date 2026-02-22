"""
Supervisor Unconscious — The Unconscious Mind.

Coordinates reflex sub-agents that run on fast/cheap models.
Each reflex evaluates the stimulus independently and in parallel (conceptually).
Aggregates results and pushes emotion spikes to the EmotionEngine.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from agentic_emo.emotions import EmotionEngine
from agentic_emo.llm import LLMConfig, chat_json, fast_model
from agentic_emo.prompts.reflexes import REFLEX_PROMPTS, SUPERVISOR_UNCONSCIOUS_SYSTEM

log = logging.getLogger(__name__)


class SupervisorUnconscious:
    """Unconscious reflex coordinator."""

    def __init__(
        self,
        emotion_engine: EmotionEngine,
        config: LLMConfig | None = None,
    ) -> None:
        self.emotions = emotion_engine
        self.config = config or fast_model()

    # ── single reflex evaluation ─────────────────────────────────────

    def _eval_reflex(self, reflex_name: str, stimulus: str) -> dict | None:
        """Run one reflex sub-agent against the stimulus."""
        reflex = REFLEX_PROMPTS[reflex_name]
        prompt = f"Stimulus: {stimulus}"
        try:
            result = chat_json(self.config, reflex["system"], prompt)
            if result.get("fires"):
                result["reflex"] = reflex_name
                return result
        except (json.JSONDecodeError, KeyError) as exc:
            log.warning("Reflex %s failed to parse: %s", reflex_name, exc)
        return None

    # ── main entry point ─────────────────────────────────────────────

    def process(self, stimulus: str) -> dict:
        """
        Evaluate all reflexes against the stimulus.

        Returns aggregated result dict and applies emotion spikes.
        """
        fired: list[dict] = []

        # Run reflexes in parallel threads (they're IO-bound LLM calls)
        with ThreadPoolExecutor(max_workers=len(REFLEX_PROMPTS)) as pool:
            futures = {
                pool.submit(self._eval_reflex, name, stimulus): name
                for name in REFLEX_PROMPTS
            }
            for future in as_completed(futures):
                result = future.result()
                if result is not None:
                    fired.append(result)

        # Apply emotion spikes to the shared emotion engine
        for reflex_result in fired:
            for spike in reflex_result.get("emotion_spikes", []):
                emotion = spike.get("emotion", "")
                amount = float(spike.get("amount", 0))
                if emotion and amount > 0:
                    self.emotions.spike(emotion, amount)
                    log.info(
                        "Reflex [%s] spiked %s by %.2f",
                        reflex_result["reflex"], emotion, amount,
                    )

        # Build aggregate motor response
        motors = [r.get("motor") for r in fired if r.get("motor")]
        overall_motor = "; ".join(motors) if motors else None

        return {
            "fired_reflexes": fired,
            "overall_motor": overall_motor,
            "emotion_snapshot": self.emotions.snapshot(),
        }
