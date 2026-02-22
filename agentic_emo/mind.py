"""
The Human Mind — top-level orchestrator.

Wires together:
  - Supervisor-1 (unconscious reflexes)
  - Supervisor-2 (conscious reasoning)
  - Emotion Engine (shared dynamic state)
  - Short-term and Long-term Memory

Processing pipeline for each stimulus:
  1. Emotion engine ticks (natural decay).
  2. Supervisor-1 evaluates reflexes → emotion spikes + motor responses.
  3. Supervisor-2 receives stimulus + reflex output + emotional context
     → conscious response (with tool use for memory & emotion adjustment).
  4. Result is returned: conscious response + internal state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agentic_emo.emotions import EmotionEngine
from agentic_emo.llm import LLMConfig, fast_model, strong_model
from agentic_emo.memory.short_term import ShortTermMemory
from agentic_emo.memory.long_term import LongTermMemory
from agentic_emo.supervisor1 import Supervisor1
from agentic_emo.supervisor2 import Supervisor2

log = logging.getLogger(__name__)


@dataclass
class MindConfig:
    sex: str = "male"
    fast_llm: LLMConfig = field(default_factory=fast_model)
    strong_llm: LLMConfig = field(default_factory=strong_model)
    stm_capacity: int = 20
    ltm_path: str = "data/long_term_memory.json"


class HumanMind:
    """The complete simulated human mind."""

    def __init__(self, config: MindConfig | None = None) -> None:
        cfg = config or MindConfig()

        # Shared subsystems
        self.emotions = EmotionEngine()
        self.stm = ShortTermMemory(capacity=cfg.stm_capacity)
        self.ltm = LongTermMemory(path=cfg.ltm_path)

        # Supervisors
        self.unconscious = Supervisor1(
            emotion_engine=self.emotions,
            config=cfg.fast_llm,
        )
        self.conscious = Supervisor2(
            sex=cfg.sex,
            emotion_engine=self.emotions,
            stm=self.stm,
            ltm=self.ltm,
            config=cfg.strong_llm,
        )

        self.sex = cfg.sex
        self._turn = 0

    def perceive(self, stimulus: str) -> dict:
        """
        Full processing pipeline for an external stimulus.

        Returns:
            {
                "turn": int,
                "stimulus": str,
                "reflex_result": dict,
                "conscious_result": dict,
                "emotion_snapshot": dict,
            }
        """
        self._turn += 1
        log.info("=== Turn %d | Stimulus: %s ===", self._turn, stimulus[:80])

        # 1. Tick emotions (natural decay)
        self.emotions.tick_all()

        # 2. Unconscious reflexes
        log.info("Running Supervisor-1 (unconscious reflexes)...")
        reflex_result = self.unconscious.process(stimulus)
        log.info("Reflexes fired: %s", [r["reflex"] for r in reflex_result["fired_reflexes"]])

        # 3. Conscious processing
        log.info("Running Supervisor-2 (conscious mind)...")
        conscious_result = self.conscious.process(stimulus, reflex_result)
        log.info("Conscious response: %s", conscious_result["response"][:100])

        return {
            "turn": self._turn,
            "stimulus": stimulus,
            "reflex_result": reflex_result,
            "conscious_result": conscious_result,
            "emotion_snapshot": self.emotions.snapshot(),
        }

    def introspect(self) -> str:
        """Return a human-readable summary of the mind's current state."""
        lines = [
            f"🧠 Human Mind ({self.sex}) — Turn {self._turn}",
            f"   Emotions: {self.emotions}",
            f"   STM items: {len(self.stm)}",
            f"   LTM entries: {len(self.ltm)}",
        ]
        return "\n".join(lines)
