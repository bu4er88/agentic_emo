"""
Supervisor-2 — The Conscious Mind.

A powerful reasoning agent that:
- Receives the stimulus + reflex results + emotional state as context.
- Has tools: short-term memory, long-term memory, emotion adjustment.
- Thinks, reasons, learns, and produces the human's conscious response.
- Uses a sex-specific personality profile as its base prompt.
"""

from __future__ import annotations

import json
import logging
import re

from agentic_emo.emotions import EmotionEngine
from agentic_emo.llm import LLMConfig, chat, strong_model
from agentic_emo.memory.short_term import ShortTermMemory
from agentic_emo.memory.long_term import LongTermMemory
from agentic_emo.prompts.profiles import get_profile

log = logging.getLogger(__name__)

# Maximum tool-use loops before we force a final answer
MAX_TOOL_ROUNDS = 5

TOOL_INSTRUCTIONS = """\

## Available tools

You can call tools by writing a line starting with `TOOL_CALL:` followed by JSON.
You may call multiple tools per turn. After all tool calls are processed you will
receive the results and can continue reasoning.

### stm_store
Store something in short-term memory.
```
TOOL_CALL: {"tool": "stm_store", "content": "...", "source": "conscious"}
```

### stm_recall
Recall recent short-term memories.
```
TOOL_CALL: {"tool": "stm_recall", "n": 5}
```

### ltm_store
Store something in long-term memory with tags and importance.
```
TOOL_CALL: {"tool": "ltm_store", "content": "...", "tags": ["tag1"], "importance": 0.7}
```

### ltm_search
Search long-term memory by keyword.
```
TOOL_CALL: {"tool": "ltm_search", "query": "..."}
```

### emotion_adjust
Gently shift an emotion (long-lasting conscious adjustment).
Amount can be negative (suppress) or positive (amplify), range -0.3 to +0.3.
```
TOOL_CALL: {"tool": "emotion_adjust", "emotion": "joy", "amount": 0.1}
```

### think
Internal monologue — reason step by step before acting. This is private.
```
TOOL_CALL: {"tool": "think", "thought": "Let me consider..."}
```

When you are done thinking and using tools, produce your FINAL conscious response
on a line starting with `RESPONSE:`. This is what the human says or does outwardly.
"""


class Supervisor2:
    """Conscious reasoning agent with memory and emotion tools."""

    def __init__(
        self,
        sex: str,
        emotion_engine: EmotionEngine,
        stm: ShortTermMemory,
        ltm: LongTermMemory,
        config: LLMConfig | None = None,
    ) -> None:
        self.profile = get_profile(sex)
        self.emotions = emotion_engine
        self.stm = stm
        self.ltm = ltm
        self.config = config or strong_model()
        self._history: list[dict] = []

    # ── system prompt assembly ───────────────────────────────────────

    def _build_system(self) -> str:
        """Assemble the full system prompt with dynamic emotion injection."""
        parts = [
            self.profile["base_system"],
            "",
            self.profile["thinking_style"],
            "",
            self.profile["social_style"],
            "",
            "--- CURRENT EMOTIONAL STATE ---",
            self.emotions.render_prompt(),
            "",
            "--- MEMORY CONTEXT ---",
            self.stm.render(n=10),
            "",
            TOOL_INSTRUCTIONS,
        ]
        return "\n".join(parts)

    # ── tool execution ───────────────────────────────────────────────

    def _exec_tool(self, call: dict) -> str:
        """Execute a single tool call and return the result string."""
        tool = call.get("tool", "")

        if tool == "stm_store":
            self.stm.store(call.get("content", ""), source=call.get("source", "conscious"))
            return "Stored in short-term memory."

        if tool == "stm_recall":
            return self.stm.render(n=call.get("n"))

        if tool == "ltm_store":
            self.ltm.store(
                content=call.get("content", ""),
                tags=call.get("tags", []),
                importance=call.get("importance", 0.5),
            )
            return "Stored in long-term memory."

        if tool == "ltm_search":
            return self.ltm.render_search(call.get("query", ""))

        if tool == "emotion_adjust":
            emotion = call.get("emotion", "")
            amount = float(call.get("amount", 0))
            amount = max(-0.3, min(0.3, amount))  # clamp
            self.emotions.shift(emotion, amount)
            snap = self.emotions.snapshot()
            return f"Adjusted {emotion} by {amount:+.2f}. Current: {snap}"

        if tool == "think":
            # Internal monologue — no side effect, just acknowledged
            return f"[internal thought noted]"

        return f"Unknown tool: {tool}"

    # ── tool-call parsing ────────────────────────────────────────────

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict]:
        calls = []
        for line in text.split("\n"):
            line = line.strip()
            if line.startswith("TOOL_CALL:"):
                json_str = line[len("TOOL_CALL:"):].strip()
                try:
                    calls.append(json.loads(json_str))
                except json.JSONDecodeError:
                    log.warning("Bad tool call JSON: %s", json_str)
        return calls

    @staticmethod
    def _extract_response(text: str) -> str | None:
        for line in text.split("\n"):
            if line.strip().startswith("RESPONSE:"):
                return line.strip()[len("RESPONSE:"):].strip()
        # Also check for multi-line response after RESPONSE:
        match = re.search(r"RESPONSE:\s*(.+)", text, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None

    # ── main entry point ─────────────────────────────────────────────

    def process(
        self,
        stimulus: str,
        reflex_result: dict | None = None,
    ) -> dict:
        """
        Run the conscious reasoning loop.

        Returns {response: str, tool_log: list, emotion_snapshot: dict}.
        """
        # Build context from stimulus + reflex output
        context_parts = [f"External stimulus: {stimulus}"]
        if reflex_result:
            fired = reflex_result.get("fired_reflexes", [])
            if fired:
                names = [r["reflex"] for r in fired]
                context_parts.append(f"Reflexes fired: {', '.join(names)}")
                motor = reflex_result.get("overall_motor")
                if motor:
                    context_parts.append(f"Instinctive body reaction: {motor}")
        context = "\n".join(context_parts)

        system = self._build_system()
        tool_log: list[dict] = []

        # Agentic tool-use loop
        current_user_msg = context
        for _round in range(MAX_TOOL_ROUNDS):
            raw = chat(self.config, system, current_user_msg, history=self._history)

            # Check for final response
            final = self._extract_response(raw)
            tool_calls = self._parse_tool_calls(raw)

            # Execute any tool calls
            results = []
            for tc in tool_calls:
                result = self._exec_tool(tc)
                tool_log.append({"call": tc, "result": result})
                results.append(f"[{tc.get('tool')}] → {result}")

            if final:
                # Store the exchange in STM
                self.stm.store(f"Stimulus: {stimulus}", source="external")
                self.stm.store(f"My response: {final}", source="conscious")
                # Keep conversation history (trimmed)
                self._history.append({"role": "user", "content": context})
                self._history.append({"role": "assistant", "content": raw})
                if len(self._history) > 20:
                    self._history = self._history[-20:]
                return {
                    "response": final,
                    "tool_log": tool_log,
                    "emotion_snapshot": self.emotions.snapshot(),
                }

            # No final response yet — feed tool results back and loop
            if results:
                current_user_msg = "Tool results:\n" + "\n".join(results) + "\n\nContinue reasoning. When ready, output RESPONSE: <your response>"
            else:
                current_user_msg = "Please produce your RESPONSE: <your conscious response>"

        # Fallback if we exceeded max rounds
        return {
            "response": "(conscious mind timed out — could not form a response)",
            "tool_log": tool_log,
            "emotion_snapshot": self.emotions.snapshot(),
        }
