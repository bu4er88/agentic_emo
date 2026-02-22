"""
Short-Term Memory (STM) — a sliding window of recent experiences.

Works like a bounded deque: newest items push oldest out.
Used by Supervisor-2 as a tool to recall recent context.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field


@dataclass
class MemoryItem:
    content: str
    timestamp: float = field(default_factory=time.time)
    source: str = "unknown"   # "conscious", "reflex", "external"

    def __str__(self) -> str:
        age = time.time() - self.timestamp
        if age < 60:
            ago = f"{age:.0f}s ago"
        else:
            ago = f"{age / 60:.1f}m ago"
        return f"[{ago} | {self.source}] {self.content}"


class ShortTermMemory:
    """Fixed-capacity sliding window memory."""

    def __init__(self, capacity: int = 20) -> None:
        self._items: deque[MemoryItem] = deque(maxlen=capacity)

    def store(self, content: str, source: str = "unknown") -> None:
        self._items.append(MemoryItem(content=content, source=source))

    def recall(self, n: int | None = None) -> list[MemoryItem]:
        """Return the most recent *n* items (all if n is None)."""
        items = list(self._items)
        if n is not None:
            items = items[-n:]
        return items

    def render(self, n: int | None = None) -> str:
        items = self.recall(n)
        if not items:
            return "Short-term memory is empty."
        lines = ["Recent memories (short-term):"]
        for item in items:
            lines.append(f"  • {item}")
        return "\n".join(lines)

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)
