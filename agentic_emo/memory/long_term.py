"""
Long-Term Memory (LTM) — persistent JSON-file-backed knowledge store.

No vector DB needed: we store tagged entries and let the conscious agent
search them by keyword / tag.  Simple but effective for a simulation.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


DEFAULT_PATH = Path("data/long_term_memory.json")


@dataclass
class LTMEntry:
    content: str
    tags: list[str] = field(default_factory=list)
    source: str = "conscious"
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # 0.0 = trivial … 1.0 = life-changing


class LongTermMemory:
    """Keyword-searchable persistent memory backed by a JSON file."""

    def __init__(self, path: Path | str = DEFAULT_PATH) -> None:
        self._path = Path(path)
        self._entries: list[LTMEntry] = []
        self._load()

    # ── persistence ──────────────────────────────────────────────────

    def _load(self) -> None:
        if self._path.exists():
            with open(self._path) as f:
                raw = json.load(f)
            self._entries = [LTMEntry(**e) for e in raw]

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w") as f:
            json.dump([asdict(e) for e in self._entries], f, indent=2)

    # ── public API ───────────────────────────────────────────────────

    def store(
        self,
        content: str,
        tags: list[str] | None = None,
        source: str = "conscious",
        importance: float = 0.5,
    ) -> None:
        entry = LTMEntry(
            content=content,
            tags=tags or [],
            source=source,
            importance=importance,
        )
        self._entries.append(entry)
        self._save()

    def search(self, query: str, limit: int = 5) -> list[LTMEntry]:
        """Simple keyword search across content and tags."""
        query_lower = query.lower()
        scored: list[tuple[float, LTMEntry]] = []
        for entry in self._entries:
            text = (entry.content + " " + " ".join(entry.tags)).lower()
            if query_lower in text:
                scored.append((entry.importance, entry))
        scored.sort(key=lambda x: -x[0])
        return [e for _, e in scored[:limit]]

    def recent(self, limit: int = 5) -> list[LTMEntry]:
        return sorted(self._entries, key=lambda e: -e.timestamp)[:limit]

    def render_search(self, query: str, limit: int = 5) -> str:
        results = self.search(query, limit)
        if not results:
            return f"No long-term memories found for '{query}'."
        lines = [f"Long-term memories matching '{query}':"]
        for e in results:
            tags = ", ".join(e.tags) if e.tags else "none"
            lines.append(
                f"  • [{e.source}, importance={e.importance:.1f}, tags={tags}] {e.content}"
            )
        return "\n".join(lines)

    def __len__(self) -> int:
        return len(self._entries)
