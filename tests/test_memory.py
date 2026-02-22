"""Tests for Short-Term and Long-Term Memory."""

import json
import time
from pathlib import Path

import pytest

from agentic_emo.memory.short_term import ShortTermMemory, MemoryItem
from agentic_emo.memory.long_term import LongTermMemory, LTMEntry


# ── MemoryItem tests ────────────────────────────────────────────────


class TestMemoryItem:
    def test_creation(self):
        item = MemoryItem(content="hello", source="test")
        assert item.content == "hello"
        assert item.source == "test"
        assert item.timestamp <= time.time()

    def test_str_recent(self):
        item = MemoryItem(content="hello", source="test")
        s = str(item)
        assert "test" in s
        assert "hello" in s
        assert "s ago" in s

    def test_str_older(self):
        item = MemoryItem(content="old", source="test")
        item.timestamp -= 120  # 2 minutes ago
        s = str(item)
        assert "m ago" in s


# ── ShortTermMemory tests ──────────────────────────────────────────


class TestShortTermMemory:
    def test_empty(self):
        stm = ShortTermMemory(capacity=5)
        assert len(stm) == 0
        assert stm.recall() == []

    def test_store_and_recall(self):
        stm = ShortTermMemory(capacity=5)
        stm.store("first", source="test")
        stm.store("second", source="test")
        items = stm.recall()
        assert len(items) == 2
        assert items[0].content == "first"
        assert items[1].content == "second"

    def test_recall_with_limit(self):
        stm = ShortTermMemory(capacity=10)
        for i in range(5):
            stm.store(f"item-{i}")
        items = stm.recall(n=2)
        assert len(items) == 2
        assert items[0].content == "item-3"
        assert items[1].content == "item-4"

    def test_capacity_eviction(self):
        stm = ShortTermMemory(capacity=3)
        for i in range(5):
            stm.store(f"item-{i}")
        assert len(stm) == 3
        items = stm.recall()
        assert items[0].content == "item-2"
        assert items[2].content == "item-4"

    def test_clear(self):
        stm = ShortTermMemory(capacity=5)
        stm.store("something")
        stm.clear()
        assert len(stm) == 0

    def test_render_empty(self):
        stm = ShortTermMemory()
        rendered = stm.render()
        assert "empty" in rendered.lower()

    def test_render_with_items(self):
        stm = ShortTermMemory()
        stm.store("hello world", source="conscious")
        rendered = stm.render()
        assert "hello world" in rendered
        assert "short-term" in rendered.lower()

    def test_render_with_limit(self):
        stm = ShortTermMemory()
        stm.store("a")
        stm.store("b")
        stm.store("c")
        rendered = stm.render(n=1)
        assert "c" in rendered
        # 'a' should not be in the render since we limited to 1
        lines = rendered.split("\n")
        item_lines = [l for l in lines if l.strip().startswith("•")]
        assert len(item_lines) == 1


# ── LTMEntry tests ─────────────────────────────────────────────────


class TestLTMEntry:
    def test_creation(self):
        entry = LTMEntry(content="learned something")
        assert entry.content == "learned something"
        assert entry.tags == []
        assert entry.source == "conscious"
        assert entry.importance == 0.5

    def test_creation_with_tags(self):
        entry = LTMEntry(content="fact", tags=["science", "physics"])
        assert entry.tags == ["science", "physics"]


# ── LongTermMemory tests ──────────────────────────────────────────


class TestLongTermMemory:
    def test_empty(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        assert len(ltm) == 0

    def test_store_and_search(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        ltm.store("Python is a programming language", tags=["python", "programming"])
        results = ltm.search("python")
        assert len(results) == 1
        assert "Python" in results[0].content

    def test_search_no_results(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        ltm.store("hello world")
        results = ltm.search("nonexistent")
        assert results == []

    def test_search_by_tag(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        ltm.store("some fact", tags=["science"])
        results = ltm.search("science")
        assert len(results) == 1

    def test_search_case_insensitive(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        ltm.store("Python Tutorial")
        results = ltm.search("PYTHON")
        assert len(results) == 1

    def test_search_sorted_by_importance(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        ltm.store("less important python", importance=0.2)
        ltm.store("very important python", importance=0.9)
        results = ltm.search("python")
        assert len(results) == 2
        assert results[0].importance == 0.9

    def test_search_with_limit(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        for i in range(10):
            ltm.store(f"fact-{i} about cats", importance=i / 10)
        results = ltm.search("cats", limit=3)
        assert len(results) == 3

    def test_recent(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        ltm.store("old")
        ltm.store("newer")
        ltm.store("newest")
        recent = ltm.recent(limit=2)
        assert len(recent) == 2
        assert recent[0].content == "newest"

    def test_persistence(self, tmp_path):
        path = tmp_path / "ltm.json"
        ltm1 = LongTermMemory(path=path)
        ltm1.store("persistent data", tags=["test"])

        # Reload from same file
        ltm2 = LongTermMemory(path=path)
        assert len(ltm2) == 1
        results = ltm2.search("persistent")
        assert len(results) == 1
        assert results[0].tags == ["test"]

    def test_render_search_found(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        ltm.store("Python is great", tags=["python"])
        rendered = ltm.render_search("python")
        assert "python" in rendered.lower()
        assert "Python is great" in rendered

    def test_render_search_not_found(self, tmp_path):
        ltm = LongTermMemory(path=tmp_path / "ltm.json")
        rendered = ltm.render_search("nonexistent")
        assert "No long-term memories found" in rendered

    def test_json_format(self, tmp_path):
        path = tmp_path / "ltm.json"
        ltm = LongTermMemory(path=path)
        ltm.store("test entry", tags=["t1"], importance=0.8)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["content"] == "test entry"
        assert data[0]["tags"] == ["t1"]
        assert data[0]["importance"] == 0.8
