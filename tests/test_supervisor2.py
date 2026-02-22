"""Tests for Supervisor-2 (conscious mind)."""

from unittest.mock import patch

from agentic_emo.emotions import EmotionEngine
from agentic_emo.llm import LLMConfig
from agentic_emo.memory.short_term import ShortTermMemory
from agentic_emo.memory.long_term import LongTermMemory
from agentic_emo.supervisor2 import Supervisor2


def _make_config():
    return LLMConfig(model="test-model", api_key="test-key")


def _make_supervisor2(tmp_path, sex="male"):
    engine = EmotionEngine()
    stm = ShortTermMemory()
    ltm = LongTermMemory(path=tmp_path / "ltm.json")
    s2 = Supervisor2(sex=sex, emotion_engine=engine, stm=stm, ltm=ltm, config=_make_config())
    return s2, engine, stm, ltm


# ── Parse helpers tests ─────────────────────────────────────────────


class TestParseToolCalls:
    def test_single_tool_call(self):
        text = 'TOOL_CALL: {"tool": "stm_store", "content": "hello"}'
        calls = Supervisor2._parse_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["tool"] == "stm_store"

    def test_multiple_tool_calls(self):
        text = (
            'Let me think...\n'
            'TOOL_CALL: {"tool": "think", "thought": "hmm"}\n'
            'TOOL_CALL: {"tool": "stm_recall", "n": 3}\n'
        )
        calls = Supervisor2._parse_tool_calls(text)
        assert len(calls) == 2

    def test_no_tool_calls(self):
        text = "Just some regular text without any tool calls."
        calls = Supervisor2._parse_tool_calls(text)
        assert calls == []

    def test_bad_json_ignored(self):
        text = 'TOOL_CALL: {not valid json}'
        calls = Supervisor2._parse_tool_calls(text)
        assert calls == []


class TestExtractResponse:
    def test_simple_response(self):
        text = "RESPONSE: Hello, how are you?"
        resp = Supervisor2._extract_response(text)
        assert resp == "Hello, how are you?"

    def test_response_with_prefix_text(self):
        text = "Let me think...\nTOOL_CALL: {...}\nRESPONSE: I'm fine."
        resp = Supervisor2._extract_response(text)
        assert resp == "I'm fine."

    def test_no_response(self):
        text = "Just thinking, no response yet."
        resp = Supervisor2._extract_response(text)
        assert resp is None

    def test_response_with_whitespace(self):
        text = "  RESPONSE:   spaced out   "
        resp = Supervisor2._extract_response(text)
        assert resp == "spaced out"


# ── Tool execution tests ────────────────────────────────────────────


class TestExecTool:
    def test_stm_store(self, tmp_path):
        s2, _, stm, _ = _make_supervisor2(tmp_path)
        result = s2._exec_tool({"tool": "stm_store", "content": "test memory"})
        assert "Stored" in result
        assert len(stm) == 1

    def test_stm_recall(self, tmp_path):
        s2, _, stm, _ = _make_supervisor2(tmp_path)
        stm.store("item1")
        stm.store("item2")
        result = s2._exec_tool({"tool": "stm_recall", "n": 5})
        assert "item1" in result
        assert "item2" in result

    def test_ltm_store(self, tmp_path):
        s2, _, _, ltm = _make_supervisor2(tmp_path)
        result = s2._exec_tool({
            "tool": "ltm_store",
            "content": "important fact",
            "tags": ["test"],
            "importance": 0.9,
        })
        assert "Stored" in result
        assert len(ltm) == 1

    def test_ltm_search(self, tmp_path):
        s2, _, _, ltm = _make_supervisor2(tmp_path)
        ltm.store("Python programming", tags=["python"])
        result = s2._exec_tool({"tool": "ltm_search", "query": "python"})
        assert "Python" in result

    def test_emotion_adjust(self, tmp_path):
        s2, engine, _, _ = _make_supervisor2(tmp_path)
        result = s2._exec_tool({"tool": "emotion_adjust", "emotion": "joy", "amount": 0.2})
        assert "Adjusted" in result
        snap = engine.snapshot()
        assert "joy" in snap

    def test_emotion_adjust_clamps(self, tmp_path):
        s2, engine, _, _ = _make_supervisor2(tmp_path)
        # Try to adjust beyond allowed range
        s2._exec_tool({"tool": "emotion_adjust", "emotion": "joy", "amount": 0.9})
        # Should be clamped to 0.3
        snap = engine.snapshot()
        assert snap["joy"] <= 0.31  # Allow tiny floating point margin

    def test_think(self, tmp_path):
        s2, _, _, _ = _make_supervisor2(tmp_path)
        result = s2._exec_tool({"tool": "think", "thought": "Let me reason..."})
        assert "thought noted" in result.lower()

    def test_unknown_tool(self, tmp_path):
        s2, _, _, _ = _make_supervisor2(tmp_path)
        result = s2._exec_tool({"tool": "nonexistent"})
        assert "Unknown tool" in result


# ── System prompt building tests ────────────────────────────────────


class TestBuildSystem:
    def test_includes_profile(self, tmp_path):
        s2, _, _, _ = _make_supervisor2(tmp_path, sex="male")
        system = s2._build_system()
        assert "male" in system.lower()

    def test_includes_emotions(self, tmp_path):
        s2, engine, _, _ = _make_supervisor2(tmp_path)
        engine.spike("fear", 0.9)
        system = s2._build_system()
        assert "FEAR" in system

    def test_includes_stm(self, tmp_path):
        s2, _, stm, _ = _make_supervisor2(tmp_path)
        stm.store("recent event")
        system = s2._build_system()
        assert "recent event" in system

    def test_includes_tool_instructions(self, tmp_path):
        s2, _, _, _ = _make_supervisor2(tmp_path)
        system = s2._build_system()
        assert "TOOL_CALL" in system
        assert "stm_store" in system
        assert "ltm_search" in system

    def test_female_profile(self, tmp_path):
        s2, _, _, _ = _make_supervisor2(tmp_path, sex="female")
        system = s2._build_system()
        assert "female" in system.lower()


# ── Process integration tests ──────────────────────────────────────


class TestProcess:
    @patch("agentic_emo.supervisor2.chat")
    def test_direct_response(self, mock_chat, tmp_path):
        """LLM returns a RESPONSE immediately."""
        mock_chat.return_value = "RESPONSE: I feel startled but I'm okay."

        s2, _, stm, _ = _make_supervisor2(tmp_path)
        result = s2.process("A bird flies by")

        assert result["response"] == "I feel startled but I'm okay."
        assert "emotion_snapshot" in result
        # STM should have the stimulus + response stored
        assert len(stm) == 2

    @patch("agentic_emo.supervisor2.chat")
    def test_with_tool_use_then_response(self, mock_chat, tmp_path):
        """LLM uses a tool first, then responds."""
        mock_chat.side_effect = [
            'TOOL_CALL: {"tool": "think", "thought": "Let me consider..."}\nI need to respond.',
            "RESPONSE: After thinking, I feel calm.",
        ]

        s2, _, _, _ = _make_supervisor2(tmp_path)
        result = s2.process("What do you think?")

        assert result["response"] == "After thinking, I feel calm."
        assert len(result["tool_log"]) == 1
        assert result["tool_log"][0]["call"]["tool"] == "think"

    @patch("agentic_emo.supervisor2.chat")
    def test_with_reflex_result(self, mock_chat, tmp_path):
        """Process includes reflex results in context."""
        mock_chat.return_value = "RESPONSE: I jumped!"

        s2, _, _, _ = _make_supervisor2(tmp_path)
        reflex_result = {
            "fired_reflexes": [{"reflex": "startle", "motor": "flinch"}],
            "overall_motor": "flinch",
        }
        result = s2.process("Loud bang!", reflex_result)
        assert result["response"] == "I jumped!"

        # Verify the context passed to LLM includes reflex info
        call_args = mock_chat.call_args
        user_msg = call_args[0][2] if len(call_args[0]) > 2 else call_args[1].get("user", "")
        assert "startle" in user_msg.lower()

    @patch("agentic_emo.supervisor2.chat")
    def test_max_rounds_fallback(self, mock_chat, tmp_path):
        """If LLM never produces RESPONSE, we get a timeout fallback."""
        mock_chat.return_value = "I'm still thinking..."

        s2, _, _, _ = _make_supervisor2(tmp_path)
        result = s2.process("Complex problem")

        assert "timed out" in result["response"]

    @patch("agentic_emo.supervisor2.chat")
    def test_history_trimming(self, mock_chat, tmp_path):
        """Conversation history is trimmed to 20 entries."""
        mock_chat.return_value = "RESPONSE: ok"

        s2, _, _, _ = _make_supervisor2(tmp_path)
        # Fill history beyond 20
        s2._history = [{"role": "user", "content": f"msg-{i}"} for i in range(25)]

        s2.process("new stimulus")
        # After process, history should be trimmed
        assert len(s2._history) <= 22  # 20 old + 2 new (user + assistant)
