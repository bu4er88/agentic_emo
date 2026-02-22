"""Tests for SupervisorUnconscious (unconscious reflexes)."""

import json
from unittest.mock import patch, MagicMock

from agentic_emo.emotions import EmotionEngine
from agentic_emo.llm import LLMConfig
from agentic_emo.supervisor_unconscious import SupervisorUnconscious


def _make_config():
    return LLMConfig(model="test-model", api_key="test-key")


class TestSupervisorUnconscious:
    def test_init(self):
        engine = EmotionEngine()
        s1 = SupervisorUnconscious(engine, config=_make_config())
        assert s1.emotions is engine

    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_process_no_reflexes_fire(self, mock_chat_json):
        """When no reflexes fire, result should have empty lists."""
        mock_chat_json.return_value = {"fires": False}

        engine = EmotionEngine()
        s1 = SupervisorUnconscious(engine, config=_make_config())
        result = s1.process("A gentle breeze blows")

        assert result["fired_reflexes"] == []
        assert result["overall_motor"] is None
        assert result["emotion_snapshot"] == {}

    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_process_reflex_fires(self, mock_chat_json):
        """When a reflex fires, it should spike emotions."""

        def side_effect(config, system, prompt):
            if "FIGHT-OR-FLIGHT" in system:
                return {
                    "fires": True,
                    "response": "flight",
                    "motor": "jumps back",
                    "emotion_spikes": [{"emotion": "fear", "amount": 0.8}],
                }
            return {"fires": False}

        mock_chat_json.side_effect = side_effect

        engine = EmotionEngine()
        s1 = SupervisorUnconscious(engine, config=_make_config())
        result = s1.process("A bear charges at you")

        assert len(result["fired_reflexes"]) == 1
        assert result["fired_reflexes"][0]["reflex"] == "fight_or_flight"
        assert result["overall_motor"] == "jumps back"
        # Fear should be spiked
        assert "fear" in result["emotion_snapshot"]
        assert result["emotion_snapshot"]["fear"] > 0.5

    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_process_multiple_reflexes_fire(self, mock_chat_json):
        """Multiple reflexes can fire for the same stimulus."""

        def side_effect(config, system, prompt):
            if "FIGHT-OR-FLIGHT" in system:
                return {
                    "fires": True,
                    "response": "fight",
                    "motor": "clenches fists",
                    "emotion_spikes": [{"emotion": "anger", "amount": 0.6}],
                }
            if "STARTLE" in system:
                return {
                    "fires": True,
                    "motor": "flinches",
                    "emotion_spikes": [{"emotion": "surprise", "amount": 0.7}],
                }
            return {"fires": False}

        mock_chat_json.side_effect = side_effect

        engine = EmotionEngine()
        s1 = SupervisorUnconscious(engine, config=_make_config())
        result = s1.process("A loud bang right behind you")

        assert len(result["fired_reflexes"]) == 2
        assert result["overall_motor"] is not None
        assert "anger" in result["emotion_snapshot"]
        assert "surprise" in result["emotion_snapshot"]

    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_process_handles_json_error(self, mock_chat_json):
        """Reflex that returns invalid JSON should be silently skipped."""
        mock_chat_json.side_effect = json.JSONDecodeError("bad", "", 0)

        engine = EmotionEngine()
        s1 = SupervisorUnconscious(engine, config=_make_config())
        result = s1.process("something")

        assert result["fired_reflexes"] == []
        assert result["emotion_snapshot"] == {}

    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_eval_reflex_returns_none_when_not_fired(self, mock_chat_json):
        mock_chat_json.return_value = {"fires": False}
        engine = EmotionEngine()
        s1 = SupervisorUnconscious(engine, config=_make_config())
        result = s1._eval_reflex("startle", "nothing happened")
        assert result is None

    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_eval_reflex_returns_dict_when_fired(self, mock_chat_json):
        mock_chat_json.return_value = {
            "fires": True,
            "motor": "smile",
            "emotion_spikes": [{"emotion": "joy", "amount": 0.5}],
        }
        engine = EmotionEngine()
        s1 = SupervisorUnconscious(engine, config=_make_config())
        result = s1._eval_reflex("social_bonding", "a warm hug")
        assert result is not None
        assert result["reflex"] == "social_bonding"

    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_emotion_spikes_with_zero_amount_ignored(self, mock_chat_json):
        mock_chat_json.return_value = {
            "fires": True,
            "motor": None,
            "emotion_spikes": [{"emotion": "fear", "amount": 0}],
        }
        engine = EmotionEngine()
        s1 = SupervisorUnconscious(engine, config=_make_config())
        result = s1.process("mild event")
        # fear amount is 0, so it should not be spiked
        assert result["emotion_snapshot"].get("fear", 0) == 0
