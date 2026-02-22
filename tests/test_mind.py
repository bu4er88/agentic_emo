"""Tests for the HumanMind orchestrator."""

from unittest.mock import patch, MagicMock

from agentic_emo.llm import LLMConfig
from agentic_emo.mind import HumanMind, MindConfig


def _make_config(tmp_path):
    return MindConfig(
        sex="male",
        fast_llm=LLMConfig(model="test-fast", api_key="test-key"),
        strong_llm=LLMConfig(model="test-strong", api_key="test-key"),
        ltm_path=str(tmp_path / "ltm.json"),
    )


class TestMindConfig:
    def test_defaults(self):
        cfg = MindConfig()
        assert cfg.sex == "male"
        assert cfg.stm_capacity == 20

    def test_custom_sex(self):
        cfg = MindConfig(sex="female")
        assert cfg.sex == "female"


class TestHumanMind:
    def test_init_male(self, tmp_path):
        mind = HumanMind(config=_make_config(tmp_path))
        assert mind.sex == "male"
        assert mind._turn == 0

    def test_init_female(self, tmp_path):
        cfg = _make_config(tmp_path)
        cfg.sex = "female"
        mind = HumanMind(config=cfg)
        assert mind.sex == "female"

    def test_subsystems_wired(self, tmp_path):
        mind = HumanMind(config=_make_config(tmp_path))
        # All subsystems should share the same emotion engine
        assert mind.unconscious.emotions is mind.emotions
        assert mind.conscious.emotions is mind.emotions
        assert mind.conscious.stm is mind.stm
        assert mind.conscious.ltm is mind.ltm

    @patch("agentic_emo.supervisor_conscious.chat")
    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_perceive_full_pipeline(self, mock_chat_json, mock_chat, tmp_path):
        """Full pipeline: stimulus → reflexes → conscious response."""
        # Supervisor-1: startle reflex fires
        def reflex_side_effect(config, system, prompt):
            if "STARTLE" in system:
                return {
                    "fires": True,
                    "motor": "flinch",
                    "emotion_spikes": [{"emotion": "surprise", "amount": 0.6}],
                }
            return {"fires": False}

        mock_chat_json.side_effect = reflex_side_effect

        # Supervisor-2: direct response
        mock_chat.return_value = "RESPONSE: Whoa, that scared me!"

        mind = HumanMind(config=_make_config(tmp_path))
        result = mind.perceive("A loud crash nearby")

        assert result["turn"] == 1
        assert result["stimulus"] == "A loud crash nearby"
        assert len(result["reflex_result"]["fired_reflexes"]) == 1
        assert result["conscious_result"]["response"] == "Whoa, that scared me!"
        assert "surprise" in result["emotion_snapshot"]

    @patch("agentic_emo.supervisor_conscious.chat")
    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_perceive_increments_turn(self, mock_chat_json, mock_chat, tmp_path):
        mock_chat_json.return_value = {"fires": False}
        mock_chat.return_value = "RESPONSE: ok"

        mind = HumanMind(config=_make_config(tmp_path))
        mind.perceive("first")
        mind.perceive("second")
        assert mind._turn == 2

    @patch("agentic_emo.supervisor_conscious.chat")
    @patch("agentic_emo.supervisor_unconscious.chat_json")
    def test_perceive_no_reflexes(self, mock_chat_json, mock_chat, tmp_path):
        mock_chat_json.return_value = {"fires": False}
        mock_chat.return_value = "RESPONSE: Nothing special happening."

        mind = HumanMind(config=_make_config(tmp_path))
        result = mind.perceive("A calm day")

        assert result["reflex_result"]["fired_reflexes"] == []
        assert result["conscious_result"]["response"] == "Nothing special happening."

    def test_introspect(self, tmp_path):
        mind = HumanMind(config=_make_config(tmp_path))
        text = mind.introspect()
        assert "male" in text
        assert "Turn 0" in text
        assert "STM" in text
        assert "LTM" in text
