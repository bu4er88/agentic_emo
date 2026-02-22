"""Tests for the LLM client wrapper."""

import json
from unittest.mock import patch, MagicMock

from agentic_emo.llm import LLMConfig, fast_model, strong_model, chat, chat_json


class TestLLMConfig:
    def test_defaults(self):
        cfg = LLMConfig(model="test-model", api_key="test-key")
        assert cfg.model == "test-model"
        assert cfg.api_key == "test-key"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 1024

    def test_api_key_from_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            cfg = LLMConfig(model="test")
            assert cfg.api_key == "env-key"

    def test_base_url_from_env(self):
        with patch.dict("os.environ", {"OPENAI_BASE_URL": "http://localhost:8000"}):
            cfg = LLMConfig(model="test", api_key="k")
            assert cfg.base_url == "http://localhost:8000"

    def test_explicit_overrides_env(self):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "env-key"}):
            cfg = LLMConfig(model="test", api_key="explicit-key")
            assert cfg.api_key == "explicit-key"


class TestModelFactories:
    def test_fast_model(self):
        cfg = fast_model()
        assert cfg.temperature == 0.3
        assert cfg.max_tokens == 256

    def test_strong_model(self):
        cfg = strong_model()
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048

    def test_fast_model_env_override(self):
        with patch.dict("os.environ", {"AGENTIC_FAST_MODEL": "my-fast-model"}):
            cfg = fast_model()
            assert cfg.model == "my-fast-model"

    def test_strong_model_env_override(self):
        with patch.dict("os.environ", {"AGENTIC_STRONG_MODEL": "my-strong-model"}):
            cfg = strong_model()
            assert cfg.model == "my-strong-model"


class TestChat:
    @patch("agentic_emo.llm.OpenAI")
    def test_chat_basic(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "Hello back!"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        cfg = LLMConfig(model="test", api_key="key")
        result = chat(cfg, "You are helpful.", "Hi")
        assert result == "Hello back!"

    @patch("agentic_emo.llm.OpenAI")
    def test_chat_with_history(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = "response"
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        cfg = LLMConfig(model="test", api_key="key")
        history = [{"role": "user", "content": "prev"}, {"role": "assistant", "content": "prev-resp"}]
        result = chat(cfg, "system", "new msg", history=history)

        # Verify messages include system + history + new user msg
        call_kwargs = mock_client.chat.completions.create.call_args
        messages = call_kwargs[1]["messages"] if "messages" in call_kwargs[1] else call_kwargs[0][0]
        assert len(messages) == 4  # system + 2 history + user

    @patch("agentic_emo.llm.OpenAI")
    def test_chat_none_content_returns_empty(self, mock_openai_cls):
        mock_client = MagicMock()
        mock_openai_cls.return_value = mock_client

        mock_choice = MagicMock()
        mock_choice.message.content = None
        mock_response = MagicMock()
        mock_response.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_response

        cfg = LLMConfig(model="test", api_key="key")
        result = chat(cfg, "system", "user")
        assert result == ""


class TestChatJson:
    @patch("agentic_emo.llm.chat")
    def test_parse_json(self, mock_chat):
        mock_chat.return_value = '{"fires": true, "emotion": "fear"}'
        cfg = LLMConfig(model="test", api_key="key")
        result = chat_json(cfg, "system", "user")
        assert result == {"fires": True, "emotion": "fear"}

    @patch("agentic_emo.llm.chat")
    def test_strip_code_fences(self, mock_chat):
        mock_chat.return_value = '```json\n{"fires": false}\n```'
        cfg = LLMConfig(model="test", api_key="key")
        result = chat_json(cfg, "system", "user")
        assert result == {"fires": False}

    @patch("agentic_emo.llm.chat")
    def test_invalid_json_raises(self, mock_chat):
        mock_chat.return_value = "not json at all"
        cfg = LLMConfig(model="test", api_key="key")
        try:
            chat_json(cfg, "system", "user")
            assert False, "Should have raised"
        except json.JSONDecodeError:
            pass
