"""Tests for prompts and profiles."""

import pytest

from agentic_emo.prompts.profiles import (
    MALE_PROFILE,
    FEMALE_PROFILE,
    get_profile,
)
from agentic_emo.prompts.reflexes import REFLEX_PROMPTS, SUPERVISOR_UNCONSCIOUS_SYSTEM


# ── Profile tests ──────────────────────────────────────────────────


class TestProfiles:
    def test_male_profile_has_required_keys(self):
        assert "sex" in MALE_PROFILE
        assert "base_system" in MALE_PROFILE
        assert "thinking_style" in MALE_PROFILE
        assert "social_style" in MALE_PROFILE
        assert MALE_PROFILE["sex"] == "male"

    def test_female_profile_has_required_keys(self):
        assert "sex" in FEMALE_PROFILE
        assert "base_system" in FEMALE_PROFILE
        assert "thinking_style" in FEMALE_PROFILE
        assert "social_style" in FEMALE_PROFILE
        assert FEMALE_PROFILE["sex"] == "female"

    def test_profiles_are_distinct(self):
        assert MALE_PROFILE["base_system"] != FEMALE_PROFILE["base_system"]
        assert MALE_PROFILE["thinking_style"] != FEMALE_PROFILE["thinking_style"]
        assert MALE_PROFILE["social_style"] != FEMALE_PROFILE["social_style"]

    def test_male_profile_content(self):
        assert "male" in MALE_PROFILE["base_system"].lower()
        assert "masculine" in MALE_PROFILE["base_system"].lower()

    def test_female_profile_content(self):
        assert "female" in FEMALE_PROFILE["base_system"].lower()
        assert "feminine" in FEMALE_PROFILE["base_system"].lower()


class TestGetProfile:
    def test_get_male(self):
        assert get_profile("male") is MALE_PROFILE

    def test_get_female(self):
        assert get_profile("female") is FEMALE_PROFILE

    def test_get_male_shorthand(self):
        assert get_profile("m") is MALE_PROFILE

    def test_get_female_shorthand(self):
        assert get_profile("f") is FEMALE_PROFILE

    def test_case_insensitive(self):
        assert get_profile("Male") is MALE_PROFILE
        assert get_profile("FEMALE") is FEMALE_PROFILE

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown sex"):
            get_profile("other")


# ── Reflex prompt tests ────────────────────────────────────────────


class TestReflexPrompts:
    EXPECTED_REFLEXES = [
        "fight_or_flight",
        "startle",
        "disgust_withdrawal",
        "social_bonding",
        "curiosity_orienting",
    ]

    def test_all_reflexes_present(self):
        for name in self.EXPECTED_REFLEXES:
            assert name in REFLEX_PROMPTS

    def test_each_reflex_has_system_prompt(self):
        for name, reflex in REFLEX_PROMPTS.items():
            assert "system" in reflex, f"{name} missing 'system' key"
            assert isinstance(reflex["system"], str)
            assert len(reflex["system"]) > 50  # Non-trivial prompt

    def test_each_reflex_has_emotions_list(self):
        for name, reflex in REFLEX_PROMPTS.items():
            assert "emotions" in reflex, f"{name} missing 'emotions' key"
            assert isinstance(reflex["emotions"], list)
            assert len(reflex["emotions"]) > 0

    def test_reflex_prompts_mention_json(self):
        for name, reflex in REFLEX_PROMPTS.items():
            assert "json" in reflex["system"].lower(), (
                f"{name} system prompt should mention JSON output format"
            )

    def test_reflex_prompts_mention_fires(self):
        for name, reflex in REFLEX_PROMPTS.items():
            assert "fires" in reflex["system"].lower(), (
                f"{name} system prompt should mention 'fires' field"
            )

    def test_supervisor_unconscious_system_prompt(self):
        assert "Supervisor-1" in SUPERVISOR_UNCONSCIOUS_SYSTEM
        assert "reflexes" in SUPERVISOR_UNCONSCIOUS_SYSTEM.lower()
        assert "JSON" in SUPERVISOR_UNCONSCIOUS_SYSTEM
