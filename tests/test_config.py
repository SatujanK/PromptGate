"""Tests for loading and checking config.toml and questions.toml."""

import pytest
from pydantic import ValidationError

from promptgate.config import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_QUESTIONS_PATH,
    PROJECT_ROOT,
    Config,
    load_config,
    read_toml,
)


@pytest.fixture
def raw_config() -> dict:
    """The real settings files as a plain dictionary, ready to break in a test."""
    data = read_toml(DEFAULT_CONFIG_PATH)
    data["questions"] = read_toml(DEFAULT_QUESTIONS_PATH)
    return data


# ---------------------------------------------------------------------------
# The real files match CLAUDE.md
# ---------------------------------------------------------------------------


def test_real_files_load():
    config = load_config()
    assert config.models.extraction == "claude-haiku-4-5-20251001"
    assert config.models.execution == "claude-sonnet-5"
    assert config.models.jev == "jev-1.13.0"
    assert config.limits.plan_max_tokens == 3000
    assert config.fallback.effort == "medium"


def test_thresholds_match_spec():
    thresholds = load_config().thresholds
    assert thresholds.present_threshold == 0.5
    assert thresholds.problem_confidence_min == 0.6
    assert thresholds.task_type_confidence_min == 0.5
    assert thresholds.large_output_threshold == 0.6


# Copied from the profile table in CLAUDE.md, so this test fails if config.toml drifts.
# (weight, required, optional, checkpoint, effort)
SPEC_PROFILES = {
    "quick_question": (1, ["objective"], ["context"], "off", "low"),
    "transformation": (2, ["objective", "inputs"], ["output_format", "constraints"], "off", "low"),
    "explanation": (
        3,
        ["objective", "context"],
        ["output_format", "constraints", "examples"],
        "off",
        "medium",
    ),
    "generation": (
        4,
        ["objective", "context", "requirements", "output_format"],
        ["constraints", "resources", "success_criteria", "examples"],
        "if_large_output",
        "medium",
    ),
    "analysis": (
        5,
        ["objective", "context", "inputs", "requirements", "output_format", "success_criteria"],
        ["constraints", "resources", "examples"],
        "on",
        "high",
    ),
    "multi_step_project": (
        6,
        [
            "objective",
            "context",
            "requirements",
            "constraints",
            "resources",
            "output_format",
            "success_criteria",
        ],
        ["inputs", "examples"],
        "on",
        "high",
    ),
}


@pytest.mark.parametrize("task_type", SPEC_PROFILES)
def test_profiles_match_spec(task_type):
    weight, required, optional, checkpoint, effort = SPEC_PROFILES[task_type]
    profile = load_config().profiles[task_type]
    assert profile.weight == weight
    assert profile.required == required
    assert profile.optional == optional
    assert profile.checkpoint == checkpoint
    assert profile.effort == effort


def test_explanation_has_the_spec_default_role():
    profile = load_config().profiles["explanation"]
    assert profile.default_role.startswith("You are a patient Python tutor")


def test_self_check_follows_the_execution_model():
    config = load_config()
    assert config.self_check is True  # claude-sonnet-5
    config.models.execution = "claude-opus-5"
    assert config.self_check is False


def test_relative_log_path_is_resolved_next_to_config():
    log_file = load_config().paths.log_file
    assert log_file.is_absolute()
    assert log_file == PROJECT_ROOT / "logs" / "runs.jsonl"


# ---------------------------------------------------------------------------
# Broken settings fail loudly
# ---------------------------------------------------------------------------


def test_valid_raw_config_passes(raw_config):
    Config.model_validate(raw_config)  # the baseline the tests below break


def test_typo_in_key_is_rejected(raw_config):
    raw_config["thresholds"]["present_treshold"] = 0.5
    with pytest.raises(ValidationError, match="present_treshold"):
        Config.model_validate(raw_config)


def test_missing_profile_is_rejected(raw_config):
    del raw_config["profiles"]["analysis"]
    with pytest.raises(ValidationError, match="no profile for"):
        Config.model_validate(raw_config)


def test_unknown_task_type_is_rejected(raw_config):
    raw_config["profiles"]["poem"] = raw_config["profiles"]["generation"]
    with pytest.raises(ValidationError):
        Config.model_validate(raw_config)


def test_misspelled_slot_is_rejected(raw_config):
    raw_config["profiles"]["quick_question"]["required"] = ["objectiv"]
    with pytest.raises(ValidationError):
        Config.model_validate(raw_config)


def test_slot_both_required_and_optional_is_rejected(raw_config):
    raw_config["profiles"]["quick_question"]["optional"] = ["objective"]
    with pytest.raises(ValidationError, match="both required and optional"):
        Config.model_validate(raw_config)


def test_role_in_a_profile_is_rejected(raw_config):
    raw_config["profiles"]["explanation"]["optional"].append("role")
    with pytest.raises(ValidationError, match="default_role"):
        Config.model_validate(raw_config)


def test_required_slot_without_a_jev_check_is_rejected(raw_config):
    raw_config["profiles"]["explanation"]["required"].append("examples")
    raw_config["profiles"]["explanation"]["optional"].remove("examples")
    with pytest.raises(ValidationError, match="no has_<slot> check"):
        Config.model_validate(raw_config)


def test_duplicate_weights_are_rejected(raw_config):
    raw_config["profiles"]["analysis"]["weight"] = 1
    with pytest.raises(ValidationError, match="different weight"):
        Config.model_validate(raw_config)


def test_threshold_above_one_is_rejected(raw_config):
    raw_config["thresholds"]["present_threshold"] = 1.5
    with pytest.raises(ValidationError):
        Config.model_validate(raw_config)


def test_unknown_effort_is_rejected(raw_config):
    raw_config["profiles"]["analysis"]["effort"] = "extreme"
    with pytest.raises(ValidationError):
        Config.model_validate(raw_config)


def test_unknown_checkpoint_mode_is_rejected(raw_config):
    raw_config["profiles"]["analysis"]["checkpoint"] = "sometimes"
    with pytest.raises(ValidationError):
        Config.model_validate(raw_config)


def test_execution_model_without_settings_is_rejected(raw_config):
    raw_config["models"]["execution"] = "claude-new-model"
    with pytest.raises(ValidationError, match="execution_models"):
        Config.model_validate(raw_config)


def test_missing_question_is_rejected(raw_config):
    del raw_config["questions"]["examples"]
    with pytest.raises(ValidationError, match="no question for 'examples'"):
        Config.model_validate(raw_config)
