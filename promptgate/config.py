"""Load PromptGate settings from config.toml and questions.toml.

Everything you tune lives in those two files: models, thresholds, task-type
profiles, and the pre-written questions. This module reads them, checks them
with pydantic, and returns one Config object.
"""

import tomllib
from pathlib import Path
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from promptgate.schemas import (
    JEV_CHECKED_SLOTS,
    TASK_TYPES,
    CheckpointMode,
    Effort,
    Probability,
    SlotName,
    TaskType,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.toml"
DEFAULT_QUESTIONS_PATH = PROJECT_ROOT / "questions.toml"


class StrictModel(BaseModel):
    """Base class that rejects unknown keys, so a typo in the TOML fails loudly."""

    model_config = ConfigDict(extra="forbid")


class ModelNames(StrictModel):
    extraction: str
    execution: str
    jev: str


class ExecutionModelSettings(StrictModel):
    self_check: bool


class Thresholds(StrictModel):
    present_threshold: Probability
    problem_confidence_min: Probability
    task_type_confidence_min: Probability
    large_output_threshold: Probability


class Limits(StrictModel):
    plan_max_tokens: int = Field(gt=0)
    execute_max_tokens: int = Field(gt=0)
    extract_max_tokens: int = Field(gt=0)
    quote_grounding_tokens: int = Field(gt=0)


class Paths(StrictModel):
    log_file: Path


class Fallback(StrictModel):
    """Settings used when Jev can't be reached and the prompt is sent as-is."""

    effort: Effort


class Profile(StrictModel):
    """The settings for one task type."""

    weight: int  # lower means lighter; used to pick the lighter of two task types
    required: list[SlotName]
    optional: list[SlotName]
    checkpoint: CheckpointMode
    effort: Effort
    default_role: str = ""

    @model_validator(mode="after")
    def check_slots(self) -> Self:
        overlap = set(self.required) & set(self.optional)
        if overlap:
            raise ValueError(f"slots listed as both required and optional: {sorted(overlap)}")
        if "role" in self.required or "role" in self.optional:
            raise ValueError("role is never required or optional; set default_role instead")
        # A slot can only be required if Jev checks for it; otherwise it would always look missing.
        unchecked = [slot for slot in self.required if slot not in JEV_CHECKED_SLOTS]
        if unchecked:
            raise ValueError(f"Jev has no has_<slot> check for required slots: {unchecked}")
        return self


class Config(StrictModel):
    """All settings, checked as a whole."""

    models: ModelNames
    execution_models: dict[str, ExecutionModelSettings]
    thresholds: Thresholds
    limits: Limits
    paths: Paths
    fallback: Fallback
    profiles: dict[TaskType, Profile]
    questions: dict[SlotName, str]

    @model_validator(mode="after")
    def check_consistency(self) -> Self:
        missing = [task_type for task_type in TASK_TYPES if task_type not in self.profiles]
        if missing:
            raise ValueError(f"config.toml has no profile for: {missing}")

        if self.models.execution not in self.execution_models:
            raise ValueError(
                f'add an [execution_models."{self.models.execution}"] section to config.toml'
            )

        # "Use the lighter profile" only makes sense if no two profiles weigh the same.
        weights = [profile.weight for profile in self.profiles.values()]
        if len(set(weights)) != len(weights):
            raise ValueError("each profile needs a different weight")

        for task_type, profile in self.profiles.items():
            for slot in profile.required + profile.optional:
                if slot not in self.questions:
                    raise ValueError(
                        f"questions.toml has no question for '{slot}' (used by {task_type})"
                    )
        return self

    @property
    def self_check(self) -> bool:
        """Whether the configured execution model gets the self-check line."""
        return self.execution_models[self.models.execution].self_check


def read_toml(path: Path) -> dict[str, Any]:
    """Read a TOML file into a dictionary."""
    with open(path, "rb") as file:  # tomllib needs the file opened in binary mode
        return tomllib.load(file)


def load_config(
    config_path: Path = DEFAULT_CONFIG_PATH,
    questions_path: Path = DEFAULT_QUESTIONS_PATH,
) -> Config:
    """Read and check both settings files and return one Config."""
    data = read_toml(config_path)
    data["questions"] = read_toml(questions_path)
    config = Config.model_validate(data)

    # A relative log path is relative to config.toml's folder, so the log
    # ends up in the same place whichever folder you run promptgate from.
    if not config.paths.log_file.is_absolute():
        config.paths.log_file = config_path.resolve().parent / config.paths.log_file
    return config
