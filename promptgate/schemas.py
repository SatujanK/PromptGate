"""Data shapes shared across PromptGate, written as pydantic models.

pydantic checks data when a model is created. Bad data, such as a misspelled
slot name or a probability of 1.7, fails right away with a clear error
instead of causing a confusing bug later in the pipeline.
"""

from typing import Annotated, Literal, Self, get_args

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Names used everywhere
# ---------------------------------------------------------------------------

SlotName = Literal[
    "role",
    "inputs",
    "context",
    "objective",
    "requirements",
    "constraints",
    "resources",
    "output_format",
    "success_criteria",
    "examples",
]

# The order slots appear in the user message. Role goes in the system prompt instead.
USER_MESSAGE_ORDER: tuple[SlotName, ...] = (
    "inputs",
    "context",
    "objective",
    "requirements",
    "constraints",
    "resources",
    "output_format",
    "success_criteria",
    "examples",
)

ALL_SLOTS: tuple[SlotName, ...] = ("role",) + USER_MESSAGE_ORDER

# Slots Jev checks with a has_<slot> question. There is no check for examples,
# which is never required. has_role only decides whether default_role is sent.
JEV_CHECKED_SLOTS: tuple[SlotName, ...] = (
    "role",
    "objective",
    "context",
    "inputs",
    "requirements",
    "constraints",
    "resources",
    "output_format",
    "success_criteria",
)

TaskType = Literal[
    "quick_question",
    "transformation",
    "explanation",
    "generation",
    "analysis",
    "multi_step_project",
]
TASK_TYPES: tuple[TaskType, ...] = get_args(TaskType)

ProblemType = Literal["none", "unclear_intent", "contradictory", "multiple_tasks", "unclear"]
PROBLEM_TYPES: tuple[ProblemType, ...] = get_args(ProblemType)

# Jev cannot abstain, so every Choice question also offers "unclear".
UNCLEAR = "unclear"

Route = Literal["as_is", "fill_missing", "full_form", "split"]
Effort = Literal["low", "medium", "high"]
CheckpointMode = Literal["on", "off", "if_large_output"]
SlotStatus = Literal["stated", "missing", "conflict"]

# A number from 0 to 1. pydantic rejects anything outside that range.
Probability = Annotated[float, Field(ge=0, le=1)]


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------


class ChoiceAnswer(BaseModel):
    """Jev's answer to one Choice question.

    Example: choice="analysis", confidence=0.82,
    probabilities={"analysis": 0.9, "generation": 0.05, ...}
    """

    choice: str  # the label with the highest probability
    confidence: Probability  # how concentrated the probabilities are (1 = all on one label)
    probabilities: dict[str, Probability]  # every label and its probability


class JevVerdict(BaseModel):
    """Everything PromptGate needs from the single Jev call.

    Noul answers are plain probabilities with no confidence, so they are
    stored as floats and there is nothing to misread.
    """

    model: str  # the versioned Jev model that answered, e.g. "jev-1.13.0"
    task_type: ChoiceAnswer
    problem_type: ChoiceAnswer
    presence: dict[SlotName, Probability]  # has_<slot> answers, e.g. {"objective": 0.93, ...}
    expects_large_output: Probability

    @model_validator(mode="after")
    def check_labels(self) -> Self:
        missing = set(JEV_CHECKED_SLOTS) - set(self.presence)
        if missing:
            raise ValueError(f"presence is missing slots: {sorted(missing)}")

        allowed_task_labels = set(TASK_TYPES) | {UNCLEAR}
        task_labels = set(self.task_type.probabilities) | {self.task_type.choice}
        unknown = task_labels - allowed_task_labels
        if unknown:
            raise ValueError(f"unknown task_type labels: {sorted(unknown)}")

        problem_labels = set(self.problem_type.probabilities) | {self.problem_type.choice}
        unknown = problem_labels - set(PROBLEM_TYPES)
        if unknown:
            raise ValueError(f"unknown problem_type labels: {sorted(unknown)}")
        return self


# ---------------------------------------------------------------------------
# Inputs and extraction
# ---------------------------------------------------------------------------


class Document(BaseModel):
    """One item of material Claude works on, passed through word for word."""

    source: str  # a file name, or "prompt" when it was copied out of the prompt
    content: str


class SlotResult(BaseModel):
    """What Haiku found for one slot.

    - stated:   the prompt says it; value holds the user's own words.
    - missing:  the prompt doesn't say it; value stays empty.
    - conflict: the prompt says it two or more ways; options holds each version.
    """

    status: SlotStatus
    value: str | list[str] | None = None
    options: list[str] = []

    @model_validator(mode="after")
    def check_options(self) -> Self:
        if self.status == "conflict" and len(self.options) < 2:
            raise ValueError("a conflict needs at least two options to choose from")
        if self.status != "conflict":
            # options only mean something for a conflict, so drop any extras
            self.options = []
        return self


class Extraction(BaseModel):
    """Haiku's copy of the prompt, split into slots. It copies and never invents."""

    role: SlotResult
    inputs: SlotResult
    context: SlotResult
    objective: SlotResult
    requirements: SlotResult
    constraints: SlotResult
    resources: SlotResult
    output_format: SlotResult
    success_criteria: SlotResult
    examples: SlotResult
    candidate_objectives: list[str] = []  # 2-3 options, used only when the objective is vague

    def slot(self, name: SlotName) -> SlotResult:
        """Return the result for one slot by name, e.g. extraction.slot("objective")."""
        return getattr(self, name)


class SplitResult(BaseModel):
    """Haiku's split of a prompt that holds several tasks. Everything is copied, never written."""

    parts: list[str]  # each task, in the user's own words
    shared_context: str | None = None  # background that applies to every part
    shared_inputs: list[str] = []  # material every part works on, copied word for word


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


class TokenUsage(BaseModel):
    """Token counts for one API call."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0  # tokens written to the cache (charged a little extra)
    cache_read_input_tokens: int = 0  # tokens read from the cache (charged much less)


class LogRecord(BaseModel):
    """One line of logs/runs.jsonl, describing one run."""

    run_id: str
    parent_run_id: str | None = None  # set on each part of a split
    timestamp: str  # ISO 8601 in UTC, e.g. "2026-09-24T14:03:11+00:00"
    dry_run: bool = False
    prompt: str
    input_sources: list[str] = []  # file names given with --input
    task_type: TaskType | None = None  # the profile actually used, after any fallback
    jev: JevVerdict | None = None
    jev_error: str | None = None
    route: Route  # the route actually taken
    split_declined: bool = False  # Jev saw several tasks but you chose to send it as-is
    missing_slots: list[SlotName] = []
    skipped_slots: list[SlotName] = []  # questions you left blank
    checkpoint: bool = False
    plan_edited: bool = False
    plan_truncated: bool = False
    cancelled: bool = False
    effort: Effort | None = None
    models: dict[str, str] = {}
    tokens: dict[str, TokenUsage] = {}  # keys: "jev", "extract", "plan", "execute"
    warnings: list[str] = []  # problems that were worked around, e.g. a failed extraction
    error: str | None = None  # the problem that stopped the run, if one did
    corrected: bool | None = None  # set later with: promptgate mark <run_id> --corrected
