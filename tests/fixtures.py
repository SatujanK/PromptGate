"""Faked Jev verdicts and the sample prompts, shared by several test files.

These stand in for real Jev responses, so tests never call the API.
"""

from dataclasses import dataclass

from promptgate.schemas import (
    ALL_SLOTS,
    JEV_CHECKED_SLOTS,
    PROBLEM_TYPES,
    TASK_TYPES,
    UNCLEAR,
    Extraction,
    JevVerdict,
)

PRESENT = 0.9  # what a fake has_<slot> answer returns for a slot the prompt has
ABSENT = 0.1  # ... and for a slot it doesn't have


def make_verdict(
    task_type: str = "quick_question",
    task_confidence: float = 0.9,
    task_probabilities: dict[str, float] | None = None,
    problem_type: str = "none",
    problem_confidence: float = 0.9,
    present: tuple[str, ...] = ("objective",),
    has_role: float = ABSENT,
    expects_large_output: float = 0.1,
) -> JevVerdict:
    """Build a fake Jev verdict.

    present: the slots the prompt has; every other checked slot scores ABSENT.
    task_probabilities: set this to control the fallback; by default the
        chosen task type gets most of the probability.
    """
    if task_probabilities is None:
        task_probabilities = {label: 0.02 for label in TASK_TYPES + (UNCLEAR,)}
        task_probabilities[task_type] = 0.86

    problem_probabilities = {label: 0.02 for label in PROBLEM_TYPES}
    problem_probabilities[problem_type] = 0.92

    presence = {slot: (PRESENT if slot in present else ABSENT) for slot in JEV_CHECKED_SLOTS}
    presence["role"] = has_role

    return JevVerdict(
        model="jev-1.13.0",
        task_type={
            "choice": task_type,
            "confidence": task_confidence,
            "probabilities": task_probabilities,
        },
        problem_type={
            "choice": problem_type,
            "confidence": problem_confidence,
            "probabilities": problem_probabilities,
        },
        presence=presence,
        expects_large_output=expects_large_output,
    )


def make_extraction(candidate_objectives: list[str] | None = None, **slots) -> Extraction:
    """Build a fake Haiku extraction. Every slot is missing unless given.

    A slot can be given as text or a list (meaning "stated"), or as a full
    dict such as {"status": "conflict", "options": ["a", "b"]}.
    """
    data: dict = {name: {"status": "missing"} for name in ALL_SLOTS}
    for name, value in slots.items():
        if isinstance(value, dict):
            data[name] = value
        else:
            data[name] = {"status": "stated", "value": value}
    data["candidate_objectives"] = candidate_objectives or []
    return Extraction.model_validate(data)


@dataclass
class Sample:
    """One sample prompt, the verdict Jev is faked to return, and what should happen."""

    prompt: str
    verdict: JevVerdict
    input_files: list[str]
    expected_task_type: str | None  # None means "any"
    expected_route: str
    expected_checkpoint: bool | None  # None means "depends on each part"
    expected_missing: list[str]


ARTICLE = "article.txt"

SAMPLES = [
    Sample(
        prompt="What does df.shape return?",
        verdict=make_verdict(task_type="quick_question", present=("objective",)),
        input_files=[],
        expected_task_type="quick_question",
        expected_route="as_is",
        expected_checkpoint=False,
        expected_missing=[],
    ),
    Sample(
        prompt="Translate this into French:",
        verdict=make_verdict(task_type="transformation", present=("objective",)),
        input_files=[],
        expected_task_type="transformation",
        expected_route="fill_missing",
        expected_checkpoint=False,
        expected_missing=["inputs"],
    ),
    Sample(
        prompt="Summarize this article and also write me a cover letter",
        verdict=make_verdict(
            task_type="generation",
            task_confidence=0.4,
            task_probabilities={"generation": 0.45, "analysis": 0.35, "transformation": 0.2},
            problem_type="multiple_tasks",
            problem_confidence=0.85,
            present=("objective",),
        ),
        input_files=[],
        expected_task_type=None,
        expected_route="split",
        expected_checkpoint=None,
        expected_missing=["context", "requirements", "output_format"],
    ),
    Sample(
        prompt="Summarize every section in full detail, in under 50 words",
        verdict=make_verdict(
            task_type="transformation",
            problem_type="contradictory",
            problem_confidence=0.8,
            present=("objective", "output_format"),
        ),
        input_files=[ARTICLE],
        expected_task_type="transformation",
        expected_route="full_form",
        expected_checkpoint=False,
        expected_missing=[],
    ),
    Sample(
        prompt="Build me a study planner app",
        verdict=make_verdict(
            task_type="multi_step_project",
            present=("objective",),
            expects_large_output=0.9,
        ),
        input_files=[],
        expected_task_type="multi_step_project",
        expected_route="fill_missing",
        expected_checkpoint=True,
        expected_missing=[
            "context",
            "requirements",
            "constraints",
            "resources",
            "output_format",
            "success_criteria",
        ],
    ),
]
