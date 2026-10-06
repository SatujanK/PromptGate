"""Decide what to do with a prompt, using Jev's verdict and the task-type profiles.

Everything here is a pure function: no network, no files, no input().
The same inputs always give the same answer, so every rule can be tested
without spending anything.
"""

from collections.abc import Collection
from typing import cast

from pydantic import BaseModel

from promptgate.config import Config, Profile
from promptgate.schemas import (
    UNCLEAR,
    Effort,
    JevVerdict,
    ProblemType,
    Probability,
    Route,
    SlotName,
    TaskType,
)


class Decision(BaseModel):
    """Everything the rest of the pipeline needs to know about one prompt."""

    task_type: TaskType | None  # the profile used; None only when Jev failed
    task_type_uncertain: bool  # True when Jev was unsure and the lighter of its top two was used
    problem_type: ProblemType  # after the confidence check, so low confidence reads as "none"
    route: Route
    missing_slots: list[SlotName]  # required slots Jev says the prompt doesn't have
    form_slots: list[SlotName]  # required + optional slots, shown in the full form
    checkpoint: bool  # make a planning call and wait for approval first
    effort: Effort  # used for both the planning and execution calls
    system_prompt: str | None  # the profile's default_role, unless the prompt states a role
    ambiguity_rule: bool  # append the ambiguity rule (every task type except quick_question)


def choose_task_type(
    verdict: JevVerdict, profiles: dict[TaskType, Profile], min_confidence: float
) -> tuple[TaskType, bool]:
    """Pick the task type to use. Returns (task_type, was_uncertain).

    If Jev is confident, use its choice. If not, or if it chose "unclear",
    use the lighter of its top two real task types.
    """
    answer = verdict.task_type
    if answer.choice != UNCLEAR and answer.confidence >= min_confidence:
        return cast(TaskType, answer.choice), False

    # Rank the real task types by probability, highest first, skipping "unclear".
    ranked = sorted(
        (label for label in answer.probabilities if label != UNCLEAR),
        key=lambda label: answer.probabilities[label],
        reverse=True,
    )
    top_two = ranked[:2]
    if not top_two:
        # Jev gave no usable labels at all, so consider every profile.
        top_two = list(profiles)

    # Uncertainty should add less friction, not more: take the lighter profile.
    lighter = min(top_two, key=lambda task_type: profiles[task_type].weight)
    return cast(TaskType, lighter), True


def effective_problem_type(verdict: JevVerdict, min_confidence: float) -> ProblemType:
    """Return Jev's problem_type, or "none" if Jev isn't confident or chose "unclear"."""
    answer = verdict.problem_type
    if answer.choice == UNCLEAR or answer.confidence < min_confidence:
        return "none"
    return cast(ProblemType, answer.choice)


def find_missing_slots(
    profile: Profile, presence: dict[SlotName, Probability], present_threshold: float
) -> list[SlotName]:
    """Return the profile's required slots that Jev scored below the threshold, in profile order."""
    return [slot for slot in profile.required if presence[slot] < present_threshold]


def needs_checkpoint(profile: Profile, expects_large_output: float, threshold: float) -> bool:
    """Whether this run gets a planning call and waits for your approval."""
    if profile.checkpoint == "on":
        return True
    if profile.checkpoint == "if_large_output":
        return expects_large_output >= threshold
    return False


def choose_route(problem_type: ProblemType, missing_slots: list[SlotName], allow_split: bool) -> Route:
    """Pick the route. The first matching rule wins."""
    if problem_type == "multiple_tasks" and allow_split:
        return "split"
    if problem_type in ("unclear_intent", "contradictory"):
        return "full_form"
    if missing_slots:
        return "fill_missing"
    return "as_is"


def pick_system_prompt(profile: Profile, has_role: float, present_threshold: float) -> str | None:
    """Return the default role, unless the prompt already states a role (or there is none)."""
    if has_role >= present_threshold:
        return None
    return profile.default_role or None


def decide(
    verdict: JevVerdict,
    config: Config,
    known_slots: Collection[SlotName] = (),
    allow_split: bool = True,
) -> Decision:
    """Turn Jev's verdict into a Decision.

    known_slots: slots code already has, which Jev never saw. For example
        "inputs" when --input files were given, or "context" for a part of a
        split. They count as present.
    allow_split: False for the parts of a split, so a part is never split again.
    """
    thresholds = config.thresholds

    presence = dict(verdict.presence)  # a copy, so the verdict itself is never changed
    for slot in known_slots:
        presence[slot] = 1.0

    task_type, uncertain = choose_task_type(
        verdict, config.profiles, thresholds.task_type_confidence_min
    )
    profile = config.profiles[task_type]
    problem_type = effective_problem_type(verdict, thresholds.problem_confidence_min)
    missing = find_missing_slots(profile, presence, thresholds.present_threshold)

    return Decision(
        task_type=task_type,
        task_type_uncertain=uncertain,
        problem_type=problem_type,
        route=choose_route(problem_type, missing, allow_split),
        missing_slots=missing,
        form_slots=profile.required + profile.optional,
        checkpoint=needs_checkpoint(
            profile, verdict.expects_large_output, thresholds.large_output_threshold
        ),
        effort=profile.effort,
        system_prompt=pick_system_prompt(profile, presence["role"], thresholds.present_threshold),
        ambiguity_rule=task_type != "quick_question",
    )


def decide_without_jev(config: Config) -> Decision:
    """The Decision used when Jev can't be reached: send the prompt as-is, add no friction."""
    return Decision(
        task_type=None,
        task_type_uncertain=True,
        problem_type="none",
        route="as_is",
        missing_slots=[],
        form_slots=[],
        checkpoint=False,
        effort=config.fallback.effort,
        system_prompt=None,
        ambiguity_rule=True,
    )
