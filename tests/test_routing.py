"""Tests for routing.py. Everything here is pure, so no fakes are needed beyond verdicts."""

import pytest

from promptgate.config import load_config
from promptgate.routing import (
    choose_route,
    choose_task_type,
    decide,
    decide_without_jev,
    effective_problem_type,
)
from promptgate.schemas import JEV_CHECKED_SLOTS, TASK_TYPES
from tests.fixtures import SAMPLES, make_verdict

CONFIG = load_config()


# ---------------------------------------------------------------------------
# The sample prompts from the build request
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sample", SAMPLES, ids=[sample.prompt[:30] for sample in SAMPLES])
def test_sample_prompts_route_as_expected(sample):
    known = ["inputs"] if sample.input_files else []
    decision = decide(sample.verdict, CONFIG, known_slots=known)
    if sample.expected_task_type is not None:
        assert decision.task_type == sample.expected_task_type
    assert decision.route == sample.expected_route
    if sample.expected_checkpoint is not None:
        assert decision.checkpoint == sample.expected_checkpoint
    assert decision.missing_slots == sample.expected_missing


# ---------------------------------------------------------------------------
# Every route, and the order the rules are checked in
# ---------------------------------------------------------------------------


def test_route_as_is_when_nothing_is_missing():
    assert choose_route("none", [], allow_split=True) == "as_is"


def test_route_fill_missing_when_a_required_slot_is_missing():
    assert choose_route("none", ["inputs"], allow_split=True) == "fill_missing"


@pytest.mark.parametrize("problem_type", ["unclear_intent", "contradictory"])
def test_route_full_form_for_unclear_or_contradictory(problem_type):
    assert choose_route(problem_type, [], allow_split=True) == "full_form"


def test_route_split_for_multiple_tasks():
    assert choose_route("multiple_tasks", [], allow_split=True) == "split"


def test_split_wins_over_missing_slots():
    assert choose_route("multiple_tasks", ["context"], allow_split=True) == "split"


def test_full_form_wins_over_missing_slots():
    assert choose_route("contradictory", ["context"], allow_split=True) == "full_form"


def test_a_part_of_a_split_is_never_split_again():
    assert choose_route("multiple_tasks", [], allow_split=False) == "as_is"
    assert choose_route("multiple_tasks", ["context"], allow_split=False) == "fill_missing"


def test_decide_passes_allow_split_through():
    verdict = make_verdict(problem_type="multiple_tasks")
    assert decide(verdict, CONFIG).route == "split"
    assert decide(verdict, CONFIG, allow_split=False).route == "as_is"


# ---------------------------------------------------------------------------
# problem_type confidence
# ---------------------------------------------------------------------------


def test_low_confidence_problem_counts_as_none():
    verdict = make_verdict(problem_type="contradictory", problem_confidence=0.59)
    assert effective_problem_type(verdict, 0.6) == "none"
    assert decide(verdict, CONFIG).route == "as_is"


def test_problem_confidence_exactly_at_the_minimum_counts():
    verdict = make_verdict(problem_type="contradictory", problem_confidence=0.6)
    assert effective_problem_type(verdict, 0.6) == "contradictory"


def test_confident_unclear_problem_counts_as_none():
    verdict = make_verdict(problem_type="unclear", problem_confidence=0.95)
    assert effective_problem_type(verdict, 0.6) == "none"


# ---------------------------------------------------------------------------
# Every task-type profile
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_profile_with_every_slot_present_goes_as_is(task_type):
    verdict = make_verdict(task_type=task_type, present=JEV_CHECKED_SLOTS)
    decision = decide(verdict, CONFIG)
    assert decision.task_type == task_type
    assert decision.route == "as_is"
    assert decision.missing_slots == []


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_profile_with_no_slots_present_asks_for_every_required_slot(task_type):
    verdict = make_verdict(task_type=task_type, present=())
    decision = decide(verdict, CONFIG)
    assert decision.route == "fill_missing"
    assert decision.missing_slots == CONFIG.profiles[task_type].required


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_profile_sets_effort_and_form_slots(task_type):
    profile = CONFIG.profiles[task_type]
    decision = decide(make_verdict(task_type=task_type), CONFIG)
    assert decision.effort == profile.effort
    assert decision.form_slots == profile.required + profile.optional


def test_presence_exactly_at_the_threshold_counts_as_present():
    verdict = make_verdict(task_type="transformation", present=("objective",))
    verdict.presence["inputs"] = 0.5
    assert decide(verdict, CONFIG).missing_slots == []


def test_input_files_mark_inputs_as_present():
    verdict = make_verdict(task_type="transformation", present=("objective",))
    assert decide(verdict, CONFIG).missing_slots == ["inputs"]
    assert decide(verdict, CONFIG, known_slots=["inputs"]).missing_slots == []
    # The verdict itself is not changed, so the log still shows what Jev said.
    assert verdict.presence["inputs"] == 0.1


def test_known_context_is_not_asked_for_again():
    verdict = make_verdict(task_type="explanation", present=("objective",))
    assert decide(verdict, CONFIG).missing_slots == ["context"]
    assert decide(verdict, CONFIG, known_slots=["context"]).missing_slots == []


# ---------------------------------------------------------------------------
# Plan checkpoint on and off
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "task_type, expected",
    [
        ("quick_question", False),
        ("transformation", False),
        ("explanation", False),
        ("analysis", True),
        ("multi_step_project", True),
    ],
)
def test_checkpoint_follows_the_profile(task_type, expected):
    # expects_large_output is high, to show it only matters for generation.
    verdict = make_verdict(task_type=task_type, expects_large_output=0.95)
    assert decide(verdict, CONFIG).checkpoint is expected


@pytest.mark.parametrize("large_output, expected", [(0.59, False), (0.6, True), (0.9, True)])
def test_generation_checkpoint_needs_large_output(large_output, expected):
    verdict = make_verdict(task_type="generation", expects_large_output=large_output)
    assert decide(verdict, CONFIG).checkpoint is expected


# ---------------------------------------------------------------------------
# Low-confidence task type falls back to the lighter of the top two
# ---------------------------------------------------------------------------


def test_confident_task_type_is_used_as_is():
    verdict = make_verdict(task_type="analysis", task_confidence=0.8)
    assert choose_task_type(verdict, CONFIG.profiles, 0.5) == ("analysis", False)


def test_task_confidence_exactly_at_the_minimum_counts():
    verdict = make_verdict(task_type="analysis", task_confidence=0.5)
    assert choose_task_type(verdict, CONFIG.profiles, 0.5) == ("analysis", False)


def test_low_confidence_uses_the_lighter_of_the_top_two():
    verdict = make_verdict(
        task_type="analysis",
        task_confidence=0.3,
        task_probabilities={"analysis": 0.45, "explanation": 0.40, "quick_question": 0.15},
    )
    # analysis (weight 5) and explanation (weight 3) are the top two; quick_question is third.
    assert choose_task_type(verdict, CONFIG.profiles, 0.5) == ("explanation", True)


def test_lighter_profile_is_used_even_when_it_is_second():
    verdict = make_verdict(
        task_type="multi_step_project",
        task_confidence=0.2,
        task_probabilities={"multi_step_project": 0.5, "generation": 0.4, "analysis": 0.1},
    )
    decision = decide(verdict, CONFIG)
    assert decision.task_type == "generation"
    assert decision.task_type_uncertain is True
    assert decision.effort == "medium"  # generation's effort, not multi_step_project's


def test_unclear_is_skipped_when_ranking_the_top_two():
    verdict = make_verdict(
        task_type="analysis",
        task_confidence=0.3,
        task_probabilities={"analysis": 0.4, "unclear": 0.35, "multi_step_project": 0.25},
    )
    assert choose_task_type(verdict, CONFIG.profiles, 0.5) == ("analysis", True)


def test_confident_unclear_counts_as_low_confidence():
    verdict = make_verdict(
        task_type="unclear",
        task_confidence=0.9,
        task_probabilities={"unclear": 0.9, "generation": 0.06, "explanation": 0.04},
    )
    assert choose_task_type(verdict, CONFIG.profiles, 0.5) == ("explanation", True)


def test_only_unclear_falls_back_to_the_lightest_profile():
    verdict = make_verdict(
        task_type="unclear", task_confidence=1.0, task_probabilities={"unclear": 1.0}
    )
    assert choose_task_type(verdict, CONFIG.profiles, 0.5) == ("quick_question", True)


# ---------------------------------------------------------------------------
# Role and fixed-line flags
# ---------------------------------------------------------------------------


def test_default_role_is_sent_when_the_prompt_has_no_role():
    decision = decide(make_verdict(task_type="explanation", has_role=0.1), CONFIG)
    assert decision.system_prompt == CONFIG.profiles["explanation"].default_role


def test_stated_role_overrides_the_default_role():
    decision = decide(make_verdict(task_type="explanation", has_role=0.5), CONFIG)
    assert decision.system_prompt is None


def test_profile_without_a_default_role_sends_no_system_prompt():
    decision = decide(make_verdict(task_type="analysis", has_role=0.1), CONFIG)
    assert decision.system_prompt is None


@pytest.mark.parametrize("task_type", TASK_TYPES)
def test_ambiguity_rule_for_every_task_type_except_quick_question(task_type):
    decision = decide(make_verdict(task_type=task_type), CONFIG)
    assert decision.ambiguity_rule is (task_type != "quick_question")


# ---------------------------------------------------------------------------
# Jev failure
# ---------------------------------------------------------------------------


def test_without_jev_the_prompt_goes_as_is_with_no_friction():
    decision = decide_without_jev(CONFIG)
    assert decision.task_type is None
    assert decision.route == "as_is"
    assert decision.checkpoint is False
    assert decision.missing_slots == []
    assert decision.effort == "medium"
    assert decision.ambiguity_rule is True
    assert decision.system_prompt is None
