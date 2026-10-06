"""Tests for the shared pydantic models."""

import pytest
from pydantic import ValidationError

from promptgate.schemas import (
    ALL_SLOTS,
    JEV_CHECKED_SLOTS,
    Extraction,
    JevVerdict,
    LogRecord,
    SlotResult,
    TokenUsage,
)


def make_verdict_data() -> dict:
    """A valid JevVerdict as a plain dictionary."""
    return {
        "model": "jev-1.13.0",
        "task_type": {
            "choice": "quick_question",
            "confidence": 0.9,
            "probabilities": {"quick_question": 0.95, "explanation": 0.05},
        },
        "problem_type": {
            "choice": "none",
            "confidence": 0.9,
            "probabilities": {"none": 0.95, "unclear": 0.05},
        },
        "presence": {slot: 0.1 for slot in JEV_CHECKED_SLOTS},
        "expects_large_output": 0.1,
    }


# ---------------------------------------------------------------------------
# JevVerdict
# ---------------------------------------------------------------------------


def test_valid_verdict_loads():
    verdict = JevVerdict.model_validate(make_verdict_data())
    assert verdict.task_type.choice == "quick_question"
    assert verdict.presence["objective"] == 0.1


def test_verdict_needs_every_checked_slot():
    data = make_verdict_data()
    del data["presence"]["role"]
    with pytest.raises(ValidationError, match="missing slots"):
        JevVerdict.model_validate(data)


def test_verdict_rejects_probability_above_one():
    data = make_verdict_data()
    data["presence"]["objective"] = 1.7
    with pytest.raises(ValidationError):
        JevVerdict.model_validate(data)


def test_verdict_rejects_unknown_task_type_label():
    data = make_verdict_data()
    data["task_type"]["probabilities"]["poem"] = 0.0
    with pytest.raises(ValidationError, match="unknown task_type labels"):
        JevVerdict.model_validate(data)


def test_verdict_accepts_unclear_task_type():
    data = make_verdict_data()
    data["task_type"]["probabilities"]["unclear"] = 0.0
    JevVerdict.model_validate(data)


# ---------------------------------------------------------------------------
# SlotResult and Extraction
# ---------------------------------------------------------------------------


def test_conflict_needs_two_options():
    with pytest.raises(ValidationError, match="at least two options"):
        SlotResult(status="conflict", options=["in full detail"])


def test_conflict_keeps_its_options():
    result = SlotResult(status="conflict", options=["in full detail", "in under 50 words"])
    assert result.options == ["in full detail", "in under 50 words"]


def test_options_are_dropped_unless_conflict():
    result = SlotResult(status="stated", value="Summarize it", options=["stray"])
    assert result.options == []


def test_value_can_be_text_or_list():
    assert SlotResult(status="stated", value="one").value == "one"
    assert SlotResult(status="stated", value=["a", "b"]).value == ["a", "b"]


def test_extraction_slot_lookup():
    data = {slot: {"status": "missing"} for slot in ALL_SLOTS}
    data["objective"] = {"status": "stated", "value": "Translate this into French"}
    extraction = Extraction.model_validate(data)
    assert extraction.slot("objective").value == "Translate this into French"
    assert extraction.slot("inputs").status == "missing"
    assert extraction.candidate_objectives == []


# ---------------------------------------------------------------------------
# LogRecord
# ---------------------------------------------------------------------------


def test_log_record_survives_a_json_round_trip():
    record = LogRecord(
        run_id="abc123",
        timestamp="2026-09-24T14:03:11+00:00",
        prompt="What does df.shape return?",
        task_type="quick_question",
        jev=JevVerdict.model_validate(make_verdict_data()),
        route="as_is",
        effort="low",
        models={"execution": "claude-sonnet-5"},
        tokens={"execute": TokenUsage(input_tokens=20, output_tokens=80)},
    )
    assert LogRecord.model_validate_json(record.model_dump_json()) == record


def test_log_record_rejects_unknown_route():
    with pytest.raises(ValidationError):
        LogRecord(run_id="x", timestamp="t", prompt="p", route="somewhere")
