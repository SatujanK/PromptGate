"""Tests for log.py. Every test writes to a temporary folder, never to your real log."""

import json
from datetime import datetime, timedelta

import pytest

from promptgate.log import (
    RunNotFoundError,
    append_record,
    mark_corrected,
    new_run_id,
    now_timestamp,
    read_records,
)
from promptgate.schemas import LogRecord, TokenUsage
from tests.fixtures import make_verdict


def make_record(run_id: str = "run1", prompt: str = "What does df.shape return?") -> LogRecord:
    return LogRecord(
        run_id=run_id,
        timestamp="2026-09-30T10:00:00+00:00",
        prompt=prompt,
        task_type="quick_question",
        jev=make_verdict(),
        route="as_is",
        effort="low",
        models={"execution": "claude-sonnet-5", "jev": "jev-1.13.0"},
        tokens={"execute": TokenUsage(input_tokens=12, output_tokens=40)},
    )


@pytest.fixture
def log_file(tmp_path):
    return tmp_path / "logs" / "runs.jsonl"


def read_lines(log_file) -> list[str]:
    return log_file.read_text(encoding="utf-8").splitlines()


# ---------------------------------------------------------------------------
# Run IDs and timestamps
# ---------------------------------------------------------------------------


def test_run_ids_are_short_and_unique():
    ids = {new_run_id() for _ in range(1000)}
    assert len(ids) == 1000
    assert all(len(run_id) == 12 for run_id in ids)


def test_timestamp_is_iso_8601_in_utc():
    parsed = datetime.fromisoformat(now_timestamp())
    assert parsed.utcoffset() == timedelta(0)


# ---------------------------------------------------------------------------
# Appending
# ---------------------------------------------------------------------------


def test_append_creates_the_folder_and_writes_one_valid_line(log_file):
    append_record(make_record(), log_file)
    lines = read_lines(log_file)
    assert len(lines) == 1
    json.loads(lines[0])  # valid JSON
    assert LogRecord.model_validate_json(lines[0]) == make_record()


def test_each_run_adds_exactly_one_line_in_order(log_file):
    for run_id in ["a", "b", "c"]:
        append_record(make_record(run_id), log_file)
    assert [json.loads(line)["run_id"] for line in read_lines(log_file)] == ["a", "b", "c"]


def test_a_multi_line_prompt_still_takes_one_line(log_file):
    append_record(make_record(prompt="line one\nline two\n\nline four"), log_file)
    lines = read_lines(log_file)
    assert len(lines) == 1
    assert json.loads(lines[0])["prompt"] == "line one\nline two\n\nline four"


def test_non_ascii_text_survives(log_file):
    append_record(make_record(prompt="Traduis « café » en anglais"), log_file)
    assert read_records(log_file)[0].prompt == "Traduis « café » en anglais"


def test_lines_use_unix_line_endings(log_file):
    append_record(make_record("a"), log_file)
    append_record(make_record("b"), log_file)
    assert b"\r\n" not in log_file.read_bytes()


def test_read_records_on_a_missing_log_is_empty(log_file):
    assert read_records(log_file) == []


# ---------------------------------------------------------------------------
# promptgate mark <run_id> --corrected
# ---------------------------------------------------------------------------


def test_mark_sets_corrected_on_the_matching_run_only(log_file):
    for run_id in ["a", "b", "c"]:
        append_record(make_record(run_id), log_file)
    mark_corrected("b", log_file)
    assert [record.corrected for record in read_records(log_file)] == [None, True, None]


def test_mark_leaves_other_lines_byte_for_byte(log_file):
    for run_id in ["a", "b", "c"]:
        append_record(make_record(run_id), log_file)
    before = read_lines(log_file)
    mark_corrected("b", log_file)
    after = read_lines(log_file)
    assert after[0] == before[0]
    assert after[2] == before[2]
    assert after[1] != before[1]


def test_marked_line_is_still_one_valid_record(log_file):
    append_record(make_record("a", prompt="multi\nline « é »"), log_file)
    mark_corrected("a", log_file)
    lines = read_lines(log_file)
    assert len(lines) == 1
    record = LogRecord.model_validate_json(lines[0])
    assert record.corrected is True
    assert record.prompt == "multi\nline « é »"


def test_marking_twice_is_harmless(log_file):
    append_record(make_record("a"), log_file)
    mark_corrected("a", log_file)
    mark_corrected("a", log_file)
    assert read_records(log_file)[0].corrected is True


def test_mark_unknown_run_raises_and_changes_nothing(log_file):
    append_record(make_record("a"), log_file)
    before = log_file.read_bytes()
    with pytest.raises(RunNotFoundError, match="no run with id 'zzz'"):
        mark_corrected("zzz", log_file)
    assert log_file.read_bytes() == before


def test_mark_with_no_log_yet_raises(log_file):
    with pytest.raises(RunNotFoundError, match="no log yet"):
        mark_corrected("a", log_file)


def test_mark_leaves_no_temporary_file_behind(log_file):
    append_record(make_record("a"), log_file)
    mark_corrected("a", log_file)
    assert [path.name for path in log_file.parent.iterdir()] == ["runs.jsonl"]


def test_mark_does_not_split_a_prompt_containing_unicode_line_separators(log_file):
    tricky = "para one para two end\x85"
    append_record(make_record("a", prompt=tricky), log_file)
    append_record(make_record("b"), log_file)
    mark_corrected("a", log_file)
    records = read_records(log_file)
    assert [record.run_id for record in records] == ["a", "b"]
    assert records[0].prompt == tricky
    assert records[0].corrected is True


def test_mark_keeps_damaged_lines_as_they_were(log_file):
    append_record(make_record("a"), log_file)
    with open(log_file, "a", encoding="utf-8", newline="\n") as file:
        file.write("{not valid json\n")
    append_record(make_record("b"), log_file)
    mark_corrected("b", log_file)
    lines = read_lines(log_file)
    assert lines[1] == "{not valid json"
    assert json.loads(lines[2])["corrected"] is True
