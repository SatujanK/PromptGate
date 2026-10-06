"""Write one JSON line per run to logs/runs.jsonl, and mark runs as corrected later.

The log is the project's measuring stick: the main success metric is fewer
corrected answers per 100 prompts. Each line is one LogRecord as JSON.
"""

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

from promptgate.schemas import LogRecord


class RunNotFoundError(LookupError):
    """Raised when `promptgate mark` is given a run_id that isn't in the log."""


def new_run_id() -> str:
    """A short random ID that is easy to copy, e.g. "3f9a1c07b2e4"."""
    return uuid.uuid4().hex[:12]


def now_timestamp() -> str:
    """The current time in UTC as ISO 8601 text, e.g. "2026-09-30T14:03:11+00:00"."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def append_record(record: LogRecord, log_file: Path) -> None:
    """Add one record to the end of the log as a single line of JSON.

    Creates the logs folder the first time. Newlines inside the prompt are
    escaped as \\n by the JSON encoder, so one run is always one line.
    """
    log_file.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps Unix line endings on Windows too, so every line looks the same.
    with open(log_file, "a", encoding="utf-8", newline="\n") as file:
        file.write(record.model_dump_json() + "\n")


def read_records(log_file: Path) -> list[LogRecord]:
    """Read every record in the log, oldest first. Blank lines are skipped."""
    if not log_file.exists():
        return []
    records = []
    with open(log_file, encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(LogRecord.model_validate_json(line))
    return records


def mark_corrected(run_id: str, log_file: Path) -> None:
    """Set "corrected": true on the line for run_id.

    Every other line is copied back exactly as it was. The new version is
    written to a temporary file first and then swapped in, so a crash
    part-way through can never leave a half-written log.
    """
    if not log_file.exists():
        raise RunNotFoundError(f"there is no log yet at {log_file}")

    with open(log_file, encoding="utf-8", newline="\n") as file:
        # Split on "\n" only. splitlines() would also split on characters such as
        # U+2028 that JSON leaves unescaped inside a prompt, breaking that line in two.
        lines = file.read().split("\n")
    if lines and lines[-1] == "":
        lines.pop()  # the file ends with "\n", which leaves one empty piece at the end

    found = False
    new_lines = []
    for line in lines:
        data = parse_line(line)
        if data is not None and data.get("run_id") == run_id:
            data["corrected"] = True
            line = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
            found = True
        new_lines.append(line)

    if not found:
        raise RunNotFoundError(f"no run with id {run_id!r} in {log_file}")

    temp_file = log_file.with_name(log_file.name + ".tmp")
    with open(temp_file, "w", encoding="utf-8", newline="\n") as file:
        file.write("\n".join(new_lines) + "\n")
    temp_file.replace(log_file)  # swaps the files in one step


def parse_line(line: str) -> dict | None:
    """Parse one log line as JSON. Returns None for blank or damaged lines, which are kept as-is."""
    if not line.strip():
        return None
    try:
        data = json.loads(line)
    except json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None
