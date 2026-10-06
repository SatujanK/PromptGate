"""Tests for env.py, the .env files, and how the command loads them.

Variable names here start with PROMPTGATE_TEST_ so they can't clash with
real ones. monkeypatch.delenv removes them again after each test.
"""

import os

import pytest

from promptgate import cli
from promptgate.config import PROJECT_ROOT
from promptgate.env import load_env_file, parse_env_line

NAMES = ["PROMPTGATE_TEST_A", "PROMPTGATE_TEST_B", "PROMPTGATE_TEST_C"]


@pytest.fixture(autouse=True)
def clean_test_variables(monkeypatch):
    for name in NAMES:
        monkeypatch.delenv(name, raising=False)  # also undoes anything a test sets


def write_env(tmp_path, text):
    path = tmp_path / ".env"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Reading lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line, expected",
    [
        ("NAME=value", ("NAME", "value")),
        ("  NAME = value  ", ("NAME", "value")),
        ('NAME="quoted value"', ("NAME", "quoted value")),
        ("NAME='single'", ("NAME", "single")),
        ("NAME=abc=def", ("NAME", "abc=def")),
        ("NAME=", ("NAME", "")),
        ("# a comment", None),
        ("", None),
        ("no equals sign", None),
    ],
)
def test_parse_env_line(line, expected):
    assert parse_env_line(line) == expected


# ---------------------------------------------------------------------------
# Loading a file
# ---------------------------------------------------------------------------


def test_loads_variables_that_are_not_set(tmp_path, monkeypatch):
    path = write_env(tmp_path, "# keys\nPROMPTGATE_TEST_A=one\n\nPROMPTGATE_TEST_B=two\n")
    assert load_env_file(path) == ["PROMPTGATE_TEST_A", "PROMPTGATE_TEST_B"]
    assert os.environ["PROMPTGATE_TEST_A"] == "one"
    assert os.environ["PROMPTGATE_TEST_B"] == "two"


def test_the_real_environment_wins_over_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PROMPTGATE_TEST_A", "from-windows")
    path = write_env(tmp_path, "PROMPTGATE_TEST_A=from-file\n")
    assert load_env_file(path) == []
    assert os.environ["PROMPTGATE_TEST_A"] == "from-windows"


def test_blank_values_are_left_unset(tmp_path):
    path = write_env(tmp_path, "PROMPTGATE_TEST_A=\n")
    assert load_env_file(path) == []
    assert "PROMPTGATE_TEST_A" not in os.environ


def test_a_missing_file_sets_nothing(tmp_path):
    assert load_env_file(tmp_path / "nope.env") == []


def test_a_file_saved_by_notepad_still_loads(tmp_path):
    path = tmp_path / ".env"
    path.write_bytes("PROMPTGATE_TEST_C=ok\r\n".encode("utf-8-sig"))  # BOM + Windows line endings
    assert load_env_file(path) == ["PROMPTGATE_TEST_C"]


# ---------------------------------------------------------------------------
# The project's files
# ---------------------------------------------------------------------------


def read_pairs(path) -> dict[str, str]:
    pairs = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parsed = parse_env_line(line)
        if parsed:
            pairs[parsed[0]] = parsed[1]
    return pairs


def test_env_example_lists_both_keys_with_only_fake_values():
    pairs = read_pairs(PROJECT_ROOT / ".env.example")
    assert set(pairs) == set(cli.API_KEY_VARIABLES)
    for value in pairs.values():
        assert cli.PLACEHOLDER_MARKER in value  # fake, and recognised as fake


def test_env_is_ignored_by_git_but_the_example_is_not():
    ignored = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in ignored
    assert not any(line.strip().startswith(".env.") or line.strip() == ".env*" for line in ignored)


# ---------------------------------------------------------------------------
# How the command uses it
# ---------------------------------------------------------------------------


def test_the_command_loads_env_before_running(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, "load_env_file", lambda: calls.append("load_env"))
    monkeypatch.setattr(cli, "main", lambda: calls.append("main") or 0)
    with pytest.raises(SystemExit) as exit_info:
        cli.entry_point()
    assert calls == ["load_env", "main"]
    assert exit_info.value.code == 0


def test_main_itself_never_reads_env(monkeypatch, tmp_path):
    """Tests call main() directly, so they must never pick up your real keys from .env."""
    monkeypatch.setattr(cli, "load_env_file", lambda: pytest.fail("main() read .env"))
    config = cli.load_config()
    config.paths.log_file = tmp_path / "runs.jsonl"
    with pytest.raises(SystemExit):  # stops at the missing-key check, after not reading .env
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
        cli.main(["What does df.shape return?"], config=config)


def test_placeholder_keys_count_as_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-fake-key-replace-me")
    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-fake-key-replace-me")
    config = cli.load_config()
    config.paths.log_file = tmp_path / "runs.jsonl"
    with pytest.raises(SystemExit, match="Set ANTHROPIC_API_KEY and TYPESAFE_API_KEY"):
        cli.main(["What does df.shape return?"], config=config)
