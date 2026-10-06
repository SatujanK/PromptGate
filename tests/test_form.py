"""Tests for form.py: the real terminal (with input() replaced) and the small helpers."""

import pytest

from promptgate.form import TerminalUI, parse_answer, preview, slots_to_show
from tests.fixtures import make_extraction


@pytest.fixture
def typed(monkeypatch):
    """Replace input() with a list of lines, as if typed."""

    def set_lines(lines):
        remaining = list(lines)

        def fake_input(prompt=""):
            if not remaining:
                raise EOFError
            return remaining.pop(0)

        monkeypatch.setattr("builtins.input", fake_input)

    return set_lines


# ---------------------------------------------------------------------------
# TerminalUI
# ---------------------------------------------------------------------------


def test_choose_asks_again_until_a_valid_number(typed, capsys):
    typed(["x", "5", "2"])
    assert TerminalUI().choose("Pick one", ["a", "b", "c"]) == 1
    assert "Type a number from 1 to 3." in capsys.readouterr().out


def test_piped_input_with_a_powershell_bom_still_counts(typed):
    # PowerShell puts an invisible marker before piped text; found live in M6.
    typed(["﻿2"])
    assert TerminalUI().choose("Pick one", ["a", "b", "c"]) == 1


def test_ask_multiline_stops_at_the_end_marker(typed):
    typed(["first line", "", "  third line", ".", "not read"])
    assert TerminalUI().ask_multiline("Paste it") == "first line\n\n  third line"


def test_ask_multiline_stops_at_end_of_input(typed):
    typed(["only line"])
    assert TerminalUI().ask_multiline("Paste it") == "only line"


def test_ask_returns_blank_at_end_of_input(typed):
    typed([])
    assert TerminalUI().ask("Anything?") == ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_examples_are_split_on_separator_lines():
    text = "print(1)\n---\nfor i in range(3):\n    print(i)\n---\n\n"
    assert parse_answer("examples", text) == ["print(1)", "for i in range(3):\n    print(i)"]


def test_blank_answers_become_none():
    assert parse_answer("context", "   ") is None
    assert parse_answer("examples", "\n---\n") is None


def test_other_answers_are_kept_as_typed_apart_from_outer_whitespace():
    assert parse_answer("constraints", "  Use Python 3.11,  because of uni  ") == (
        "Use Python 3.11,  because of uni"
    )


def test_preview_shortens_long_text():
    shown = preview("x" * 1000)
    assert "(1000 characters in total)" in shown
    assert len(shown) < 400


def test_form_shows_profile_slots_then_stated_extras_and_skips_known_ones():
    extraction = make_extraction(objective="Summarize", resources="pandas", inputs="text")
    shown = slots_to_show(
        extraction, ["objective", "inputs", "output_format"], skip={"inputs"}
    )
    assert shown == ["objective", "output_format", "resources"]
