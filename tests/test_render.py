"""Tests for render.py. Everything here is pure: text in, text out."""

import re

import pytest

from promptgate.config import PROJECT_ROOT
from promptgate.render import (
    AMBIGUITY_RULE,
    CHECKPOINT_INSTRUCTION,
    FOLLOW_PLAN,
    QUOTE_GROUNDING,
    SELF_CHECK,
    FixedLines,
    approved_plan_block,
    choose_fixed_lines,
    documents_from_value,
    plan_request_block,
    render_documents,
    render_full,
    render_with_answers,
)
from promptgate.schemas import Document

NO_LINES = FixedLines(ambiguity_rule=False, quote_grounding=False, self_check=False)
AMBIGUITY_ONLY = FixedLines(ambiguity_rule=True, quote_grounding=False, self_check=False)
ALL_LINES = FixedLines(ambiguity_rule=True, quote_grounding=True, self_check=True)

ARTICLE = Document(source="article.txt", content="Section 1. Cats.\nSection 2. Dogs.")


def has_any_tag(text: str) -> bool:
    return re.search(r"</?[a-z_]+[^>]*>", text) is not None


# ---------------------------------------------------------------------------
# Fixed-line text matches CLAUDE.md
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "line", [AMBIGUITY_RULE, CHECKPOINT_INSTRUCTION, FOLLOW_PLAN, QUOTE_GROUNDING, SELF_CHECK]
)
def test_fixed_lines_match_the_spec_exactly(line):
    spec = (PROJECT_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    assert line in spec


# ---------------------------------------------------------------------------
# as_is route
# ---------------------------------------------------------------------------


def test_quick_question_renders_with_no_tags_at_all():
    rendered = render_with_answers("What does df.shape return?", {}, [], None, NO_LINES)
    assert rendered.user == "What does df.shape return?"
    assert rendered.system is None
    assert not has_any_tag(rendered.user)


def test_as_is_keeps_the_prompt_word_for_word_and_adds_the_ambiguity_rule():
    prompt = "Explain  list comprehensions,\nbriefly."
    rendered = render_with_answers(prompt, {}, [], None, AMBIGUITY_ONLY)
    assert rendered.user == f"{prompt}\n\n{AMBIGUITY_RULE}"


def test_system_prompt_is_passed_through():
    rendered = render_with_answers("Explain recursion.", {}, [], "You are a tutor.", NO_LINES)
    assert rendered.system == "You are a tutor."


# ---------------------------------------------------------------------------
# fill_missing route: answers attached to the original prompt
# ---------------------------------------------------------------------------


def test_answers_go_below_the_prompt_as_tagged_blocks_in_slot_order():
    answers = {"output_format": "A table.", "context": "For my exam revision."}
    rendered = render_with_answers("Compare lists and tuples.", answers, [], None, NO_LINES)
    assert rendered.user == (
        "Compare lists and tuples.\n\n"
        "<context>\nFor my exam revision.\n</context>\n\n"
        "<output_format>\nA table.\n</output_format>"
    )


def test_blank_answers_are_left_out():
    answers = {"context": "   ", "constraints": None, "resources": []}
    rendered = render_with_answers("Do it.", answers, [], None, NO_LINES)
    assert rendered.user == "Do it."


def test_inputs_answer_goes_on_top_as_documents():
    documents = documents_from_value("Bonjour is hello.", source="answer")
    rendered = render_with_answers("Translate this into French:", {}, documents, None, AMBIGUITY_ONLY)
    assert rendered.user.startswith("<documents>")
    assert rendered.user.index("</documents>") < rendered.user.index("Translate this")
    assert "<source>answer</source>" in rendered.user
    assert rendered.user.endswith(AMBIGUITY_RULE)


def test_inputs_as_an_answer_is_refused():
    with pytest.raises(ValueError, match="documents"):
        render_with_answers("Translate:", {"inputs": "text"}, [], None, NO_LINES)


# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------


def test_documents_use_the_anthropic_format():
    assert render_documents([ARTICLE]) == (
        "<documents>\n"
        '<document index="1">\n'
        "<source>article.txt</source>\n"
        "<document_content>\n"
        "Section 1. Cats.\nSection 2. Dogs.\n"
        "</document_content>\n"
        "</document>\n"
        "</documents>"
    )


def test_documents_are_numbered_from_one():
    notes = Document(source="notes.md", content="Notes")
    text = render_documents([ARTICLE, notes])
    assert '<document index="1">\n<source>article.txt</source>' in text
    assert '<document index="2">\n<source>notes.md</source>' in text


def test_document_content_is_passed_through_word_for_word():
    content = "  indented line\n<b>not a tag of ours</b>\n\ntrailing  "
    text = render_documents([Document(source="raw.txt", content=content)])
    assert f"<document_content>\n{content}\n</document_content>" in text


def test_documents_from_a_list_value_make_one_document_each():
    documents = documents_from_value(["first", "  ", "second"], source="prompt")
    assert [document.content for document in documents] == ["first", "second"]


def test_long_inputs_put_documents_first_and_add_quote_grounding():
    fixed = choose_fixed_lines(
        ambiguity_rule=True,
        input_tokens=25_000,
        quote_grounding_tokens=20_000,
        has_success_criteria=False,
        self_check_enabled=True,
    )
    rendered = render_with_answers("Summarize every section.", {}, [ARTICLE], None, fixed)
    assert rendered.user.startswith("<documents>")
    assert QUOTE_GROUNDING in rendered.user


# ---------------------------------------------------------------------------
# full_form route: the full template
# ---------------------------------------------------------------------------


def test_full_template_follows_the_render_order_and_skips_empty_slots():
    slots = {
        "examples": ["Mon: maths", "Tue: physics"],
        "success_criteria": "I can follow it for a week.",
        "objective": "Write a study plan.",
        "context": "First-year engineering student.",
        "constraints": "",
        "resources": None,
    }
    rendered = render_full(slots, [ARTICLE], None, AMBIGUITY_ONLY)
    order = ["<documents>", "<context>", "<objective>", "<success_criteria>", "<examples>"]
    positions = [rendered.user.index(tag) for tag in order]
    assert positions == sorted(positions)
    assert "<constraints>" not in rendered.user
    assert "<resources>" not in rendered.user
    assert rendered.user.endswith(AMBIGUITY_RULE)


def test_examples_are_each_wrapped_inside_examples():
    rendered = render_full({"examples": ["one", "two"]}, [], None, NO_LINES)
    assert rendered.user == (
        "<examples>\n<example>\none\n</example>\n<example>\ntwo\n</example>\n</examples>"
    )


def test_list_values_go_on_separate_lines():
    rendered = render_full({"requirements": ["Covers maths", "Covers physics"]}, [], None, NO_LINES)
    assert rendered.user == "<requirements>\nCovers maths\nCovers physics\n</requirements>"


def test_stated_role_goes_to_the_system_prompt_and_beats_the_default():
    slots = {"role": "You are a strict examiner.", "objective": "Mark my answer."}
    rendered = render_full(slots, [], "You are a patient tutor.", NO_LINES)
    assert rendered.system == "You are a strict examiner."
    assert "<role>" not in rendered.user


def test_default_role_is_used_when_the_form_has_none():
    rendered = render_full({"objective": "Explain it."}, [], "You are a patient tutor.", NO_LINES)
    assert rendered.system == "You are a patient tutor."


def test_no_role_and_no_default_means_no_system_prompt():
    rendered = render_full({"role": "  ", "objective": "Explain it."}, [], None, NO_LINES)
    assert rendered.system is None


def test_inputs_as_a_slot_is_refused():
    with pytest.raises(ValueError, match="documents"):
        render_full({"inputs": "text"}, [], None, NO_LINES)


# ---------------------------------------------------------------------------
# Choosing and ordering the fixed lines
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tokens, expected", [(0, False), (19_999, False), (20_000, True)])
def test_quote_grounding_starts_at_the_threshold(tokens, expected):
    fixed = choose_fixed_lines(True, tokens, 20_000, False, False)
    assert fixed.quote_grounding is expected


@pytest.mark.parametrize(
    "has_criteria, enabled, expected",
    [(True, True, True), (True, False, False), (False, True, False), (False, False, False)],
)
def test_self_check_needs_criteria_and_the_model_setting(has_criteria, enabled, expected):
    fixed = choose_fixed_lines(True, 0, 20_000, has_criteria, enabled)
    assert fixed.self_check is expected


def test_fixed_lines_come_last_with_the_ambiguity_rule_at_the_end():
    rendered = render_full({"objective": "Analyse it."}, [ARTICLE], None, ALL_LINES)
    assert rendered.user.endswith(f"{QUOTE_GROUNDING}\n\n{SELF_CHECK}\n\n{AMBIGUITY_RULE}")


# ---------------------------------------------------------------------------
# Plan checkpoint blocks
# ---------------------------------------------------------------------------


def test_plan_request_block_is_the_checkpoint_instruction():
    assert plan_request_block() == CHECKPOINT_INSTRUCTION


def test_approved_plan_block_wraps_the_plan_and_asks_to_follow_it():
    block = approved_plan_block("1. Outline\n2. Build\n")
    assert block == "<approved_plan>\n1. Outline\n2. Build\n</approved_plan>\nFollow the approved plan."
