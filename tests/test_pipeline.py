"""End-to-end tests for pipeline.py, with fake clients and scripted answers.

No test here calls an API or waits for typing. Each checks what reached
"Claude" (the FakeClaude records it) and what was written to the log.
"""

import json

import pytest

from promptgate.checkpoint import TRUNCATION_WARNING
from promptgate.config import load_config
from promptgate.pipeline import Pipeline
from promptgate.render import AMBIGUITY_RULE, QUOTE_GROUNDING, SELF_CHECK
from promptgate.schemas import JEV_CHECKED_SLOTS, Document, SplitResult
from tests.fakes import FakeClaude, FakeExtractor, FakeJev, ScriptedUI, make_clients
from tests.fixtures import SAMPLES, make_extraction, make_verdict

QUICK, TRANSLATE, SPLIT, CONFLICT, PROJECT = SAMPLES  # the five sample prompts, in order
ARTICLE = Document(source="article.txt", content="Section 1. Cats.\nSection 2. Dogs.")


@pytest.fixture
def config(tmp_path):
    config = load_config()
    config.paths.log_file = tmp_path / "logs" / "runs.jsonl"  # never the real log
    return config


def run(config, prompt, ui_answers=(), documents=(), dry_run=False, jev=None, extractor=None, claude=None):
    """Run the pipeline once with fakes. Returns (records, ui, claude, log lines)."""
    claude = claude or FakeClaude()
    clients = make_clients(jev=jev, extractor=extractor, claude=claude)
    ui = ScriptedUI(list(ui_answers))
    records = Pipeline(config, clients, ui).run(prompt, list(documents), dry_run=dry_run)
    assert ui.answers == [], f"scripted answers were not all used: {ui.answers}"
    lines = config.paths.log_file.read_text(encoding="utf-8").splitlines()
    return records, ui, claude, lines


def sent_prompt(claude: FakeClaude):
    """The rendered prompt from the one execution call."""
    assert len(claude.execute_calls) == 1
    return claude.execute_calls[0][0]


# ---------------------------------------------------------------------------
# as_is
# ---------------------------------------------------------------------------


def test_quick_question_goes_as_is_with_no_tags_and_no_plan(config):
    records, ui, claude, lines = run(config, QUICK.prompt, jev=FakeJev(QUICK.verdict))
    rendered = sent_prompt(claude)
    assert rendered.user == "What does df.shape return?"
    assert rendered.system is None
    assert claude.plan_calls == []
    assert claude.execute_calls[0][1] == "low"  # effort
    assert ui.questions == []  # nothing was asked
    assert ui.streamed == "Here is the answer."

    assert len(lines) == 1
    record = records[0]
    assert json.loads(lines[0])["run_id"] == record.run_id
    assert record.route == "as_is"
    assert record.task_type == "quick_question"
    assert set(record.tokens) == {"jev", "execute"}


def test_explanation_gets_the_default_role_unless_the_prompt_has_one(config):
    verdict = make_verdict(task_type="explanation", present=JEV_CHECKED_SLOTS[1:], has_role=0.1)
    _, _, claude, _ = run(config, "Explain recursion to me.", jev=FakeJev(verdict))
    assert sent_prompt(claude).system == config.profiles["explanation"].default_role

    verdict = make_verdict(task_type="explanation", present=JEV_CHECKED_SLOTS, has_role=0.9)
    _, _, claude, _ = run(config, "You are a chef. Explain recursion.", jev=FakeJev(verdict))
    assert sent_prompt(claude).system is None


# ---------------------------------------------------------------------------
# fill_missing
# ---------------------------------------------------------------------------


def test_translate_asks_for_the_inputs_and_puts_them_first(config):
    records, ui, claude, lines = run(
        config, TRANSLATE.prompt, ui_answers=["Good morning, everyone."], jev=FakeJev(TRANSLATE.verdict)
    )
    rendered = sent_prompt(claude)
    assert rendered.user.startswith("<documents>")
    assert "<source>answer</source>" in rendered.user
    assert "Good morning, everyone." in rendered.user
    assert rendered.user.index("</documents>") < rendered.user.index("Translate this into French:")
    assert rendered.user.endswith(AMBIGUITY_RULE)
    assert "Inputs:" in ui.questions[0]
    assert claude.plan_calls == []
    assert records[0].route == "fill_missing"
    assert records[0].missing_slots == ["inputs"]
    assert records[0].skipped_slots == []
    assert len(lines) == 1


def test_skipped_questions_are_logged(config):
    records, _, claude, _ = run(config, TRANSLATE.prompt, ui_answers=[""], jev=FakeJev(TRANSLATE.verdict))
    assert records[0].skipped_slots == ["inputs"]
    assert sent_prompt(claude).user == f"Translate this into French:\n\n{AMBIGUITY_RULE}"


# ---------------------------------------------------------------------------
# Plan checkpoint
# ---------------------------------------------------------------------------

PROJECT_ANSWERS = [
    "I'm a first-year student.",  # context
    "Weekly view",  # requirements
    "Python only, because that's what I know",  # constraints
    "tkinter",  # resources
    "One .py file",  # output_format
    "I can plan a real week with it",  # success_criteria
]
APPROVE, EDIT, CANCEL = 0, 1, 2


def test_multi_step_project_plans_first_and_waits_for_approval(config):
    records, _, claude, lines = run(
        config, PROJECT.prompt, ui_answers=PROJECT_ANSWERS + [APPROVE], jev=FakeJev(PROJECT.verdict)
    )
    assert len(claude.plan_calls) == 1
    plan_prompt, plan_effort = claude.plan_calls[0]
    exec_prompt, exec_effort, approved_plan = claude.execute_calls[0]

    assert plan_prompt == exec_prompt  # the same shared prefix, so it can be cached
    assert plan_effort == exec_effort == "high"  # the same effort, so the cache still works
    assert approved_plan == "1. Outline\n2. Build\n3. Test"
    assert SELF_CHECK in exec_prompt.user  # success criteria answered, sonnet-5 self_check on

    record = records[0]
    assert record.checkpoint is True
    assert record.plan_edited is False
    assert set(record.tokens) == {"jev", "plan", "execute"}
    assert len(lines) == 1


def test_edited_plan_is_the_one_sent(config):
    records, _, claude, _ = run(
        config,
        PROJECT.prompt,
        ui_answers=PROJECT_ANSWERS + [EDIT, "1. Only build the weekly view", APPROVE],
        jev=FakeJev(PROJECT.verdict),
    )
    assert len(claude.plan_calls) == 1  # editing never calls Claude again
    assert claude.execute_calls[0][2] == "1. Only build the weekly view"
    assert records[0].plan_edited is True


def test_blank_edit_keeps_the_current_plan(config):
    records, _, claude, _ = run(
        config,
        PROJECT.prompt,
        ui_answers=PROJECT_ANSWERS + [EDIT, "", APPROVE],
        jev=FakeJev(PROJECT.verdict),
    )
    assert claude.execute_calls[0][2] == "1. Outline\n2. Build\n3. Test"
    assert records[0].plan_edited is False


def test_cancel_stops_before_execution_and_is_logged(config):
    records, _, claude, lines = run(
        config, PROJECT.prompt, ui_answers=PROJECT_ANSWERS + [CANCEL], jev=FakeJev(PROJECT.verdict)
    )
    assert claude.execute_calls == []
    assert records[0].cancelled is True
    assert "execute" not in records[0].tokens
    assert len(lines) == 1


def test_cut_off_plan_shows_a_warning(config):
    records, ui, _, _ = run(
        config,
        PROJECT.prompt,
        ui_answers=PROJECT_ANSWERS + [APPROVE],
        jev=FakeJev(PROJECT.verdict),
        claude=FakeClaude(plan_truncated=True),
    )
    assert TRUNCATION_WARNING in ui.shown
    assert records[0].plan_truncated is True


def test_failed_planning_call_stops_the_run_with_an_error(config):
    records, _, claude, lines = run(
        config,
        PROJECT.prompt,
        ui_answers=PROJECT_ANSWERS,
        jev=FakeJev(PROJECT.verdict),
        claude=FakeClaude(fail_plan=True),
    )
    assert claude.execute_calls == []
    assert "planning call failed" in records[0].error
    assert len(lines) == 1


# ---------------------------------------------------------------------------
# split
# ---------------------------------------------------------------------------

SPLIT_RESULT = SplitResult(
    parts=["Summarize this article", "write me a cover letter"],
    shared_context="I'm applying for a data analyst job.",
    shared_inputs=["ARTICLE TEXT"],
)
PART_VERDICTS = {
    # Jev wrongly says this part has several tasks too; a part must never split again.
    "Summarize this article": make_verdict(
        task_type="transformation", problem_type="multiple_tasks", present=("objective",)
    ),
    "write me a cover letter": make_verdict(task_type="quick_question", present=("objective",)),
}


def split_jev():
    return FakeJev(by_prompt={SPLIT.prompt: SPLIT.verdict, **PART_VERDICTS})


def test_split_runs_each_part_with_the_shared_inputs_and_context(config):
    extractor = FakeExtractor(split_result=SPLIT_RESULT)
    jev = split_jev()
    records, _, claude, lines = run(
        config, SPLIT.prompt, ui_answers=[0], jev=jev, extractor=extractor
    )
    parent, first, second = records
    assert parent.route == "split"
    assert parent.checkpoint is False
    assert "extract" in parent.tokens
    assert "execute" not in parent.tokens
    assert first.parent_run_id == second.parent_run_id == parent.run_id
    assert first.route == second.route == "as_is"  # the first part was not split again

    assert jev.calls == [SPLIT.prompt, "Summarize this article", "write me a cover letter"]
    assert len(claude.execute_calls) == 2
    for (rendered, _, _), part in zip(claude.execute_calls, SPLIT_RESULT.parts):
        assert rendered.user.startswith("<documents>")
        assert "ARTICLE TEXT" in rendered.user
        assert "<context>\nI'm applying for a data analyst job.\n</context>" in rendered.user
        assert part in rendered.user

    # One line per run, parent first.
    assert [json.loads(line)["run_id"] for line in lines] == [r.run_id for r in records]


def test_declining_the_split_sends_the_prompt_as_is(config):
    extractor = FakeExtractor(split_result=SPLIT_RESULT)
    records, _, claude, lines = run(
        config, SPLIT.prompt, ui_answers=[1], jev=FakeJev(SPLIT.verdict), extractor=extractor
    )
    assert extractor.split_calls == []
    assert records[0].route == "as_is"
    assert records[0].split_declined is True
    assert sent_prompt(claude).user.startswith(SPLIT.prompt)
    assert len(lines) == 1


def test_failed_split_falls_back_to_as_is(config):
    records, _, claude, _ = run(
        config,
        SPLIT.prompt,
        ui_answers=[0],
        jev=FakeJev(SPLIT.verdict),
        extractor=FakeExtractor(fail=True),
    )
    assert records[0].route == "as_is"
    assert "split failed" in records[0].warnings[0]
    assert len(claude.execute_calls) == 1


# ---------------------------------------------------------------------------
# full_form
# ---------------------------------------------------------------------------


def test_contradiction_shows_the_conflict_and_renders_the_full_template(config):
    extraction = make_extraction(
        objective="Summarize every section",
        output_format={"status": "conflict", "options": ["in full detail", "in under 50 words"]},
    )
    extractor = FakeExtractor(extraction=extraction)
    records, ui, claude, lines = run(
        config,
        CONFLICT.prompt,
        # objective: keep | output_format: pick "in under 50 words" | constraints: skip
        ui_answers=["", 1, ""],
        documents=[ARTICLE],
        jev=FakeJev(CONFLICT.verdict),
        extractor=extractor,
    )
    rendered = sent_prompt(claude)
    assert rendered.user.startswith("<documents>")
    assert "<source>article.txt</source>" in rendered.user
    assert "<objective>\nSummarize every section\n</objective>" in rendered.user
    assert "<output_format>\nin under 50 words\n</output_format>" in rendered.user
    assert "in full detail" not in rendered.user
    assert "<constraints>" not in rendered.user
    assert rendered.user.endswith(AMBIGUITY_RULE)

    assert extractor.extract_calls == [CONFLICT.prompt]
    assert not any("Inputs" in question for question in ui.questions)  # --input covers it
    record = records[0]
    assert record.route == "full_form"
    assert record.skipped_slots == ["constraints"]
    assert record.input_sources == ["article.txt"]
    assert len(lines) == 1


def test_vague_objective_offers_candidate_objectives(config):
    verdict = make_verdict(task_type="explanation", problem_type="unclear_intent")
    extraction = make_extraction(
        candidate_objectives=["Explain what recursion is", "Write a recursive function"],
        context="I'm new to Python.",
    )
    _, _, claude, _ = run(
        config,
        "recursion??",
        # objective: pick 2nd candidate | context: keep | output_format, constraints: skip | examples: skip
        ui_answers=[1, "", "", "", ""],
        jev=FakeJev(verdict),
        extractor=FakeExtractor(extraction=extraction),
    )
    rendered = sent_prompt(claude)
    assert "<objective>\nWrite a recursive function\n</objective>" in rendered.user
    assert "<context>\nI'm new to Python.\n</context>" in rendered.user


def test_stated_slot_outside_the_profile_is_kept(config):
    verdict = make_verdict(task_type="quick_question", problem_type="contradictory")
    extraction = make_extraction(objective="Show me a groupby", resources="pandas only")
    _, ui, claude, _ = run(
        config,
        "Show me a groupby, pandas only",
        # objective: keep | context: skip | resources: keep
        ui_answers=["", "", ""],
        jev=FakeJev(verdict),
        extractor=FakeExtractor(extraction=extraction),
    )
    assert "<resources>\npandas only\n</resources>" in sent_prompt(claude).user


def test_stated_role_becomes_the_system_prompt(config):
    verdict = make_verdict(task_type="explanation", problem_type="contradictory", has_role=0.9)
    extraction = make_extraction(
        role="You are a strict examiner.", objective="Mark my answer", context="Exam prep"
    )
    _, _, claude, _ = run(
        config,
        "You are a strict examiner. Mark my answer.",
        ui_answers=["", "", "", "", ""],  # keep, keep, then skip the three optional slots
        jev=FakeJev(verdict),
        extractor=FakeExtractor(extraction=extraction),
    )
    rendered = sent_prompt(claude)
    assert rendered.system == "You are a strict examiner."
    assert "<role>" not in rendered.user


def test_failed_extraction_falls_back_to_questions(config):
    verdict = make_verdict(task_type="explanation", problem_type="contradictory", present=("objective",))
    records, _, claude, _ = run(
        config,
        "Explain it briefly but in full detail",
        ui_answers=["Revising for an exam"],  # the missing context question
        jev=FakeJev(verdict),
        extractor=FakeExtractor(fail=True),
    )
    assert records[0].route == "fill_missing"
    assert "extraction failed" in records[0].warnings[0]
    assert "<context>\nRevising for an exam\n</context>" in sent_prompt(claude).user


# ---------------------------------------------------------------------------
# Failures and fallbacks
# ---------------------------------------------------------------------------


def test_jev_failure_sends_the_prompt_as_is_with_the_ambiguity_rule(config):
    records, _, claude, lines = run(config, "Build me an app", jev=FakeJev(fail=True))
    rendered = sent_prompt(claude)
    assert rendered.user == f"Build me an app\n\n{AMBIGUITY_RULE}"
    assert claude.execute_calls[0][1] == "medium"  # the fallback effort
    record = records[0]
    assert record.jev is None
    assert record.jev_error == "Jev is down"
    assert record.task_type is None
    assert len(lines) == 1


def test_failed_execution_is_logged_as_an_error(config):
    records, _, _, lines = run(
        config, QUICK.prompt, jev=FakeJev(QUICK.verdict), claude=FakeClaude(fail_execute=True)
    )
    assert "execution call failed" in records[0].error
    assert len(lines) == 1


def test_ctrl_c_is_logged_as_cancelled_and_passed_on(config):
    class InterruptingUI(ScriptedUI):
        def ask_multiline(self, question):
            raise KeyboardInterrupt

    clients = make_clients(jev=FakeJev(TRANSLATE.verdict))
    with pytest.raises(KeyboardInterrupt):
        Pipeline(config, clients, InterruptingUI()).run(TRANSLATE.prompt, [])
    lines = config.paths.log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["cancelled"] is True


# ---------------------------------------------------------------------------
# Fixed lines that depend on the documents
# ---------------------------------------------------------------------------


def test_long_inputs_get_the_quote_grounding_line(config):
    _, _, claude, _ = run(
        config,
        QUICK.prompt,
        documents=[ARTICLE],
        jev=FakeJev(QUICK.verdict),
        claude=FakeClaude(token_count=25_000),
    )
    rendered = sent_prompt(claude)
    assert rendered.user.startswith("<documents>")
    assert QUOTE_GROUNDING in rendered.user


def test_short_inputs_do_not(config):
    _, _, claude, _ = run(
        config, QUICK.prompt, documents=[ARTICLE], jev=FakeJev(QUICK.verdict), claude=FakeClaude(token_count=500)
    )
    assert QUOTE_GROUNDING not in sent_prompt(claude).user


def test_tokens_are_only_counted_when_there_are_documents(config):
    _, _, claude, _ = run(config, QUICK.prompt, jev=FakeJev(QUICK.verdict))
    assert claude.count_calls == []


def test_jev_never_sees_the_input_files(config):
    jev = FakeJev(QUICK.verdict)
    run(config, QUICK.prompt, documents=[ARTICLE], jev=jev)
    assert jev.calls == [QUICK.prompt]


# ---------------------------------------------------------------------------
# Dry run
# ---------------------------------------------------------------------------


def test_dry_run_never_calls_claude_but_shows_everything_and_logs(config):
    records, ui, claude, lines = run(
        config, PROJECT.prompt, ui_answers=PROJECT_ANSWERS, dry_run=True, jev=FakeJev(PROJECT.verdict)
    )
    assert claude.plan_calls == []
    assert claude.execute_calls == []

    output = ui.all_output
    assert "DRY RUN" in output
    assert "Jev (jev-1.13.0)" in output
    assert "Route: fill_missing" in output
    assert "Missing slots: context, requirements, constraints" in output
    assert "Checkpoint: on (planning call skipped in dry run)" in output
    assert "Build me a study planner app" in output  # the rendered user message

    record = records[0]
    assert record.dry_run is True
    assert record.checkpoint is True
    assert len(lines) == 1


@pytest.mark.parametrize(
    "sample, answers",
    [(QUICK, []), (TRANSLATE, ["Bonjour"]), (SPLIT, [1]), (PROJECT, PROJECT_ANSWERS)],
    ids=["quick", "translate", "split-declined", "project"],
)
def test_every_sample_writes_exactly_one_valid_line(config, sample, answers):
    run(config, sample.prompt, ui_answers=answers, dry_run=True, jev=FakeJev(sample.verdict))
    lines = config.paths.log_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    json.loads(lines[0])
