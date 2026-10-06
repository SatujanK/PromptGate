"""Tests for the real clients, with the SDKs replaced by small fakes.

The fakes record the exact arguments each client sends and return canned
replies, so these tests check request shapes and reply handling without
any network call or API key.
"""

from types import SimpleNamespace

import anthropic
import pytest
from typesafe_sdk import Choice, Noul, TypeSafeError

from promptgate.clients import ClientError, token_usage
from promptgate.executor import CUT_OFF_NOTE, ClaudeExecutor, build_request
from promptgate.extractor import (
    EXTRACT_INSTRUCTION,
    SPLIT_INSTRUCTION,
    ExtractionReply,
    HaikuExtractor,
    SlotCopy,
    SplitReply,
    to_extraction,
    to_slot_result,
    to_split_result,
)
from promptgate.jev_client import JevClient, build_questions, verdict_from_response
from promptgate.render import (
    CHECKPOINT_INSTRUCTION,
    FOLLOW_PLAN,
    RenderedPrompt,
)
from promptgate.schemas import ALL_SLOTS, JEV_CHECKED_SLOTS, PROBLEM_TYPES, TASK_TYPES, Document

RENDERED = RenderedPrompt(system="You are a tutor.", user="<objective>\nBuild it\n</objective>")


def usage(**counts):
    return SimpleNamespace(**counts)


# ---------------------------------------------------------------------------
# token_usage
# ---------------------------------------------------------------------------


def test_token_usage_copies_counts_and_treats_missing_as_zero():
    result = token_usage(usage(input_tokens=10, output_tokens=None, cache_read_input_tokens=7))
    assert result.input_tokens == 10
    assert result.output_tokens == 0
    assert result.cache_read_input_tokens == 7
    assert result.cache_creation_input_tokens == 0


# ---------------------------------------------------------------------------
# Jev
# ---------------------------------------------------------------------------


def test_one_request_holds_every_question():
    questions = build_questions()
    expected = {"task_type", "problem_type", "expects_large_output"} | {
        f"has_{slot}" for slot in JEV_CHECKED_SLOTS
    }
    assert set(questions) == expected
    assert isinstance(questions["task_type"], Choice)
    assert isinstance(questions["has_role"], Noul)


def test_every_choice_offers_unclear_and_only_known_labels():
    questions = build_questions()
    assert set(questions["task_type"].criteria) == set(TASK_TYPES) | {"unclear"}
    assert set(questions["problem_type"].criteria) == set(PROBLEM_TYPES)


def fake_jev_response(model="jev-1.13.0"):
    """Shaped like typesafe_sdk's SystemOneResponse. Noul answers have no confidence field."""
    choice = lambda name, labels: SimpleNamespace(  # noqa: E731
        choice=name, confidence=0.8, probabilities={label: 0.1 for label in labels} | {name: 0.9}
    )
    return SimpleNamespace(
        model=model,
        choices={
            "task_type": choice("analysis", TASK_TYPES),
            "problem_type": choice("none", PROBLEM_TYPES),
        },
        nouls={f"has_{slot}": SimpleNamespace(noul=0.7) for slot in JEV_CHECKED_SLOTS}
        | {"expects_large_output": SimpleNamespace(noul=0.2)},
        usage=usage(input_tokens=321, output_tokens=None),
    )


def test_verdict_is_copied_from_the_response():
    verdict = verdict_from_response(fake_jev_response())
    assert verdict.model == "jev-1.13.0"
    assert verdict.task_type.choice == "analysis"
    assert verdict.task_type.probabilities["analysis"] == 0.9
    assert verdict.presence == {slot: 0.7 for slot in JEV_CHECKED_SLOTS}
    assert verdict.expects_large_output == 0.2


class FakeTypeSafe:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = []

    def system_one(self, state, questions, *, model=None):
        self.calls.append({"state": state, "questions": questions, "model": model})
        if self.error:
            raise self.error
        return self.response


def test_jev_gets_only_the_prompt_and_the_pinned_model():
    fake = FakeTypeSafe(fake_jev_response())
    verdict, used = JevClient("jev-1.13.0", client=fake).classify("What does df.shape return?")
    call = fake.calls[0]
    assert call["state"] == "What does df.shape return?"
    assert call["model"] == "jev-1.13.0"
    assert set(call["questions"]) == set(build_questions())
    assert used.input_tokens == 321


def test_jev_sdk_errors_become_client_errors():
    fake = FakeTypeSafe(error=TypeSafeError("service unavailable"))
    with pytest.raises(ClientError, match="Jev call failed"):
        JevClient("jev-1.13.0", client=fake).classify("hi")


def test_unexpected_jev_answers_become_client_errors():
    response = fake_jev_response()
    del response.nouls["has_role"]
    with pytest.raises(ClientError, match="unexpected answer"):
        JevClient("jev-1.13.0", client=FakeTypeSafe(response)).classify("hi")


# ---------------------------------------------------------------------------
# Haiku: converting replies
# ---------------------------------------------------------------------------


def copy(status, value=(), options=()):
    return SlotCopy(status=status, value=list(value), options=list(options))


CONFLICT_PROMPT = "Summarize every section in full detail, in under 50 words"


@pytest.mark.parametrize(
    "slot_copy, status, value, options",
    [
        (copy("stated", ["Translate it"]), "stated", "Translate it", []),
        (copy("stated", ["a", "", "b"]), "stated", ["a", "b"], []),
        (copy("stated", ["  "]), "missing", None, []),
        (copy("missing", ["made up"]), "missing", None, []),  # never use words Haiku put in a missing slot
        (copy("conflict", [], ["in full detail", "in under 50 words"]), "conflict", None, ["in full detail", "in under 50 words"]),
        (copy("conflict", [], ["In Full  Detail", "under 50 words."]), "conflict", None, ["In Full  Detail", "under 50 words."]),
        (copy("conflict", [], ["in full detail", "at most fifty words"]), "stated", "in full detail", []),  # reworded option dropped
        (copy("conflict", [], []), "missing", None, []),
    ],
    ids=["one-piece", "several", "blank", "missing-ignores-value", "conflict", "case-and-spacing", "reworded-option", "no-options"],
)
def test_slot_replies_are_converted_and_corrected(slot_copy, status, value, options):
    result = to_slot_result(slot_copy, CONFLICT_PROMPT)
    assert (result.status, result.value, result.options) == (status, value, options)


def make_reply(candidates=(), **slots) -> ExtractionReply:
    data = {name: copy("missing") for name in ALL_SLOTS}
    data.update(slots)
    return ExtractionReply(**data, candidate_objectives=list(candidates))


def test_reworded_candidate_objectives_are_dropped():
    # The real reply Haiku gave for sample 4 in M6: two of the three were rewordings.
    reply = make_reply(
        candidates=[
            "Summarize every section in full detail, in under 50 words",
            "Summarize every section with full detail within a 50-word limit",
            "Create detailed summaries of each section not exceeding 50 words",
        ],
        objective=copy("stated", ["Summarize every section in full detail, in under 50 words"]),
    )
    extraction = to_extraction(reply, CONFLICT_PROMPT)
    # The first is your own objective repeated, the other two are reworded: none survive.
    assert extraction.candidate_objectives == []


def test_copied_candidate_objectives_are_kept_up_to_three():
    prompt = "recursion?? explain it or write one or test one or time one"
    reply = make_reply(candidates=["explain it", "write one", "test one", "time one", " "])
    assert to_extraction(reply, prompt).candidate_objectives == ["explain it", "write one", "test one"]


def test_split_reply_conversion():
    result = to_split_result(SplitReply(parts=["A", " ", "B"], shared_context="  ", shared_inputs=["text"]))
    assert result.parts == ["A", "B"]
    assert result.shared_context is None
    assert result.shared_inputs == ["text"]


def test_the_extraction_instruction_contains_the_three_copy_rules():
    assert "1. Copy the user's words into slots. Do not paraphrase." in EXTRACT_INSTRUCTION
    assert "2. If two instructions conflict, list both and choose neither." in EXTRACT_INSTRUCTION
    assert "even when they would go in different slots" in EXTRACT_INSTRUCTION
    assert "3. If something is not stated, mark it missing and never fill it." in EXTRACT_INSTRUCTION


def test_reply_schemas_have_no_optional_fields():
    # Every field required keeps the structured-output grammar small.
    for model in (SlotCopy, ExtractionReply, SplitReply):
        schema = model.model_json_schema()
        assert set(schema["required"]) == set(schema["properties"]), model.__name__


# ---------------------------------------------------------------------------
# Haiku: calls
# ---------------------------------------------------------------------------


class FakeMessages:
    """Stands in for anthropic.Anthropic().messages and records every call."""

    def __init__(self, reply=None, error=None, stream_pieces=(), token_count=0):
        self.reply = reply
        self.error = error
        self.stream_pieces = list(stream_pieces)
        self.token_count = token_count
        self.calls = []

    def parse(self, **kwargs):
        self.calls.append(("parse", kwargs))
        if self.error:
            raise self.error
        return self.reply

    def create(self, **kwargs):
        self.calls.append(("create", kwargs))
        if self.error:
            raise self.error
        return self.reply

    def count_tokens(self, **kwargs):
        self.calls.append(("count_tokens", kwargs))
        if self.error:
            raise self.error
        return SimpleNamespace(input_tokens=self.token_count)

    def stream(self, **kwargs):
        self.calls.append(("stream", kwargs))
        if self.error:
            raise self.error
        return FakeStream(self.stream_pieces, self.reply)


class FakeStream:
    def __init__(self, pieces, final):
        self.text_stream = pieces
        self.final = final

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def get_final_message(self):
        return self.final


def fake_anthropic(**kwargs):
    return SimpleNamespace(messages=FakeMessages(**kwargs))


def parsed_reply(parsed, stop_reason="end_turn"):
    return SimpleNamespace(
        parsed_output=parsed, stop_reason=stop_reason, usage=usage(input_tokens=400, output_tokens=150)
    )


def test_extract_sends_the_instruction_and_tagged_prompt_and_asks_for_the_schema():
    client = fake_anthropic(reply=parsed_reply(make_reply(objective=copy("stated", ["Do it"]))))
    extractor = HaikuExtractor(client, "claude-haiku-4-5-20251001", 16000)
    extraction, used = extractor.extract("Do it")
    kind, kwargs = client.messages.calls[0]
    assert kind == "parse"
    assert kwargs["model"] == "claude-haiku-4-5-20251001"
    assert kwargs["max_tokens"] == 16000
    assert kwargs["system"] == EXTRACT_INSTRUCTION
    assert kwargs["messages"] == [{"role": "user", "content": "<prompt>\nDo it\n</prompt>"}]
    assert kwargs["output_format"] is ExtractionReply
    assert extraction.objective.value == "Do it"
    assert used.output_tokens == 150


def test_split_uses_its_own_instruction_and_schema():
    reply = SplitReply(parts=["A", "B"], shared_context="ctx", shared_inputs=[])
    client = fake_anthropic(reply=parsed_reply(reply))
    result, _ = HaikuExtractor(client, "haiku", 16000).split("A and B")
    _, kwargs = client.messages.calls[0]
    assert kwargs["system"] == SPLIT_INSTRUCTION
    assert kwargs["output_format"] is SplitReply
    assert result.parts == ["A", "B"]
    assert result.shared_context == "ctx"


@pytest.mark.parametrize(
    "reply, message",
    [
        (parsed_reply(None, stop_reason="max_tokens"), "cut off"),
        (parsed_reply(None, stop_reason="refusal"), "declined"),
        (parsed_reply(None), "no structured reply"),
    ],
)
def test_bad_haiku_replies_become_client_errors(reply, message):
    with pytest.raises(ClientError, match=message):
        HaikuExtractor(fake_anthropic(reply=reply), "haiku", 16000).extract("x")


def test_anthropic_errors_become_client_errors():
    client = fake_anthropic(error=anthropic.AnthropicError("overloaded"))
    with pytest.raises(ClientError, match="Haiku call failed"):
        HaikuExtractor(client, "haiku", 16000).extract("x")


# ---------------------------------------------------------------------------
# Claude: request shape and caching
# ---------------------------------------------------------------------------


def test_request_puts_the_cache_marker_on_the_shared_prompt_only():
    request = build_request("claude-sonnet-5", RENDERED, "EXTRA", "high", 3000, cache=True)
    first, second = request["messages"][0]["content"]
    assert first == {"type": "text", "text": RENDERED.user, "cache_control": {"type": "ephemeral"}}
    assert second == {"type": "text", "text": "EXTRA"}
    assert request["system"] == "You are a tutor."
    assert request["thinking"] == {"type": "adaptive"}
    assert request["output_config"] == {"effort": "high"}
    assert request["max_tokens"] == 3000


def test_no_cache_marker_and_no_system_when_not_needed():
    rendered = RenderedPrompt(system=None, user="What does df.shape return?")
    request = build_request("claude-sonnet-5", rendered, None, "low", 64000, cache=False)
    assert request["messages"][0]["content"] == [{"type": "text", "text": "What does df.shape return?"}]
    assert "system" not in request


def test_planning_and_execution_share_everything_before_the_last_block():
    plan = build_request("claude-sonnet-5", RENDERED, CHECKPOINT_INSTRUCTION, "high", 3000, cache=True)
    execute = build_request("claude-sonnet-5", RENDERED, "<approved_plan>...", "high", 64000, cache=True)
    for key in ("model", "system", "thinking", "output_config"):
        assert plan[key] == execute[key]
    assert plan["messages"][0]["content"][0] == execute["messages"][0]["content"][0]


# ---------------------------------------------------------------------------
# Claude: calls
# ---------------------------------------------------------------------------


def claude_reply(text_blocks, stop_reason="end_turn"):
    content = [SimpleNamespace(type="thinking", thinking="")]
    content += [SimpleNamespace(type="text", text=text) for text in text_blocks]
    return SimpleNamespace(
        content=content,
        stop_reason=stop_reason,
        usage=usage(input_tokens=30, output_tokens=200, cache_creation_input_tokens=1100),
    )


def executor(client):
    return ClaudeExecutor(client, "claude-sonnet-5", plan_max_tokens=3000, execute_max_tokens=64000)


def test_plan_sends_the_checkpoint_instruction_with_the_capped_budget():
    client = fake_anthropic(reply=claude_reply(["1. Outline\n", "2. Build"]))
    draft, used = executor(client).plan(RENDERED, "high")
    kind, kwargs = client.messages.calls[0]
    assert kind == "create"
    assert kwargs["max_tokens"] == 3000
    assert kwargs["messages"][0]["content"][1]["text"] == CHECKPOINT_INSTRUCTION
    assert "cache_control" in kwargs["messages"][0]["content"][0]
    assert draft.text == "1. Outline\n2. Build"  # thinking blocks are skipped
    assert draft.truncated is False
    assert used.cache_creation_input_tokens == 1100


def test_plan_that_hits_the_cap_is_marked_truncated():
    client = fake_anthropic(reply=claude_reply(["1. Outl"], stop_reason="max_tokens"))
    draft, _ = executor(client).plan(RENDERED, "high")
    assert draft.truncated is True


def test_refused_plan_is_a_client_error():
    client = fake_anthropic(reply=claude_reply([], stop_reason="refusal"))
    with pytest.raises(ClientError, match="declined"):
        executor(client).plan(RENDERED, "high")


def test_execute_streams_the_answer_and_adds_the_approved_plan():
    client = fake_anthropic(reply=claude_reply(["Hello world"]), stream_pieces=["Hello", " world"])
    pieces = []
    text, _ = executor(client).execute(RENDERED, "high", "1. Build", pieces.append)
    kind, kwargs = client.messages.calls[0]
    assert kind == "stream"
    assert kwargs["max_tokens"] == 64000
    first, second = kwargs["messages"][0]["content"]
    assert "cache_control" in first  # a planning call came first, so the cache can be read
    assert second["text"] == f"<approved_plan>\n1. Build\n</approved_plan>\n{FOLLOW_PLAN}"
    assert pieces == ["Hello", " world"]
    assert text == "Hello world"


def test_execute_without_a_plan_has_no_cache_marker():
    client = fake_anthropic(reply=claude_reply(["Hi"]), stream_pieces=["Hi"])
    executor(client).execute(RENDERED, "low", None, lambda piece: None)
    _, kwargs = client.messages.calls[0]
    assert kwargs["messages"][0]["content"] == [{"type": "text", "text": RENDERED.user}]


def test_cut_off_answer_gets_a_note():
    client = fake_anthropic(reply=claude_reply(["Part"], stop_reason="max_tokens"), stream_pieces=["Part"])
    pieces = []
    executor(client).execute(RENDERED, "low", None, pieces.append)
    assert pieces[-1] == CUT_OFF_NOTE


def test_execute_errors_become_client_errors():
    client = fake_anthropic(error=anthropic.AnthropicError("overloaded"))
    with pytest.raises(ClientError):
        executor(client).execute(RENDERED, "low", None, lambda piece: None)


def test_count_tokens_counts_the_documents_as_they_will_be_sent():
    client = fake_anthropic(token_count=25_000)
    documents = [Document(source="a.txt", content="Body")]
    assert executor(client).count_tokens(documents) == 25_000
    kind, kwargs = client.messages.calls[0]
    assert kind == "count_tokens"
    assert kwargs["model"] == "claude-sonnet-5"
    assert kwargs["messages"][0]["content"].startswith("<documents>")
