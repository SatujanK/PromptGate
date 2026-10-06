"""Haiku extraction (for the full form) and task splitting (for the split route).

Haiku only copies. It never paraphrases, never fills a missing slot, and
never resolves a conflict.

Structured outputs make Haiku's reply match a schema exactly. The API turns
that schema into a grammar, and optional fields and "either/or" (union)
types make the grammar much more expensive to build. So Haiku fills in the
flat reply models below, where every field is required and every value is
a list of strings. Code then converts them into the project's Extraction
and SplitResult.
"""

from typing import Literal

import anthropic
from pydantic import BaseModel, ValidationError

from promptgate.clients import ClientError, token_usage
from promptgate.render import SlotValue, as_text
from promptgate.schemas import Extraction, SlotResult, SplitResult, TokenUsage

MAX_CANDIDATE_OBJECTIVES = 3

EXTRACT_INSTRUCTION = """\
You copy the parts of a user's prompt into labelled slots. You never write new content.

Rules:
1. Copy the user's words into slots. Do not paraphrase.
2. If two instructions conflict, list both and choose neither. This applies even when \
they would go in different slots, such as a level of detail and a word limit: put both, \
copied exactly, as the options of the one slot they compete over (usually output_format), \
and leave them out of the other slots.
3. If something is not stated, mark it missing and never fill it.

The slots:
- role: a role or persona the user gives Claude, such as "You are a ...".
- inputs: material Claude should work on, such as pasted text, code, or data. Copy it in full.
- context: background, such as who the result is for, the situation, or why it matters.
- objective: the one thing the user wants done.
- requirements: what the result must include or do.
- constraints: limits the result must stay within.
- resources: tools, libraries, or sources to use or stay within.
- output_format: the structure, length, or file type of the result.
- success_criteria: how the user will judge whether the result is right.
- examples: examples of what the user wants.

For each slot, set status to one of:
- "stated": value holds the exact words from the prompt, as one item or one item per \
separate piece. options is empty.
- "missing": value and options are both empty.
- "conflict": value is empty, and options holds each conflicting version, copied exactly.

candidate_objectives: only if the objective is vague, list 2 or 3 possible objectives, \
each an exact phrase copied from the prompt. Otherwise leave it empty.
"""

SPLIT_INSTRUCTION = """\
You divide a prompt that asks for several separate tasks. You never write new content; you only copy.

Rules:
1. Copy the user's words. Do not paraphrase.
2. parts: one item per separate task, each in the user's own words, in the order they appear.
3. shared_context: background that applies to every task, copied exactly. Use an empty \
string if there is none.
4. shared_inputs: material every task works on, such as pasted text, code, or data, copied \
in full. Leave it empty if there is none. Do not repeat this material inside the parts.
5. Never add anything that is not in the prompt.
"""


# ---------------------------------------------------------------------------
# The reply shapes Haiku fills in (flat, every field required)
# ---------------------------------------------------------------------------


class SlotCopy(BaseModel):
    status: Literal["stated", "missing", "conflict"]
    value: list[str]  # the user's words; empty unless stated
    options: list[str]  # the conflicting versions; empty unless conflict


class ExtractionReply(BaseModel):
    role: SlotCopy
    inputs: SlotCopy
    context: SlotCopy
    objective: SlotCopy
    requirements: SlotCopy
    constraints: SlotCopy
    resources: SlotCopy
    output_format: SlotCopy
    success_criteria: SlotCopy
    examples: SlotCopy
    candidate_objectives: list[str]


class SplitReply(BaseModel):
    parts: list[str]
    shared_context: str  # "" when there is none
    shared_inputs: list[str]


# ---------------------------------------------------------------------------
# Converting replies into the project's models (pure)
# ---------------------------------------------------------------------------


def non_blank(items: list[str]) -> list[str]:
    return [item for item in items if item.strip()]


def normalize(text: str) -> str:
    """Lowercase and collapse spacing, so a copy still matches if only capitals or spaces differ."""
    return " ".join(text.lower().split())


def copied_from(prompt: str, items: list[str]) -> list[str]:
    """Keep only the items that appear word for word in the prompt (ignoring case and spacing).

    This is code enforcing principle 3: Haiku may only copy. Anything it
    reworded is dropped here rather than shown to you as if you had written it.
    """
    source = normalize(prompt)
    return [item for item in non_blank(items) if normalize(item).strip(" .,;:!?\"'") in source]


def to_slot_value(pieces: list[str]) -> SlotValue:
    """No pieces -> None, one piece -> text, several -> the list."""
    pieces = non_blank(pieces)
    if not pieces:
        return None
    if len(pieces) == 1:
        return pieces[0]
    return pieces


def to_slot_result(copy: SlotCopy, prompt: str) -> SlotResult:
    """Convert one slot, correcting replies that don't make sense rather than failing.

    - "stated" with no words counts as missing.
    - "conflict" keeps only versions copied word for word from the prompt.
      With one left it counts as stated; with none, as missing.
    - "missing" always ends up empty, so nothing Haiku wrote there is used.
    """
    if copy.status == "conflict":
        options = copied_from(prompt, copy.options)
        if len(options) >= 2:
            return SlotResult(status="conflict", options=options)
        if len(options) == 1:
            return SlotResult(status="stated", value=options[0])
        return SlotResult(status="missing")

    if copy.status == "stated":
        value = to_slot_value(copy.value)
        if value is not None:
            return SlotResult(status="stated", value=value)

    return SlotResult(status="missing")


def to_extraction(reply: ExtractionReply, prompt: str) -> Extraction:
    """Convert Haiku's reply. prompt is the original text, used to check what was really copied."""
    objective = to_slot_result(reply.objective, prompt)

    # Candidates must be copied from the prompt, and repeating your own
    # objective back to you as a "candidate" adds nothing.
    stated = normalize(as_text(objective.value)) if objective.status == "stated" else None
    candidates = [
        candidate
        for candidate in copied_from(prompt, reply.candidate_objectives)
        if normalize(candidate) != stated
    ]

    return Extraction(
        role=to_slot_result(reply.role, prompt),
        inputs=to_slot_result(reply.inputs, prompt),
        context=to_slot_result(reply.context, prompt),
        objective=objective,
        requirements=to_slot_result(reply.requirements, prompt),
        constraints=to_slot_result(reply.constraints, prompt),
        resources=to_slot_result(reply.resources, prompt),
        output_format=to_slot_result(reply.output_format, prompt),
        success_criteria=to_slot_result(reply.success_criteria, prompt),
        examples=to_slot_result(reply.examples, prompt),
        candidate_objectives=candidates[:MAX_CANDIDATE_OBJECTIVES],
    )


def to_split_result(reply: SplitReply) -> SplitResult:
    return SplitResult(
        parts=non_blank(reply.parts),
        shared_context=reply.shared_context.strip() or None,
        shared_inputs=non_blank(reply.shared_inputs),
    )


def build_request(model: str, max_tokens: int, instruction: str, prompt: str) -> dict:
    """The arguments for one Haiku call. The prompt goes in tags so Haiku treats it as data."""
    return {
        "model": model,
        "max_tokens": max_tokens,
        "system": instruction,
        "messages": [{"role": "user", "content": f"<prompt>\n{prompt}\n</prompt>"}],
    }


# ---------------------------------------------------------------------------
# The client
# ---------------------------------------------------------------------------


class HaikuExtractor:
    """Copies a prompt into slots, or splits it into tasks, with Claude Haiku."""

    def __init__(self, client: anthropic.Anthropic, model: str, max_tokens: int):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def call(self, instruction: str, prompt: str, reply_model: type[BaseModel]):
        """Make one structured-output call and return (parsed reply, usage)."""
        request = build_request(self.model, self.max_tokens, instruction, prompt)
        try:
            response = self.client.messages.parse(**request, output_format=reply_model)
        except anthropic.AnthropicError as error:
            raise ClientError(f"Haiku call failed: {error}") from error
        except ValidationError as error:  # the reply didn't match the schema
            raise ClientError(f"Haiku's reply didn't match the schema: {error}") from error

        if response.stop_reason == "max_tokens":
            raise ClientError("Haiku's reply was cut off; raise extract_max_tokens in config.toml")
        if response.stop_reason == "refusal":
            raise ClientError("Haiku declined to process the prompt")
        if response.parsed_output is None:
            raise ClientError("Haiku returned no structured reply")
        return response.parsed_output, token_usage(response.usage)

    def extract(self, prompt: str) -> tuple[Extraction, TokenUsage]:
        reply, usage = self.call(EXTRACT_INSTRUCTION, prompt, ExtractionReply)
        return to_extraction(reply, prompt), usage

    def split(self, prompt: str) -> tuple[SplitResult, TokenUsage]:
        reply, usage = self.call(SPLIT_INSTRUCTION, prompt, SplitReply)
        return to_split_result(reply), usage
