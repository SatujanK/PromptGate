"""Build the final prompt text from slots, answers, documents, and fixed lines.

Everything here is a pure function: text in, text out. Nothing is rewritten.
Your words and your documents pass through exactly as given; code only adds
XML tags around them and the fixed lines at the end.
"""

from pydantic import BaseModel

from promptgate.schemas import USER_MESSAGE_ORDER, Document, SlotName

# ---------------------------------------------------------------------------
# Fixed lines (exact text from CLAUDE.md; tests check they still match it)
# ---------------------------------------------------------------------------

AMBIGUITY_RULE = (
    "If anything above is unclear or conflicting, ask me before starting. "
    "For minor details, state your assumption and continue."
)
CHECKPOINT_INSTRUCTION = (
    "Before doing the task, reply only with: (1) a numbered plan of at most 8 steps, "
    "(2) the assumptions you are making, (3) any questions. Do not start the task."
)
FOLLOW_PLAN = "Follow the approved plan."
QUOTE_GROUNDING = (
    "First quote the parts of the documents relevant to the task in <quotes> tags, "
    "then do the task."
)
SELF_CHECK = "Before you finish, check your answer against the success criteria."

# A slot value is the user's own words: one piece of text, or a list of pieces.
SlotValue = str | list[str] | None


class RenderedPrompt(BaseModel):
    """The prompt as Claude will receive it."""

    system: str | None  # the role, or None for no system prompt
    user: str  # the whole user message


class FixedLines(BaseModel):
    """Which optional fixed lines to add at the end of the user message."""

    ambiguity_rule: bool
    quote_grounding: bool
    self_check: bool


# ---------------------------------------------------------------------------
# Deciding the fixed lines
# ---------------------------------------------------------------------------


def choose_fixed_lines(
    ambiguity_rule: bool,
    input_tokens: int,
    quote_grounding_tokens: int,
    has_success_criteria: bool,
    self_check_enabled: bool,
) -> FixedLines:
    """Work out which fixed lines this prompt gets.

    ambiguity_rule: from the routing Decision (off only for quick_question).
    input_tokens: the counted size of the documents, or 0 if there are none.
    quote_grounding_tokens: the threshold from config.toml (20000).
    has_success_criteria: whether the prompt ends up with success criteria.
    self_check_enabled: the execution model's self_check setting.
    """
    return FixedLines(
        ambiguity_rule=ambiguity_rule,
        quote_grounding=input_tokens >= quote_grounding_tokens,
        self_check=has_success_criteria and self_check_enabled,
    )


# ---------------------------------------------------------------------------
# Small building blocks
# ---------------------------------------------------------------------------


def is_empty(value: SlotValue) -> bool:
    """True for None, blank text, or a list with nothing but blank text in it."""
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return all(item.strip() == "" for item in value)


def as_text(value: SlotValue) -> str:
    """Turn a slot value into text. List items go on separate lines, unchanged."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return "\n".join(item.strip() for item in value if item.strip())


def as_list(value: SlotValue) -> list[str]:
    """Turn a slot value into a list of non-blank pieces."""
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return [item.strip() for item in value if item.strip()]


def wrap(tag: str, text: str) -> str:
    """Put text inside an XML tag, each on its own line."""
    return f"<{tag}>\n{text}\n</{tag}>"


def documents_from_value(value: SlotValue, source: str) -> list[Document]:
    """Turn an inputs value (from a form answer or extraction) into Documents."""
    return [Document(source=source, content=piece) for piece in as_list(value)]


def render_documents(documents: list[Document]) -> str:
    """Wrap documents in the <documents> format Anthropic recommends, numbered from 1."""
    parts = []
    for index, document in enumerate(documents, start=1):
        parts.append(
            f'<document index="{index}">\n'
            f"<source>{document.source}</source>\n"
            f"{wrap('document_content', document.content)}\n"
            f"</document>"
        )
    return wrap("documents", "\n".join(parts))


def render_examples(value: SlotValue) -> str:
    """Wrap each example in <example> tags, all inside <examples>."""
    examples = [wrap("example", example) for example in as_list(value)]
    return wrap("examples", "\n".join(examples))


def render_slot(name: SlotName, value: SlotValue) -> str:
    """Wrap one slot in a tag of the same name. Examples get their own nested format."""
    if name == "examples":
        return render_examples(value)
    return wrap(name, as_text(value))


def render_fixed_lines(fixed: FixedLines) -> list[str]:
    """The fixed lines to add, in order. The ambiguity rule is last because it refers to "above"."""
    lines = []
    if fixed.quote_grounding:
        lines.append(QUOTE_GROUNDING)
    if fixed.self_check:
        lines.append(SELF_CHECK)
    if fixed.ambiguity_rule:
        lines.append(AMBIGUITY_RULE)
    return lines


def join_blocks(blocks: list[str]) -> str:
    """Join the parts of the user message with a blank line between each."""
    return "\n\n".join(block for block in blocks if block)


# ---------------------------------------------------------------------------
# The two ways a prompt is built
# ---------------------------------------------------------------------------


def render_with_answers(
    original_prompt: str,
    answers: dict[SlotName, SlotValue],
    documents: list[Document],
    system_prompt: str | None,
    fixed: FixedLines,
) -> RenderedPrompt:
    """Build the prompt for the as_is and fill_missing routes.

    Your original prompt is kept exactly as written. Documents go above it,
    because long material works best at the top. Answers to the pre-written
    questions go below it as tagged blocks, in the standard slot order.
    For as_is, answers is empty.

    An "inputs" answer is not allowed here: pass it in documents instead
    (see documents_from_value), so it lands at the top.
    """
    if "inputs" in answers:
        raise ValueError("pass an inputs answer as documents, not as an answer")

    blocks = []
    if documents:
        blocks.append(render_documents(documents))
    blocks.append(original_prompt.strip())
    for name in USER_MESSAGE_ORDER:
        value = answers.get(name)
        if not is_empty(value):
            blocks.append(render_slot(name, value))
    blocks.extend(render_fixed_lines(fixed))
    return RenderedPrompt(system=system_prompt, user=join_blocks(blocks))


def render_full(
    slots: dict[SlotName, SlotValue],
    documents: list[Document],
    default_role: str | None,
    fixed: FixedLines,
) -> RenderedPrompt:
    """Build the full template for the full_form route.

    slots: the completed form, e.g. {"objective": "...", "context": "..."}.
        Empty slots are left out. A "role" slot goes in the system prompt.
    documents: every document (from --input and from the inputs slot).
    default_role: used only when the form has no role.
    """
    if "inputs" in slots:
        raise ValueError("pass the inputs slot as documents, not as a slot")

    role = slots.get("role")
    system = as_text(role) if not is_empty(role) else default_role

    blocks = []
    if documents:
        blocks.append(render_documents(documents))
    for name in USER_MESSAGE_ORDER:
        value = slots.get(name)
        if not is_empty(value):
            blocks.append(render_slot(name, value))
    blocks.extend(render_fixed_lines(fixed))
    return RenderedPrompt(system=system or None, user=join_blocks(blocks))


# ---------------------------------------------------------------------------
# Plan checkpoint blocks (sent after the shared, cached prompt)
# ---------------------------------------------------------------------------


def plan_request_block() -> str:
    """The extra block for the planning call."""
    return CHECKPOINT_INSTRUCTION


def approved_plan_block(plan: str) -> str:
    """The extra block for the execution call after you approve a plan."""
    return f"{wrap('approved_plan', plan.strip())}\n{FOLLOW_PLAN}"
