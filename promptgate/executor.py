"""Claude calls: token counting, the planning call, and the one execution call.

Prompt caching: when a run has a plan checkpoint, the planning and execution
calls send the same system prompt and the same first user block (your
rendered prompt). A cache marker on that block lets the execution call read
it from the cache at a fraction of the price. Only the block after it
differs: the checkpoint instruction, or your approved plan.

For a cache hit, both calls must also use the same thinking and effort
settings, which is why build_request always sets both the same way.
Claude Sonnet 5 only caches prompts of 1,024 tokens or more; shorter ones
simply miss, with no error and no extra charge.

Runs without a checkpoint make only one call, so they get no cache marker.
Writing to the cache costs a little extra and nothing would ever read it.
"""

from collections.abc import Callable

import anthropic

from promptgate.clients import ClientError, PlanDraft, token_usage
from promptgate.render import RenderedPrompt, approved_plan_block, plan_request_block, render_documents
from promptgate.schemas import Document, Effort, TokenUsage

CUT_OFF_NOTE = "\n\n[The answer hit the token limit and may be cut off.]"


def build_request(
    model: str,
    rendered: RenderedPrompt,
    extra_block: str | None,
    effort: Effort,
    max_tokens: int,
    cache: bool,
) -> dict:
    """The arguments for one planning or execution call.

    rendered: the shared prompt (system prompt + user message).
    extra_block: the checkpoint instruction, the approved plan, or None.
    cache: put a cache marker on the shared prompt block.
    """
    prompt_block: dict = {"type": "text", "text": rendered.user}
    if cache:
        prompt_block["cache_control"] = {"type": "ephemeral"}
    content = [prompt_block]
    if extra_block:
        content.append({"type": "text", "text": extra_block})

    request: dict = {
        "model": model,
        "max_tokens": max_tokens,  # thinking tokens count toward this too
        "thinking": {"type": "adaptive"},  # the same on every call, so the cache holds
        "output_config": {"effort": effort},  # the same for planning and execution
        "messages": [{"role": "user", "content": content}],
    }
    if rendered.system:
        request["system"] = rendered.system
    return request


def text_of(message) -> str:
    """Join the text blocks of a reply. Thinking blocks are skipped."""
    return "".join(block.text for block in message.content if block.type == "text")


class ClaudeExecutor:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        plan_max_tokens: int,
        execute_max_tokens: int,
    ):
        self.client = client
        self.model = model
        self.plan_max_tokens = plan_max_tokens
        self.execute_max_tokens = execute_max_tokens

    def count_tokens(self, documents: list[Document]) -> int:
        """Count the documents exactly as they'll be sent, with the execution model's tokenizer."""
        try:
            response = self.client.messages.count_tokens(
                model=self.model,
                messages=[{"role": "user", "content": render_documents(documents)}],
            )
        except anthropic.AnthropicError as error:
            raise ClientError(f"token counting failed: {error}") from error
        return response.input_tokens

    def plan(self, rendered: RenderedPrompt, effort: Effort) -> tuple[PlanDraft, TokenUsage]:
        """The planning call: the shared prompt plus the checkpoint instruction."""
        request = build_request(
            self.model, rendered, plan_request_block(), effort, self.plan_max_tokens, cache=True
        )
        try:
            message = self.client.messages.create(**request)
        except anthropic.AnthropicError as error:
            raise ClientError(str(error)) from error
        if message.stop_reason == "refusal":
            raise ClientError("Claude declined to plan this task")
        draft = PlanDraft(text=text_of(message), truncated=message.stop_reason == "max_tokens")
        return draft, token_usage(message.usage)

    def execute(
        self,
        rendered: RenderedPrompt,
        effort: Effort,
        approved_plan: str | None,
        on_text: Callable[[str], None],
    ) -> tuple[str, TokenUsage]:
        """The execution call. The answer is streamed to on_text as it arrives."""
        extra = approved_plan_block(approved_plan) if approved_plan else None
        request = build_request(
            self.model,
            rendered,
            extra,
            effort,
            self.execute_max_tokens,
            cache=approved_plan is not None,  # a cache hit is only possible after a planning call
        )
        try:
            # Streaming shows the answer as it is written, and avoids the time
            # limit the SDK puts on long non-streaming requests.
            with self.client.messages.stream(**request) as stream:
                for piece in stream.text_stream:
                    on_text(piece)
                message = stream.get_final_message()
        except anthropic.AnthropicError as error:
            raise ClientError(str(error)) from error

        if message.stop_reason == "refusal":
            raise ClientError("Claude declined this request")
        if message.stop_reason == "max_tokens":
            on_text(CUT_OFF_NOTE)
        return text_of(message), token_usage(message.usage)
