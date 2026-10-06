"""The small interfaces the pipeline expects from its network clients.

The pipeline never talks to an SDK directly, only to these interfaces.
The real clients (jev_client.py, extractor.py, executor.py) and the test
fakes (tests/fakes.py) both follow them, so tests can run the whole
pipeline without spending anything.

A Protocol lists the methods a class must have. Any class with matching
methods fits, without inheriting from it.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel

from promptgate.render import RenderedPrompt
from promptgate.schemas import Document, Effort, Extraction, JevVerdict, SplitResult, TokenUsage


class ClientError(Exception):
    """Any failure talking to Jev or Claude. Real clients turn SDK errors into this."""


def token_usage(usage: object) -> TokenUsage:
    """Copy the token counts from an SDK's usage object. Counts the SDK leaves out become 0.

    Works for both the Anthropic and TypeSafe SDKs, because it only reads
    fields by name; getattr returns None for a field an object doesn't have.
    """
    return TokenUsage(
        input_tokens=getattr(usage, "input_tokens", None) or 0,
        output_tokens=getattr(usage, "output_tokens", None) or 0,
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None) or 0,
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None) or 0,
    )


class PlanDraft(BaseModel):
    """Claude's reply to the planning call."""

    text: str
    truncated: bool  # True if the reply stopped because it hit plan_max_tokens


class JevAPI(Protocol):
    def classify(self, prompt: str) -> tuple[JevVerdict, TokenUsage]:
        """Ask Jev every question about the prompt in one call."""
        ...


class ExtractorAPI(Protocol):
    def extract(self, prompt: str) -> tuple[Extraction, TokenUsage]:
        """Copy the prompt into slots (Haiku)."""
        ...

    def split(self, prompt: str) -> tuple[SplitResult, TokenUsage]:
        """Divide a prompt that holds several tasks (Haiku)."""
        ...


class ClaudeAPI(Protocol):
    def count_tokens(self, documents: list[Document]) -> int:
        """Count the tokens in the documents, for the quote-grounding rule."""
        ...

    def plan(self, rendered: RenderedPrompt, effort: Effort) -> tuple[PlanDraft, TokenUsage]:
        """The planning call: the prompt plus the checkpoint instruction."""
        ...

    def execute(
        self,
        rendered: RenderedPrompt,
        effort: Effort,
        approved_plan: str | None,
        on_text: Callable[[str], None],
    ) -> tuple[str, TokenUsage]:
        """The execution call. on_text receives the answer piece by piece as it streams in."""
        ...


@dataclass
class Clients:
    """The three clients the pipeline needs, bundled so they can be passed around together."""

    jev: JevAPI
    extractor: ExtractorAPI
    claude: ClaudeAPI
