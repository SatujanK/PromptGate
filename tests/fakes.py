"""Fake clients and a scripted user interface, so tests never call an API or wait for typing.

Each fake records the calls it receives, so tests can check, for example,
that a dry run never called Claude.
"""

from promptgate.clients import ClientError, Clients, PlanDraft
from promptgate.render import RenderedPrompt
from promptgate.schemas import Document, Extraction, JevVerdict, SplitResult, TokenUsage


class FakeJev:
    """Returns a fixed verdict, or a different one per prompt (for split parts)."""

    def __init__(
        self,
        verdict: JevVerdict | None = None,
        by_prompt: dict[str, JevVerdict] | None = None,
        fail: bool = False,
    ):
        self.verdict = verdict
        self.by_prompt = by_prompt or {}
        self.fail = fail
        self.calls: list[str] = []

    def classify(self, prompt: str) -> tuple[JevVerdict, TokenUsage]:
        self.calls.append(prompt)
        if self.fail:
            raise ClientError("Jev is down")
        verdict = self.by_prompt.get(prompt, self.verdict)
        assert verdict is not None, f"FakeJev has no verdict for {prompt!r}"
        return verdict, TokenUsage(input_tokens=50)


class FakeExtractor:
    def __init__(
        self,
        extraction: Extraction | None = None,
        split_result: SplitResult | None = None,
        fail: bool = False,
    ):
        self.extraction = extraction
        self.split_result = split_result
        self.fail = fail
        self.extract_calls: list[str] = []
        self.split_calls: list[str] = []

    def extract(self, prompt: str) -> tuple[Extraction, TokenUsage]:
        self.extract_calls.append(prompt)
        if self.fail:
            raise ClientError("Haiku is down")
        assert self.extraction is not None, "FakeExtractor was not given an extraction"
        return self.extraction, TokenUsage(input_tokens=300, output_tokens=200)

    def split(self, prompt: str) -> tuple[SplitResult, TokenUsage]:
        self.split_calls.append(prompt)
        if self.fail:
            raise ClientError("Haiku is down")
        assert self.split_result is not None, "FakeExtractor was not given a split result"
        return self.split_result, TokenUsage(input_tokens=200, output_tokens=100)


class FakeClaude:
    """Records every planning and execution call instead of making it."""

    def __init__(
        self,
        plan_text: str = "1. Outline\n2. Build\n3. Test",
        plan_truncated: bool = False,
        answer: str = "Here is the answer.",
        token_count: int = 100,
        fail_plan: bool = False,
        fail_execute: bool = False,
    ):
        self.plan_text = plan_text
        self.plan_truncated = plan_truncated
        self.answer = answer
        self.token_count = token_count
        self.fail_plan = fail_plan
        self.fail_execute = fail_execute
        self.count_calls: list[list[Document]] = []
        self.plan_calls: list[tuple[RenderedPrompt, str]] = []
        self.execute_calls: list[tuple[RenderedPrompt, str, str | None]] = []

    def count_tokens(self, documents: list[Document]) -> int:
        self.count_calls.append(documents)
        return self.token_count

    def plan(self, rendered: RenderedPrompt, effort: str) -> tuple[PlanDraft, TokenUsage]:
        self.plan_calls.append((rendered, effort))
        if self.fail_plan:
            raise ClientError("planning failed")
        draft = PlanDraft(text=self.plan_text, truncated=self.plan_truncated)
        return draft, TokenUsage(input_tokens=1200, output_tokens=150, cache_creation_input_tokens=1100)

    def execute(self, rendered, effort, approved_plan, on_text) -> tuple[str, TokenUsage]:
        self.execute_calls.append((rendered, effort, approved_plan))
        if self.fail_execute:
            raise ClientError("execution failed")
        on_text(self.answer)
        return self.answer, TokenUsage(input_tokens=1250, output_tokens=900, cache_read_input_tokens=1100)


class ScriptedUI:
    """Answers questions from a list, in order, and records everything shown.

    Answers for ask/ask_multiline are strings; answers for choose are option
    numbers counted from 0. If the pipeline asks more than was scripted, the
    test fails and names the unexpected question.
    """

    def __init__(self, answers: list[str | int] | None = None):
        self.answers = list(answers or [])
        self.shown: list[str] = []
        self.questions: list[str] = []
        self.streamed = ""

    def next_answer(self, question: str) -> str | int:
        self.questions.append(question)
        assert self.answers, f"UI asked a question the test didn't script: {question!r}"
        return self.answers.pop(0)

    def show(self, text: str) -> None:
        self.shown.append(text)

    def ask(self, question: str) -> str:
        answer = self.next_answer(question)
        assert isinstance(answer, str), f"expected a text answer for {question!r}"
        return answer

    def ask_multiline(self, question: str) -> str:
        return self.ask(question)

    def choose(self, question: str, options: list[str]) -> int:
        answer = self.next_answer(question)
        assert isinstance(answer, int), f"expected an option number for {question!r}"
        assert 0 <= answer < len(options), f"option {answer} doesn't exist for {question!r}"
        return answer

    def stream(self, text: str) -> None:
        self.streamed += text

    @property
    def all_output(self) -> str:
        return "\n".join(self.shown)


def make_clients(
    jev: FakeJev | None = None,
    extractor: FakeExtractor | None = None,
    claude: FakeClaude | None = None,
) -> Clients:
    return Clients(
        jev=jev or FakeJev(),
        extractor=extractor or FakeExtractor(),
        claude=claude or FakeClaude(),
    )
