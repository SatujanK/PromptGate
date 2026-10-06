"""Run one prompt through every step and log it.

    Jev -> routing -> questions or form -> render -> plan checkpoint -> execute -> log

The pipeline talks to the outside world only through the clients in
clients.py and the UserInterface in form.py, so tests can run it end to end
with fakes. Each run fills in a LogRecord as it goes, and that record is
written exactly once, even if something goes wrong part-way through.
"""

from pydantic import BaseModel

from promptgate.checkpoint import run_checkpoint
from promptgate.clients import ClientError, Clients
from promptgate.config import Config
from promptgate.form import (
    UserInterface,
    ask_missing,
    ask_split,
    complete_form,
    describe_decision,
    describe_dry_run,
)
from promptgate.log import append_record, new_run_id, now_timestamp
from promptgate.render import (
    FixedLines,
    RenderedPrompt,
    SlotValue,
    choose_fixed_lines,
    documents_from_value,
    is_empty,
    render_full,
    render_with_answers,
)
from promptgate.routing import Decision, decide, decide_without_jev
from promptgate.schemas import (
    Document,
    Extraction,
    JevVerdict,
    LogRecord,
    Route,
    SlotName,
    SlotResult,
)


class PartToRun(BaseModel):
    """One part of a split prompt, waiting to go through the pipeline on its own."""

    prompt: str  # the part's own words
    documents: list[Document]  # shared by every part
    context: str | None  # shared by every part


class Pipeline:
    def __init__(self, config: Config, clients: Clients, ui: UserInterface):
        self.config = config
        self.clients = clients
        self.ui = ui

    # -----------------------------------------------------------------------
    # Entry point
    # -----------------------------------------------------------------------

    def run(self, prompt: str, documents: list[Document], dry_run: bool = False) -> list[LogRecord]:
        """Run a prompt. Returns the log records written: one, or one plus one per part after a split."""
        record, parts = self.run_single(prompt, documents, None, dry_run, None, allow_split=True)
        records = [record]
        for number, part in enumerate(parts, start=1):
            self.ui.show(f"\n===== Part {number} of {len(parts)}: {part.prompt} =====")
            child, _ = self.run_single(
                part.prompt, part.documents, part.context, dry_run, record.run_id, allow_split=False
            )
            records.append(child)
        return records

    def run_single(
        self,
        prompt: str,
        documents: list[Document],
        context: str | None,
        dry_run: bool,
        parent_run_id: str | None,
        allow_split: bool,
    ) -> tuple[LogRecord, list[PartToRun]]:
        """Run one prompt (or one part) and write its log line. Returns (record, parts to run next)."""
        record = LogRecord(
            run_id=new_run_id(),
            parent_run_id=parent_run_id,
            timestamp=now_timestamp(),
            dry_run=dry_run,
            prompt=prompt,
            input_sources=[document.source for document in documents],
            route="as_is",  # replaced as soon as routing decides
            models={
                "jev": self.config.models.jev,
                "extraction": self.config.models.extraction,
                "execution": self.config.models.execution,
            },
        )
        parts: list[PartToRun] = []
        try:
            parts = self.process(record, prompt, documents, context, dry_run, allow_split)
        except KeyboardInterrupt:  # Ctrl+C: still log the run, as cancelled
            record.cancelled = True
            raise
        finally:
            append_record(record, self.config.paths.log_file)  # exactly one line per run
        return record, parts

    # -----------------------------------------------------------------------
    # The steps, in order
    # -----------------------------------------------------------------------

    def process(
        self,
        record: LogRecord,
        prompt: str,
        documents: list[Document],
        context: str | None,
        dry_run: bool,
        allow_split: bool,
    ) -> list[PartToRun]:
        """Every step for one prompt, filling in the record. Returns parts only after a split."""
        # 1. Jev, then routing (pure code)
        verdict = self.classify(record, prompt)
        known_slots: list[SlotName] = []
        if documents:
            known_slots.append("inputs")
        if context:
            known_slots.append("context")
        if verdict is None:
            decision = decide_without_jev(self.config)
        else:
            decision = decide(verdict, self.config, known_slots, allow_split)

        record.task_type = decision.task_type
        record.route = decision.route
        record.missing_slots = decision.missing_slots
        record.checkpoint = decision.checkpoint
        record.effort = decision.effort
        self.ui.show(describe_decision(decision))

        # 2. Split, if Jev saw several tasks and you agree
        route: Route = decision.route
        if route == "split":
            parts = self.try_split(record, prompt, documents)
            if parts:
                record.checkpoint = False  # the parts get their own checkpoints
                return parts
            route = "as_is"
            record.route = route

        # 3. Questions or form, then render
        rendered, route = self.build_prompt(record, decision, verdict, route, prompt, documents, context)
        record.route = route

        if dry_run:
            self.ui.show(describe_dry_run(verdict, decision, route, rendered))
            return []

        # 4. Plan checkpoint
        approved_plan = None
        if decision.checkpoint:
            try:
                outcome = run_checkpoint(self.clients.claude, self.ui, rendered, decision.effort)
            except ClientError as error:
                record.error = f"planning call failed: {error}"
                self.ui.show(f"Error: {record.error}")
                return []
            record.tokens["plan"] = outcome.usage
            record.plan_edited = outcome.edited
            record.plan_truncated = outcome.truncated
            if outcome.approved_plan is None:
                record.cancelled = True
                self.ui.show("Cancelled. Nothing was sent for execution.")
                return []
            approved_plan = outcome.approved_plan

        # 5. Execute once
        self.ui.show("")
        try:
            _answer, usage = self.clients.claude.execute(
                rendered, decision.effort, approved_plan, self.ui.stream
            )
        except ClientError as error:
            record.error = f"execution call failed: {error}"
            self.ui.show(f"\nError: {record.error}")
            return []
        record.tokens["execute"] = usage
        self.ui.show(
            f"\n\n[Run {record.run_id}] If you had to correct this answer, run:\n"
            f"  promptgate mark {record.run_id} --corrected"
        )
        return []

    # -----------------------------------------------------------------------
    # Network steps, each with a fallback that adds less friction, not more
    # -----------------------------------------------------------------------

    def classify(self, record: LogRecord, prompt: str) -> JevVerdict | None:
        """The single Jev call. Returns None if Jev can't be reached."""
        try:
            verdict, usage = self.clients.jev.classify(prompt)
        except ClientError as error:
            record.jev_error = str(error)
            self.ui.show("Jev could not be reached, so the prompt will be sent as-is.")
            return None
        record.jev = verdict
        record.tokens["jev"] = usage
        record.models["jev"] = verdict.model  # the version that actually answered
        return verdict

    def extract(self, record: LogRecord, prompt: str) -> Extraction | None:
        """The Haiku extraction for the full form. Returns None if it fails."""
        try:
            extraction, usage = self.clients.extractor.extract(prompt)
        except ClientError as error:
            record.warnings.append(f"extraction failed: {error}")
            self.ui.show("The prompt couldn't be split into slots, so the form is skipped.")
            return None
        record.tokens["extract"] = usage
        return extraction

    def try_split(
        self, record: LogRecord, prompt: str, documents: list[Document]
    ) -> list[PartToRun]:
        """Ask before splitting, then have Haiku split. Returns [] to send the prompt as-is."""
        if not ask_split(self.ui):
            record.split_declined = True
            return []
        try:
            result, usage = self.clients.extractor.split(prompt)
        except ClientError as error:
            record.warnings.append(f"split failed: {error}")
            self.ui.show("The prompt couldn't be split, so it will be sent as-is.")
            return []
        record.tokens["extract"] = usage

        parts = [part for part in result.parts if part.strip()]
        if len(parts) < 2:
            record.warnings.append("split returned fewer than two parts")
            self.ui.show("Only one task was found, so the prompt will be sent as-is.")
            return []

        # Every part gets the shared material: your --input files plus anything
        # the whole prompt works on, and the shared background.
        shared_documents = documents + documents_from_value(result.shared_inputs, source="prompt")
        return [
            PartToRun(prompt=part, documents=shared_documents, context=result.shared_context)
            for part in parts
        ]

    def count_input_tokens(self, documents: list[Document]) -> int:
        """Count the documents' tokens. If counting fails, return 0 (no quote line) rather than stop."""
        if not documents:
            return 0
        try:
            return self.clients.claude.count_tokens(documents)
        except ClientError:
            return 0

    # -----------------------------------------------------------------------
    # Building the prompt for each route
    # -----------------------------------------------------------------------

    def fixed_lines(
        self, decision: Decision, documents: list[Document], has_success_criteria: bool
    ) -> FixedLines:
        return choose_fixed_lines(
            ambiguity_rule=decision.ambiguity_rule,
            input_tokens=self.count_input_tokens(documents),
            quote_grounding_tokens=self.config.limits.quote_grounding_tokens,
            has_success_criteria=has_success_criteria,
            self_check_enabled=self.config.self_check,
        )

    def build_prompt(
        self,
        record: LogRecord,
        decision: Decision,
        verdict: JevVerdict | None,
        route: Route,
        prompt: str,
        documents: list[Document],
        context: str | None,
    ) -> tuple[RenderedPrompt, Route]:
        """Ask what's needed for the route and render. Returns (prompt, route actually taken)."""
        if route == "full_form":
            extraction = self.extract(record, prompt)
            if extraction is not None:
                return self.build_from_form(record, decision, extraction, documents, context), route
            # Without an extraction there's no form: fall back to the next rule.
            route = "fill_missing" if decision.missing_slots else "as_is"

        answers: dict[SlotName, SlotValue] = {}
        if context:
            answers["context"] = context
        if route == "fill_missing":
            asked, skipped = ask_missing(self.ui, decision.missing_slots, self.config.questions)
            answers.update(asked)
            record.skipped_slots = skipped

        # An inputs answer becomes a document, so it goes at the top.
        all_documents = documents + documents_from_value(answers.pop("inputs", None), "answer")

        jev_says_criteria = (
            verdict is not None
            and verdict.presence["success_criteria"] >= self.config.thresholds.present_threshold
        )
        has_criteria = jev_says_criteria or not is_empty(answers.get("success_criteria"))

        fixed = self.fixed_lines(decision, all_documents, has_criteria)
        rendered = render_with_answers(prompt, answers, all_documents, decision.system_prompt, fixed)
        return rendered, route

    def build_from_form(
        self,
        record: LogRecord,
        decision: Decision,
        extraction: Extraction,
        documents: list[Document],
        context: str | None,
    ) -> RenderedPrompt:
        """The full_form route: you check every slot, then code renders the full template."""
        if context and extraction.context.status == "missing":
            extraction.context = SlotResult(status="stated", value=context)

        skip: set[SlotName] = {"inputs"} if documents else set()
        slots, skipped = complete_form(
            self.ui, extraction, decision.form_slots, self.config.questions, skip
        )
        record.skipped_slots = skipped

        if extraction.role.status == "stated":  # a role you wrote beats the default
            slots["role"] = extraction.role.value

        all_documents = documents + documents_from_value(slots.pop("inputs", None), "prompt")
        has_criteria = not is_empty(slots.get("success_criteria"))
        fixed = self.fixed_lines(decision, all_documents, has_criteria)
        return render_full(slots, all_documents, decision.system_prompt, fixed)
