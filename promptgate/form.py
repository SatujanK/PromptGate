"""Everything you see and type in the terminal.

The question and form logic here only uses the small UserInterface below,
never input() or print() directly. TerminalUI connects it to the real
terminal; tests connect it to a list of scripted answers instead.
"""

from typing import Protocol

from promptgate.render import RenderedPrompt, SlotValue, as_text, is_empty
from promptgate.routing import Decision
from promptgate.schemas import USER_MESSAGE_ORDER, Extraction, JevVerdict, Route, SlotName, SlotResult

# Slots whose answers usually span several lines (a pasted article, code examples).
MULTILINE_SLOTS: set[SlotName] = {"inputs", "examples"}
END_MARKER = "."  # a line with only this ends a multi-line answer
EXAMPLE_SEPARATOR = "---"  # a line with only this separates two examples
PREVIEW_LENGTH = 300  # characters of a long value shown on screen


class UserInterface(Protocol):
    def show(self, text: str) -> None:
        """Print text on its own line(s)."""
        ...

    def ask(self, question: str) -> str:
        """Ask for a one-line answer. An empty answer means skip or keep."""
        ...

    def ask_multiline(self, question: str) -> str:
        """Ask for an answer that can span several lines."""
        ...

    def choose(self, question: str, options: list[str]) -> int:
        """Show numbered options and return the index (from 0) of the one picked."""
        ...

    def stream(self, text: str) -> None:
        """Print a piece of Claude's answer as it arrives, without a line break."""
        ...


BOM = "\ufeff"  # the invisible byte-order mark


def read_line(prompt: str = "") -> str:
    """input(), minus the invisible marker (a BOM) PowerShell puts in front of piped text."""
    return input(prompt).lstrip(BOM)


class TerminalUI:
    """The real terminal."""

    def show(self, text: str) -> None:
        print(text)

    def ask(self, question: str) -> str:
        print(question)
        try:
            return read_line("> ")
        except EOFError:  # input was closed (e.g. piped from a file): treat as blank
            return ""

    def ask_multiline(self, question: str) -> str:
        print(question)
        print(f"(Finish with a line containing only '{END_MARKER}')")
        lines = []
        while True:
            try:
                line = read_line()
            except EOFError:
                break
            if line.strip() == END_MARKER:
                break
            lines.append(line)
        return "\n".join(lines)

    def choose(self, question: str, options: list[str]) -> int:
        print(question)
        for number, option in enumerate(options, start=1):
            print(f"  {number}. {option}")
        while True:
            try:
                answer = read_line("> ").strip()
            except EOFError:
                return 0  # no way to ask again, so take the first option
            if answer.isdigit() and 1 <= int(answer) <= len(options):
                return int(answer) - 1
            print(f"Type a number from 1 to {len(options)}.")

    def stream(self, text: str) -> None:
        print(text, end="", flush=True)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def label(name: SlotName) -> str:
    """Turn a slot name into a label, e.g. "output_format" becomes "Output format"."""
    return name.replace("_", " ").capitalize()


def preview(value: SlotValue) -> str:
    """The value as text, shortened if it is long, indented for display."""
    text = as_text(value)
    if len(text) > PREVIEW_LENGTH:
        text = text[:PREVIEW_LENGTH] + f"... ({len(text)} characters in total)"
    return "\n".join("    " + line for line in text.splitlines())


def parse_answer(name: SlotName, text: str) -> SlotValue:
    """Turn typed text into a slot value. Examples are split on '---' lines into a list."""
    if name == "examples":
        examples: list[str] = []
        current: list[str] = []
        for line in text.splitlines():
            if line.strip() == EXAMPLE_SEPARATOR:
                examples.append("\n".join(current))
                current = []
            else:
                current.append(line)
        examples.append("\n".join(current))
        examples = [example.strip() for example in examples if example.strip()]
        return examples or None
    return text.strip() or None


def ask_slot(ui: UserInterface, name: SlotName, question: str) -> SlotValue:
    """Ask the pre-written question for one slot and return the answer (None if skipped)."""
    prompt = f"{label(name)}: {question}"
    if name == "examples":
        prompt += f" Separate examples with a line containing only '{EXAMPLE_SEPARATOR}'."
    if name in MULTILINE_SLOTS:
        text = ui.ask_multiline(prompt)
    else:
        text = ui.ask(prompt)
    return parse_answer(name, text)


# ---------------------------------------------------------------------------
# fill_missing: one question per missing slot
# ---------------------------------------------------------------------------


def ask_missing(
    ui: UserInterface, missing: list[SlotName], questions: dict[SlotName, str]
) -> tuple[dict[SlotName, SlotValue], list[SlotName]]:
    """Ask the question for each missing slot. Returns (answers, skipped slots)."""
    answers: dict[SlotName, SlotValue] = {}
    skipped: list[SlotName] = []
    if missing:
        ui.show("\nA few details are missing. Press Enter to skip any question.")
    for name in missing:
        value = ask_slot(ui, name, questions[name])
        if is_empty(value):
            skipped.append(name)
        else:
            answers[name] = value
    return answers, skipped


# ---------------------------------------------------------------------------
# split: ask before splitting
# ---------------------------------------------------------------------------


def ask_split(ui: UserInterface) -> bool:
    """Ask whether to split a prompt that looks like several tasks. True means split."""
    choice = ui.choose(
        "\nThis looks like more than one task. How should it be sent?",
        ["Split it into separate runs", "Send it as-is"],
    )
    return choice == 0


# ---------------------------------------------------------------------------
# full_form: check and complete every slot
# ---------------------------------------------------------------------------


def slots_to_show(
    extraction: Extraction, form_slots: list[SlotName], skip: set[SlotName]
) -> list[SlotName]:
    """The profile's slots, then any other slot where the prompt already says something.

    Showing those extra slots means nothing you wrote is silently dropped
    from the final prompt.
    """
    shown = [name for name in form_slots if name not in skip]
    for name in USER_MESSAGE_ORDER:
        if name in form_slots or name in skip:
            continue
        if extraction.slot(name).status != "missing":
            shown.append(name)
    return shown


def review_stated(ui: UserInterface, name: SlotName, value: SlotValue) -> SlotValue:
    """Show what the prompt says for a slot and let you keep or replace it."""
    ui.show(f"\n{label(name)}:\n{preview(value)}")
    if name in MULTILINE_SLOTS:
        if ui.choose("Keep this?", ["Keep", "Replace"]) == 0:
            return value
        return parse_answer(name, ui.ask_multiline(f"{label(name)}: type the replacement."))
    answer = ui.ask("Press Enter to keep it, or type a replacement.")
    return answer.strip() or value


def resolve_conflict(
    ui: UserInterface, name: SlotName, options: list[str], question: str
) -> SlotValue:
    """Show the conflicting versions and let you pick one or type your own."""
    choice = ui.choose(
        f"\n{label(name)}: your prompt says this in different ways. Which do you mean?",
        options + ["None of these (type my own)"],
    )
    if choice < len(options):
        return options[choice]
    return ask_slot(ui, name, question)


def choose_objective(
    ui: UserInterface, result: SlotResult, candidates: list[str], question: str
) -> SlotValue:
    """Offer Haiku's candidate objectives (built from your own words) or your own."""
    choices: list[tuple[str, SlotValue]] = []
    if result.status == "stated":
        choices.append((f"Keep what I wrote: {as_text(result.value)}", result.value))
    for option in result.options:  # a conflict's versions, if any
        choices.append((option, option))
    for candidate in candidates:
        choices.append((candidate, candidate))
    choices.append(("None of these (type my own)", None))

    index = ui.choose(
        "\nObjective: which of these is what you want Claude to do?",
        [text for text, _ in choices],
    )
    value = choices[index][1]
    if value is None:
        value = ask_slot(ui, "objective", question)
    return value


def complete_form(
    ui: UserInterface,
    extraction: Extraction,
    form_slots: list[SlotName],
    questions: dict[SlotName, str],
    skip: set[SlotName],
) -> tuple[dict[SlotName, SlotValue], list[SlotName]]:
    """Walk through the form. Returns (completed slots, skipped slots).

    - stated slots are shown and can be kept or replaced
    - conflict slots show the versions to pick from
    - missing slots show the pre-written question
    skip: slots not to show, e.g. "inputs" when --input files were given.
    """
    ui.show("\nCheck and complete the prompt. Your own words are kept exactly as written.")
    slots: dict[SlotName, SlotValue] = {}
    skipped: list[SlotName] = []
    for name in slots_to_show(extraction, form_slots, skip):
        result = extraction.slot(name)
        question = questions.get(name, f"What should go in {label(name)}?")
        if name == "objective" and extraction.candidate_objectives:
            value = choose_objective(ui, result, extraction.candidate_objectives, question)
        elif result.status == "stated":
            value = review_stated(ui, name, result.value)
        elif result.status == "conflict":
            value = resolve_conflict(ui, name, result.options, question)
        else:
            value = ask_slot(ui, name, question)

        if is_empty(value):
            skipped.append(name)
        else:
            slots[name] = value
    return slots, skipped


# ---------------------------------------------------------------------------
# What the pipeline prints
# ---------------------------------------------------------------------------


def describe_decision(decision: Decision) -> str:
    """One status line, e.g. "Task: analysis | Route: fill_missing | Checkpoint: on | Effort: high"."""
    if decision.task_type is None:
        task = "unknown (Jev unavailable)"
    elif decision.task_type_uncertain:
        task = f"{decision.task_type} (Jev unsure, lighter of its top two)"
    else:
        task = decision.task_type
    checkpoint = "on" if decision.checkpoint else "off"
    return (
        f"Task: {task} | Route: {decision.route} | "
        f"Checkpoint: {checkpoint} | Effort: {decision.effort}"
    )


def describe_verdict(verdict: JevVerdict | None) -> str:
    """Every Jev answer, for --dry-run."""
    if verdict is None:
        return "Jev: no verdict (Jev could not be reached)"

    def ranked(probabilities: dict[str, float]) -> str:
        pairs = sorted(probabilities.items(), key=lambda pair: pair[1], reverse=True)
        return ", ".join(f"{name} {probability:.2f}" for name, probability in pairs)

    lines = [
        f"Jev ({verdict.model})",
        f"  task_type: {verdict.task_type.choice} "
        f"(confidence {verdict.task_type.confidence:.2f})",
        f"    {ranked(verdict.task_type.probabilities)}",
        f"  problem_type: {verdict.problem_type.choice} "
        f"(confidence {verdict.problem_type.confidence:.2f})",
        f"    {ranked(verdict.problem_type.probabilities)}",
    ]
    for slot, probability in verdict.presence.items():
        lines.append(f"  has_{slot}: {probability:.2f}")
    lines.append(f"  expects_large_output: {verdict.expects_large_output:.2f}")
    return "\n".join(lines)


def describe_dry_run(
    verdict: JevVerdict | None, decision: Decision, route: Route, rendered: RenderedPrompt
) -> str:
    """Everything --dry-run prints after the questions: verdicts, route, slots, checkpoint, prompt."""
    missing = ", ".join(decision.missing_slots) or "none"
    checkpoint = "on (planning call skipped in dry run)" if decision.checkpoint else "off"
    return "\n".join(
        [
            "",
            "===== DRY RUN (Claude is not called) =====",
            describe_verdict(verdict),
            "",
            f"Task type used: {decision.task_type or 'unknown'}"
            + (" (lighter of Jev's top two)" if decision.task_type_uncertain else ""),
            f"Route: {route}",
            f"Missing slots: {missing}",
            f"Checkpoint: {checkpoint}",
            f"Effort: {decision.effort}",
            "",
            "----- System prompt -----",
            rendered.system or "(none)",
            "----- User message -----",
            rendered.user,
            "-------------------------",
        ]
    )
