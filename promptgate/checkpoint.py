"""The plan checkpoint: one planning call, then approve, edit, or cancel.

There is exactly one planning call. Editing the plan never calls Claude
again; your edited text simply becomes the approved plan.
"""

from pydantic import BaseModel

from promptgate.clients import ClaudeAPI
from promptgate.form import UserInterface
from promptgate.render import RenderedPrompt
from promptgate.schemas import Effort, TokenUsage

TRUNCATION_WARNING = (
    "Warning: the plan hit the token limit and may be cut off. Check it before approving."
)
APPROVE, EDIT, CANCEL = 0, 1, 2  # positions of the three options in the menu


class CheckpointOutcome(BaseModel):
    approved_plan: str | None  # None means you cancelled
    edited: bool
    truncated: bool
    usage: TokenUsage  # tokens used by the planning call


def run_checkpoint(
    claude: ClaudeAPI, ui: UserInterface, rendered: RenderedPrompt, effort: Effort
) -> CheckpointOutcome:
    """Make the planning call, show the plan, and loop until you approve or cancel."""
    draft, usage = claude.plan(rendered, effort)
    plan = draft.text.strip()
    edited = False

    while True:
        ui.show("\n===== Proposed plan =====")
        ui.show(plan or "(Claude returned no plan text.)")
        if draft.truncated and not edited:
            ui.show(TRUNCATION_WARNING)

        choice = ui.choose("\nWhat next?", ["Approve and run", "Edit the plan", "Cancel"])
        if choice == APPROVE:
            return CheckpointOutcome(
                approved_plan=plan, edited=edited, truncated=draft.truncated, usage=usage
            )
        if choice == CANCEL:
            return CheckpointOutcome(
                approved_plan=None, edited=edited, truncated=draft.truncated, usage=usage
            )

        # EDIT: your text replaces the plan. A blank answer keeps the current plan.
        new_plan = ui.ask_multiline("Type the plan you want Claude to follow.").strip()
        if new_plan:
            plan = new_plan
            edited = True
