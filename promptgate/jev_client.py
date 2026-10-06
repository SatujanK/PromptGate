"""The single Jev call, through the TypeSafe SDK.

Every question goes in one request, because Jev evaluates them in parallel
and independently. Jev sees only the prompt text: never the --input files
and never any conversation history, since unrelated material lowers its
accuracy.

Jev 1.13 reads questions literally, so each one states its exact condition,
and each Choice label has a description. Every Choice also offers "unclear",
because Jev cannot abstain.
"""

from pydantic import ValidationError
from typesafe_sdk import Choice, Noul, TypeSafeClient, TypeSafeError

from promptgate.clients import ClientError, token_usage
from promptgate.schemas import JevVerdict, TokenUsage

TASK_TYPE_CRITERIA = {
    "quick_question": "A short question that needs a brief, direct answer.",
    "transformation": (
        "Turn material the user gives into another form: translate, reformat, "
        "summarize, shorten, or convert it."
    ),
    "explanation": "Explain a concept, a piece of code, or an idea so the user understands it.",
    "generation": "Write something new, such as code, an essay, a letter, or a list.",
    "analysis": (
        "Examine material the user gives in depth and report findings, "
        "judged against some standard."
    ),
    "multi_step_project": (
        "Build something substantial that needs several stages of work, such as an app or a tool."
    ),
    "unclear": "None of the above clearly fits.",
}

PROBLEM_TYPE_CRITERIA = {
    "none": (
        "The prompt asks for one task, and its instructions agree with each other. "
        "Missing details or missing material do not count as a problem."
    ),
    "unclear_intent": (
        "It is not clear what the user wants Claude to do. A clear request that is only "
        "missing the material to work on, or some details, does not count."
    ),
    "contradictory": "Two instructions in the prompt conflict, so both cannot be followed.",
    "multiple_tasks": "The prompt asks for two or more separate tasks.",
    "unclear": "None of the above clearly fits.",
}

# One Noul per slot Jev checks, keyed by slot name.
PRESENCE_QUESTIONS = {
    "role": (
        "Does the prompt tell Claude what role or persona to take, "
        'for example "You are a ..." or "Act as ..."?'
    ),
    "objective": "Does the prompt say what the user wants Claude to do?",
    "context": (
        "Does the prompt give background, such as who the result is for, "
        "the situation, or why the task matters?"
    ),
    "inputs": (
        "Does the prompt itself contain the material Claude should work on, "
        "such as pasted text, code, or data?"
    ),
    "requirements": "Does the prompt state what the result must include or do?",
    "constraints": (
        "Does the prompt state limits the result must stay within, "
        "such as length, style, or things to avoid?"
    ),
    "resources": "Does the prompt name tools, libraries, or sources Claude should use or stay within?",
    "output_format": (
        "Does the prompt say what form the result should take, "
        "such as its structure, length, or file type?"
    ),
    "success_criteria": "Does the prompt say how the user will judge whether the result is right?",
}

LARGE_OUTPUT_QUESTION = "Will a complete answer to this prompt be long or have several parts?"


def build_questions() -> dict[str, Choice | Noul]:
    """Every Jev question, keyed by the name used to read its answer."""
    questions: dict[str, Choice | Noul] = {
        "task_type": Choice(
            instructions="What kind of task does this prompt ask Claude to do?",
            criteria=TASK_TYPE_CRITERIA,
        ),
        "problem_type": Choice(
            instructions="Does this prompt have a problem that would stop Claude doing it well?",
            criteria=PROBLEM_TYPE_CRITERIA,
        ),
        "expects_large_output": Noul(instructions=LARGE_OUTPUT_QUESTION),
    }
    for slot, instructions in PRESENCE_QUESTIONS.items():
        questions[f"has_{slot}"] = Noul(instructions=instructions)
    return questions


def verdict_from_response(response) -> JevVerdict:
    """Copy the answers out of the SDK response into a JevVerdict.

    Choice answers have choice, confidence, and probabilities. Noul answers
    are just a probability (.noul); they have no confidence, so none is read.
    """
    choices = response.choices
    nouls = response.nouls
    return JevVerdict(
        model=response.model,
        task_type={
            "choice": choices["task_type"].choice,
            "confidence": choices["task_type"].confidence,
            "probabilities": dict(choices["task_type"].probabilities),
        },
        problem_type={
            "choice": choices["problem_type"].choice,
            "confidence": choices["problem_type"].confidence,
            "probabilities": dict(choices["problem_type"].probabilities),
        },
        presence={slot: nouls[f"has_{slot}"].noul for slot in PRESENCE_QUESTIONS},
        expects_large_output=nouls["expects_large_output"].noul,
    )


class JevClient:
    """Asks Jev every question about a prompt in one call."""

    def __init__(self, model: str, client: TypeSafeClient | None = None):
        self.model = model  # pinned, e.g. "jev-1.13.0", because the thresholds depend on it
        # TypeSafeClient reads TYPESAFE_API_KEY from the environment.
        self.client = client if client is not None else TypeSafeClient(model=model)

    def classify(self, prompt: str) -> tuple[JevVerdict, TokenUsage]:
        try:
            response = self.client.system_one(prompt, build_questions(), model=self.model)
            verdict = verdict_from_response(response)
        except TypeSafeError as error:
            raise ClientError(f"Jev call failed: {error}") from error
        except (KeyError, ValidationError) as error:
            raise ClientError(f"Jev returned an unexpected answer: {error}") from error
        return verdict, token_usage(response.usage)
