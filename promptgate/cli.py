"""The promptgate command.

    promptgate "your prompt"
    promptgate --file prompt.txt
    promptgate --input article.txt --input notes.md "Summarize these"
    promptgate --dry-run "your prompt"
    promptgate mark <run_id> --corrected
"""

import argparse
import os
import sys
from pathlib import Path

import anthropic
from typesafe_sdk import TypeSafeError

from promptgate.clients import Clients
from promptgate.config import Config, load_config
from promptgate.env import load_env_file
from promptgate.executor import ClaudeExecutor
from promptgate.extractor import HaikuExtractor
from promptgate.form import TerminalUI, UserInterface
from promptgate.jev_client import JevClient
from promptgate.log import RunNotFoundError, mark_corrected
from promptgate.pipeline import Pipeline
from promptgate.schemas import Document

API_KEY_VARIABLES = ("ANTHROPIC_API_KEY", "TYPESAFE_API_KEY")
PLACEHOLDER_MARKER = "replace-me"  # the fake values in .env.example contain this


def build_run_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptgate",
        description="Check a prompt, fill any gaps with you, then send it to Claude once.",
        epilog="To mark a run whose answer you had to correct: promptgate mark <run_id> --corrected",
    )
    parser.add_argument("prompt", nargs="?", help="the prompt text (or use --file)")
    parser.add_argument("--file", type=Path, help="read the prompt from this file")
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        default=[],
        help="attach a document word for word; repeat for several",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="show Jev's verdicts, the route, and the final prompt without calling Claude",
    )
    return parser


def build_mark_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="promptgate mark", description="Record that a run's answer needed correcting."
    )
    parser.add_argument("run_id", help="the run id printed after the answer")
    parser.add_argument(
        "--corrected", action="store_true", required=True, help="mark the run as corrected"
    )
    return parser


def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def make_real_clients(config: Config) -> Clients:
    """Build the real Jev, Haiku, and Claude clients. API keys come from the environment only."""
    missing = [
        name
        for name in API_KEY_VARIABLES
        if not os.environ.get(name) or PLACEHOLDER_MARKER in os.environ[name]
    ]
    if missing:
        raise SystemExit(
            f"Set {' and '.join(missing)} in your .env file or environment first (see README.md)."
        )

    try:
        jev = JevClient(config.models.jev)
    except TypeSafeError as error:  # the SDK checks the key when the client is created
        raise SystemExit(f"TYPESAFE_API_KEY was rejected: {error}") from error

    anthropic_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    return Clients(
        jev=jev,
        extractor=HaikuExtractor(
            anthropic_client, config.models.extraction, config.limits.extract_max_tokens
        ),
        claude=ClaudeExecutor(
            anthropic_client,
            config.models.execution,
            plan_max_tokens=config.limits.plan_max_tokens,
            execute_max_tokens=config.limits.execute_max_tokens,
        ),
    )


def run_mark(argv: list[str], config: Config) -> int:
    args = build_mark_parser().parse_args(argv)
    try:
        mark_corrected(args.run_id, config.paths.log_file)
    except RunNotFoundError as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    print(f"Marked {args.run_id} as corrected.")
    return 0


def run_prompt(
    argv: list[str], config: Config, clients: Clients | None, ui: UserInterface | None
) -> int:
    parser = build_run_parser()
    args = parser.parse_args(argv)

    if (args.prompt is None) == (args.file is None):
        parser.error("give the prompt either as text or with --file (exactly one)")
    try:
        prompt = read_text(args.file) if args.file else args.prompt
        documents = [Document(source=path.name, content=read_text(path)) for path in args.input]
    except OSError as error:
        parser.error(f"can't read {error.filename}: {error.strerror}")
    if not prompt.strip():
        parser.error("the prompt is empty")

    if ui is None:
        ui = TerminalUI()
    if clients is None:
        clients = make_real_clients(config)

    try:
        records = Pipeline(config, clients, ui).run(prompt, documents, dry_run=args.dry_run)
    except KeyboardInterrupt:
        print("\nCancelled.", file=sys.stderr)
        return 130  # the usual exit code for Ctrl+C
    return 1 if any(record.error for record in records) else 0


def main(
    argv: list[str] | None = None,
    config: Config | None = None,
    clients: Clients | None = None,
    ui: UserInterface | None = None,
) -> int:
    """Run the command. Tests call this directly with fakes, so it never reads .env itself."""
    if argv is None:
        argv = sys.argv[1:]
    if config is None:
        config = load_config()

    if argv and argv[0] == "mark":
        return run_mark(argv[1:], config)
    return run_prompt(argv, config, clients, ui)


def entry_point() -> None:
    """What `promptgate` and `python -m promptgate` run: load .env, then main()."""
    load_env_file()
    sys.exit(main())


if __name__ == "__main__":
    entry_point()
