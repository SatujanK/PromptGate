# PromptGate

PromptGate checks a prompt before it reaches Claude. It spots what's missing or unclear, asks you only for those parts, builds a well-structured prompt, and then sends it to Claude once.

The aim is fewer misread prompts and fewer retries, at a lower total cost than rewriting a prompt by hand and running it again.

## How it works

1. **Jev** (a fast classifier from TypeSafe) reads your prompt and answers a set of questions in one call: what kind of task it is, which details it already has, and whether it has a problem such as two tasks in one or conflicting instructions.
2. **Code** (no AI) picks a route:

   | Route | When | What you see |
   |---|---|---|
   | as_is | Nothing important is missing | Nothing. The prompt goes straight to Claude. |
   | fill_missing | A required detail is missing | One short question per missing detail. |
   | full_form | The prompt is unclear or contradicts itself | A form showing what your prompt says, so you can fix it. |
   | split | The prompt asks for several separate tasks | A choice to split it into separate runs. |

3. For bigger tasks, Claude first proposes a **plan**. You approve it, edit it, or cancel.
4. Claude runs the prompt **once**, and the answer streams into your terminal.
5. Every run is saved as one line in `logs/runs.jsonl`.

The design, and the reasons behind it, are in [CLAUDE.md](CLAUDE.md).

## Setup

You need **Python 3.11 or newer** and API keys for Anthropic and TypeSafe. These commands are for PowerShell on Windows, run in the project folder.

### 1. Create a virtual environment and install

A virtual environment keeps this project's packages separate from the rest of your computer.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
```

This installs the four packages the project uses: `anthropic`, `typesafe-sdk`, `pydantic`, and `pytest` (tests only).

> Use `python -m pip` rather than `pip`. On some school or work computers, `pip.exe` is blocked by a security policy, but `python -m pip` still works.

### 2. Add your API keys

PromptGate reads two keys:

| Variable | Where to get it |
|---|---|
| `ANTHROPIC_API_KEY` | [console.anthropic.com](https://console.anthropic.com) → API Keys |
| `TYPESAFE_API_KEY` | [console.typesafe.ai](https://console.typesafe.ai) |

Open the `.env` file in the project folder and paste each key after its `=`:

```
ANTHROPIC_API_KEY=sk-ant-...
TYPESAFE_API_KEY=...
```

If there is no `.env` file (for example, after a fresh copy of the project), create it from the example first:

```powershell
Copy-Item .env.example .env
```

About the two files:

- **`.env`** holds your real keys. It is listed in `.gitignore`, so git never commits it. Don't share it or paste its contents anywhere.
- **`.env.example`** shows the format with fake values. It is safe to commit. PromptGate recognises its fake values and treats them as "not set".

You can also set the keys as Windows environment variables instead. A key set in Windows takes priority over the `.env` file.

```powershell
[Environment]::SetEnvironmentVariable("ANTHROPIC_API_KEY", "your-key", "User")
```

Restart VS Code (or your terminal) afterwards so it picks up the change.

### 3. Check it works

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m promptgate --help
```

The tests never call a real API and pass without any keys, so they cost nothing to run.

## Usage

The examples use `python -m promptgate`, which works everywhere. If your computer allows it, the shorter `promptgate` command does the same thing (run it from the activated virtual environment: `.\.venv\Scripts\Activate.ps1`).

```powershell
# A prompt as text
python -m promptgate "What does df.shape return?"

# A prompt from a file
python -m promptgate --file prompt.txt

# Attach documents. Each is passed to Claude word for word; repeat --input for more.
python -m promptgate --input article.txt --input notes.md "Summarize these"

# See what would happen, without calling Claude
python -m promptgate --dry-run "Build me a study planner app"
```

### Answering questions

- **Short answers:** type one line and press Enter. Press Enter on an empty line to skip a question.
- **Pasted material and examples:** these can span several lines. Finish with a line containing only `.`, and separate examples with a line containing only `---`.
- **Choices:** type the number of the option you want.
- **In the form:** press Enter to keep what your prompt already says, or type a replacement.
- **At the plan step:** choose 1 to approve, 2 to edit (type your own plan), or 3 to cancel.

### Dry run

`--dry-run` shows:

- Jev's answers
- the route
- the missing details
- whether a plan step would happen
- the exact final prompt

It still asks its questions, so it can show you the final prompt. It calls Jev, and Haiku if the route needs it, but **never Claude's planning or answer step**. Dry runs are logged with `"dry_run": true`.

### Marking answers you had to correct

After each answer, PromptGate prints a run ID. If you had to fix or redo that answer, mark it:

```powershell
python -m promptgate mark 3f9a1c07b2e4 --corrected
```

This is how the project is measured: fewer corrected answers per 100 prompts, at equal or lower cost.

## Settings

- **`config.toml`**: the models, the thresholds Jev's answers are compared against, the token limits, and the six task-type profiles (which details each task type needs, whether it gets a plan step, and the effort level).
- **`questions.toml`**: the question asked for each missing detail.

Edit either file and the change applies on the next run. If a setting is invalid, for example a misspelled name, PromptGate stops with an error that names it.

## Logs

Each run adds one line of JSON to `logs/runs.jsonl`. A line records:

- the prompt
- Jev's answers
- the route taken
- which questions you skipped
- the tokens used by each call
- whether you marked the answer as corrected

Prompt-cache reads and writes are recorded too. `logs/` is in `.gitignore`, because the lines contain your full prompts.

## Troubleshooting

| Message or problem | What to do |
|---|---|
| `Set ANTHROPIC_API_KEY ... in your .env file or environment first` | Paste the key into `.env`, or replace the fake `replace-me` value that's still there. |
| `TYPESAFE_API_KEY was rejected` | Check the key at console.typesafe.ai and paste it again. |
| `pip.exe` or `promptgate.exe` is blocked | Use `python -m pip ...` and `python -m promptgate ...` instead. |
| `Jev could not be reached, so the prompt will be sent as-is.` | The run continues without checks. The error is saved in the log's `jev_error`. |

## Project layout

```
promptgate/        the package (each file explains itself at the top)
config.toml        models, thresholds, task-type profiles
questions.toml     one question per missing detail
.env.example       the format of .env, with fake keys (committed)
.env               your real keys (never committed)
samples/           the five sample prompts, an article, and a long project prompt
tests/             pytest tests; fake clients replace every API
logs/runs.jsonl    one line per run (never committed)
```
