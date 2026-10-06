# PromptGate

A personal command-line tool that checks a prompt before it reaches Claude, fills any gaps with the user, builds a structured prompt from a fixed template, and then calls Claude to execute it.

**Goal:** fewer misinterpreted prompts and retries, at a lower total token cost than manually rewriting a prompt in one turn and executing it in another.

---

## Core principles (do not break these without asking)

1. **No large model touches the prompt before execution.** Only Jev (classification) and Haiku (extraction) run before Claude.
2. **Models classify and extract; code decides and renders.** Profile lookup, missing-slot calculation, routing, and final prompt rendering are all deterministic Python.
3. **The extraction model copies and never invents.** It does not paraphrase, fill a missing slot, or resolve a conflict.
4. **Uncertainty adds less friction, not more.** Low confidence means send as-is, or use the lighter profile.
5. **Claude executes once.** The only extra call is one short planning call for task types that require a plan checkpoint.
6. **Every run is logged.**

---

## Pipeline

1. **Jev (one call)** returns `task_type`, the `has_*` checks, `problem_type`, and `expects_large_output`. All questions go in a single request because Jev evaluates them in parallel and independently. Jev sees only the prompt text, never the `--input` files; when `--input` is given, code sets `has_inputs` to 1.
2. **Code** looks up the task-type profile, computes `missing = required slots where has_X < present_threshold`, and decides whether a plan checkpoint is on.
3. **Route**, first match wins:

   | Condition | Route | What happens |
   |---|---|---|
   | `problem_type` is `multiple_tasks` (confident) | split | Ask the user: split or send as-is. On split, Haiku divides the prompt and each part re-enters the pipeline together with the shared inputs and context from the original prompt. A part is never split again. |
   | `problem_type` is `unclear_intent` or `contradictory` (confident) | full_form | Haiku extracts all slots with statuses, the user completes a form, and code renders the full template. |
   | Required slots missing | fill_missing | Ask the pre-written question for each missing required slot. Code attaches the answers to the original prompt as tagged blocks. |
   | Otherwise | as_is | Send the original prompt. |

   If Jev fails (an error, or a prompt too long for it), the route is as_is with the ambiguity rule, no checkpoint, and `[fallback] effort` from `config.toml`. The error is logged.

4. **Fixed lines.** Code appends the ambiguity rule to every task type except `quick_question`.
5. **Plan checkpoint**, if the profile turns it on:
   - A planning call with the checkpoint instruction and a capped `max_tokens` (default 3000, because thinking tokens count toward the cap). If the response stops at `max_tokens`, warn the user that the plan may be cut off.
   - The user approves, edits the plan, or cancels.
   - The execution call sends the same prompt plus `<approved_plan>`. Mark the shared prompt prefix for prompt caching so the repeated input is cheaper. The planning and execution calls use the same effort and thinking settings, because changing either one invalidates the cache. Claude Sonnet 5 only caches prefixes of 1024 tokens or more; shorter prompts are expected to miss.
6. **Execute** with a single Claude call in a fresh context.
7. **Log** one JSON line.

---

## Slots

The structure is based on Anthropic's prompting best practices (platform.claude.com/docs/en/build-with-claude/prompt-engineering/claude-prompting-best-practices). The "Source" column shows which slots come from that guidance and which are this project's additions.

| Slot | Where it goes | What goes in it | Source |
|---|---|---|---|
| `role` | System prompt | One sentence setting Claude's role, e.g. "You are a Python tutor for a first-year engineering student." Optional. | Anthropic: give Claude a role |
| `inputs` | Top of user message | Material Claude works **on**, passed through word for word (from `--input` files, a form answer, or extraction). Each item goes in `<document index="n">` with `<source>` and `<document_content>`, inside `<documents>`. | Anthropic: long data at the top; document tags |
| `context` | User message | Background: who, why, what situation, including **why** the task matters | Anthropic: add context and motivation |
| `objective` | User message | The one thing to accomplish, starting with an action verb ("Change…", "Write…", "Explain…") | Anthropic: be clear and direct; explicit action verbs |
| `requirements` | User message | What the output must include or do, most important first. Written as numbered steps only when order matters; otherwise say *what*, not *how*. | Anthropic: numbered steps when order matters; general instructions over prescriptive steps |
| `constraints` | User message | Limits, each with its reason, phrased as what to do rather than what not to do where possible | Anthropic: explain motivation; say what to do instead of what not to do |
| `resources` | User message | Tools, libraries, or sources Claude works **with** or must stay within | Project addition |
| `output_format` | User message | Structure, length, file type. Includes scope: "minimal" or "go beyond the basics." | Anthropic: be specific about output format; request "above and beyond" explicitly |
| `success_criteria` | User message | How the user will judge whether the result is right | Anthropic: define success criteria first |
| `examples` | User message | 3–5 relevant, varied examples, each in `<example>` tags inside `<examples>`. Optional. | Anthropic: use examples effectively |

**Render order:**
- **System prompt:** `role`.
- **User message:** `<documents>` (the inputs), `context`, `objective`, `requirements`, `constraints`, `resources`, `output_format`, `success_criteria`, `examples`, then the fixed lines.
- **as_is and fill_missing:** `<documents>` (from `--input` or an inputs answer), then the original prompt word for word, then any answers as tagged blocks in the order above, then the fixed lines.
- **Fixed-line order:** quote grounding, self-check, then the ambiguity rule last, because it refers to everything "above".

Each slot is wrapped in an XML tag of the same name, and empty slots are left out. Tag names never change, because Anthropic recommends consistent tag names.

**Pre-written questions** (`questions.toml`) should nudge toward Anthropic's phrasing. For example:
- Objective: "What should Claude do? Start with a verb."
- Constraints: "Any limits? Say what you want rather than what to avoid, and why."

The extraction model still only copies. The better phrasing comes from you, not from a rewrite.

---

## Task-type profiles

Profiles are stored in `config.toml`, not hard-coded.

| Task type | Weight | Required | Optional | Plan checkpoint | Effort |
|---|---|---|---|---|---|
| `quick_question` | 1 | objective | context | off | low |
| `transformation` | 2 | objective, inputs | output_format, constraints | off | low |
| `explanation` | 3 | objective, context | output_format, constraints, examples | off | medium |
| `generation` | 4 | objective, context, requirements, output_format | constraints, resources, success_criteria, examples | on only if `expects_large_output` | medium |
| `analysis` | 5 | objective, context, inputs, requirements, output_format, success_criteria | constraints, resources, examples | on | high |
| `multi_step_project` | 6 | objective, context, requirements, constraints, resources, output_format, success_criteria | inputs, examples | on | high |

- **Low-confidence `task_type`:** use the lower-weight profile of Jev's top two choices. `unclear` is never used as a profile: skip it and take the next choice. A confident `unclear` counts as low confidence.
- **Form display:** the full form shows only required and optional slots for the task type.
- **Effort:** sent as `output_config.effort` on the planning and execution calls. Both calls in one run use the profile's effort, so the cache still works.
- **Role:** never required. `config.toml` holds an optional `default_role` per task type (e.g. `explanation` → "You are a patient Python tutor for a first-year engineering student."), sent as the system prompt. A role the user states in the prompt overrides it: code skips `default_role` when `has_role >= present_threshold`.

---

## Jev questions (one request)

- `task_type` (Choice): the six types above, plus `unclear`.
- `has_objective`, `has_context`, `has_inputs`, `has_requirements`, `has_constraints`, `has_resources`, `has_output_format`, `has_success_criteria` (Noul each). There is no check for `examples`, since it is never required.
- `has_role` (Noul): does the prompt state a role for Claude? Used only to decide whether to send `default_role`.
- `problem_type` (Choice): `none`, `unclear_intent`, `contradictory`, `multiple_tasks`, `unclear`. A confident `unclear` counts as `none`.
- `expects_large_output` (Noul): will the answer be long or multi-part?

**Starting thresholds** (in `config.toml`, to be tuned from logs):

| Setting | Value |
|---|---|
| `present_threshold` | 0.5 |
| `problem_confidence_min` | 0.6 (below this, `problem_type` counts as `none`) |
| `task_type_confidence_min` | 0.5 (below this, use the lighter of the top two) |
| `large_output_threshold` | 0.6 |

**Jev limitations to design around:**
- **It cannot abstain.** Every Choice must include `unclear`.
- **It cannot do arithmetic.** Token counting happens in code.
- **Its accuracy drops as the state fills with irrelevant material.** Send the prompt only, never conversation history.
- **Noul answers have no `confidence` field.** Do not read `.confidence` on binaries.
- **Pin the Jev model version**, because the thresholds depend on it.
- **Confirm the SDK's current class and method names** at docs.typesafe.ai before coding against them.

---

## Extraction (Haiku)

Structured output, validated with pydantic:

- `SlotResult`:
  - `status`: `stated`, `missing`, or `conflict`
  - `value`: text or list
  - `options`: list of the conflicting versions, only for `conflict`
- `Extraction`: one `SlotResult` per slot, plus `candidate_objectives` (2–3 options, used only when the objective is vague).

The instruction must contain these rules:
1. Copy the user's words into slots. Do not paraphrase.
2. If two instructions conflict, list both and choose neither.
3. If something is not stated, mark it missing and never fill it.

**Form behaviour:**
- `stated` slots are pre-filled and editable.
- `missing` slots show the pre-written question.
- `conflict` slots show the options to pick from, or the user types their own.
- `candidate_objectives` are shown with a "none of these" option.
- A slot outside the profile is also shown if the prompt already states it, so nothing the user wrote is silently dropped.
- Code enforces principle 3: conflict options and candidate objectives are kept only if they appear word for word in the prompt (ignoring case and spacing), and a candidate that repeats the stated objective is dropped. Rule 2 of the instruction also covers conflicts between instructions that would go in different slots.
- Known limitation (seen in M6): Haiku may also leave one side of a cross-slot conflict inside the objective. The form shows the objective first so the user can edit it, and the ambiguity rule makes Claude ask.
- If extraction fails, the route falls back to fill_missing (or as_is if nothing is missing). If the split fails, the prompt is sent as-is.

---

## Fixed lines (exact text)

- **Ambiguity rule:** `If anything above is unclear or conflicting, ask me before starting. For minor details, state your assumption and continue.`
- **Checkpoint instruction** (planning call only): `Before doing the task, reply only with: (1) a numbered plan of at most 8 steps, (2) the assumptions you are making, (3) any questions. Do not start the task.`
- **Execution addition** after approval: `<approved_plan>…</approved_plan>` followed by `Follow the approved plan.`
- **Quote grounding** (added when `inputs` is about 20k tokens or more; count with the Anthropic token-counting endpoint): `First quote the parts of the documents relevant to the task in <quotes> tags, then do the task.` Source: Anthropic's long-context guidance.
- **Self-check** (added when `success_criteria` is present and the execution model's `self_check` setting is true): `Before you finish, check your answer against the success criteria.` Anthropic recommends this in general but advises removing it for Claude Opus 5, where it causes over-verification. The per-model setting lives in `config.toml`, and each model's prompting page should be checked before turning it on.

Fixed lines use normal, calm wording. No capital-letter emphasis or "CRITICAL/MUST", because Anthropic notes that current models overreact to aggressive language.

---

## Project layout

```
promptgate/
  cli.py          # argparse entry point: prompt text or --file, repeatable --input, --dry-run
  __main__.py     # lets `python -m promptgate` run the command
  env.py          # loads API keys from .env without overriding the real environment
  config.py       # loads config.toml and questions.toml (tomllib)
  schemas.py      # pydantic models: JevVerdict, SlotResult, Extraction, LogRecord
  pipeline.py     # runs one prompt through every step; keeps cli.py thin
  clients.py      # the small interfaces (Protocols) every network client and fake follows
  form.py         # terminal questions, the full form, and the split choice
  jev_client.py   # thin wrapper around the TypeSafe SDK, behind a small interface
  extractor.py    # Haiku extraction and task splitting
  routing.py      # PURE: profile lookup, missing slots, route choice, checkpoint flag
  render.py       # PURE: template rendering, fixed lines, answer attachment
  checkpoint.py   # planning call and approve/edit/cancel loop
  executor.py     # Claude execution call (prompt caching on the shared prefix)
  log.py          # appends one JSON line per run to logs/runs.jsonl
config.toml       # models, thresholds, profiles
questions.toml    # one pre-written question per slot
.env.example      # the .env format with fake keys (committed)
.env              # real API keys (gitignored)
README.md         # setup, API keys, and usage
tests/            # pytest; fakes (tests/fakes.py) replace every network client
```

`logs/` and `.env` are in `.gitignore`: log lines contain full prompts, and `.env` holds real keys.

---

## Models (in `config.toml`)

| Role | Model |
|---|---|
| Extraction | `claude-haiku-4-5-20251001` |
| Execution | `claude-sonnet-5` (configurable), with a per-model `self_check` flag (see Fixed lines) |
| Jev | pinned version, e.g. `jev-1.13.0`; confirm the current one |

API keys: `ANTHROPIC_API_KEY` and `TYPESAFE_API_KEY`, from the environment or from a `.env` file in the project folder (read by `env.py`, no extra dependency). A variable already set in the environment wins over `.env`. `.env` is gitignored; `.env.example` holds fake values containing `replace-me` and is committed. Only the command's `entry_point()` reads `.env`, never `main()`, so tests never see real keys.

---

## Log record (one line per run)

```json
{"run_id": "...", "timestamp": "...", "task_type": "...", "jev": {...all probabilities...},
 "route": "as_is|fill_missing|full_form|split", "missing_slots": [...], "checkpoint": true,
 "plan_edited": false, "models": {...}, "tokens": {"extract": {...}, "plan": {...}, "execute": {...}},
 "corrected": null}
```

Additional fields: `parent_run_id` (set on each part of a split), `dry_run`, `prompt`, `input_sources`, `effort`, `split_declined`, `skipped_slots` (for tuning profiles), `plan_truncated`, `cancelled`, `jev_error`, `warnings` (problems worked around, e.g. a failed extraction), `error` (the problem that stopped the run), `tokens.jev`, and cache read/write counts inside each token entry. A split writes one line for the split decision and one line per part.

`corrected` is set later with `promptgate mark <run_id> --corrected`. It is the main success metric: fewer corrected answers per 100 prompts at equal or lower cost.

---

## Conventions

- Python 3.11+, type hints everywhere, pydantic v2, `tomllib` for config.
- Keep dependencies minimal: `anthropic`, `typesafe-sdk`, `pydantic`; `pytest` for development only.
- Routing and rendering are pure functions with no I/O. Every network call sits behind a small client class, so tests can use fakes.
- **Tests never call real APIs** and must pass with no API keys set.
- `--dry-run` prints the Jev verdicts, route, missing slots, checkpoint flag, and final rendered prompt, and does not call Claude for execution or planning.
- Prefer readable code over clever code. The owner is learning Python.

## Commands

```bash
python -m pip install -e ".[dev]"
pytest
promptgate "your prompt"
promptgate --file prompt.txt
promptgate --input article.txt --input notes.md "Summarize these"
promptgate --dry-run "your prompt"
promptgate mark <run_id> --corrected
```

On this machine `pip.exe` is blocked by an Application Control policy; use `python -m pip`. `promptgate.exe` runs, and `python -m promptgate` works as a fallback.

---

## Explaining code to the owner

When explaining a function or module, use this order:
1. **Inputs:** accepted types and shapes.
2. **Processing:** what it does.
3. **Output:** the type and shape it returns.
4. **Examples:** 2–3 concrete input → output pairs.
5. **Common mistakes:** incompatible or problem inputs.
6. **When to use it:** why it exists in this project.

Keep it concise. Do not walk through the code line by line.

---

## Open decisions

- Tune the thresholds from logs.
- Tune the profiles: slots whose questions the user keeps skipping should become optional for that task type.
- Command-line form vs. a simple web form (command line for v1).
- Optional Langfuse tracing (out of scope for v1).
