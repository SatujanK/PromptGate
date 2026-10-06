# Sample prompts

The five sample prompts used to check PromptGate's routing, with what each should do. Run any of them with `--dry-run` to see the route without calling Claude.

| # | Command | Expected task type | Expected route | Checkpoint |
|---|---|---|---|---|
| 1 | `promptgate --dry-run "What does df.shape return?"` | quick_question | as_is | off |
| 2 | `promptgate --dry-run "Translate this into French:"` | transformation | fill_missing (asks for inputs) | off |
| 3 | `promptgate --dry-run "Summarize this article and also write me a cover letter"` | any | split | depends on each part |
| 4 | `promptgate --dry-run --input samples/article.txt "Summarize every section in full detail, in under 50 words"` | analysis or transformation | full_form (conflict shown) | per profile |
| 5 | `promptgate --dry-run "Build me a study planner app"` | multi_step_project | fill_missing | on |

`study_planner_project.txt` is a detailed project prompt of more than 1,024 tokens. It is long enough for Claude Sonnet 5 to cache the shared prompt between the planning and execution calls:

```powershell
promptgate --file samples/study_planner_project.txt
```
