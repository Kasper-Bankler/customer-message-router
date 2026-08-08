# Decision log

Judgment calls only: choices a reviewer could reasonably have made differently,
with what was given up. Routine choices are not logged — if the alternative was
obviously worse, it does not belong here.

The point of this file is that "why did you do it that way?" has a written answer
before the question is asked.

Assumptions live in `ASSUMPTIONS.md`; this file records choices, not premises.

| Date | Decision | Rationale | Alternatives considered |
| --- | --- | --- | --- |
| 2026-08-08 | Orchestration is plain Python function calls in `src/pipeline.py`, not LangGraph | Every line is explainable under questioning, and the agents stay pure functions, so swapping in LangGraph later is a wiring change rather than a rewrite | LangGraph — named in the job ad and maps 1:1 to the architecture diagram, but adds framework behaviour I would have to defend as my own |
| 2026-08-08 | `proposed_action`, `escalation` and `token_cost` are typed Pydantic submodels, not bare `dict` | The pitch is typed I/O contracts; untyped escape hatches in the most important model would undercut it on the schema slide | Bare `dict` as originally specced — less upfront design, but defers the field set to whoever writes the agent |
| 2026-08-08 | `ollama` is the only LLM SDK installed; the Azure OpenAI path is described in `llm.py` but not depended on | The prototype runs fully locally, so no customer data leaves the machine — a genuine asset in a banking interview | Installing `openai` too for a hosted demo fallback; rejected for now to keep the "runs offline" claim clean |
| 2026-08-08 | Numeric fields carry explicit bounds (`injection_score`, `confidence`, `sla_minutes`, `latency_ms`) | A malformed structured-output response fails validation at the boundary instead of silently propagating a nonsense score into the policy matrix | Unbounded floats as originally specced |
| 2026-08-08 | `similarity` and `relevance` are left unbounded while the other floats are bounded to [0,1] | Cosine similarity can be negative and RRF scores are not in [0,1]; bounding them would reject valid values | Bounding every float for consistency — rejected as wrong for these two specifically |
| 2026-08-08 | Models do **not** set `extra="forbid"` | Required fields are still enforced, so a malformed response is still caught; silently dropping a hallucinated extra key is the preferred failure mode over raising on one | `extra="forbid"`, which is stricter on LLM structured output but turns a harmless extra key into an exception on the customer path |
| 2026-08-08 | `ProposedAction.arguments` stays `dict[str, Any]` | Per-tool argument types are declared in the tool registry as `args_schema`; restating them here would make `schemas.py` import every tool and invert the dependency | A generic `ProposedAction[ArgsT]` — rejected because it does not survive JSON serialisation into `RoutingDecision`, and is hard to defend line by line |
| 2026-08-08 | `ProposedAction.risk` reuses `RiskTier` instead of a separate tool-risk enum | Same four levels with the same meaning; two parallel enums invite mapping bugs between them | A distinct `ToolRisk` enum — cleaner separation of message risk from tool risk, at the cost of a second concept to explain |
| 2026-08-08 | `Escalation.priority` is a three-value Literal, not an integer | Integer priority has no self-evident direction (is 1 highest or lowest?), and "elevated" matches the wording of assumption 5 exactly | `int` 1–5, as most ticketing systems use; rejected for direction ambiguity |
| 2026-08-08 | Enums for `Disposition` and `RiskTier`, Literals for `channel`, `priority`, `requires_auth_level` | Enums for values the policy matrix branches on and other modules import; Literals for closed sets used in one place only | Enums throughout — more uniform, more boilerplate for values nothing branches on |
| 2026-08-08 | `grounded` is `bool \| None` rather than defaulting to `False` | `None` means no reply was generated; `False` means one was generated and failed verification. Collapsing them corrupts the groundedness-pass-rate safety metric | `bool = False`, which reads simpler but silently merges two different states |
| 2026-08-08 | `CLARIFY` gets its own `clarifying_question` field on `RoutingDecision` | An explicit field beats an implicit convention that `reply_text` sometimes holds a question; the two have different guardrail obligations, since a question is not grounded output | Reusing `reply_text` with `grounded=None` — fewer fields, but the convention lives only in someone's head |
| 2026-08-08 | `Escalation.suggested_reply` added now rather than in a later session | PLAN §2 already has the handoff agent producing a draft for the human agent, so the field is known; adding it now avoids a schema change once agents are written against the contract | Holding strictly to the eight models in §5 and amending later |
| 2026-08-08 | Added `src/cli.py` as the demo entry point | The demo shows `RoutingDecision` JSON, and the Streamlit UI is first on the cut list, so the primary entry point must not depend on it | A `__main__` block in `pipeline.py`; rejected to keep wiring separate from presentation |
| 2026-08-08 | `pip install -e .` via a minimal `pyproject.toml` is the documented install | Makes `src` importable from `eval/`, `tests/` and `app/` in one move; running `python eval/run_eval.py` otherwise fails outright, which I confirmed rather than assumed | A `sys.path` shim at the top of each script — works, but is the kind of line that invites "why is this here?" in every file it appears in |
| 2026-08-08 | `requirements.txt` carries lower bounds only for now, with exact pins to follow from `pip freeze` | The upper bounds I had written were guesses I had not checked against PyPI, and a wrong guess blocks a clean-clone install — the exact pins land once the stack is known to work | Keeping guessed major-version caps; rejected because an unverified cap is worse than none |
| 2026-08-08 | Language detection uses `langdetect`, with its weakness on short strings documented rather than engineered around | Banking77 messages are short, so this will misfire; measuring that in the adversarial suite is a better presentation than a heavier dependency and an unmeasured claim | `lingua` (more accurate on short text, much larger) or a hosted detector; both add weight for a gap that is more honest to quantify |
| 2026-08-08 | Package structure flattened: no single-file packages. A directory exists only where there is more than one file and a real reason to group them | `guardrails/`, `retrieval/`, `tools/` and `observability/` each held one or two files, so the directory added a path segment and an `__init__.py` without adding meaning. `agents/` keeps four genuinely parallel files and survives | Keeping PLAN §4 verbatim — tidier on a slide, but four packages that exist only to hold one module each is exactly the ceremony the complexity budget rules out |
| 2026-08-08 | The action agent lives in `tools.py` with the registry, not as a separate `agents/action_agent.py` | The agent is a thin layer over the registry — it picks a tool and fills its argument schema. Splitting them puts two halves of one idea in two files that always change together | A standalone action agent module, parallel to the RAG and handoff agents; rejected because the symmetry is cosmetic |
| 2026-08-08 | `graph.py` renamed to `pipeline.py` | "Graph" is LangGraph vocabulary for something that is a straight line of function calls; the name would invite a question the code cannot support | Keeping `graph.py` to match the architecture diagram's shape |
| 2026-08-08 | `notebooks/` and `slides/` are not tracked | Neither is code, and both would be committed as large binary or output-laden files that no reviewer reads from the repo | Tracking them as §4 specifies; rejected because the exploration notebook's only durable output is the coverage chart, which belongs in `eval/results/` |

## Divergence from PLAN.md

`PLAN.md` is the design document written before any code existed. Where the
as-built repo now differs, the repo is correct and this table is the record:

- **§3, orchestration.** The plan names LangGraph. The pipeline is plain Python
  function calls, per the complexity budget in `CLAUDE.md`. §3's stack table is
  left unedited as the original reasoning; §4's tree has been updated.
- **§4, repository structure.** Flattened as above. §4 has been updated to match
  the as-built layout; `notebooks/` and `slides/` are removed from it, and
  `pyproject.toml`, `DECISIONS.md` and `tests/` are added.

## Open questions

1. **The FAQ coverage map has no home.** PLAN §6 and §15 put the "31 of 77
   intents have grounding" analysis in `notebooks/`, which is no longer tracked.
   The chart is on the critical path — §15 calls it the thing that anchors the
   whole narrative and never cuts it — so it needs a tracked script, most
   naturally `eval/coverage_map.py` writing to `eval/results/`. Decide in
   Session 2, before the taxonomy work starts.
