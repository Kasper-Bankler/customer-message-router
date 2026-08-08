# Decision log

Design choices that a reviewer could reasonably have made differently, with what
was given up. The point of this file is that "why did you do it that way?" has a
written answer before the question is asked.

Assumptions live in `ASSUMPTIONS.md`; this file records choices, not premises.

| Date | Decision | Rationale | Alternatives considered |
| --- | --- | --- | --- |
| 2026-08-08 | Orchestration is a plain Python state machine in `src/graph.py`, not LangGraph | Every line is explainable under questioning, and the agents stay pure functions, so swapping in LangGraph later is a wiring change rather than a rewrite | LangGraph — named in the job ad and maps 1:1 to the architecture diagram, but adds framework behaviour I would have to defend as my own |
| 2026-08-08 | `proposed_action`, `escalation` and `token_cost` are typed Pydantic submodels, not bare `dict` | The pitch is typed I/O contracts; untyped escape hatches in the most important model would undercut it on the schema slide | Bare `dict` as originally specced — less upfront design, but defers the field set to whoever writes the agent |
| 2026-08-08 | `ollama` is the only LLM SDK installed; the Azure OpenAI path is described in `llm.py` but not depended on | The prototype runs fully locally, so no customer data leaves the machine — a genuine asset in a banking interview | Installing `openai` too for a hosted demo fallback; rejected for now to keep the "runs offline" claim clean |
| 2026-08-08 | Numeric fields carry explicit bounds (`injection_score`, `confidence`, `sla_minutes`, `latency_ms`) | A malformed structured-output response fails validation at the boundary instead of silently propagating a nonsense score into the policy matrix | Unbounded floats as originally specced |
| 2026-08-08 | Added `src/cli.py` as the demo entry point | The demo shows `RoutingDecision` JSON, and the Streamlit UI is first on the cut list, so the primary entry point must not depend on it | A `__main__` block in `graph.py`; rejected to keep wiring separate from presentation |
| 2026-08-08 | `ProposedAction.arguments` stays `dict[str, Any]` | Per-tool argument types are declared in the tool registry as `args_schema`; restating them here would make `schemas.py` import every tool and invert the dependency | A generic `ProposedAction[ArgsT]` — rejected because it does not survive JSON serialisation into `RoutingDecision`, and it is hard to defend line by line |
| 2026-08-08 | `ProposedAction.risk` reuses `RiskTier` instead of a separate tool-risk enum | Same four levels with the same meaning; two parallel enums invite mapping bugs between them | A distinct `ToolRisk` enum — cleaner separation of "how risky is this message" from "how risky is this tool", at the cost of a second concept |
| 2026-08-08 | `Escalation.priority` is a three-value Literal, not an integer | Integer priority has no self-evident direction (is 1 highest or lowest?), and "elevated" matches the wording of assumption 5 exactly | `int` 1–5, as most ticketing systems use; rejected for direction ambiguity |
| 2026-08-08 | Enums for `Disposition` and `RiskTier`, Literals for `channel`, `priority`, `requires_auth_level` | Enums for values the policy matrix branches on and other modules import; Literals for closed sets used in one place only | Enums throughout — more uniform, more boilerplate for values nothing branches on |
| 2026-08-08 | `similarity` and `relevance` are left unbounded while other floats are bounded to [0,1] | Cosine similarity can be negative and RRF scores are not in [0,1]; bounding them would reject valid values at the boundary | Bounding every float for consistency — rejected as wrong for these two specifically |
| 2026-08-08 | `grounded` is `bool \| None` rather than defaulting to `False` | `None` means no reply was generated; `False` means one was generated and failed verification. Collapsing them corrupts the groundedness-pass-rate safety metric | `bool = False`, which reads simpler but silently merges two different states |
| 2026-08-08 | `default_factory=list` rather than `= []` | Pydantic v2 deep-copies either form (verified), so this is purely for the reader who expects Python's mutable-default trap | `= []` exactly as specced in the plan — functionally identical |
| 2026-08-08 | `CLAUDE.md` condenses the architecture to numbered stages instead of the ASCII diagram | The diagram is ~60 lines and `CLAUDE.md` is reloaded into context every session, so it pays a recurring cost for something already in `PLAN.md` | Pasting the §2 ASCII art verbatim — visually closer to the slide |
| 2026-08-08 | `CLAUDE.md` states two hard rules beyond the three requested: fail-closed, and nothing executes | Both are §9 requirements that only hold if written down before the code exists; `PLAN.md` is not in context while code is being written | Leaving them in `PLAN.md` §9 and trusting they get applied |
| 2026-08-08 | `ASSUMPTIONS.md` adds a scope-boundary section drawn from §9 | "This system stays outside EU AI Act high-risk and GDPR Art. 22" is itself an assumption, and it is the load-bearing reason loans and limit changes route to humans | Keeping only the ten items from §1 |
| 2026-08-08 | `README.md` opens with an explicit "Status: scaffolding" line | A quickstart that describes commands which do not exist yet is worse than one that admits it | Writing the quickstart in future tense, or omitting it until the code lands |
| 2026-08-08 | Added a `tests/` directory, which §4 does not contain | Follows from selecting pytest; the guardrail regexes and the policy matrix are the two things that must never silently break | No test directory, keeping strictly to §4 |
| 2026-08-08 | `eval/goldset.csv` is not created as a placeholder | It is a generated artifact of `build_goldset.py`, and an empty file would be mistaken for a real one | Committing an empty stub to match the §4 listing |

## Open questions

Recorded rather than guessed. Each needs a decision before the relevant module is
written.

1. **`eval/` and `app/` cannot import `src`.** Verified: running
   `python eval/run_eval.py` puts `eval/` on `sys.path`, not the repo root, so
   `import src.schemas` fails. The README quickstart is wrong as written. Fix is
   a `pyproject.toml` plus `pip install -e .`, or `python -m eval.run_eval` with
   an `eval/__init__.py`. Decide before writing `run_eval.py`.
2. **No `extra="forbid"` on the models.** Left off so a stray key from an LLM
   does not hard-fail, but structured LLM output is exactly where strictness
   pays. Revisit when `llm.py` and the retry-once-then-escalate path are written.
3. **Upper version bounds in `requirements.txt` are unverified.**
   `sentence-transformers<6`, `chromadb<2` and `datasets<5` were chosen for
   headroom, not checked against PyPI. Resolve on first real install.
4. **No language-detection dependency.** `ingress.py` needs one and §3 names
   none. `lingua` is the most accurate on short text; `py3langid` is far lighter.
5. **Nothing carries the CLARIFY question.** `Disposition.CLARIFY` exists but no
   field holds the question text. `reply_text` can serve, but that is an
   implicit convention that should be written down or given its own field.
6. **`Escalation` has no agent-facing draft field**, though §2 has the handoff
   agent producing "draft for agent". Scoped to the four fields named in §5.
7. **The sample `RoutingDecision` in `README.md` contains invented numbers.**
   `doc_028` is verified against the corpus; `confidence` and `latency_ms` are
   illustrative and must be replaced with real output before the presentation.

