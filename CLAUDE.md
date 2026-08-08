# CLAUDE.md

## Project

A multi-agent system that routes inbound Banking77 customer messages to one of
four dispositions: an auto-reply grounded in a 30-article FAQ corpus, a proposed
action, a human handoff, or a single clarifying question. The thesis is _"a
router that knows when it doesn't know, and proves it"_ — abstention is the
designed behaviour, not a failure mode, because the FAQ corpus covers only a
fraction of the 77 intents and the remainder are structurally unanswerable.

**Done means:** a clean clone runs the CLI demo in under five minutes; five demo
messages each exercise a different lane (forced fraud escalation, grounded
auto-reply, honest abstention, action proposal, injection block); every decision
emits a complete `RoutingDecision` reconstructable from the trace log alone; and
`eval/` reports metrics with bootstrap confidence intervals rather than a bare
accuracy number.

## Architecture

Six stages, wired as an explicit Python state machine in `src/graph.py`:

1. **Ingress / sanitiser** (`guardrails/ingress.py`) — language detect, PII
   detect and redact (CPR, PAN via Luhn, IBAN), prompt-injection scan, abuse and
   vulnerability flags. Emits `SanitisedMessage`.
2. **Intent resolver** (`agents/intent_resolver.py`) — Stage A: embedding kNN
   over Banking77 train gives top-5 candidates plus a calibrated similarity.
   Stage B: an LLM adjudicator picks among those five or returns
   `none_of_these`. Small decision space, so it is accurate and auditable.
   Emits `IntentResult`.
3. **Orchestrator** (`agents/orchestrator.py`) — intent to domain, then a
   **policy matrix in code** to a disposition, with confidence thresholds and
   deterministic overrides.
4. **One of three agents** — RAG (hybrid retrieve, ground, cite, abstain),
   action (schema-validated, dry-run only, needs approval), or handoff
   (summarise, queue, priority, SLA).
5. **Verifier / egress guard** (`guardrails/egress.py`) — groundedness, citation
   validity, no invented contact details, no-advice check, AI disclosure
   appended. Returns PASS, REWRITE or ESCALATE.
6. **Observability** (`observability/trace.py`) — cross-cutting `trace_id`,
   spans, cost, latency, decisions.

The load-bearing idea: **the LLM decides what the customer wants; deterministic
code decides what we are allowed to do about it.**

## Conventions

- Python 3.11+.
- Pydantic v2 for all inter-agent I/O. An agent boundary is a schema, never a
  dict or a free-text blob.
- Type hints everywhere, including return types.
- No function over ~40 lines. No file over ~250 lines. If a file is growing past
  that, the seam it needs is usually obvious.
- Every tunable number (thresholds, budgets, model names) lives in
  `config/settings.yaml`, never inline.

## Hard rules

- **Never log raw customer text or PII values.** Trace records carry PII _types_
  (`["CPR", "PAN"]`) and hashes, never values. `InboundMessage.text` must not
  reach a log sink or an LLM prompt; only `SanitisedMessage.text_redacted`
  travels downstream.
- **Every LLM call goes through the `LLMClient` interface** in `src/llm.py`.
  Never import or call a provider SDK directly from an agent. This is
  concentration-risk management, not tidiness.
- **Routing policy is deterministic code, never prompt text.** No prompt may be
  the thing that stops a fraud message being auto-replied to. An LLM can never
  downgrade a disposition, because it is never asked to.
- **Fail closed.** Any component exception routes to HUMAN. Never drop a
  message, never guess.
- **Nothing executes.** Actions are proposed with a simulated receipt. No tool
  moves money, changes credentials, or alters credit terms.

## A note on style

This is an interview case, and the panel will point at a line and ask why it is
there. Readability beats cleverness every time. Prefer the obvious
implementation over the compact one, name things in full, and if a design choice
is non-obvious, record it in `DECISIONS.md` rather than in a comment. Code that
cannot be explained out loud in two sentences is a liability here.

## Complexity budget (hard constraint)

This is interview work built by a 3rd-semester BSc student who must
explain every line under questioning. Simple and explainable beats
clever and impressive.

- Prefer plain Python functions over framework abstractions. Do NOT use
  LangGraph — wire the pipeline as explicit function calls in graph.py.
- No decorators, metaclasses, async, or dependency injection unless I
  ask for it. Standard library over new dependencies.
- No abstraction introduced for a single use case. Two implementations
  before an interface, except LLMClient (which has a stated reason).
- If a solution needs more than ~40 lines, stop and offer me a simpler
  version alongside it, and say what the simpler one gives up.
- Comment WHY, not WHAT. Every non-obvious threshold or constant needs a
  one-line comment explaining how it was chosen.
- At the end of each session, explain what you wrote in plain language
  as if to someone who has not seen the code. If that explanation is
  hard to give, the code is too complex — simplify it.

## Session close protocol

At the end of every session:

1. Walk me through what you wrote, file by file, in plain language.
2. Log only genuine judgment calls in DECISIONS.md — not routine choices.
3. List anything you're unsure about, capped at 3 items. If there are
   more, you built too much in one session.
4. Do not commit. I review and commit myself.
