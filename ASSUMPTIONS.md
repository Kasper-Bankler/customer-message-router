# Assumptions

## Data and scope

1. **Banking77 messages are inbound written contacts** (secure inbox or chat
   transcript), not voice — because voice would need ASR confidence handling and
   a different abstention calculus, which is out of scope for this case.

2. **Each message is a standalone contact** with no conversation history and no
   authenticated customer identity — because Banking77 supplies neither. Real
   systems have both; `InboundMessage.customer_ref` marks where identity would
   plug in, and it is deliberately unused.

3. **The 30-article FAQ corpus is the only approved grounding source.** Anything
   outside it is out-of-knowledge by definition — because the model's parametric
   knowledge of banking is untrusted, and a plausible answer sourced from
   pre-training is indistinguishable from a fabricated one at review time.

4. **Banking77 is English-only** Language is detected and non-English routed through the same
   pipeline — this is a real production gap, not a solved problem, because
   Danish retrieval quality and CPR-number formats are untested here.

   _Known limitation, stated deliberately:_ detection uses `langdetect`, which is
   unreliable on very short strings — and Banking77 messages are short. I chose
   to measure that failure rate in the adversarial suite rather than reach for a
   heavier detector, because at this scope an honest, quantified limitation is
   worth more than a marginally better number. In production this would be a
   dedicated service with a confidence threshold, and low-confidence detection
   would itself be a reason to route to a human.

## Business rules

5. **Any fraud or unauthorised-transaction signal goes to a human, always**,
   regardless of model confidence, with elevated priority and no auto-reply —
   because the cost of one wrong automated answer about a compromised card
   exceeds the total savings from automating the whole domain.

6. **No action agent may move money, change credentials, or alter credit
   terms.** Actions are limited to reversible, low-blast-radius operations, and
   every action is _proposed_, never executed — because a prototype has no
   business touching a real system, and the approval boundary is the interesting
   part of the design anyway.

7. **Automated replies must not constitute financial or investment advice**
   (MiFID II territory). Rate and product-suitability questions get informational
   grounding plus a human handoff, never a recommendation — because advice
   carries a suitability obligation this system cannot discharge.

8. **The operating point is high precision on auto-reply, deliberately low
   recall.** I would rather escalate 40% of answerable messages than send one
   wrong answer about a blocked card — because escalation costs an agent minute
   and a wrong answer costs trust and possibly money.

## Technical

9. **Latency budget: p95 under 3s for the routing decision, under 6s including a
   generated reply** — because anything slower degrades agent-desktop UX to the
   point where agents route around the tool.

10. **The LLM is a swappable dependency behind an interface** — local Ollama for
    development, Azure OpenAI for production. EU data residency is a hard
    requirement, not a preference, and under DORA the LLM provider is a critical
    ICT third party, so the abstraction layer is concentration-risk management
    rather than gold-plating.

## Reconciliation with the implementation

Measured on the 80-message gold set and the adversarial suite; see
`eval/results/`.

**4 — non-English handling is weaker than written.** The assumption says language
is detected and the failure rate measured in the adversarial suite. Detection is
implemented and `detected_language` is recorded on every trace, but **no code
reads it** — nothing routes on language. The two non-English adversarial cases do
pass, and they pass _by accident_: the Danish one via `low_confidence` and the
German one via `out_of_taxonomy`, neither of which is a language rule. The
promised measurement of langdetect's failure rate on short strings was never
built. Either the assumption should be narrowed to "language is recorded but not
acted on", or a language gate belongs in `ingress.py`. **Unresolved — your call.**

**8 — the operating point overshoots its own target.** The assumption says "I
would rather escalate 40% of answerable messages". Measured auto-reply recall is
**0.308**, so the system escalates roughly **69%** of the messages a human
labelled answerable. The direction of the trade is right and precision held at
1.000, but the stated 40% was a guess made before any measurement and the real
figure is nearly double it. The threshold sweep in
`eval/results/threshold_sweep.csv` shows what it would cost to move.

**9 — the latency budget is met for routing and missed for replies.** The
assumption sets p95 under 3 s for a routing decision and under 6 s including a
generated reply. Escalation and blocked lanes land at **2.0–2.5 s** (inside
budget); a grounded auto-reply measures **9.7–12.7 s** on an 8 GB M1 running
`qwen2.5:3b-instruct` locally, which is **twice the stated budget**. Two LLM
calls happen on that path (adjudication, then generation) and both are local CPU
inference. A hosted model would very likely meet it; the assumption should not be
relaxed to match a laptop.

**6 — one registered tool sits on the wrong side of the rule.** The assumption
forbids any action that alters credit terms, and limits actions to reversible,
low-blast-radius operations. `request_card_limit_change` is in the registry and
alters a credit term, and `block_card` is not reversible by the customer. Both
require human approval, no executor does anything but return a dry-run receipt,
and no intent currently maps to either — so nothing can propose them today. They
exist to demonstrate the approval gate. The tension is real and is noted in the
code, but the assumption as written does not permit them.

## Scope boundary, stated deliberately

This system makes **no creditworthiness decisions and no decisions with legal
effect**. That keeps it a limited-risk transparency case under the EU AI Act
rather than a high-risk Annex III system, and keeps it outside GDPR Art. 22's
prohibition on solely automated decisions with significant effect. The boundary
is a design constraint, not an accident — which is why loans and limit changes
route to humans.
