# Assumptions

The case gave no business rules, so these are mine. Every one has a *because*,
and every one is falsifiable — if a Danske Bank stakeholder disagrees with an
assumption here, the system changes in a known place rather than everywhere.

Last reviewed: 2026-08-08.

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

4. **Banking77 is English-only, but Danske Bank is not.** The bank operates in
   DA/SV/NO/FI/EN. Language is detected and non-English routed through the same
   pipeline — this is a real production gap, not a solved problem, because
   Danish retrieval quality and CPR-number formats are untested here.

   *Known limitation, stated deliberately:* detection uses `langdetect`, which is
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
   every action is *proposed*, never executed — because a prototype has no
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

## Scope boundary, stated deliberately

This system makes **no creditworthiness decisions and no decisions with legal
effect**. That keeps it a limited-risk transparency case under the EU AI Act
rather than a high-risk Annex III system, and keeps it outside GDPR Art. 22's
prohibition on solely automated decisions with significant effect. The boundary
is a design constraint, not an accident — which is why loans and limit changes
route to humans.
