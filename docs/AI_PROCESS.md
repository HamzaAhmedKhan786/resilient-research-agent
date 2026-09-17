# AI-assisted development process

This repository was built interactively with Codex from the challenge specification. This document summarizes the collaboration; the original conversation is not included in this repository.

## What AI proposed

- The narrow evidence-backed research domain and initially four-action loop, later extended with one explicit `skip` transition after live traces exposed irrelevant-source trapping.
- Dataclass-based durable state, atomic checkpoint replacement, and JSONL traces.
- Deterministic scripted planners and failure injection for reproducible evals.
- The first CLI, Wikipedia adapter, local corpus, tests, documentation, and localhost UI implementation.

## Decisions I directed or changed

- Keep the system framework-free and standard-library-only.
- Add a minimal browser UI while retaining the CLI.
- Let users supply API keys through a masked UI field.
- Use a session-only key lifecycle instead of encrypted persistent storage.
- Separate evaluation-scenario counts from unit-test behavior counts.
- Expand the repository into a publication checklist with explicit security and development-process artifacts.

## Where AI-generated work failed

The first deterministic example assumed search ranking would assign the retry document to `S2`. Trace inspection showed that `S2` was a different document, so a note failed exact-excerpt validation. The initial final-answer validator also accepted an answer when any citation matched saved evidence, allowing a second unsupported citation. The validator was corrected to require every cited source ID to be admitted evidence, and the example now discovers the two sources explicitly.

The initial status summary said “4/4 evaluation scenarios” and then listed additional unit-test behaviors without distinguishing them. The current documentation reports seven controlled evaluation scenarios and 55 unit tests separately.

The first live run to reach `complete` also revealed that structural citation validation was not claim-level grounding. The answer cited admitted sources but included uncited background and unsupported uncertainty. The original run is preserved with an adverse manual review; the harness now rejects uncited factual sentences when the goal explicitly requires every substantive claim to be cited.

A later run retrieved the right two sources but failed because the model emitted `Claim. [S2]`, leaving the citation outside the sentence recognized by the validator. The repair is deliberately narrow: only IDs already present in the evidence ledger are moved before punctuation. The same finish path now rejects an answer that omits an explicit input comparison term and logs goal coverage, citation coverage, and evidence-word overlap. The overlap score is not presented as proof of entailment.

## How suggestions were validated

Later live traces revealed two provider-specific defects that controlled tests had missed: OpenAI received a double-encoded request body, and Groq frequently failed forced function calls, first for long final answers and then for short note actions. The fixes were driven by persisted traces. The final Groq design removes provider tools entirely: the harness owns phase/source constraints and plain completions supply the research content. Regression tests cover request shape, deterministic reads, source-bound notes, terminal errors, synthesis, and evidence coverage. A first lexical coverage proposal was rejected as too broad and narrowed to explicit comparison concepts plus a small fallback.

- Unit tests exercise unread evidence rejection/recovery and budget termination.
- Deterministic evals inject search and read timeouts and an invalid citation.
- Every relevant change is followed by compilation, tests, evals, or an HTTP smoke test.
- The UI security smoke test submits a sentinel key and verifies it is absent from the status response.
- A separately proposed expanded implementation was executed against the existing regression suite before reuse. It failed the offline demo, so only its retrieval validation and observability counters were selectively incorporated.
- Raw example traces were inspected manually, which caught the ranking/citation defect.

## What I designed and prioritized

The central product judgment is to demonstrate resilience through observable state transitions instead of tool breadth. Evidence must be admitted through objective checks; transient and semantic failures use different recovery paths; run state remains inspectable; and deterministic tests provide a stable baseline.

## Overengineering explicitly rejected

- Agent frameworks and multi-agent orchestration.
- Embeddings or a vector database for a three-document eval corpus.
- React or another frontend build system for a local form-and-trace UI.
- Distributed workers and a database for a single-process demonstration.
- Persistently encrypted API keys, which would introduce a key-management problem when session-only memory is sufficient.

## Original session export

The original AI conversation is not published in this repository. The take-home asks for unedited AI conversations, so this deliverable is currently missing.
