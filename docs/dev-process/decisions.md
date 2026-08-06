# Decision log

## One action per decision

The agent chooses one of `search`, `read`, `note`, `skip`, or `finish`. This costs more decisions than executing a large generated plan but makes changed behavior after errors visible. `skip` exists only to persist and trace that a read source had no admissible evidence.

## Evidence admission instead of unrestricted notes

A note must quote an excerpt present in a source already read by the agent. This is deliberately simpler than semantic entailment and blocks fabricated quotations.

## Two failure paths

Timeout and connection failures receive bounded low-level retries. Semantic failures—invalid IDs, unread evidence, bad excerpts, or invalid citations—become state errors so the next model decision can change course.

Permanent provider errors now terminate immediately, while rate limits and server faults receive bounded retries inside one logical planning step. Live traces showed that `openai/gpt-oss-20b` intermittently rejected forced function calls even with one valid tool. The Groq adapter therefore uses phase-specific plain text: the harness selects read/source transitions, while the model supplies search queries, evidence excerpts, and final synthesis.

The first plain-text search budget was only 80 tokens. Provider logs showed the reasoning model consuming that entire allowance without producing visible content. Groq requests now use low reasoning effort, exclude returned reasoning, place instructions in the user message, and use phase-specific output limits.

Wikipedia search also exposed an adaptation boundary: a verbose model query returned zero results, and deterministic sampling repeated it. The harness now normalizes search text and uses a small, distinct fallback ladder derived from the goal and uncovered terms. This is retrieval guardrail logic, not a general planner framework.

A later Tacoma run had healthy provider responses but spent its budget reading weak initial results. The harness now performs a focused search when admitted evidence leaves an explicit goal term uncovered and no unread result's title or snippet matches it. Read context is an exact bounded window around the densest uncovered-term passage, so relevant later sections are not hidden by introduction-only truncation.

Another healthy-provider run showed that topical terms alone were insufficient: resonance from Broughton Bridge and aeroelasticity from an aircraft wing falsely appeared to cover the Tacoma goal. Evidence admission now requires a named-subject anchor, source ranking weights subject terms, and final synthesis receives no unsaved source metadata. Groq calls are also paced at a minimum 750 ms interval as a conservative guard against bursts, although that run itself had no 429 response.

## Lightweight evidence coverage

The finish validator checks explicit minimum-source requests against both admitted evidence and distinct citations, plus a bounded lexical set of important goal terms. This is intentionally an observable heuristic rather than an LLM judge; it caught the Tacoma run's missing resonance evidence, but synonyms can still produce false gaps.

The same bounded goal-term check now applies to the final answer, not only the evidence ledger. This establishes a deterministic input/output relevance link for explicit comparison terms. The trace also reports cited-sentence coverage and lexical overlap with cited excerpts. Overlap is diagnostic rather than terminal because it cannot establish entailment and may underrate a faithful paraphrase.

Wikipedia extracts are retained locally up to 100,000 characters and only then compacted to a 3,600-character relevance-centered model view. This prevents relevant passages late in long articles from disappearing while keeping model context bounded. Once evidence exists, a follow-up search that omits every uncovered concept is replaced with a focused subject-plus-concept query; this prevents broad fallbacks such as `resonance` from consuming the remaining step budget on unrelated medical or nuclear pages.

For goals that explicitly request a citation for every substantive claim, the validator also requires an inline admitted-source citation in every factual-looking sentence. This improves citation coverage but still does not prove entailment; the live example therefore includes a separate manual quality review.

One Groq run produced correct source numbers as Unicode superscripts (`¹`, `³`) instead of the required `[S1]`, `[S3]` form on its final budgeted step. The harness now repairs only superscript numbers that map unambiguously to admitted evidence, traces that repair, and leaves unknown numbers invalid. Once evidence exists, unread-source ranking also excludes results that do not mention the currently uncovered comparison concept. A source whose title names a different subject must name the requested subject near the start of the proposed excerpt, preventing evidence about modifications to another bridge from becoming a claim about Tacoma.

## Durable JSON and JSONL

Atomic JSON checkpoints and append-only JSONL traces were chosen over a database. They are inspectable and sufficient for one local process, but not safe coordination for distributed workers.

Trace events now include UTC timestamps and run IDs, and trace/checkpoint error strings pass through credential-pattern redaction. Server console logs intentionally contain only run lifecycle metadata rather than goals, evidence, or credentials.

Provider retries, tool retries, and validation failures are stored as small aggregate counters and shown in the CLI and UI. These do not replace the trace; they make run health immediately visible without adding a metrics service.

## Deterministic evaluation first

Scripted decisions and a local corpus isolate harness behavior from model and network variance. The limitation is explicit: these tests do not measure general research quality.

## Session-only API keys

The initial environment-only design was expanded to accept a user key in the UI. A proposal to “encrypt and store” it was rejected: a local app would also need to protect the decryption key. The implemented design transmits the key to localhost for one run, retains it only in process memory, and excludes it from artifacts and responses.

The company later supplied access to `gpt-5.6-sol` through a private OpenAI-compatible Azure endpoint. It is represented as a separate `libra` provider so its credential can never be routed to the public OpenAI or Groq endpoints. The endpoint and model are configuration; the company-issued key remains session-only.

## Simplicity over expansion

Browser automation, embeddings, a task database, streaming tokens, multi-agent delegation, and a frontend framework were rejected as unjustified for this narrow submission.
