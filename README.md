# Resilient Research Agent

Resilient Research Agent is a narrow, evidence-backed research assistant for showing how a small agent makes decisions and recovers across multiple steps. A user supplies a goal through a CLI or dependency-free localhost UI; the agent searches Wikipedia, reads sources, admits verified excerpts, and produces a cited answer. The scope is intentionally limited so the decision loop, state, retries, checkpoints, and evaluation remain understandable without an agent framework.

## Quick start

Requirements: Python 3.11+ and, for live mode, an OpenAI or Groq API key.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e .

# Offline deterministic demonstration
research-agent --demo --run-dir .runs/example

# Minimal local UI
research-agent-ui

# Tests and evaluations
python -m unittest discover -s tests -v
python evals/run_evals.py
```

Open the UI at `http://127.0.0.1:8765`. Offline demo mode is selected by default. Clear it, choose OpenAI or Groq, and enter that provider's API key. Groq defaults to `openai/gpt-oss-20b`; the harness selects the valid phase while Groq supplies plain-text search queries, evidence excerpts, and final synthesis.

For CLI live mode:

```powershell
$env:OPENAI_API_KEY="your-key"
research-agent "Compare two explanations for why the Tacoma Narrows Bridge failed" --model gpt-5-mini

# Alternative live provider
$env:GROQ_API_KEY="your-key"
research-agent "Compare two explanations for why the Tacoma Narrows Bridge failed" --provider groq
```

## Architecture

```text
User goal
   |
   v
Context builder ---- durable state + recent errors + bounded extracts
   |
   v
Provider decision -- one validated action
   |
   v
Action validator
   |
   v
Tool executor ------ search / read / note / skip / finish
   |
   v
Evidence and state update
   |
   v
Checkpoint + trace -----> next decision or terminal result
```

The core is `src/research_agent/agent.py`. `model.py` contains the thin Responses API adapter, `tools.py` contains Wikipedia/local-corpus tools and failure injection, and `web.py` provides the localhost interface. No agent framework is used.

GitHub renders the component and decision-sequence diagrams in [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md). That document also shows persisted, transient, and model-visible data boundaries.

## Agent loop

```text
state = new_state(goal) or load_checkpoint()

while state is running and steps remain:
    context = compact(state, current_read_extracts, last_3_errors)
    action = model.choose_one_action(context)
    validate(action, state)

    if action calls a tool:
        retry only transient transport failures, within a bound
    execute(action)
    verify evidence or citations before admitting the result
    append_trace_event()
    atomically_save_checkpoint()

if no steps remain:
    terminate as budget_exhausted
```

The action space is deliberately small: `search`, `read`, `note`, `skip`, and `finish`. Search results get stable IDs such as `S1`. A source must be read before it supports a note, and the note excerpt must appear verbatim in the retrieved document. A read source with no relevant excerpt is explicitly abandoned and traced so it cannot trap the loop. Every source ID cited by a final answer must exist in the evidence ledger.

## State and context handling

Persisted in `checkpoint.json`:

- goal, run ID, step count, and terminal status;
- discovered source metadata and stable IDs;
- IDs of sources read;
- admitted evidence excerpts;
- provider retry, tool retry, and validation-failure counters;
- semantic errors and final answer.

Sent to the model:

- goal and remaining budget;
- discovered source metadata;
- one exact relevance-centered window of up to 3,600 characters from read documents that do not yet have admitted evidence;
- admitted evidence;
- a small lexical coverage summary for important goal terms;
- only the three most recent errors.

Deliberately omitted:

- API keys and authorization headers;
- raw chain-of-thought;
- complete trace history;
- unbounded documents and tool payloads;
- previous model responses beyond their structured effects on state.

Long read results are compacted to bounded, relevance-centered extracts in model context rather than blindly taking the article introduction. Durable state stores evidence and metadata rather than entire pages. On resume, the agent re-fetches previously read sources, traces each restoration, and then continues from the checkpoint.

## Failure handling

| Failure | Behavior |
|---|---|
| Tool timeout/connection failure | Retry with bounded exponential backoff; trace every attempt. |
| Invalid model action or arguments | Reject before execution and add a semantic error to state. |
| Unknown/unread source | Reject the evidence action; allow a later changed decision. |
| Fabricated evidence excerpt | Match only harmless case/whitespace/typography variants, recover the exact source span, and reject semantic changes. |
| Missing or invalid final citation | Reject the finish action; require another decision. |
| Missing requested source count, distinct citations, or goal-term coverage | Reject finish and direct research toward uncovered requirements. |
| Step budget exhausted | Stop deterministically with `budget_exhausted`. |
| Process restart | Load durable state with `--resume` and re-fetch previously read source content. |
| Provider 429/5xx | Retry inside the current planning cycle using provider delay guidance; rate limits have a separate bounded cooldown budget and do not spend a logical research step. |
| Bursty Groq calls | Keep at least 750 ms between request starts; this pacing is not a logical research step. |
| Permanent provider 4xx | Fail immediately with a terminal provider event instead of repeating the same invalid request. |
| Groq final synthesis | Generate ordinary answer text after evidence collection instead of embedding a long answer in tool-call JSON. |
| Groq operational generation | Use phase-specific plain text with no function tools; deterministic read/source targeting avoids provider tool-call failures. |
| Reasoning-only Groq response | Use low reasoning effort, exclude returned reasoning, and reserve separate output budgets for query, excerpt, and final text. |
| Empty or repeated search | Normalize verbose queries and choose a distinct title-like fallback locally after zero results. |

`trace.jsonl` records decisions, tool outcomes, retries, validation errors, and the termination reason. Checkpoints are written through a temporary file followed by atomic replacement, with a short bounded retry for transient Windows file-lock contention. The CLI and UI surface aggregate retry and validation counters while the trace retains event-level detail.

## Security

The UI binds to `127.0.0.1` by default. The selected OpenAI or Groq key is transmitted from the browser to the local HTTP server for the selected run, held only in process memory, and excluded from persistent state, logs, status responses, and browser storage. The password field is masked by default, offers an explicit Show/Hide control, and is cleared after submission.

Keys are not encrypted and stored: doing that securely would require a separate protected decryption key. Avoiding persistence is simpler and safer for this local demonstration. `.env`, run directories, caches, logs, and virtual environments are ignored by Git. Live run directories can contain user goals and retrieved content and must not be committed.

This is a local development UI, not a hardened multi-user service. It has no authentication, TLS, isolation, or rate limiting and should not be exposed on a public interface.

## Evaluation

Exact command:

```powershell
python evals/run_evals.py
```

Checked-in result: **6/6 controlled evaluation scenarios passing**.

| Scenario | Fault or behavior | Assertions | Result |
|---|---|---|---|
| Happy path | Search, read, evidence, cited finish | completion, evidence, checkpoint, terminal trace | Pass, 4 steps |
| Transient search failure | One injected timeout | all base assertions plus visible retry | Pass, 4 steps |
| Transient read failure | Two injected timeouts | all base assertions plus visible retries | Pass, 4 steps |
| Invalid citation recovery | First finish has no citation, next changes | rejection followed by successful completion | Pass, 5 steps |
| Adaptive long horizon | State-driven decisions, injected search/read faults, invalid finish | retries, error-conditioned correction, two-source completion | Pass, 8 steps |
| Irrelevant source recovery | First read has no relevant evidence | source abandonment, changed source, cited completion | Pass, 7 steps |

Detailed output is in `evals/results.json`. Separately, **41/41 unit tests pass** across agent, model, tool, and web-security coverage. They include OpenAI request serialization and pacing, a full multi-step Groq plain-text run, zero-result and uncovered-term query fallback, source abandonment, deterministic subject-ranked reads, relevance-centered context extraction, typography-tolerant exact evidence recovery, title-aware subject grounding, distinct-citation and claim-level citation validation, interruption/resume, circuit breaking, permanent quota errors, persisted retry accounting, corpus boundaries, Wikipedia response validation, locator construction, and API-key non-persistence. These unit tests are not part of the six-scenario evaluation count.

The controlled harness remains the stable offline baseline. A separate live model-in-the-loop evaluation uses a fixed public research goal and a key that is never written to disk:

```powershell
python evals/run_live_eval.py
```

It checks multi-step completion, admission of at least two sources, citation membership, sentence-level citation presence, checkpoint creation, and trace creation. After each attempt it writes a sanitized `evals/live-results.json`; the raw run remains under ignored `.runs/`. Live results are reported separately because network, account quota, and model behavior are non-deterministic.

The latest recorded Groq run completed in six steps with two sources and no operational failures. Manual review nevertheless marked it as not fully passing: several factual sentences lacked their own citation, one statement strengthened the source wording, and the unsupported uncertainty sentence had no evidence. The sanitized original and review are in `examples/live-run.md` and `examples/live-trace.jsonl`. That observed failure motivated sentence-level citation validation and a stricter synthesis prompt. A post-fix live rerun is still needed; the controlled suite verifies the new guard deterministically.

The deterministic planner and three-document local corpus isolate harness mechanics from sampling and network variation. The suite checks control flow, evidence invariants, recovery, checkpoint creation, and observability. It does not establish live research accuracy, citation entailment, cross-model performance, latency, or cost.

### Lessons and changes after evaluation

- Transport and semantic failures need different recovery paths: local retry for the former, planner-visible feedback for the latter.
- Stable source IDs and strict evidence admission added more audit value than expanding the tool set.
- Trace inspection caught a real example bug: source ranking made `S2` differ from the assumed document.
- That discovery exposed overly permissive citation validation; validation was changed from “at least one citation is saved” to “every cited source must be saved evidence.”
- Evaluation counts are now reported separately from unit-test counts.
- Saved traces exposed an OpenAI double-serialization defect and Groq final-tool JSON failures. Regression tests now cover both paths; raw read context is dropped after evidence admission, and Groq synthesizes final text without a tool wrapper.
- Further live runs showed that forced Groq function calls also failed for short note actions. The Groq path now uses no provider tools: the harness owns phase/source constraints and the model supplies only research content.
- Groq dashboard traces then showed three successful HTTP responses consuming the exact 80-token cap without visible content. GPT-OSS reasoning is now set to low and excluded from the response, with larger phase-specific completion budgets.
- A later live trace returned zero Wikipedia results for an over-specified query and then repeated it. Queries are now capped at six meaningful terms; after an empty search, the harness selects a distinct title-like fallback without another model call.
- Another live trace showed a read source with no single relevant excerpt trapping the note phase. The state now persists abandoned source IDs, traces `source_abandoned`, and continues with a different source.
- A later code comparison contributed only bounded improvements: Wikipedia snippets and empty-response checks, persisted operational counters, and UI/CLI metric visibility. Its larger planner rewrite was rejected because it removed source abandonment and failed the deterministic demo.
- The same comparison exposed a final-answer gap: two saved sources could satisfy the source minimum even if the answer cited only one. Finish validation now requires the requested number of distinct cited evidence IDs.
- A successful-provider Tacoma run still exhausted its budget because it read every initial result while the needed resonance discussion lay beyond the article-introduction context. Read context now centers on uncovered terms, and the phase policy searches for an uncovered requirement before consuming steps on unrelated results.
- The next live run had no provider errors but admitted evidence about Broughton Bridge and aircraft wings, then synthesized an invalid `[S9]` citation. Evidence must now mention the goal subject, final synthesis receives only admitted evidence and allowed citation IDs, and Groq request starts are paced by 750 ms as a conservative burst guard.
- The following run reached the correct 1940 article but Groq changed capitalization, whitespace, and non-breaking dash/space characters while copying evidence. Verification now matches only those harmless typography variants and stores the corresponding exact source substring; paraphrases remain invalid.
- That exact article then exposed a context issue: a valid excerpt used “the bridge” rather than repeating the title. Subject grounding now accepts contextual language when the source title itself identifies the named subject, while generic or analogous source titles still require the excerpt to name it.

## Example run

```powershell
research-agent --demo --run-dir .runs/example
```

Result:

> Checkpointing limits repeated work after interruption [S1], while bounded retries absorb transient faults without allowing endless loops [S2]. Together they address different failure modes.

The seven-step sanitized transcript is in `examples/offline-demo.md`, with its machine-readable trace in `examples/offline-trace.jsonl`. It is a real execution of the harness with deterministic decisions, not a live-model quality claim.

The repository also preserves a real six-step Groq run in `examples/live-run.md` and `examples/live-trace.jsonl`. Its manual review deliberately records a claim-grounding failure even though the harness originally returned `complete`; this is the clearest example of how evaluation changed the implementation. Raw `.runs/` artifacts remain local because they may contain user content.

## Trade-offs

- Wikipedia instead of arbitrary web browsing keeps retrieval narrow and predictable.
- CLI plus dependency-free web UI avoids a production frontend and build chain.
- JSON checkpoints and JSONL traces are inspectable but not a concurrent database.
- A synchronous core loop is easier to reason about than distributed task workers.
- Deterministic corpus evals are reproducible but do not replace live model evaluations.
- Verbatim evidence verification is cheap and objective but does not prove claim entailment.
- One action per turn improves observability at the cost of additional model calls.

## Time spent

The work extended beyond the suggested 4–6 hours across several debugging and documentation sessions. No reliable timer was running, so inventing an exact total would be misleading. The best retrospective allocation is:

| Work | Approximate share |
|---|---:|
| Planning and scope | 10% |
| Core loop, state, checkpoints, and traces | 30% |
| Tools and evidence grounding | 15% |
| Evaluation and testing | 25% |
| Web UI and documentation | 20% |

The extra time went primarily into reproducing real OpenAI/Groq failures, inspecting persisted traces, and adding regression tests. The priority was harness behavior and failure visibility. Retrieval breadth, UI sophistication, and production infrastructure were deliberately traded away.

## Limitations and future work

- Live research is limited to Wikipedia.
- There is no source-authority ranking beyond prompt guidance.
- Search results and evidence have no semantic deduplication.
- The evaluation set is small and mostly deterministic.
- There is no live cross-model quality or cost comparison yet.
- One process owns in-memory UI run state; server restart loses that status index.
- There is no run cancellation mechanism.
- Resume does not persist full retrieved pages.
- Claim-level citation entailment is not evaluated.
- Output quality depends on the selected model.

Next priorities are 20–30 recorded live tasks scored for correctness and citation coverage, cost/latency reporting, cancellation, a bounded source cache to avoid resume re-fetches, and domain-specific primary-source retrieval.

## Repository structure

```text
README.md                overview and run instructions
LICENSE                  MIT license
.env.example             placeholder only; never a real key
pyproject.toml           package metadata and console commands
src/research_agent/      loop, prompts, state, tools, CLI, and UI
tests/                   focused invariant tests
evals/                   corpus, fault-injection harness, measured results
docs/                    architecture, design, video outline, and development process
examples/                sanitized offline and live runs with quality review
```

## Development process and submission status

The original specification, plan, decisions, and exact planner prompt are in `docs/dev-process/`. `docs/AI_PROCESS.md` records AI contributions, corrections, validation, and rejected overengineering.

The required unedited AI conversation export is intentionally not fabricated. Add it under `docs/dev-process/ai-sessions/` following that directory's redaction policy. The 3–5 minute video must also be recorded and linked before final submission.
