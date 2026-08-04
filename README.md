# Resilient Research Agent

Resilient Research Agent is a narrow, evidence-backed research assistant for showing how a small agent makes decisions and recovers across multiple steps. A user supplies a goal through a CLI or dependency-free localhost UI; the agent searches Wikipedia, reads sources, admits verified excerpts, and produces a cited answer. The scope is intentionally limited so the decision loop, state, retries, checkpoints, and evaluation remain understandable without an agent framework.

## Quick start

Requirements: Python 3.11+ and, for live mode, an OpenAI API key.

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

Open the UI at `http://127.0.0.1:8765`. Offline demo mode is selected by default. Clear it to enter an API key and execute a live OpenAI/Wikipedia task.

For CLI live mode:

```powershell
$env:OPENAI_API_KEY="your-key"
research-agent "Compare two explanations for why the Tacoma Narrows Bridge failed" --model gpt-5-mini
```

## Architecture

```text
User goal
   |
   v
Context builder ---- durable state + recent errors + bounded extracts
   |
   v
OpenAI decision ---- one structured action
   |
   v
Action validator
   |
   v
Tool executor ------ search / read / note / finish
   |
   v
Evidence and state update
   |
   v
Checkpoint + trace -----> next decision or terminal result
```

The core is `src/research_agent/agent.py`. `model.py` contains the thin Responses API adapter, `tools.py` contains Wikipedia/local-corpus tools and failure injection, and `web.py` provides the localhost interface. No agent framework is used.

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

The action space is deliberately small: `search`, `read`, `note`, and `finish`. Search results get stable IDs such as `S1`. A source must be read before it supports a note, and the note excerpt must appear verbatim in the retrieved document. Every source ID cited by a final answer must exist in the evidence ledger.

## State and context handling

Persisted in `checkpoint.json`:

- goal, run ID, step count, and terminal status;
- discovered source metadata and stable IDs;
- IDs of sources read;
- admitted evidence excerpts;
- semantic errors and final answer.

Sent to the model:

- goal and remaining budget;
- discovered source metadata;
- up to 3,000 characters from currently available read documents;
- admitted evidence;
- only the three most recent errors.

Deliberately omitted:

- API keys and authorization headers;
- raw chain-of-thought;
- complete trace history;
- unbounded documents and tool payloads;
- previous model responses beyond their structured effects on state.

Long read results are compacted to bounded extracts in model context. Durable state stores evidence and metadata rather than entire pages. On restart, source content is not rehydrated from the checkpoint; the agent must re-read a source before using its extract again. This is a known resume limitation.

## Failure handling

| Failure | Behavior |
|---|---|
| Tool timeout/connection failure | Retry with bounded exponential backoff; trace every attempt. |
| Invalid model action or arguments | Reject before execution and add a semantic error to state. |
| Unknown/unread source | Reject the evidence action; allow a later changed decision. |
| Fabricated evidence excerpt | Reject unless the text occurs verbatim in the read source. |
| Missing or invalid final citation | Reject the finish action; require another decision. |
| Step budget exhausted | Stop deterministically with `budget_exhausted`. |
| Process restart | Load durable state with `--resume`; read content may need retrieval again. |

`trace.jsonl` records decisions, tool outcomes, retries, validation errors, and the termination reason. Checkpoints are written through a temporary file followed by atomic replacement.

## Security

The UI binds to `127.0.0.1` by default. The key is transmitted from the browser to the local HTTP server for the selected run, held only in process memory, and excluded from persistent state, logs, status responses, and browser storage. The password field is masked by default, offers an explicit Show/Hide control, and is cleared after submission.

Keys are not encrypted and stored: doing that securely would require a separate protected decryption key. Avoiding persistence is simpler and safer for this local demonstration. `.env`, run directories, caches, logs, and virtual environments are ignored by Git. Live run directories can contain user goals and retrieved content and must not be committed.

This is a local development UI, not a hardened multi-user service. It has no authentication, TLS, isolation, or rate limiting and should not be exposed on a public interface.

## Evaluation

Exact command:

```powershell
python evals/run_evals.py
```

Checked-in result: **4/4 evaluation scenarios passing**.

| Scenario | Fault or behavior | Assertions | Result |
|---|---|---|---|
| Happy path | Search, read, evidence, cited finish | completion, evidence, checkpoint, terminal trace | Pass, 4 steps |
| Transient search failure | One injected timeout | all base assertions plus visible retry | Pass, 4 steps |
| Transient read failure | Two injected timeouts | all base assertions plus visible retries | Pass, 4 steps |
| Invalid citation recovery | First finish has no citation, next changes | rejection followed by successful completion | Pass, 5 steps |

Detailed output is in `evals/results.json`. Separately, **2/2 unit tests pass**: one tests rejection and recovery when evidence is submitted before reading, and one tests step-budget termination. These unit tests are not part of the four-scenario evaluation count.

The deterministic planner and three-document local corpus isolate harness mechanics from sampling and network variation. The suite checks control flow, evidence invariants, recovery, checkpoint creation, and observability. It does not establish live research accuracy, citation entailment, cross-model performance, latency, or cost.

### Lessons and changes after evaluation

- Transport and semantic failures need different recovery paths: local retry for the former, planner-visible feedback for the latter.
- Stable source IDs and strict evidence admission added more audit value than expanding the tool set.
- Trace inspection caught a real example bug: source ranking made `S2` differ from the assumed document.
- That discovery exposed overly permissive citation validation; validation was changed from “at least one citation is saved” to “every cited source must be saved evidence.”
- Evaluation counts are now reported separately from unit-test counts.

## Example run

```powershell
research-agent --demo --run-dir .runs/example
```

Result:

> Checkpointing limits repeated work after interruption [S1], while bounded retries absorb transient faults without allowing endless loops [S2]. Together they address different failure modes.

The seven-step sanitized transcript is in `examples/offline-demo.md`. It is a real execution of the harness with deterministic decisions, not a live-model quality claim. Raw `.runs/` artifacts remain local because they may contain user content.

## Trade-offs

- Wikipedia instead of arbitrary web browsing keeps retrieval narrow and predictable.
- CLI plus dependency-free web UI avoids a production frontend and build chain.
- JSON checkpoints and JSONL traces are inspectable but not a concurrent database.
- A synchronous core loop is easier to reason about than distributed task workers.
- Deterministic corpus evals are reproducible but do not replace live model evaluations.
- Verbatim evidence verification is cheap and objective but does not prove claim entailment.
- One action per turn improves observability at the cost of additional model calls.

## Time spent

Approximate hands-on allocation for the current implementation:

| Work | Time |
|---|---:|
| Planning and scope | 30 minutes |
| Core loop, state, checkpoints, and traces | 2 hours |
| Tools and evidence grounding | 1 hour |
| Evaluation and testing | 1.5 hours |
| Web UI and documentation | 1 hour |
| **Total** | **about 6 hours** |

The priority was harness behavior and failure visibility. Retrieval breadth, UI sophistication, and production infrastructure were deliberately traded away.

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

Next priorities are an interruption/resume integration test, a bounded persistent source cache, 20–30 recorded live tasks scored for correctness and citation coverage, cost/latency reporting, cancellation, and domain-specific primary-source retrieval.

## Repository structure

```text
README.md                overview and run instructions
LICENSE                  MIT license
.env.example             placeholder only; never a real key
pyproject.toml           package metadata and console commands
src/research_agent/      loop, prompts, state, tools, CLI, and UI
tests/                   focused invariant tests
evals/                   corpus, fault-injection harness, measured results
docs/                    design, video outline, and development process
examples/                sanitized reproducible example
```

## Development process and submission status

The original specification, plan, decisions, and exact planner prompt are in `docs/dev-process/`. `docs/AI_PROCESS.md` records AI contributions, corrections, validation, and rejected overengineering.

The required unedited AI conversation export is intentionally not fabricated. Add it under `docs/dev-process/ai-sessions/` following that directory's redaction policy. The 3–5 minute video must also be recorded and linked before final submission.
