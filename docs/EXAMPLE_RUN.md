# Example run: checkpointing and retries

Command:

```powershell
$env:PYTHONPATH="src"
python -m research_agent.cli --demo --run-dir .runs/example
```

Goal: **Compare checkpointing and retry logic for resilient agents.**

| Step | Decision | Observed result |
|---:|---|---|
| 1 | Search `checkpoint durable state` | The checkpoint source is discovered as S1. |
| 2 | Read S1 (`Durable checkpointing`) | 289 characters retrieved. |
| 3 | Save evidence from S1 | Verbatim excerpt accepted. |
| 4 | Search `bounded retries transient failures` | The retry source is discovered as S2. |
| 5 | Read S2 (`Bounded retries`) | 279 characters retrieved. |
| 6 | Save evidence from S2 | Verbatim excerpt accepted. |
| 7 | Finish | Citation validation passes; status becomes `complete`. |

Final answer:

> Checkpointing limits repeated work after interruption [S1], while bounded retries absorb transient faults without allowing endless loops [S2]. Together they address different failure modes.

The generated `trace.jsonl` contains the full machine-readable sequence, including each short rationale and tool outcome. The generated `checkpoint.json` contains the terminal durable state and evidence ledger.

## Live run

[`examples/live-run.md`](../examples/live-run.md) records a real six-step Groq run from goal to answer, and [`examples/live-trace.jsonl`](../examples/live-trace.jsonl) preserves its sanitized event sequence. It completed structurally with two Wikipedia sources, but manual review found uncited and unsupported prose. That finding produced the claim-level citation guard and stricter synthesis prompt now covered by regression tests.
