# Offline demonstration

Goal: Compare checkpointing and retry logic for resilient agents.

| Step | Action | Outcome |
|---:|---|---|
| 1 | Search for durable checkpoint state | Source S1 discovered. |
| 2 | Read S1 | 290 characters retrieved. |
| 3 | Save an exact excerpt from S1 | Evidence accepted. |
| 4 | Search for bounded transient-failure retries | Retry source discovered as S2. |
| 5 | Read S2 | 280 characters retrieved. |
| 6 | Save an exact excerpt from S2 | Evidence accepted. |
| 7 | Finish | Both citations validate; run completes. |

Result:

> Checkpointing limits repeated work after interruption [S1], while bounded retries absorb transient faults without allowing endless loops [S2]. Together they address different failure modes.

This run uses the deterministic planner and local corpus. It demonstrates the harness mechanics but is not presented as a live-model quality evaluation.
