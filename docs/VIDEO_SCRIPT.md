# 3–5 minute video outline

**0:00 — Problem and demo.** Show `python -m research_agent.cli --demo`, the final cited answer, and the trace path.

**0:40 — The loop.** Open `agent.py`. Explain the single-action loop: compact state goes to the planner; actions are schema-checked; tools execute; state is checkpointed after every step.

**1:40 — Information flow.** Show search results becoming stable source IDs, reads staying ephemeral/trimmed in model context, and only verified verbatim excerpts becoming durable evidence. Point out final citation validation.

**2:30 — Failure behavior.** Open a failure-injection trace. Show transient tool retries, semantic errors returned to the next planning step, the hard step budget, and atomic checkpoints.

**3:20 — Evaluation and judgment.** Run `python evals/run_evals.py`. Explain that deterministic planners isolate harness behavior from model variance. Discuss the limitation: these evals test orchestration, not live research quality.

**4:10 — Extension.** Add recorded live-task evals, a resume integration test that rehydrates read content, source-quality scoring, and domain-specific web sources before adding more tools.
