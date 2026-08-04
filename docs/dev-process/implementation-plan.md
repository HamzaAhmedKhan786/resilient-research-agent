# Implementation plan

1. Implement structured state and a four-action research loop.
2. Add a minimal OpenAI Responses API adapter and Wikipedia/local-corpus tools.
3. Validate actions, exact evidence excerpts, and final source citations.
4. Add atomic checkpoints, append-only traces, retry policy, and step budget.
5. Build deterministic failure-injection evals and invariant-focused unit tests.
6. Add a dependency-free localhost UI with background work and trace polling.
7. Document architecture, security, evaluation limits, trade-offs, and examples.
8. Export unedited AI sessions, record the video, scan secrets and history, then publish.

Non-goals: general web crawling, a reusable agent framework, multi-agent execution, distributed workers, authentication, and production hosting.
