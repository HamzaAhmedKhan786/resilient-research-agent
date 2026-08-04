# Initial specification

Build a narrow, framework-free AI agent that accepts a high-level research goal, chooses and executes multiple tool actions, keeps durable state, adapts when actions fail, and returns an evidence-backed answer.

Required submission artifacts: runnable source, clear README, evaluation harness and measured results, end-to-end example, short-video material, implementation plan, and unedited AI-assisted development sessions.

Constraints:

- No LangChain, LangGraph, AutoGen, CrewAI, or equivalent agent framework.
- Prefer the simplest well-understood harness over a reusable platform.
- Make the loop, prompts, context construction, failures, and traces inspectable.
- Never persist or log a user API key.

Success criteria:

- A user can run the agent from a CLI or minimal localhost UI.
- Invalid actions and evidence do not silently enter durable state.
- Transient faults are retried within a bound; every run terminates.
- Controlled evals run without network access and produce checked-in results.
