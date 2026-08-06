# Post-fix live model-in-the-loop run and review

This is a sanitized real Groq run performed through the localhost UI after the focused-search, long-page retrieval, citation-formatting, and answer-quality changes. The API key and raw checkpoint are not included. The machine-readable event sequence is in `live-trace.jsonl`; UI screenshots are under `assets/screenshots/`.

Goal:

> Explain why the Tacoma Narrows Bridge collapsed in 1940 and distinguish the modern aeroelastic explanation from the commonly repeated resonance explanation. Use at least two relevant Wikipedia sources and cite every substantive claim.

Provider: Groq, model `openai/gpt-oss-20b`.

Outcome: `complete` in nine logical steps, with two admitted Wikipedia sources, no provider retries, no tool retries, no validation failures, 2/2 goal-term coverage, and 2/2 cited-substantive-sentence coverage.

| Step | Decision | Observation |
|---:|---|---|
| 1 | Search for the 1940 collapse and aeroelastic explanation | Five sources discovered |
| 2 | Read S2, the exact 1940 bridge article | 33,080 characters retrieved |
| 3 | Abandon S2 | Groq found no suitable verbatim excerpt in the bounded relevant view |
| 4 | Read S1, the general Tacoma Narrows Bridge article | 12,033 characters retrieved |
| 5 | Save S1 evidence | Exact collapse/date/wind excerpt admitted |
| 6 | Search specifically for the uncovered resonance concept | Three new sources discovered |
| 7 | Read S6, Mechanical resonance | 8,598 characters retrieved |
| 8 | Save S6 evidence | Exact resonance-versus-aeroelastic excerpt admitted |
| 9 | Synthesize and validate | Goal coverage, citation coverage, and evidence-overlap diagnostics passed |

Sources:

- S1: [Tacoma Narrows Bridge](https://en.wikipedia.org/wiki/Tacoma_Narrows_Bridge)
- S6: [Mechanical resonance](https://en.wikipedia.org/wiki/Mechanical_resonance)

Final answer produced by the model:

> The first Tacoma Narrows Bridge collapsed on November 7 1940, four months after opening, when a 42 mph (68 km/h) wind induced aeroelastic flutter that caused the main span to fail [S1]. The collapse is often described as a classic example of mechanical resonance, but the actual mechanism was aeroelastic flutter—a self-sustaining oscillation driven by the interaction between the wind and the bridge structure, not a simple resonant vibration at a natural frequency [S6].

## What passed

- The run was multi-step and model-in-the-loop.
- An initially unhelpful source was explicitly abandoned rather than trapping the loop.
- The follow-up search targeted the uncovered resonance requirement.
- Both cited IDs belonged to admitted, verbatim evidence.
- Both substantive sentences carried citations.
- Both explicit goal terms were present in evidence and the answer.
- Timestamped trace and atomic checkpoint creation completed.

## Honest manual quality review

The harness correctly returned `complete` for its deterministic structural checks, but manual entailment review is not a full pass:

- S1 says the collapse was **possibly** a result of aeroelastic flutter caused by the reported wind; the answer states that relationship definitively.
- S6 supports that the vibrations arose from wind/bridge interaction and are known as aeroelastic flutter, but the answer's “self-sustaining” and “not a simple resonant vibration at a natural frequency” wording is stronger than this admitted excerpt explicitly establishes.

This is a useful result rather than a hidden defect: the trace proves the harness behavior, while the review demonstrates that citation presence and lexical overlap do not prove claim entailment. A future improvement would add an independent claim-evidence grader or require extractive synthesis for high-risk factual wording.
