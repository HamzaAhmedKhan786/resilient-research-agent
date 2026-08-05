# Live model-in-the-loop run and review

This is a sanitized real Groq run performed through the localhost UI. The API key and raw checkpoint are not included. The machine-readable event sequence is in `live-trace.jsonl`.

Goal:

> Explain why the Tacoma Narrows Bridge collapsed in 1940 and distinguish the modern aeroelastic explanation from the commonly repeated resonance explanation. Use at least two relevant Wikipedia sources and cite every substantive claim.

Provider: Groq, model `openai/gpt-oss-20b`.

Outcome: the harness completed in six logical steps with two admitted Wikipedia sources, no provider retries, no tool retries, and no validation failures.

| Step | Decision | Observation |
|---:|---|---|
| 1 | Search for the 1940 collapse and aeroelastic explanation | Five sources discovered |
| 2 | Read S2, the exact 1940 bridge article | 30,000 characters retrieved |
| 3 | Save S2 evidence | Exact source span accepted after typography normalization |
| 4 | Read S1, the general Tacoma Narrows Bridge article | 12,033 characters retrieved |
| 5 | Save S1 evidence | Second exact source span accepted |
| 6 | Synthesize | Structural source-count and citation-ID validation passed |

Sources:

- S1: [Tacoma Narrows Bridge](https://en.wikipedia.org/wiki/Tacoma_Narrows_Bridge)
- S2: [Tacoma Narrows Bridge (1940)](https://en.wikipedia.org/wiki/Tacoma_Narrows_Bridge_%281940%29)

## Honest quality review

The run passed the original structural checks but did **not** fully satisfy the goal's claim-level citation requirement. Several factual sentences lacked their own citation, and the final uncertainty sentence was not supported by the admitted excerpts. The answer also strengthened “vertical movement” into “vertical and torsional movements observed by workers.”

This run is retained because it exposed a real harness gap. After this review:

- goals containing “cite every substantive claim” require a citation in every factual sentence;
- the Groq synthesis prompt forbids unsupported background definitions, causal details, and uncertainty;
- the live evaluator reports uncited substantive sentences;
- the controlled recovery scenarios assert their expected validation errors and transitions.

The original model answer is preserved inside `live-trace.jsonl`; it is not silently rewritten. A future live run should be recorded after these changes and judged with both structural checks and manual entailment review.
