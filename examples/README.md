# Examples

`offline-demo.md` is the readable sanitized demonstration, and `offline-trace.jsonl` is the corresponding machine-readable event sequence. They contain no API key, user checkpoint, browser data, or raw transient run directory.

`live-run.md` and `live-trace.jsonl` preserve a real Groq model-in-the-loop run plus an explicit manual quality review. The run completed structurally but exposed unsupported and uncited claims, which directly motivated stricter final-answer validation. Keeping that imperfection visible demonstrates the evaluation-and-adaptation process rather than presenting a polished synthetic transcript.

Reproduce it with:

```powershell
research-agent --demo --run-dir .runs/example
```

`.runs/` is deliberately ignored because live checkpoints and traces can contain user goals and retrieved material.
