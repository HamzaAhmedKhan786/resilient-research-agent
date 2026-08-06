# Examples

`offline-demo.md` is the readable sanitized demonstration, and `offline-trace.jsonl` is the corresponding machine-readable event sequence. The sanitized checked-in trace omits per-run IDs and timestamps; actual traces include both. These files contain no API key, user checkpoint, browser data, or raw transient run directory.

`live-run.md` and `live-trace.jsonl` preserve a real post-fix Groq model-in-the-loop run plus an explicit manual quality review. The run completed in nine steps with two sources and full structural coverage, while manual review still found two semantic-strengthening risks. Keeping that distinction visible demonstrates evaluation judgment rather than presenting structural checks as proof of factual entailment.

Reproduce it with:

```powershell
research-agent --demo --run-dir .runs/example
```

`.runs/` is deliberately ignored because live checkpoints and traces can contain user goals and retrieved material.
