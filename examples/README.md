# Examples

`offline-demo.md` is the sanitized, reproducible demonstration run. It contains no API key, user checkpoint, browser data, or raw transient run directory.

Reproduce it with:

```powershell
research-agent --demo --run-dir .runs/example
```

`.runs/` is deliberately ignored because live checkpoints and traces can contain user goals and retrieved material.
