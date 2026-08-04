# Decision log

## One action per decision

The model chooses one of `search`, `read`, `note`, or `finish`. This costs more calls than executing a large generated plan but makes changed decisions after errors visible.

## Evidence admission instead of unrestricted notes

A note must quote an excerpt present in a source already read by the agent. This is deliberately simpler than semantic entailment and blocks fabricated quotations.

## Two failure paths

Timeout and connection failures receive bounded low-level retries. Semantic failures—invalid IDs, unread evidence, bad excerpts, or invalid citations—become state errors so the next model decision can change course.

## Durable JSON and JSONL

Atomic JSON checkpoints and append-only JSONL traces were chosen over a database. They are inspectable and sufficient for one local process, but not safe coordination for distributed workers.

## Deterministic evaluation first

Scripted decisions and a local corpus isolate harness behavior from model and network variance. The limitation is explicit: these tests do not measure general research quality.

## Session-only API keys

The initial environment-only design was expanded to accept a user key in the UI. A proposal to “encrypt and store” it was rejected: a local app would also need to protect the decryption key. The implemented design transmits the key to localhost for one run, retains it only in process memory, and excludes it from artifacts and responses.

## Simplicity over expansion

Browser automation, embeddings, a task database, streaming tokens, multi-agent delegation, and a frontend framework were rejected as unjustified for this narrow submission.
