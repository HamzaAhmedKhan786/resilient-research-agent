# System design and architecture

The design is intentionally one process and one agent loop. The language model proposes research content; deterministic Python code owns state transitions, tool access, evidence admission, retry policy, checkpoints, and terminal conditions.

## Component architecture

```mermaid
flowchart LR
    U["User goal<br/>CLI or localhost UI"] --> C["Context builder<br/>compact state view"]
    C --> P["Planner adapter<br/>OpenAI or Groq"]
    P --> V["Action validator"]
    V --> A["Agent state machine"]
    A --> T["Research tools<br/>Wikipedia or local corpus"]
    T --> A
    A --> E["Evidence ledger<br/>verified excerpts"]
    E --> C
    A --> K["Atomic checkpoint.json"]
    A --> R["Append-only trace.jsonl"]
    A --> O["Cited answer or bounded failure"]
```

The API key crosses from the browser to the localhost server for the selected run. It is retained only in the active process call and is excluded from state, checkpoints, traces, status responses, and browser storage.

## Decision and recovery sequence

```mermaid
sequenceDiagram
    actor User
    participant RA as ResearchAgent
    participant LM as Planner
    participant AV as Validator
    participant RT as Research tool
    participant CP as Checkpoint and trace

    User->>RA: High-level research goal
    loop Until complete, failed, or budget exhausted
        RA->>LM: Goal + compact state + recent errors
        LM-->>RA: One next action
        RA->>AV: Validate lifecycle and arguments
        alt valid search/read
            AV-->>RA: accepted
            RA->>RT: execute bounded operation
            RT-->>RA: sources or document
        else valid note
            AV-->>RA: accepted
            RA->>RA: verify exact source span and subject relevance
        else valid finish
            AV-->>RA: accepted
            RA->>RA: validate source count, coverage, and citations
        else invalid or transient failure
            AV-->>RA: typed error
            RA->>RA: retry or expose error in next context
        end
        RA->>CP: append event and atomically save state
    end
    RA-->>User: cited result or explicit terminal status
```

## Persisted versus transient data

| Persisted | Process memory only | Sent to the model |
|---|---|---|
| Goal, source metadata, read/abandoned IDs, admitted evidence, recent errors, counters, answer | API key, full retrieved pages, active HTTP objects, UI thread state | Goal, ranked source metadata, bounded relevant extracts, admitted evidence, progress, last three errors |

Full pages are not placed in checkpoints. A resumed run re-fetches previously read, non-abandoned pages and reconstructs only the bounded context needed for the next decision.

## Failure boundaries

- Transport failures receive bounded retry and backoff without consuming a logical agent step.
- Provider errors are classified as rate-limit, generation, quota/authentication, or server failures.
- Invalid actions and unsupported final answers become planner-visible validation errors.
- Three identical unrecoverable errors open the circuit.
- The logical step budget prevents endless but varied behavior.
- Every iteration writes observable state before the next decision.
