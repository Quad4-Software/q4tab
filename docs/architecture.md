# Architecture

q4tab answers "what comes next" with statistics over your own code,
not inference. There is no neural network anywhere in the pipeline.

A completion request flows through four layers:

```text
document text
   |
   v
tokenize   language-agnostic lexer, identifier subtoken splitting
   |
   v
retrieve   same-file repeats, dynamic index, corpus line index,
           masked-shape index, line n-grams, file-start priors
   |
   v
generate   n-gram chain decode conditioned on the token window
   |
   v
rank       per-source weights, scope boost, calibration, MMR dedup
```

Packages:

| package | role |
|---|---|
| `internal/tokenize` | lexer; splits identifiers into subtokens (camelCase, snake_case) |
| `internal/model` | packed n-gram model with Kneser-Ney smoothing, mmap'd |
| `internal/lines` | verbatim line index, masked-shape index, line n-gram table |
| `internal/engine` | builds indexes, merges sources, ranks candidates |
| `internal/lsp` | JSON-RPC server: stdio, TCP, HTTP, MCP |
| `internal/commitmsg` | conventional-commit drafting from git history |
| `internal/esort` | external sort used by the spill builder |
| `internal/corpus` | repository collection helpers |
| `internal/symbols` | cheap identifier extraction |

## The two brains

**Retrieval** finds lines that actually exist somewhere. It is right
most of the time on boilerplate, error handling, and repeated idioms,
and it cannot hallucinate.

**Generation** decodes token-by-token from the n-gram model when no
retrieval hit fits. It generalizes to novel identifier names but is
shallower: it finishes what it has seen patterns for.

Retrieval wins ties. Generation fills the gaps retrieval misses.

## Latency shape

Each keystroke is one small request. Serving is mmap + binary search +
a short greedy decode, so warm latency is single-digit milliseconds.
There is no streaming: a request returns a ranked list and is done.

## Edge cases

- UTF-16 position mapping including astral characters and CRLF files
- Incremental `didChange` sync, not full-document resends
- Never suggests text already after the cursor; divergent mid-line
  suggestions carry a replace-to-EOL range
- Documents over 128KB index in a background worker, so keystrokes on
  huge files never block completion
- Corrupt or truncated models fail with an error; a panic in the
  completion path degrades to no suggestions
- Malformed or oversized JSON-RPC frames are skipped, not fatal
- Journal entries written before a model loads are queued and replayed
