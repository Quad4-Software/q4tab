# Ranking and learning

Every candidate carries a source tag and a score. Scores combine a
per-source weight with evidence (match counts, rebound quality,
in-scope identifiers).

## Source weights

| source | meaning |
|---|---|
| `file` | verbatim repeat from the open document |
| `dyn` | dynamic index: open docs, journal, delta |
| `learn` | boost per recorded accept of this line |
| `adapt` | masked-shape hit with rebound identifiers |
| `prior` | file-start prior for thin contexts |
| `lineBi` | line n-gram hit |
| `model` | n-gram chain decode |
| `fim` / `fimIdx` | verified fill-in-the-middle |
| `scope` | in-scope identifier reuse boost |
| `struct` / `lang` | structural and per-language table votes |

## Scope boost

Candidates that reuse identifiers already declared in the document get
a multiplicative boost scaled by how often each name appears. The
boost reorders within a source class; it cannot vault a weak source
over a strong one.

## Operator healing

A trailing partial token constrains generation. In Go, a line ending
in `:` retracts the colon and forces the first generated token to be
`:=` when the following context wants it. Identifier partials are
handled inside the chain decode.

## Confidence cliff

During multi-line decode, a structural token that lands far below the
running mean confidence ends the block. Identifiers are exempt: rare
names are legitimate. This is what stops suggestions from tailing off
into noise.

## MMR dedup

Final candidates are deduplicated on masked two-line shape, so the
list is not four variants of the same line. A variant survives if it
introduces an identifier that exists in the current scope.

## Learning from accepts

Accepts are journaled to `learned.jsonl`, replayed on startup, mixed
into the learned cache, and indexed into the dynamic line index.
Implicit rejects (shown but not accepted) feed per-source multipliers:
sources that keep getting ignored get damped. `Q4TAB_JOURNAL`
relocates the journal.

## Kill switches

Every feature can be disabled individually for debugging:

```sh
Q4TAB_DISABLE=adapt,scope,unit,cliff,prior,heal,qual,iter,imp,mmr,src,embed q4tab serve
```
