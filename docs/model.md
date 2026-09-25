# The model

## N-grams with Kneser-Ney smoothing

The core is an n-gram language model over lexed tokens. Given a window
of preceding tokens it scores candidate next tokens with modified
Kneser-Ney smoothing: continuation counts instead of raw counts, with
tiered discounts so common patterns dominate and rare ones get a fair
share of probability mass.

The build order defaults to 6 and can go higher. Deep orders would
explode the table, so orders above the base are bloom-gated: only
contexts that repeat are stored. One-off contexts cost nothing.

Identifier subtokens are modeled separately, so `httpServer` and
`http_client` share structure in the model.

## Packed storage

The trained model is a single `model.bin` file: sorted context rows
with varint-encoded token streams, a vocabulary table, and aux tables
appended as trailing sections (file-start priors, per-directory
identifier tables). The file is mmap'd read-only, so startup is a
header parse and probes fault pages in on demand.

Scale reference: 272M tokens of mixed source build into ~2.8GB packed,
load in under a second, and hold ~340MB resident.

## Spill builds

Indexing a large corpus does not scale in RAM. The default builder
lexes files in a worker pool, emits n-gram sightings into sorted run
files under `-tmpdir`, then k-way merges them into packed tables
(`internal/esort`, `internal/engine/spill.go`). Peak RSS stays around
6-7GB regardless of corpus size; scratch disk is the real cost, roughly
90 bytes per token at peak.

`-spill=false` selects the older in-memory path, which is faster on
small corpora but scales memory with token count.

## Incremental overlay

`index -incr` diffs the tree against a manifest and writes
`model.bin.delta`, a compact overlay the server merges at startup.
Deletions drop out of the delta; lines deleted long ago linger in the
base until a full rebuild folds everything back.

## Aux sidecar

`index -aux` walks the corpus extracting only the small tables -
type members, call members, directory idents, file-start priors - and
writes model.bin.aux (tens of MB, no spill). The server merges it at
startup, so corpus-wide member memory works on cold files even when a
full n-gram rebuild is not practical on the local disk budget.

## Weight tuning

`q4tab tune` runs coordinate descent over a holdout corpus and
persists a weights file only when it beats the shipped defaults.
Per-source coefficients and the order-interpolation weights (LamK)
are the knobs it moves, so ranking reflects measured hit rates rather
than guesses.
