# Retrieval layers

Retrieval is layered cheapest-to-strongest. Each layer returns
candidate lines; the ranker scores them by source.

## Same-file repeats

Lines from the open document itself. Repetitive code (table tests,
struct tags, flag blocks) is dominated by this layer. Capped by
distance so far-away duplicates lose to near ones.

## Dynamic index

Open documents, accepted completions from the journal, and the
incremental delta. This is the session memory: what you accepted ten
minutes ago is retrievable now.

## Corpus line index

A sorted index of every unique line in the training corpus, keyed by
the normalized prefix. Given `if err := load(&st` it binary-searches
lines that share the prefix and returns their continuations with
frequency counts.

## Masked-shape index

The rebind layer. Every indexed line is also stored with identifiers
masked out, keyed by shape. On a verbatim miss, the masked lookup finds
structurally identical lines from other call sites and rebinds the
stored identifiers to the names in scope.

That is what turns `use(&cfg)` into `use(&st)`: the stored line used
`cfg`, your file uses `st`, the shape matches.

Two gates keep it honest. A rebound item only emits when a rebound
name actually appears in the continuation (shape-only matches are the
model's job), and the continuation is verified against the n-gram
model, so implausible rebinds are dropped.

## Line n-grams

A separate table over whole lines: previous line(s) to likely next
line. It answers "what line follows this line" independently of token
position, which is how two-line suggestions are attested rather than
generated.

## Iterative retrieval

When the top hit is a single line, its text feeds a second line-gram
lookup to fetch an attested second line. The variant is capped below
its parent score, so a speculative extra line never outranks the
confirmed first line.

## File-start priors

Files with thin context (empty or near-empty) get opening lines
sampled from the corpus for the detected language, plus a Go package
clause inferred from the directory name. An empty `x.go` in `store/`
offers `package store`.

## Import adjacency

The build records the top identifiers per directory. At query time the
current file's directory and its import paths contribute a weak scope
prior, so idioms from a package you import surface without dominating
same-file evidence.
