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

## Member memory

The line index is textual, so a bare `x.` at the cursor used to fall
back to whatever receivers were popular in the corpus. The member
layer fixes that with lightweight fact extraction, not a typechecker:

- `func (r *T) M(` records `M` as a member of `T` and `r` as holding
  type `T`
- `type T struct { f U }` records field `f` and its type, so
  `x.field.` chains resolve; `class`/`interface`/`type T interface`
  bodies get the same treatment in other languages
- `x := NewT(`, `var x T`, `x = T(`, `x: T` annotations, and typed
  parameters record `x` as `T`
- `x []T`, `x map[K]V`, `x := make([]T)` record element types, so
  `x[0].` and `x[k].` resolve to `T` members
- `for v := range c` and `for v in c` bind `v` to `c`'s element type
- `func F() *T` records result types, so `F().` offers `T` members
- `F(...)` followed by `.M` records `M` as a member seen on `F`'s
  result, so `json.NewDecoder(r).` offers `Decode(`

Facts are extracted per open document, merged into a session table,
and persisted into the model at index time. At a dot the receiver
expression resolves doc-first, then session, then corpus. Resolved
members emit directly as candidates and boost retrieved candidates
that begin with a member name.

## Edit rules

Every document update is diffed against its previous version and the
token-level changes become rewrite rules for the session. Rename a
field from items to jobs once and two things happen: candidates
still using items gain a jobs variant, and candidates already
using jobs get a small working-set boost. Rules are capped in size
and apply only at token boundaries, so they never corrupt unrelated
text.

## Import adjacency

The build records the top identifiers per directory. At query time the
current file's directory and its import paths contribute a weak scope
prior, so idioms from a package you import surface without dominating
same-file evidence.
