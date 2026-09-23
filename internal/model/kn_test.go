package model

import (
	"math"
	"testing"
)

// knCorpus is a small repetitive token corpus with enough structure for
// continuation statistics to be meaningful: "a b c" repeats, "a b d"
// varies the tail, "x b c" gives b a second left extension, and the
// trailing "x" on some lines puts mid-stream trigrams into the data so
// higher-order rows survive continuation counting.
var knCorpus = [][]string{
	{"a", "b", "c", "d"},
	{"a", "b", "c", "d"},
	{"a", "b", "c", "e"},
	{"x", "b", "c", "d"},
	{"a", "b", "d", "d"},
	{"x", "b", "c", "e"},
	{"a", "b", "c", "f"},
	{"y", "z", "a", "b", "c"},
	{"y", "a", "b", "c", "d"},
}

func buildKN(t *testing.T, corpus [][]string, n, deep int) *Model {
	t.Helper()
	v := NewVocab()
	b := NewBuilderDeep(v, n, deep, nil)
	for _, s := range corpus {
		b.Add(v.Intern(s))
	}
	return b.Compact()
}

func TestKNFlagAndOrders(t *testing.T) {
	m := buildKN(t, knCorpus, 3, 0)
	if !m.KN {
		t.Fatal("KN model not flagged")
	}
	if m.N != 3 || len(m.Orders) != 4 {
		t.Fatalf("bad order table n=%d len=%d", m.N, len(m.Orders))
	}
	if len(m.Uni) != m.Vocab.Len() || m.UniTot == 0 {
		t.Fatalf("bad unigram array len=%d tot=%d", len(m.Uni), m.UniTot)
	}
	if len(m.UniTop) == 0 {
		t.Fatal("no unigram top candidates")
	}
}

// TestKNContinuationCounts checks that lower-order rows hold
// continuation counts: "b" appears as continuation of both "a" and "x"
// in bigram position, so its unigram continuation count is 2 while its
// raw count is 6.
func TestKNContinuationCounts(t *testing.T) {
	m := buildKN(t, knCorpus, 3, 0)
	bID, _ := m.Vocab.Lookup("b")
	if got := m.Uni[bID]; got != 2 {
		t.Fatalf("uni cont count for b = %d, want 2 (distinct predecessors a,x)", got)
	}
	cID, _ := m.Vocab.Lookup("c")
	if got := m.Uni[cID]; got != 1 {
		t.Fatalf("uni cont count for c = %d, want 1 (only after b)", got)
	}
	// Bigram continuation count for (ctx=a, tok=b): distinct left
	// extensions v of the trigram (v,a,b): "y z a b c" gives z and
	// "y a b c d" gives y, so the count is 2.
	aID, _ := m.Vocab.Lookup("a")
	toks, cnts, _, _, ok := m.Orders[2].row(hashCtx([]uint32{aID}), nil)
	if !ok {
		t.Fatal("no bigram row for ctx=a")
	}
	var cb int64 = -1
	for i, tok := range toks {
		if uint32(tok) == bID {
			cb = int64(cnts[i])
		}
	}
	if cb != 2 {
		t.Fatalf("bigram cont count for b|a = %d, want 2 (left extensions z,y)", cb)
	}
}

// TestKNRowNormalization verifies each context row's probability mass
// sums to ~1 over the whole vocabulary (id 0 included: the uniform base
// distributes over every id): direct + backoff reaches every token
// through the recursive lower orders.
func TestKNRowNormalization(t *testing.T) {
	m := buildKN(t, knCorpus, 4, 0)
	ctxs := [][]uint32{
		ids(m.Vocab, "a", "b"),
		ids(m.Vocab, "x", "b"),
		ids(m.Vocab, "y", "z"),
	}
	for _, ctx := range ctxs {
		q := m.NewQuery() // fresh memo per context
		var sum float64
		for id := 0; id < m.Vocab.Len(); id++ {
			sum += m.probKN(uint32(id), ctx, len(ctx)+1, q)
		}
		if math.Abs(sum-1) > 0.02 {
			t.Fatalf("ctx %v: prob mass sums to %f, want ~1", ctx, sum)
		}
	}
}

// TestKNRanking checks the known KN property: continuation counts stop
// high-frequency tokens from dominating sparse contexts. With raw
// counts, a token seen 6 times outranks one seen 5; with continuation
// counts the score reflects distributional diversity instead.
func TestKNRankingVsRaw(t *testing.T) {
	m := buildKN(t, knCorpus, 3, 0)
	q := m.NewQuery()
	// In ctx (a,b): corpus says c follows 4x, d once. c must win.
	cID, _ := m.Vocab.Lookup("c")
	dID, _ := m.Vocab.Lookup("d")
	pc := m.probKN(cID, ids(m.Vocab, "a", "b"), 3, q)
	pd := m.probKN(dID, ids(m.Vocab, "a", "b"), 3, q)
	if pc <= pd {
		t.Fatalf("P(c|a,b)=%g <= P(d|a,b)=%g", pc, pd)
	}
	// Unseen-in-context token still gets nonzero backoff mass.
	zID, _ := m.Vocab.Lookup("z")
	pz := m.probKN(zID, ids(m.Vocab, "a", "b"), 3, q)
	if pz <= 0 {
		t.Fatal("no backoff mass for unseen token")
	}
}

// TestDeepOrders verifies repeat-gated long contexts: a context seen
// twice lands in the deep order table, a singleton does not.
func TestDeepOrders(t *testing.T) {
	corpus := [][]string{
		{"p", "q", "r", "s", "t", "u", "v", "w"},
		{"p", "q", "r", "s", "t", "u", "v", "w"},
		{"a", "b", "c", "d", "e", "f", "g", "h"},
	}
	m := buildKN(t, corpus, 3, 2) // deep orders 4,5
	if m.N != 5 {
		t.Fatalf("N=%d want 5", m.N)
	}
	// The repeated 5-gram context (p,q,r,s) at order 5 predicts w.
	ctx := ids(m.Vocab, "p", "q", "r", "s")
	toks, _, _, _, ok := m.Orders[5].row(hashCtx(ctx), nil)
	if !ok || len(toks) == 0 {
		t.Fatal("repeat deep context missing")
	}
	// The singleton context (a,b,c,d) at order 5 was bloom-gated out.
	ctx2 := ids(m.Vocab, "a", "b", "c", "d")
	if _, _, _, _, ok := m.Orders[5].row(hashCtx(ctx2), nil); ok {
		t.Fatal("singleton deep context should be gated out")
	}
}

// TestPlainBuilder verifies the raw-count path still builds a working
// JM model (used for the subtoken aux model and v2 compatibility).
func TestPlainBuilder(t *testing.T) {
	v := NewVocab()
	b := NewBuilderPlain(v, 3, nil)
	for _, s := range knCorpus {
		b.Add(v.Intern(s))
	}
	m := b.Compact()
	if m.KN {
		t.Fatal("plain model flagged KN")
	}
	if len(m.Orders[1].Keys) == 0 {
		t.Fatal("plain model missing unigram row")
	}
	q := m.NewQuery()
	ctx := ids(m.Vocab, "a", "b")
	var sum float64
	for id := 1; id < m.Vocab.Len(); id++ {
		sum += m.probAtOrder(uint32(id), ctx, 3, 0.8, q)
	}
	if sum <= 0 {
		t.Fatal("plain model returned no probability mass")
	}
}

// TestDeletedInterpolation exercises the trainer on a held-out stream:
// it must return a clipped, normalized scale vector without panicking.
func TestDeletedInterpolation(t *testing.T) {
	m := buildKN(t, knCorpus, 4, 0)
	holdout := m.Vocab.Intern([]string{"a", "b", "c", "d", "x", "b", "c"})
	lamK := m.DeletedInterpolation(holdout)
	if len(lamK) != m.N+1 {
		t.Fatalf("lamK len %d want %d", len(lamK), m.N+1)
	}
	for k, v := range lamK {
		if v <= 0 || math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("lamK[%d]=%g invalid", k, v)
		}
	}
	// Tiny holdout: too few informative positions, defaults stand.
	m.LamK = nil
	lamK2 := m.DeletedInterpolation(m.Vocab.Intern([]string{"a", "b"}))
	for k := 2; k <= m.N; k++ {
		if lamK2[k] != 1 {
			t.Fatalf("lamK2[%d]=%g want 1 for insufficient data", k, lamK2[k])
		}
	}
}

func ids(v *Vocab, toks ...string) []uint32 {
	out := make([]uint32, len(toks))
	for i, s := range toks {
		out[i], _ = v.Lookup(s)
	}
	return out
}

// TestTightenDropsSingletons: after Tighten(2), first sightings at the
// gated orders only set bloom bits, so contexts seen exactly once never
// reach the model. Contexts seen twice before Tighten keep counting
// exactly, and contexts first seen after Tighten need a second sighting
// before they are stored.
func TestTightenDropsSingletons(t *testing.T) {
	v := NewVocab()
	b := NewBuilderDeep(v, 3, 0, nil)
	// "r s t" seen twice before tighten survives. "a b c" seen once
	// does not. Bigrams stay ungated below order 2.
	add := func(toks ...string) { b.Add(v.Intern(toks)) }
	add("r", "s", "t")
	add("r", "s", "t")
	add("a", "b", "c")
	b.Tighten(2)
	add("a", "b", "c") // first post-tighten sighting: sets the bit only
	add("a", "b", "c") // second: counted once
	add("p", "q", "w") // one sighting after tighten: never counted
	m := b.Compact()

	ctxRS := hashCtx(ids(v, "r", "s"))
	ctxAB := hashCtx(ids(v, "a", "b"))
	ctxPQ := hashCtx(ids(v, "p", "q"))
	tID, _ := v.Lookup("t")
	cID, _ := v.Lookup("c")
	wID, _ := v.Lookup("w")

	toks, _, _, _, ok := m.Orders[3].Row(ctxRS, nil)
	if !ok || len(toks) != 1 || toks[0] != int32(tID) {
		t.Fatalf("pre-tighten repeated trigram lost: %v ok=%v", toks, ok)
	}
	toks, _, _, _, ok = m.Orders[3].Row(ctxAB, nil)
	if !ok || len(toks) != 1 || toks[0] != int32(cID) {
		t.Fatalf("post-tighten repeated trigram lost: %v ok=%v", toks, ok)
	}
	if toks, _, _, _, ok := m.Orders[3].Row(ctxPQ, nil); ok {
		for _, tok := range toks {
			if tok == int32(wID) {
				t.Fatal("singleton post-tighten trigram was kept")
			}
		}
	}
}

// TestTightenIdempotentAndFloor: Tighten clamps at order 2 and a second
// call is a no-op.
func TestTightenIdempotentAndFloor(t *testing.T) {
	v := NewVocab()
	b := NewBuilderDeep(v, 3, 0, nil)
	b.Add(v.Intern([]string{"a", "b", "c"}))
	b.Tighten(1) // clamps to 2
	if b.gate != 2 {
		t.Fatalf("gate=%d want 2", b.gate)
	}
	b.Tighten(2) // already gated, must not panic or re-seed
	if b.gate != 2 {
		t.Fatalf("gate=%d want 2", b.gate)
	}
}
