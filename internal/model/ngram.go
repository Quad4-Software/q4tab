// Package model implements a count-based n-gram language model with
// per-order context tables, modified Kneser-Ney smoothing, and dynamic
// scoped caches. This is the classical approach to code modeling
// (Hindle et al. 2012, Hellendoorn & Devanbu 2017): no neural net, only
// counts, which makes training a single pass and inference microseconds.
package model

import (
	"encoding/binary"
	"sort"
)

// Order holds the context table for one n-gram order in compact CSR form:
// sorted context keys, with per-context rows of (token, count) pairs.
// For a KN model (see Model.KN), counts stored in rows below the top
// order are continuation counts (distinct left extensions), not raw
// occurrence counts.
type Order struct {
	Keys   []uint64 // sorted context keys
	Off    []int64  // row boundaries, len(Keys)+1. V2: byte offsets into Stream
	Toks   []int32  // v1: token ids, sorted by count desc within each row
	Cnts   []int32  // v1
	Totals []int64  // v1: precomputed row sums, parallel to Keys
	Stream []byte   // v2: per-row varint stream {ntoks, total, toks, cnts}
	NToks  int64    // total (tok,cnt) pairs across all rows

	// Disc holds the modified Kneser-Ney tiered discounts for this
	// order (D for counts 1, 2, and 3+). Zero value = no discount, which
	// the scorer treats as the v2 Jelinek-Mercer model.
	Disc [3]float64
}

type memRow struct {
	toks  []int32
	cnts  []int32
	tot   int64
	gamma float64
}

// Query is a per-request decode memo: the interpolated scorer revisits
// the same context row once per candidate token, so caching decoded
// rows for the duration of one completion avoids quadratic unpacking.
// Request-scoped rather than shared, so concurrent completers never
// contend on it.
type Query struct {
	memo map[*Order]map[int]memRow
}

// NewQuery returns a fresh per-request memo. Reuse within one
// completion only. Do not share across goroutines.
func (m *Model) NewQuery() *Query {
	return &Query{memo: make(map[*Order]map[int]memRow)}
}

// Row is the exported form of row for engine-side aux tables that
// share this layout but are not the main model (struct context,
// language tables).
func (o *Order) Row(ctx uint64, q *Query) ([]int32, []int32, int64, float64, bool) {
	return o.row(ctx, q)
}

// row returns the token, count, total, and Kneser-Ney backoff mass
// (gamma) for a context. gamma is D1*n1 + D2*n2 + D3*n3+ over the row's
// counts-of-counts, computed once per decode and memoized with the row.
func (o *Order) row(ctx uint64, q *Query) ([]int32, []int32, int64, float64, bool) {
	i := sort.Search(len(o.Keys), func(i int) bool { return o.Keys[i] >= ctx })
	if i >= len(o.Keys) || o.Keys[i] != ctx {
		return nil, nil, 0, 0, false
	}
	if q != nil {
		if om := q.memo[o]; om != nil {
			if rm, ok := om[i]; ok {
				return rm.toks, rm.cnts, rm.tot, rm.gamma, true
			}
		}
	}
	var toks, cnts []int32
	var tot int64
	if o.Stream != nil {
		var ok bool
		toks, cnts, tot, ok = o.rowDecoded(i)
		if !ok {
			return nil, nil, 0, 0, false
		}
	} else {
		toks = o.Toks[o.Off[i]:o.Off[i+1]]
		cnts = o.Cnts[o.Off[i]:o.Off[i+1]]
		tot = o.Totals[i]
	}
	gamma := rowGamma(o.Disc, cnts)
	if q != nil {
		om := q.memo[o]
		if om == nil {
			om = make(map[int]memRow, 8)
			q.memo[o] = om
		}
		om[i] = memRow{toks, cnts, tot, gamma}
	}
	return toks, cnts, tot, gamma, true
}

// rowGamma is D1*n1 + D2*n2 + D3*n3+ over a row's stored counts. Rows
// are sorted by count descending, so the count boundaries are two
// binary searches rather than a full scan.
func rowGamma(d [3]float64, cnts []int32) float64 {
	n := len(cnts)
	i2 := sort.Search(n, func(i int) bool { return cnts[i] < 2 })
	i3 := sort.Search(i2, func(i int) bool { return cnts[i] < 3 })
	return d[0]*float64(n-i2) + d[1]*float64(i2-i3) + d[2]*float64(i3)
}

// rowDecoded unpacks row i from the v2 varint stream. Rows are small
// (median a few entries, capped during compaction) so per-call
// allocation is cheap. Decode runs outside the memo lock so concurrent
// completions never serialize on it.
func (o *Order) rowDecoded(i int) ([]int32, []int32, int64, bool) {
	b := o.Stream[o.Off[i]:o.Off[i+1]]
	nt, w := binary.Uvarint(b)
	if w <= 0 || nt > uint64(len(b)) {
		return nil, nil, 0, false
	}
	b = b[w:]
	tot, w := binary.Uvarint(b)
	if w <= 0 {
		return nil, nil, 0, false
	}
	b = b[w:]
	n := int(nt)
	// Decode at most maxRowScan entries: rows are sorted by count
	// descending and callers never look past that bound, so a giant
	// unigram row (millions of entries) costs a bounded decode rather
	// than a full linear unpack on every lookup. Pairs are interleaved
	// in the stream so a prefix decode stays consistent.
	if n > maxRowScan {
		n = maxRowScan
	}
	toks := make([]int32, n)
	cnts := make([]int32, n)
	for i := 0; i < n; i++ {
		v, w := binary.Uvarint(b)
		if w <= 0 {
			return nil, nil, 0, false
		}
		b = b[w:]
		c, w := binary.Uvarint(b)
		if w <= 0 {
			return nil, nil, 0, false
		}
		b = b[w:]
		toks[i] = int32(v)
		cnts[i] = int32(c)
	}
	return toks, cnts, int64(tot), true
}

// Model is a static trained n-gram model.
type Model struct {
	Vocab  *Vocab
	N      int
	Orders []Order // index 0 unused. Orders[k] uses k-1 context tokens

	// KN is true when the model was built with modified Kneser-Ney
	// smoothing (format v3): rows below the top order hold continuation
	// counts and each order carries tiered discounts. When false the
	// scorer falls back to recursive Jelinek-Mercer interpolation.
	KN bool

	// Uni holds per-token continuation counts (KN builds) indexed by
	// vocab id, replacing the order-1 row for exact O(1) unigram
	// probabilities. UniTot is the total; UniGamma is the precomputed
	// unigram backoff mass D1*n1 + D2*n2 + D3*n3+.
	Uni      []int32
	UniTot   int64
	UniDisc  [3]float64
	UniGamma float64
	UniTop   []uint32 // highest-continuation tokens, for empty-context fallback

	// LamK optionally scales the backoff mass per order (deleted-
	// interpolation weights tuned offline). Nil means 1.0 everywhere.
	LamK []float64
}

// hashCtx hashes a token-id context to a 64-bit key (FNV-1a).
func hashCtx(ctx []uint32) uint64 {
	h := uint64(1469598103934665603)
	for _, t := range ctx {
		x := uint64(t)
		h ^= x & 0xff
		h *= 1099511628211
		h ^= (x >> 8) & 0xff
		h *= 1099511628211
		h ^= (x >> 16) & 0xff
		h *= 1099511628211
		h ^= x >> 24
		h *= 1099511628211
	}
	// Mix in the length so contexts of different orders never collide.
	h ^= uint64(len(ctx)) * 0x9e3779b97f4a7c15
	h *= 1099511628211
	return h
}

// nkey keys the builder's flat count maps. full is hashCtx(ctx), suf is
// hashCtx(ctx[1:]) — the order-(k-1) row this entry's k-gram continues —
// and v is ctx[0]. suf and v let compaction count each (ctx',tok) pair's
// distinct left extensions, which are the Kneser-Ney continuation counts.
type nkey struct {
	full, suf uint64
	v, tok    uint32
}

// Builder accumulates counts transiently and compacts to a Model.
type Builder struct {
	vocab  *Vocab
	n      int  // orders 1..n get continuation counts; n itself uses raw
	deep   int  // verbatim orders beyond n, bloom-gated to repeat contexts
	plain  bool // raw counts everywhere, v2-style: for small aux models
	counts []map[nkey]uint32
	// min count to keep a (ctx,tok) pair, per order index
	minCnt []uint32
	bloom  []uint64 // seen-once context bits for gated orders
	gate   int      // first order whose singletons are bloom-gated
	frozen bool     // gated orders stopped accepting new keys
	total  uint64
}

const bloomShift = 6 // log2 of bits per uint64

// NewBuilder creates a builder. minCnt[i] applies to order i+1.
func NewBuilder(v *Vocab, n int, minCnt []uint32) *Builder {
	return NewBuilderDeep(v, n, 2, minCnt)
}

// NewBuilderPlain creates a builder that produces a raw-count model
// (KN=false, Jelinek-Mercer scoring, unigram kept as an order-1 row).
// Used for small auxiliary models where continuation statistics are
// not worth the space.
func NewBuilderPlain(v *Vocab, n int, minCnt []uint32) *Builder {
	b := NewBuilderDeep(v, n, 0, minCnt)
	b.plain = true
	return b
}

// NewBuilderDeep creates a builder with deep extra verbatim orders.
// Orders n+1..n+deep only record contexts seen at least twice, so the
// model gains long-range verbatim memory without paying for millions of
// singleton long contexts.
func NewBuilderDeep(v *Vocab, n, deep int, minCnt []uint32) *Builder {
	total := n + deep
	b := &Builder{vocab: v, n: n, deep: deep}
	b.counts = make([]map[nkey]uint32, total+1)
	for i := 0; i <= total; i++ {
		b.counts[i] = make(map[nkey]uint32)
	}
	if len(minCnt) < total+1 {
		m := make([]uint32, total+1)
		copy(m, minCnt)
		for i := len(minCnt); i <= total; i++ {
			// Orders 1-3 keep singletons: rare identifier pairs are where
			// codebase-specific knowledge lives. Higher orders prune at 2
			// to bound memory on large corpora. Deep orders keep 1: the
			// bloom gate already means a stored entry was seen at least
			// twice, and the first sighting is not counted, so the raw
			// count is sightings-1.
			switch {
			case i <= 3:
				m[i] = 1
			case i <= n:
				m[i] = 2
			default:
				m[i] = 1
			}
		}
		minCnt = m
	}
	b.minCnt = minCnt
	// Every order with a min count >= 2 is bloom-gated: a context seen
	// once can never survive pruning, so the flat map only pays for
	// repeat contexts. Deep orders are always gated (verbatim-repeat
	// storage is their purpose). On a large corpus this drops peak
	// build memory by half or more.
	b.gate = total + 1
	if deep > 0 {
		b.gate = n + 1
	}
	for k := 1; k <= total && k < b.gate; k++ {
		if minCnt[k] >= 2 {
			b.gate = k
			break
		}
	}
	if b.gate <= total {
		b.bloom = make([]uint64, 1<<26) // 512Mbit of seen-once bits
	}
	return b
}

// Tighten retroactively bloom-gates orders at or above minOrder when a
// corpus outgrows the memory budget mid-build. Existing entries seed
// the bloom so re-sightings keep counting; singleton entries at the
// newly gated orders are dropped and their maps rebuilt so the memory
// is actually freed. Counts for surviving entries stay exact.
func (b *Builder) Tighten(minOrder int) {
	if minOrder < 2 {
		minOrder = 2
	}
	if minOrder >= b.gate {
		return
	}
	if b.bloom == nil {
		b.bloom = make([]uint64, 1<<26)
	}
	mask := uint64(len(b.bloom)) - 1
	top := b.n + b.deep
	for k := minOrder; k <= top && k < b.gate; k++ {
		cm := b.counts[k]
		nm := make(map[nkey]uint32, len(cm)/4)
		for key, c := range cm {
			b.bloom[(key.full>>bloomShift)&mask] |= 1 << (key.full & 63)
			if c >= 2 {
				nm[key] = c
			}
		}
		b.counts[k] = nm
	}
	b.gate = minOrder
}

// Freeze stops gated orders from accepting new (ctx,tok) keys: only
// keys already in the map keep counting. Called when Tighten alone does
// not bound memory — on very large corpora the set of repeated contexts
// still grows without limit. Entries counted only once since gating
// are dropped first so the retained map is the set of genuinely hot
// contexts.
func (b *Builder) Freeze() {
	if b.frozen {
		return
	}
	for k := b.gate; k <= b.n+b.deep; k++ {
		cm := b.counts[k]
		nm := make(map[nkey]uint32, len(cm)/2)
		for key, c := range cm {
			if c >= 2 {
				nm[key] = c
			}
		}
		b.counts[k] = nm
	}
	b.frozen = true
}

// Add counts the n-grams in a token id sequence. Contexts do not cross
// the EOF token (id of "<eof>").
func (b *Builder) Add(ids []uint32) {
	eof := eofID(b.vocab)
	maxOrder := b.n + b.deep
	bloomMask := uint64(len(b.bloom)) - 1
	for i := 0; i < len(ids); i++ {
		tok := ids[i]
		if tok == eof {
			continue
		}
		maxK := i + 1
		if maxK > maxOrder {
			maxK = maxOrder
		}
		// ctx for order k is ids[i-k+1:i]; ctx[1:] for order k equals
		// the order k-1 context, so the previous iteration's hash is
		// this iteration's suffix hash.
		hPrev := hashCtx(nil)
		for k := 1; k <= maxK; k++ {
			ctx := ids[i-k+1 : i]
			if hasEOF(ctx, b.vocab) {
				continue
			}
			var v uint32 // ctx[0], the left extension; 0 for the empty ctx
			if len(ctx) > 0 {
				v = ctx[0]
			}
			h := hashCtx(ctx)
			key := nkey{h, hPrev, v, tok}
			if k >= b.gate {
				// Orders pruned at >=2 never keep a singleton: first
				// sightings set a bloom bit, repeats get counted.
				// Stored counts run one short, which is fine since
				// pruning is by >= min and ranking is monotone.
				if b.frozen {
					if _, ok := b.counts[k][key]; !ok {
						hPrev = h
						continue
					}
				} else if b.bloom[(h>>bloomShift)&bloomMask]&(1<<(h&63)) == 0 {
					b.bloom[(h>>bloomShift)&bloomMask] |= 1 << (h & 63)
					hPrev = h
					continue
				}
			}
			b.counts[k][key]++
			b.total++
			hPrev = h
		}
	}
}

func eofID(v *Vocab) uint32 {
	id, ok := v.Lookup("<eof>")
	if !ok {
		return 0
	}
	return id
}

func hasEOF(ctx []uint32, v *Vocab) bool {
	e := eofID(v)
	for _, t := range ctx {
		if t == e {
			return true
		}
	}
	return false
}

// nent is one surviving (ctx, tok, count) triple during compaction.
// suf and v ride along so continuation counts can be computed.
type nent struct {
	full, suf uint64
	v, tok    uint32
	cnt       uint32
}

// ck is a (context-hash, token) pair used for continuation-count maps.
type ck struct {
	ctx uint64
	tok uint32
}

// knDiscounts derives the modified Kneser-Ney tiered discounts from
// counts-of-counts (Chen & Goodman): D1 = 1-2Y(n2/n1), D2 = 2-3Y(n3/n2),
// D3 = 3-4Y(n4/n3) with Y = n1/(n1+2n2).
func knDiscounts(n1, n2, n3, n4 int64) [3]float64 {
	if n1 == 0 || n2 == 0 || n3 == 0 {
		return [3]float64{0.5, 0.67, 0.8}
	}
	y := float64(n1) / float64(n1+2*n2)
	d1 := 1 - 2*y*float64(n2)/float64(n1)
	d2 := 2 - 3*y*float64(n3)/float64(n2)
	d3 := 3 - 4*y*float64(n4)/float64(n3)
	clamp := func(d float64) float64 {
		if d < 0 {
			return 0
		}
		if d > 1 {
			return 1
		}
		return d
	}
	return [3]float64{clamp(d1), clamp(d2), clamp(d3)}
}

// Compact builds the immutable Model, applying min-count pruning.
// Entries are sorted once per order (ctx asc, count desc) so each
// context's row lands contiguous and pre-sorted in the CSR arrays.
//
// Modified Kneser-Ney layout: orders 2..n-1 store continuation counts —
// for each (ctx,tok) entry, the number of distinct left extensions of
// that k-gram. Orders n..n+deep store raw counts: n is the boundary
// because deep orders are bloom-gated to repeat contexts and cannot
// supply complete continuation statistics. The unigram is a flat
// vocab-indexed continuation-count array.
//
// Continuation counts are computed over each order's UNPRUNED flat map:
// pruning the order above must not shrink the backoff distribution of
// the order below.
func (b *Builder) Compact() *Model {
	m := &Model{Vocab: b.vocab, N: b.n + b.deep, Orders: make([]Order, b.n+b.deep+1), KN: !b.plain}
	conts := make(map[ck]uint32) // continuation counts for order k-1
	for k := b.n + b.deep; k >= 1; k-- {
		cm := b.counts[k]
		// Continuation counts for the order below come from this
		// order's full flat map: entries keyed (full,suf,v,tok) with
		// the same (suf,tok) differ only in the left extension, so a
		// map increment per entry counts distinct v's. Only orders
		// that store continuation counts (2..n-1) need this, and they
		// get it from orders 3..n.
		next := make(map[ck]uint32, len(cm))
		if !b.plain && k >= 2 && k <= b.n {
			// k=2 yields the unigram continuation counts: each
			// distinct predecessor of a token is one left extension.
			for key := range cm {
				next[ck{key.suf, key.tok}]++
			}
		}
		mc := b.minCnt[k]
		if k >= b.gate && mc > 1 {
			// Gated orders store sightings-1 (first sighting sets the
			// bloom bit without counting), so the count threshold is
			// one lower to keep the same sighting threshold.
			mc--
		}
		es := make([]nent, 0, len(cm))
		for key, c := range cm {
			if c >= mc {
				es = append(es, nent{key.full, key.suf, key.v, key.tok, c})
			}
		}
		b.counts[k] = nil // release the flat map before sorting

		var n1, n2, n3, n4 int64
		if b.plain || k >= b.n {
			// Raw counts at and above the KN boundary.
			for _, e := range es {
				switch e.cnt {
				case 1:
					n1++
				case 2:
					n2++
				case 3:
					n3++
				case 4:
					n4++
				}
			}
		} else {
			// Continuation-count orders: entries never extended left
			// carry no mass and drop out of the backoff distribution.
			kept := es[:0]
			for _, e := range es {
				c := conts[ck{e.full, e.tok}]
				if c == 0 {
					continue
				}
				e.cnt = c
				switch c {
				case 1:
					n1++
				case 2:
					n2++
				case 3:
					n3++
				case 4:
					n4++
				}
				kept = append(kept, e)
			}
			es = kept
		}
		o := &m.Orders[k]
		o.Disc = knDiscounts(n1, n2, n3, n4)

		if k == 1 && b.plain {
			// Plain models keep the unigram as a normal order-1 row.
			sort.Slice(es, func(i, j int) bool {
				return es[i].cnt > es[j].cnt
			})
			o.Keys = []uint64{hashCtx(nil)}
			o.Off = []int64{0, int64(len(es))}
			for _, e := range es {
				o.Toks = append(o.Toks, int32(e.tok))
				o.Cnts = append(o.Cnts, int32(e.cnt))
			}
			o.NToks = int64(len(es))
			var tot int64
			for _, c := range o.Cnts {
				tot += int64(c)
			}
			o.Totals = []int64{tot}
		} else if k == 1 {
			// Unigram: a single row keyed by the empty context. Store
			// it as a flat vocab-indexed array for O(1) lookup.
			m.Uni = make([]int32, b.vocab.Len())
			for _, e := range es {
				if int(e.tok) < len(m.Uni) {
					m.Uni[e.tok] = int32(e.cnt)
					m.UniTot += int64(e.cnt)
				}
			}
			m.UniDisc = o.Disc
			m.UniGamma = o.Disc[0]*float64(n1) + o.Disc[1]*float64(n2) + o.Disc[2]*float64(n3+n4)
			// Top tokens by continuation count for the empty-context
			// candidate fallback.
			sort.Slice(es, func(i, j int) bool { return es[i].cnt > es[j].cnt })
			top := min(len(es), 64)
			m.UniTop = make([]uint32, top)
			for i := 0; i < top; i++ {
				m.UniTop[i] = es[i].tok
			}
		} else {
			sort.Slice(es, func(i, j int) bool {
				if es[i].full != es[j].full {
					return es[i].full < es[j].full
				}
				if es[i].cnt != es[j].cnt {
					return es[i].cnt > es[j].cnt
				}
				return es[i].tok < es[j].tok // deterministic builds
			})
			o.Keys = make([]uint64, 0, len(es))
			o.Off = append(o.Off, 0)
			var cur uint64
			for i, e := range es {
				if i == 0 || e.full != cur {
					if i != 0 {
						o.Off = append(o.Off, int64(i))
					}
					cur = e.full
					o.Keys = append(o.Keys, cur)
				}
				o.Toks = append(o.Toks, int32(e.tok))
				o.Cnts = append(o.Cnts, int32(e.cnt))
			}
			o.Off = append(o.Off, int64(len(es)))
			o.NToks = int64(len(es))
			o.Totals = make([]int64, len(o.Keys))
			for i := range o.Keys {
				for j := o.Off[i]; j < o.Off[i+1]; j++ {
					o.Totals[i] += int64(o.Cnts[j])
				}
			}
		}
		conts = next
	}
	return m
}

// maxRowScan bounds linear search within a row. Rows are sorted by count
// descending. Entries past this point carry negligible probability mass.
const maxRowScan = 4096

func (m *Model) lamK(k int) float64 {
	if m.LamK != nil && k < len(m.LamK) {
		return m.LamK[k]
	}
	return 1
}

func discFor(d [3]float64, c int64) float64 {
	switch {
	case c <= 1:
		return d[0]
	case c == 2:
		return d[1]
	default:
		return d[2]
	}
}

// probAtOrder returns the interpolated probability of tok for a context.
// KN models (v3) use modified Kneser-Ney: at each order,
// P = (max(c - D(c),0) + lam*gamma*P_lower) / rowTotal, recursing to a
// uniform 1/V base. Older models use Jelinek-Mercer:
// P_k = lam*rowProb + (1-lam)*P_{k-1}.
func (m *Model) probAtOrder(tok uint32, ctx []uint32, k int, lam float64, q *Query) float64 {
	if m.KN {
		return m.probKN(tok, ctx, k, q)
	}
	if k <= 0 || len(ctx) == 0 {
		toks, cnts, tot, _, ok := m.Orders[1].row(hashCtx(nil), q)
		if !ok {
			return 1e-9
		}
		n := len(toks)
		if n > maxRowScan {
			n = maxRowScan
		}
		for i := 0; i < n; i++ {
			if uint32(toks[i]) == tok {
				return float64(cnts[i]) / float64(tot)
			}
		}
		return 0
	}
	useCtx := ctx
	if len(useCtx) > k-1 {
		useCtx = useCtx[len(useCtx)-(k-1):]
	}
	toks, cnts, tot, _, ok := m.Orders[k].row(hashCtx(useCtx), q)
	lower := m.probAtOrder(tok, ctx[1:], k-1, lam, q)
	if !ok {
		return lower
	}
	var p float64
	n := len(toks)
	if n > maxRowScan {
		n = maxRowScan
	}
	for i := 0; i < n; i++ {
		if uint32(toks[i]) == tok {
			p = float64(cnts[i]) / float64(tot)
			break
		}
	}
	return lam*p + (1-lam)*lower
}

// probKN is the modified Kneser-Ney probability at order k.
func (m *Model) probKN(tok uint32, ctx []uint32, k int, q *Query) float64 {
	if k <= 1 || len(ctx) == 0 {
		// Unigram: continuation counts in a flat array, backoff to
		// the uniform distribution.
		var c int64
		if int(tok) < len(m.Uni) {
			c = int64(m.Uni[tok])
		}
		if m.UniTot <= 0 {
			return 1e-9
		}
		v := float64(max(len(m.Uni), 1))
		d := discFor(m.UniDisc, c)
		num := float64(c) - d
		if num < 0 {
			num = 0
		}
		return (num + m.lamK(1)*m.UniGamma*(1/v)) / float64(m.UniTot)
	}
	useCtx := ctx
	if len(useCtx) > k-1 {
		useCtx = useCtx[len(useCtx)-(k-1):]
	}
	if len(useCtx) != k-1 {
		// Not enough context for this order: drop straight down.
		return m.probKN(tok, ctx[1:], k-1, q)
	}
	toks, cnts, tot, gamma, ok := m.Orders[k].row(hashCtx(useCtx), q)
	if !ok || tot <= 0 {
		return m.probKN(tok, ctx[1:], k-1, q)
	}
	lower := m.probKN(tok, ctx[1:], k-1, q)
	var c int64
	for i := 0; i < len(toks); i++ {
		if uint32(toks[i]) == tok {
			c = int64(cnts[i])
			break
		}
	}
	direct := float64(c) - discFor(m.Orders[k].Disc, c)
	if direct < 0 {
		direct = 0
	}
	return (direct + m.lamK(k)*gamma*lower) / float64(tot)
}

// Top returns the highest-probability next tokens for ctx using
// interpolation, limited to the row at the highest available order
// (candidates not present in the row are not considered, matching how
// count models rank).
func (m *Model) Top(ctx []uint32, lam float64, maxK int, q *Query) []Cand {
	for k := m.N; k >= 1; k-- {
		var useCtx []uint32
		if k > 1 {
			if len(ctx) >= k-1 {
				useCtx = ctx[len(ctx)-(k-1):]
			} else {
				continue
			}
		}
		var toks, cnts []int32
		var tot int64
		var ok bool
		if m.KN && k == 1 {
			// Unigram row is the flat array.
			toks = make([]int32, len(m.UniTop))
			for i, t := range m.UniTop {
				toks[i] = int32(t)
			}
			ok = len(toks) > 0
		} else {
			toks, cnts, tot, _, ok = m.Orders[k].row(hashCtx(useCtx), q)
		}
		if !ok || len(toks) == 0 {
			continue
		}
		limit := len(toks)
		if limit > 64 {
			limit = 64
		}
		cands := make([]Cand, 0, limit)
		for i := 0; i < limit; i++ {
			t := uint32(toks[i])
			var p float64
			if m.KN {
				p = m.probKN(t, ctx, k, q)
			} else {
				rowP := float64(cnts[i]) / float64(tot)
				p = lam*rowP + (1-lam)*m.probAtOrder(t, ctx, k-1, lam, q)
			}
			cands = append(cands, Cand{Tok: t, P: p})
		}
		// toks are sorted by count desc. Interpolated score may reorder,
		// so re-sort by p.
		sort.Slice(cands, func(i, j int) bool { return cands[i].P > cands[j].P })
		if len(cands) > maxK {
			cands = cands[:maxK]
		}
		return cands
	}
	return nil
}

// TopUnion gathers candidate tokens from every order row that matches
// ctx, scores each with the fully interpolated probability, and returns
// the best maxK. Unlike Top, a sparse high-order row cannot hide strong
// candidates that only exist in lower-order rows.
func (m *Model) TopUnion(ctx []uint32, lam float64, maxK int, q *Query) []Cand {
	seen := make(map[uint32]bool)
	var toks []uint32
	for k := m.N; k >= 2; k-- {
		var useCtx []uint32
		if len(ctx) >= k-1 {
			useCtx = ctx[len(ctx)-(k-1):]
		} else {
			continue
		}
		row, _, _, _, ok := m.Orders[k].row(hashCtx(useCtx), q)
		if !ok {
			continue
		}
		limit := len(row)
		if limit > 32 {
			limit = 32
		}
		for i := 0; i < limit; i++ {
			t := uint32(row[i])
			if !seen[t] {
				seen[t] = true
				toks = append(toks, t)
			}
		}
	}
	if m.KN && len(toks) == 0 {
		// Empty context: fall back to top unigram tokens.
		for _, t := range m.UniTop {
			toks = append(toks, t)
		}
	}
	if !m.KN && len(toks) == 0 {
		// v2 models keep the order-1 row in the order table.
		row, _, _, _, ok := m.Orders[1].row(hashCtx(nil), q)
		if ok {
			limit := len(row)
			if limit > 32 {
				limit = 32
			}
			for i := 0; i < limit; i++ {
				t := uint32(row[i])
				if !seen[t] {
					seen[t] = true
					toks = append(toks, t)
				}
			}
		}
	}
	if len(toks) == 0 {
		return nil
	}
	k := m.N
	if len(ctx) < k-1 {
		k = len(ctx) + 1
	}
	cands := make([]Cand, 0, len(toks))
	for _, t := range toks {
		cands = append(cands, Cand{Tok: t, P: m.probAtOrder(t, ctx, k, lam, q)})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].P > cands[j].P })
	if len(cands) > maxK {
		cands = cands[:maxK]
	}
	return cands
}

// Cand is a scored next-token candidate.
type Cand struct {
	Tok uint32
	P   float64
}
