// Package model implements a count-based n-gram language model with
// per-order context tables, recursive Jelinek-Mercer interpolation, and
// dynamic scoped caches. This is the classical approach to code modeling
// (Hindle et al. 2012, Hellendoorn & Devanbu 2017): no neural net, only
// counts, which makes training a single pass and inference microseconds.
package model

import (
	"encoding/binary"
	"sort"
)

// Order holds the context table for one n-gram order in compact CSR form:
// sorted context keys, with per-context rows of (token, count) pairs.
type Order struct {
	Keys   []uint64 // sorted context hashes
	Off    []int64  // row boundaries, len(Keys)+1. V2: byte offsets into Stream
	Toks   []int32  // v1: token ids, sorted by count desc within each row
	Cnts   []int32  // v1
	Totals []int64  // v1: precomputed row sums, parallel to Keys
	Stream []byte   // v2: per-row varint stream {ntoks, total, toks, cnts}
	NToks  int64    // total (tok,cnt) pairs across all rows

	// v2 rows decode from the stream on demand. The per-request Query
	// memoizes them (see below).
}

type memRow struct {
	toks []int32
	cnts []int32
	tot  int64
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

// row returns the token, count, and total for a context.
func (o *Order) row(ctx uint64, q *Query) ([]int32, []int32, int64, bool) {
	i := sort.Search(len(o.Keys), func(i int) bool { return o.Keys[i] >= ctx })
	if i >= len(o.Keys) || o.Keys[i] != ctx {
		return nil, nil, 0, false
	}
	if o.Stream != nil {
		if q != nil {
			if om := q.memo[o]; om != nil {
				if rm, ok := om[i]; ok {
					return rm.toks, rm.cnts, rm.tot, true
				}
			}
		}
		toks, cnts, tot, ok := o.rowDecoded(i)
		if ok && q != nil {
			om := q.memo[o]
			if om == nil {
				om = make(map[int]memRow, 8)
				q.memo[o] = om
			}
			om[i] = memRow{toks, cnts, tot}
		}
		return toks, cnts, tot, ok
	}
	return o.Toks[o.Off[i]:o.Off[i+1]], o.Cnts[o.Off[i]:o.Off[i+1]], o.Totals[i], true
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

// nkey packs a context hash and token id into one flat-map key. Flat
// maps use roughly half the memory of map[ctx]map[tok]count, which
// matters on 100M+ token corpora.
type nkey struct {
	ctx uint64
	tok uint32
}

// Builder accumulates counts transiently and compacts to a Model.
type Builder struct {
	vocab *Vocab
	n     int
	// per-order: (ctx,tok) -> count
	counts []map[nkey]uint32
	// min count to keep a (ctx,tok) pair, per order index
	minCnt []uint32
	total  uint64
}

// NewBuilder creates a builder. minCnt[i] applies to order i+1.
func NewBuilder(v *Vocab, n int, minCnt []uint32) *Builder {
	b := &Builder{vocab: v, n: n}
	b.counts = make([]map[nkey]uint32, n+1)
	for i := 0; i <= n; i++ {
		b.counts[i] = make(map[nkey]uint32)
	}
	if len(minCnt) < n+1 {
		m := make([]uint32, n+1)
		copy(m, minCnt)
		for i := len(minCnt); i <= n; i++ {
			// Orders 1-3 keep singletons: rare identifier pairs are where
			// codebase-specific knowledge lives. Higher orders prune at 2
			// to bound memory on large corpora.
			if i <= 3 {
				m[i] = 1
			} else {
				m[i] = 2
			}
		}
		minCnt = m
	}
	b.minCnt = minCnt
	return b
}

// Add counts the n-grams in a token id sequence. Contexts do not cross
// the EOF token (id of "<eof>").
func (b *Builder) Add(ids []uint32) {
	for i := 0; i < len(ids); i++ {
		tok := ids[i]
		if tok == eofID(b.vocab) {
			continue
		}
		maxK := i + 1
		if maxK > b.n {
			maxK = b.n
		}
		for k := 1; k <= maxK; k++ {
			ctx := ids[i-k+1 : i]
			if hasEOF(ctx, b.vocab) {
				continue
			}
			h := hashCtx(ctx)
			b.counts[k][nkey{h, tok}]++
			b.total++
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
type nent struct {
	ctx uint64
	tok uint32
	cnt uint32
}

// Compact builds the immutable Model, applying min-count pruning.
// Entries are sorted once per order (ctx asc, count desc) so each
// context's row lands contiguous and pre-sorted in the CSR arrays.
func (b *Builder) Compact() *Model {
	m := &Model{Vocab: b.vocab, N: b.n, Orders: make([]Order, b.n+1)}
	for k := 1; k <= b.n; k++ {
		cm := b.counts[k]
		min := b.minCnt[k]
		es := make([]nent, 0, len(cm))
		for key, c := range cm {
			if c >= min {
				es = append(es, nent{key.ctx, key.tok, c})
			}
		}
		b.counts[k] = nil // release the flat map before sorting
		sort.Slice(es, func(i, j int) bool {
			if es[i].ctx != es[j].ctx {
				return es[i].ctx < es[j].ctx
			}
			return es[i].cnt > es[j].cnt
		})
		o := &m.Orders[k]
		o.Keys = make([]uint64, 0, len(es))
		o.Off = append(o.Off, 0)
		var cur uint64
		for i, e := range es {
			if i == 0 || e.ctx != cur {
				if i != 0 {
					o.Off = append(o.Off, int64(i))
				}
				cur = e.ctx
				o.Keys = append(o.Keys, cur)
			}
			o.Toks = append(o.Toks, int32(e.tok))
			o.Cnts = append(o.Cnts, int32(e.cnt))
		}
		o.Off = append(o.Off, int64(len(es)))
		o.NToks = int64(len(es))
		// Row totals cover only surviving pairs.
		o.Totals = make([]int64, len(o.Keys))
		for i := range o.Keys {
			for j := o.Off[i]; j < o.Off[i+1]; j++ {
				o.Totals[i] += int64(o.Cnts[j])
			}
		}
	}
	return m
}

// maxRowScan bounds linear search within a row. Rows are sorted by count
// descending. Entries past this point carry negligible probability mass.
const maxRowScan = 4096

// probAtOrder returns the interpolated probability of tok for a context:
// P_k = lam*rowProb + (1-lam)*P_{k-1}, recursing down to the unigram.
func (m *Model) probAtOrder(tok uint32, ctx []uint32, k int, lam float64, q *Query) float64 {
	if k <= 0 || len(ctx) == 0 {
		toks, cnts, tot, ok := m.Orders[1].row(hashCtx(nil), q)
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
	toks, cnts, tot, ok := m.Orders[k].row(hashCtx(useCtx), q)
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
		toks, cnts, tot, ok := m.Orders[k].row(hashCtx(useCtx), q)
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
			rowP := float64(cnts[i]) / float64(tot)
			p := lam*rowP + (1-lam)*m.probAtOrder(t, ctx, k-1, lam, q)
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
	for k := m.N; k >= 1; k-- {
		var useCtx []uint32
		if k > 1 {
			if len(ctx) >= k-1 {
				useCtx = ctx[len(ctx)-(k-1):]
			} else {
				continue
			}
		}
		row, _, _, ok := m.Orders[k].row(hashCtx(useCtx), q)
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
