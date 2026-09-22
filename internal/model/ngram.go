// Package model implements a count-based n-gram language model with
// per-order context tables, recursive Jelinek-Mercer interpolation, and
// dynamic scoped caches. This is the classical approach to code modeling
// (Hindle et al. 2012; Hellendoorn & Devanbu 2017): no neural net, just
// counts, which makes training a single pass and inference microseconds.
package model

import (
	"sort"
)

// Order holds the context table for one n-gram order in compact CSR form:
// sorted context keys, with per-context rows of (token, count) pairs.
type Order struct {
	Keys   []uint64 // sorted context hashes
	Off    []int64  // row boundaries into Toks/Cnts, len(Keys)+1
	Toks   []int32  // token ids, sorted by count desc within each row
	Cnts   []int32
	Totals []int64 // precomputed row sums, parallel to Keys
}

// row returns the token, count, and precomputed total for a context.
func (o *Order) row(ctx uint64) ([]int32, []int32, int64, bool) {
	i := sort.Search(len(o.Keys), func(i int) bool { return o.Keys[i] >= ctx })
	if i >= len(o.Keys) || o.Keys[i] != ctx {
		return nil, nil, 0, false
	}
	return o.Toks[o.Off[i]:o.Off[i+1]], o.Cnts[o.Off[i]:o.Off[i+1]], o.Totals[i], true
}

// Model is a static trained n-gram model.
type Model struct {
	Vocab  *Vocab
	N      int
	Orders []Order // index 0 unused; Orders[k] uses k-1 context tokens
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

// Builder accumulates counts transiently and compacts to a Model.
type Builder struct {
	vocab *Vocab
	n     int
	// per-order: ctx -> tok -> count
	counts []map[uint64]map[uint32]uint32
	// per-order: ctx -> total
	totals []map[uint64]uint32
	// min count to keep a (ctx,tok) pair, per order index
	minCnt []uint32
	total  uint64
}

// NewBuilder creates a builder. minCnt[i] applies to order i+1.
func NewBuilder(v *Vocab, n int, minCnt []uint32) *Builder {
	b := &Builder{vocab: v, n: n}
	b.counts = make([]map[uint64]map[uint32]uint32, n+1)
	b.totals = make([]map[uint64]uint32, n+1)
	for i := 0; i <= n; i++ {
		b.counts[i] = make(map[uint64]map[uint32]uint32)
		b.totals[i] = make(map[uint64]uint32)
	}
	if len(minCnt) < n+1 {
		m := make([]uint32, n+1)
		copy(m, minCnt)
		for i := len(minCnt); i <= n; i++ {
			m[i] = 2
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
			row := b.counts[k][h]
			if row == nil {
				row = make(map[uint32]uint32, 1)
				b.counts[k][h] = row
			}
			row[tok]++
			b.totals[k][h]++
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

// Compact builds the immutable Model, applying min-count pruning.
func (b *Builder) Compact() *Model {
	m := &Model{Vocab: b.vocab, N: b.n, Orders: make([]Order, b.n+1)}
	for k := 1; k <= b.n; k++ {
		cm := b.counts[k]
		keys := make([]uint64, 0, len(cm))
		for h := range cm {
			keys = append(keys, h)
		}
		sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
		o := Order{Keys: keys, Off: make([]int64, len(keys)+1), Totals: make([]int64, len(keys))}
		var toks []int32
		var cnts []int32
		for i, h := range keys {
			row := cm[h]
			// collect pairs above min count
			type pair struct {
				t uint32
				c uint32
			}
			ps := make([]pair, 0, len(row))
			var tot int64
			for t, c := range row {
				if c >= b.minCnt[k] {
					ps = append(ps, pair{t, c})
					tot += int64(c)
				}
			}
			sort.Slice(ps, func(i, j int) bool { return ps[i].c > ps[j].c })
			o.Off[i] = int64(len(toks))
			o.Totals[i] = tot
			for _, p := range ps {
				toks = append(toks, int32(p.t))
				cnts = append(cnts, int32(p.c))
			}
		}
		o.Off[len(keys)] = int64(len(toks))
		o.Toks = toks
		o.Cnts = cnts
		m.Orders[k] = o
		// release transient map
		b.counts[k] = nil
		b.totals[k] = nil
	}
	return m
}

// maxRowScan bounds linear search within a row. Rows are sorted by count
// descending; entries past this point carry negligible probability mass.
const maxRowScan = 4096

// probAtOrder returns the interpolated probability of tok for a context:
// P_k = lam*rowProb + (1-lam)*P_{k-1}, recursing down to the unigram.
func (m *Model) probAtOrder(tok uint32, ctx []uint32, k int, lam float64) float64 {
	if k <= 0 || len(ctx) == 0 {
		toks, cnts, tot, ok := m.Orders[1].row(hashCtx(nil))
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
	toks, cnts, tot, ok := m.Orders[k].row(hashCtx(useCtx))
	lower := m.probAtOrder(tok, ctx[1:], k-1, lam)
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
func (m *Model) Top(ctx []uint32, lam float64, maxK int) []Cand {
	for k := m.N; k >= 1; k-- {
		var useCtx []uint32
		if k > 1 {
			if len(ctx) >= k-1 {
				useCtx = ctx[len(ctx)-(k-1):]
			} else {
				continue
			}
		}
		toks, cnts, tot, ok := m.Orders[k].row(hashCtx(useCtx))
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
			p := lam*rowP + (1-lam)*m.probAtOrder(t, ctx, k-1, lam)
			cands = append(cands, Cand{Tok: t, P: p})
		}
		// toks are sorted by count desc; interpolated score may reorder,
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
func (m *Model) TopUnion(ctx []uint32, lam float64, maxK int) []Cand {
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
		row, _, _, ok := m.Orders[k].row(hashCtx(useCtx))
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
		cands = append(cands, Cand{Tok: t, P: m.probAtOrder(t, ctx, k, lam)})
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
