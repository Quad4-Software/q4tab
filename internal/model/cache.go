package model

import "sort"

// Cache is a small dynamic n-gram model fed from open/edited files.
// It captures the locality of software: identifiers and patterns from the
// file you are editing should dominate suggestions. It mixes with the
// static model at query time via Jelinek-Mercer interpolation, with the
// cache weight growing as the cache sees the context more often.
type Cache struct {
	order  int
	rows   []map[uint64]map[uint32]uint32 // index k: ctx(k-1 toks) -> tok -> count
	totals []map[uint64]uint32
}

func NewCache(order int) *Cache {
	c := &Cache{order: order}
	c.rows = make([]map[uint64]map[uint32]uint32, order+1)
	c.totals = make([]map[uint64]uint32, order+1)
	for i := 0; i <= order; i++ {
		c.rows[i] = make(map[uint64]map[uint32]uint32)
		c.totals[i] = make(map[uint64]uint32)
	}
	return c
}

// Add counts n-grams in the token sequence. eofTok contexts are skipped.
func (c *Cache) Add(ids []uint32, eofTok uint32) {
	for i := 0; i < len(ids); i++ {
		tok := ids[i]
		if tok == eofTok {
			continue
		}
		maxK := i + 1
		if maxK > c.order {
			maxK = c.order
		}
		for k := 1; k <= maxK; k++ {
			ctx := ids[i-k+1 : i]
			skip := false
			for _, t := range ctx {
				if t == eofTok {
					skip = true
					break
				}
			}
			if skip {
				continue
			}
			h := hashCtx(ctx)
			row := c.rows[k][h]
			if row == nil {
				row = make(map[uint32]uint32, 1)
				c.rows[k][h] = row
			}
			row[tok]++
			c.totals[k][h]++
		}
	}
}

// Dist returns the interpolated cache distribution row for ctx at the
// highest available order: candidates with cache probabilities.
// The second return value is the total count of the winning context row,
// used to derive the interpolation weight.
func (c *Cache) Dist(ctx []uint32) (map[uint32]float64, float64) {
	for k := c.order; k >= 1; k-- {
		var useCtx []uint32
		if k > 1 {
			if len(ctx) >= k-1 {
				useCtx = ctx[len(ctx)-(k-1):]
			} else {
				continue
			}
		}
		row := c.rows[k][hashCtx(useCtx)]
		if len(row) == 0 {
			continue
		}
		tot := float64(c.totals[k][hashCtx(useCtx)])
		out := make(map[uint32]float64, len(row))
		for t, n := range row {
			out[t] = float64(n) / tot
		}
		return out, tot
	}
	return nil, 0
}

// Mixed candidates: for each candidate token in the union of cache and
// static rows, compute beta*P_cache + (1-beta)*P_static, where
// beta = cacheTotal/(cacheTotal + gamma).
func MixCandidates(static []Cand, cacheDist map[uint32]float64, cacheTot, gamma float64) []Cand {
	if len(cacheDist) == 0 {
		return static
	}
	beta := cacheTot / (cacheTot + gamma)
	if beta > 0.75 {
		beta = 0.75
	}
	seen := make(map[uint32]bool, len(static))
	out := make([]Cand, 0, len(static)+len(cacheDist))
	for _, c := range static {
		seen[c.Tok] = true
		p := c.P
		if cp, ok := cacheDist[c.Tok]; ok {
			p = beta*cp + (1-beta)*p
		} else {
			p = (1 - beta) * p
		}
		out = append(out, Cand{Tok: c.Tok, P: p})
	}
	for t, cp := range cacheDist {
		if seen[t] {
			continue
		}
		out = append(out, Cand{Tok: t, P: beta * cp * 0.5})
	}
	sort.Slice(out, func(i, j int) bool { return out[i].P > out[j].P })
	return out
}
