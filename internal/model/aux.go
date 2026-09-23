package model

import (
	"math"
	"sort"
	"unsafe"
)

// aux.go: auxiliary tables that ride along with the main model.

// LangTable is a per-language low-order distribution: a unigram row and
// a bigram table. Kept small on purpose; the shared high-order model
// still does the heavy lifting, while these stop one language's habits
// (Go braces, Python colons) from bleeding into another's ranking.
type LangTable struct {
	Uni Order // single row keyed by the empty context
	Bi  Order // ctx is a single token id
}

// LangCtxKey is the context key used for language bigram rows: the
// single previous token id, mixed.
func LangCtxKey(prev uint32) uint64 {
	return uint64(prev)*0x9e3779b97f4a7c15 + 0x517cc1b727220a95
}

// Dist returns the blended low-order distribution: 3/4 bigram row for
// lastTok plus 1/4 unigram, or the unigram alone when the bigram row is
// missing. nil-safe.
func (lt *LangTable) Dist(lastTok uint32, q *Query) map[uint32]float64 {
	if lt == nil {
		return nil
	}
	uniToks, uniCnts, uniTot, _, _ := lt.Uni.row(hashCtx(nil), q)
	biToks, biCnts, biTot, _, biOK := lt.Bi.row(LangCtxKey(lastTok), q)
	out := make(map[uint32]float64, len(biToks)+len(uniToks)/8)
	const w = 0.75 // bigram share when it exists
	if biOK && biTot > 0 {
		for i, t := range biToks {
			out[uint32(t)] = w * float64(biCnts[i]) / float64(biTot)
		}
	}
	if uniTot > 0 {
		share := 1.0
		if biOK {
			share = 1 - w
		}
		for i, t := range uniToks {
			out[uint32(t)] += share * float64(uniCnts[i]) / float64(uniTot)
		}
	}
	return out
}

// IdentIndex maps a lowercase subtoken path ("parse|http|response") to
// vocabulary identifier ids, sorted so a path prefix is one binary
// search. It is the recall layer for names the n-gram model has never
// seen in the current context: the typed prefix decomposes into parts
// and matching identifiers surface as first-token candidates.
type IdentIndex struct {
	Off  []uint32 // len = n+1, offsets into Blob
	Blob []byte   // sorted lowercase "part|part|part" keys
	Ids  []int32  // main vocab id of the full identifier
	Freq []int32  // count proxy (unigram continuation count)
}

// IdentKey is the stored key form: parts lowercased, | separated.
func IdentKey(parts []string) string {
	n := 0
	for _, p := range parts {
		n += len(p) + 1
	}
	b := make([]byte, 0, n)
	for i, p := range parts {
		if i > 0 {
			b = append(b, '|')
		}
		for j := 0; j < len(p); j++ {
			c := p[j]
			if c >= 'A' && c <= 'Z' {
				c += 'a' - 'A'
			}
			b = append(b, c)
		}
	}
	return string(b)
}

// Prefix returns (id, freq) pairs whose subtoken path starts with
// prefix, best-frequency first, at most cap.
func (x *IdentIndex) Prefix(prefix string, cap int) ([]int32, []int32) {
	if x == nil || cap <= 0 {
		return nil, nil
	}
	n := len(x.Ids)
	lo := sort.Search(n, func(i int) bool { return x.key(i) >= prefix })
	type pair struct {
		id, f int32
	}
	var hits []pair
	for i := lo; i < n && len(hits) < cap*4; i++ {
		k := x.key(i)
		if len(k) < len(prefix) {
			break
		}
		match := true
		for j := 0; j < len(prefix); j++ {
			if k[j] != prefix[j] {
				match = false
				break
			}
		}
		if !match {
			break
		}
		hits = append(hits, pair{x.Ids[i], x.Freq[i]})
	}
	// Best frequency first, stable on path order for ties.
	sort.SliceStable(hits, func(i, j int) bool { return hits[i].f > hits[j].f })
	if len(hits) > cap {
		hits = hits[:cap]
	}
	ids := make([]int32, len(hits))
	fr := make([]int32, len(hits))
	for i, p := range hits {
		ids[i], fr[i] = p.id, p.f
	}
	return ids, fr
}

func (x *IdentIndex) key(i int) string {
	lo, hi := x.Off[i], x.Off[i+1]
	if lo == hi {
		return ""
	}
	return unsafe.String(&x.Blob[lo], int(hi-lo))
}

// BuildIdentIndex indexes every identifier in the vocab by its subtoken
// path. freq supplies a per-id count (pass the unigram continuation
// counts for KN models).
func BuildIdentIndex(v *Vocab, parts func(string) []string, freq func(uint32) int32) *IdentIndex {
	type ent struct {
		key  string
		id   int32
		freq int32
	}
	var ents []ent
	n := v.Len()
	for id := 1; id < n; id++ { // id 0 is <pad>
		s := v.Str(uint32(id))
		if len(s) < 2 || !isIdentByteStart(s[0]) {
			continue
		}
		ents = append(ents, ent{IdentKey(parts(s)), int32(id), freq(uint32(id))})
	}
	sort.Slice(ents, func(i, j int) bool { return ents[i].key < ents[j].key })
	x := &IdentIndex{Off: make([]uint32, len(ents)+1)}
	for i, e := range ents {
		x.Off[i] = uint32(len(x.Blob))
		x.Blob = append(x.Blob, e.key...)
		x.Ids = append(x.Ids, e.id)
		x.Freq = append(x.Freq, e.freq)
	}
	x.Off[len(ents)] = uint32(len(x.Blob))
	return x
}

func isIdentByteStart(c byte) bool {
	return c == '_' || c == '$' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80
}

// DeletedInterpolation estimates a per-order backoff-mass scale by
// held-out credit assignment. For each position where the order-k row
// exists, the mass reaching the true token splits into a direct part
// (max(c-D,0)) and a backoff part (gamma*P_lower); the backoff share is
// 1 when the token is absent from the row. The mean backoff share
// beta_k is the deleted-interpolation estimate of the backoff weight,
// and the emitted scale is beta_k over the model's own mean backoff
// fraction gamma/tot at those positions, so 1.0 means the learned and
// stored weights agree.
//
// ids must be held-out data (not the training stream). Returns a slice
// indexed by order, [0] and [1] set to 1, suitable for Model.LamK.
// Values are clipped to [0.25, 4] and normalized so the geometric mean
// is 1: a global level shift would only rescale all probabilities.
func (m *Model) DeletedInterpolation(ids []uint32) []float64 {
	n := m.N
	resp := make([]float64, n+1) // backoff responsibility mass
	gam := make([]float64, n+1)  // model's own gamma/tot mass
	hits := make([]int64, n+1)
	eof, _ := m.Vocab.Lookup("<eof>")
	q := m.NewQuery()
	for i := 0; i < len(ids); i++ {
		w := ids[i]
		if w == eof {
			continue
		}
		maxK := i + 1
		if maxK > n {
			maxK = n
		}
		for k := maxK; k >= 2; k-- {
			ctx := ids[i-k+1 : i]
			skip := false
			for _, t := range ctx {
				if t == eof {
					skip = true
					break
				}
			}
			if skip {
				break
			}
			toks, cnts, tot, gamma, ok := m.Orders[k].row(hashCtx(ctx), q)
			if !ok || tot <= 0 {
				continue
			}
			var c int64
			for j, t := range toks {
				if uint32(t) == w {
					c = int64(cnts[j])
					break
				}
			}
			direct := float64(c) - discFor(m.Orders[k].Disc, c)
			if direct < 0 {
				direct = 0
			}
			lower := m.probKN(w, ctx[1:], k-1, q)
			back := gamma * lower
			if direct+back <= 0 {
				continue
			}
			resp[k] += back / (direct + back)
			gam[k] += gamma / float64(tot)
			hits[k]++
		}
	}
	out := make([]float64, n+1)
	out[0] = 1
	out[1] = 1
	var sumLog float64
	var nSet int
	for k := 2; k <= n; k++ {
		if hits[k] < 64 || gam[k] <= 0 {
			out[k] = 1
			continue
		}
		beta := resp[k] / float64(hits[k])
		gbar := gam[k] / float64(hits[k])
		v := beta / gbar
		if v < 0.25 {
			v = 0.25
		}
		if v > 4 {
			v = 4
		}
		out[k] = v
		sumLog += math.Log(v)
		nSet++
	}
	if nSet > 0 {
		// Normalize: geometric mean 1 keeps this a reweighting, not a
		// global rescale.
		g := math.Exp(-sumLog / float64(nSet))
		for k := 2; k <= n; k++ {
			out[k] *= g
		}
	}
	return out
}
