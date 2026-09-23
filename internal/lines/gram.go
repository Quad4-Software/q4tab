package lines

import (
	"encoding/binary"
	"sort"
)

// gram.go: a line-level n-gram index. The line index answers "what does
// a line starting with X look like". The gram index answers the other
// question: "which line usually FOLLOWS this one". It stores
// (previous normalized line [, the one before it]) -> next-line counts,
// which gives multi-line completion a retrieval path the token model
// cannot reach: the first token of a fresh line is nearly free for the
// n-gram, but the WHOLE next line is exact here.

// GramKey hashes a context of previous normalized lines. Arity is mixed
// in so a bigram key never collides with a trigram key.
func GramKey(prev ...string) uint64 {
	h := uint64(1469598103934665603)
	for _, s := range prev {
		for i := 0; i < len(s); i++ {
			h ^= uint64(s[i])
			h *= 1099511628211
		}
		h ^= 0xff
		h *= 1099511628211
	}
	h ^= uint64(len(prev)) * 0x9e3779b97f4a7c15
	h *= 1099511628211
	return h
}

// GramBuilder accumulates (previous lines -> next line) counts while
// walking files. Next lines are interned to provisional ids to keep
// memory flat; Compact translates them to line-index positions.
type GramBuilder struct {
	ids    map[string]uint32
	strs   []string
	grams  map[uint64]map[uint32]uint32
	min    uint32
	frozen bool // Tighten was called: only already-known contexts count
}

func NewGramBuilder() *GramBuilder {
	return &GramBuilder{
		ids:   make(map[string]uint32),
		grams: make(map[uint64]map[uint32]uint32),
		min:   2, // a next-line pattern must repeat to be worth storing
	}
}

// AddFile walks one file's normalized line sequence and counts bigram
// and trigram transitions.
func (b *GramBuilder) AddFile(data []byte) {
	var prev1, prev2 string
	start := 0
	for i := 0; i <= len(data); i++ {
		if i != len(data) && data[i] != '\n' {
			continue
		}
		n := Normalize(string(data[start:i]))
		start = i + 1
		if len(n) < 4 || len(n) > 400 {
			continue
		}
		id, ok := b.ids[n]
		if !ok {
			id = uint32(len(b.strs))
			b.ids[n] = id
			b.strs = append(b.strs, n)
		}
		if prev1 != "" {
			add := func(key uint64) {
				row := b.grams[key]
				if row == nil {
					if b.frozen {
						return
					}
					row = make(map[uint32]uint32, 1)
					b.grams[key] = row
				}
				row[id]++
			}
			add(GramKey(prev1))
			if prev2 != "" {
				add(GramKey(prev2, prev1))
			}
		}
		prev2, prev1 = prev1, n
	}
}

// GramIndex is the compacted lookup table. Rows in Stream are
// uvarint n, uvarint rowTotal, then (lineIdx, count) pairs sorted by
// count desc. lineIdx indexes into the line Index built over the same
// corpus, so no strings are duplicated.
type GramIndex struct {
	Keys   []uint64
	Off    []int64 // len = len(Keys)+1, byte offsets into Stream
	Stream []byte
}

// Next returns (lineIdx, count) pairs for the given previous lines,
// sorted by count desc. Caller tries the longest context first.
func (g *GramIndex) Next(prev ...string) ([]int32, []int32, int64) {
	if g == nil || len(prev) == 0 {
		return nil, nil, 0
	}
	key := GramKey(prev...)
	i := sort.Search(len(g.Keys), func(i int) bool { return g.Keys[i] >= key })
	if i >= len(g.Keys) || g.Keys[i] != key {
		return nil, nil, 0
	}
	b := g.Stream[g.Off[i]:g.Off[i+1]]
	n, w := binary.Uvarint(b)
	if w <= 0 || n > uint64(len(b)) {
		return nil, nil, 0
	}
	b = b[w:]
	tot, w := binary.Uvarint(b)
	if w <= 0 {
		return nil, nil, 0
	}
	b = b[w:]
	if n > 64 {
		n = 64 // rows are count-sorted; the tail never surfaces
	}
	toks := make([]int32, n)
	cnts := make([]int32, n)
	for j := 0; j < int(n); j++ {
		v, w := binary.Uvarint(b)
		if w <= 0 {
			return nil, nil, 0
		}
		b = b[w:]
		c, w := binary.Uvarint(b)
		if w <= 0 {
			return nil, nil, 0
		}
		b = b[w:]
		toks[j] = int32(v)
		cnts[j] = int32(c)
	}
	return toks, cnts, int64(tot)
}

// Tighten bounds memory on very large corpora: rows with no repeated
// transition are dropped, and new contexts stop accumulating (only
// contexts already seen keep counting). Call once the corpus crosses a
// size budget; idempotent.
func (b *GramBuilder) Tighten() {
	// Rebuild rather than delete: Go maps keep bucket memory otherwise.
	nm := make(map[uint64]map[uint32]uint32, len(b.grams)/4)
	for key, row := range b.grams {
		for _, c := range row {
			if c >= b.min {
				nm[key] = row
				break
			}
		}
	}
	b.grams = nm
	b.frozen = true
}

// Compact freezes the gram builder against the finished line index:
// provisional next-line ids become Index positions. Next lines missing
// from the index (too short) are dropped. Rows are pruned to the top 32
// next lines and contexts kept only with a repeatable transition.
func (b *GramBuilder) Compact(li *Index) *GramIndex {
	type kv struct {
		key uint64
		row []gcent
	}
	var rows []kv
	for key, m := range b.grams {
		var row []gcent
		var tot uint64
		for id, c := range m {
			if c < b.min {
				continue
			}
			li2, ok := li.Lookup(b.strs[id])
			if !ok {
				continue
			}
			row = append(row, gcent{int32(li2), int32(c)})
			tot += uint64(c)
		}
		if len(row) == 0 {
			continue
		}
		sort.Slice(row, func(i, j int) bool {
			if row[i].cnt != row[j].cnt {
				return row[i].cnt > row[j].cnt
			}
			return row[i].tok < row[j].tok // deterministic builds
		})
		if len(row) > 32 {
			row = row[:32]
		}
		rows = append(rows, kv{key, row})
		_ = tot
	}
	sort.Slice(rows, func(i, j int) bool { return rows[i].key < rows[j].key })
	g := &GramIndex{Off: make([]int64, 0, len(rows)+1)}
	g.Off = append(g.Off, 0)
	var vbuf [binary.MaxVarintLen64]byte
	for _, r := range rows {
		var tot uint64
		for _, e := range r.row {
			tot += uint64(e.cnt)
		}
		n := binary.PutUvarint(vbuf[:], uint64(len(r.row)))
		g.Stream = append(g.Stream, vbuf[:n]...)
		n = binary.PutUvarint(vbuf[:], tot)
		g.Stream = append(g.Stream, vbuf[:n]...)
		for _, e := range r.row {
			n = binary.PutUvarint(vbuf[:], uint64(uint32(e.tok)))
			g.Stream = append(g.Stream, vbuf[:n]...)
			n = binary.PutUvarint(vbuf[:], uint64(uint32(e.cnt)))
			g.Stream = append(g.Stream, vbuf[:n]...)
		}
		g.Keys = append(g.Keys, r.key)
		g.Off = append(g.Off, int64(len(g.Stream)))
	}
	return g
}

type gcent struct {
	tok, cnt int32
}

// Len returns the number of distinct gram contexts. Used by diagnostics.
func (b *GramBuilder) Len() int { return len(b.grams) }
