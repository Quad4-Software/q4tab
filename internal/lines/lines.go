// Package lines implements the verbatim-retrieval half of the engine: a
// sorted index of normalized source lines with prefix search. Given the
// text the user has typed on the current line, it finds corpus lines that
// begin the same way and proposes their continuations, ranked by how
// often each continuation was seen.
//
// The index stores keys as a packed byte blob plus offsets rather than
// a []string: no per-string header overhead, and the blob can be an
// mmap'd region so untouched pages never become resident.
package lines

import (
	"sort"
	"strings"
	"unsafe"
)

// Index is a static sorted line index. Keys live in Blob: key i is
// Blob[Off[i]:Off[i+1]]. Keys are sorted lexicographically.
type Index struct {
	Blob []byte
	Off  []uint32 // len = key count + 1
	Cnts []int32  // occurrence count per key
}

// Key returns the i-th normalized line without copying. Safe because
// the blob is never mutated after construction (mapped or not).
func (idx *Index) Key(i int) string {
	lo, hi := idx.Off[i], idx.Off[i+1]
	if lo == hi {
		return ""
	}
	return unsafe.String(&idx.Blob[lo], int(hi-lo))
}

// Normalize collapses whitespace runs and trims leading whitespace so
// lines match regardless of indentation depth.
func Normalize(s string) string {
	s = strings.TrimLeft(s, " \t")
	if s == "" {
		return ""
	}
	var b strings.Builder
	b.Grow(len(s))
	ws := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == ' ' || c == '\t' || c == '\r' {
			ws = true
			continue
		}
		if ws {
			b.WriteByte(' ')
			ws = false
		}
		b.WriteByte(c)
	}
	return b.String()
}

// Builder accumulates line counts.
type Builder struct {
	counts map[string]int32
	minLen int
	maxLen int
	frozen bool // Tighten was called: only known lines keep counting
}

func NewBuilder() *Builder {
	return &Builder{counts: make(map[string]int32), minLen: 4, maxLen: 400}
}

// AddFile adds every line of a source file.
func (b *Builder) AddFile(data []byte) {
	start := 0
	for i := 0; i <= len(data); i++ {
		if i == len(data) || data[i] == '\n' {
			line := data[start:i]
			start = i + 1
			n := Normalize(string(line))
			if len(n) >= b.minLen && len(n) <= b.maxLen {
				if b.frozen {
					if _, ok := b.counts[n]; ok {
						b.counts[n]++
					}
				} else {
					b.counts[n]++
				}
			}
		}
	}
}

// AddLine adds a single already-normalized or raw line.
func (b *Builder) AddLine(raw string) {
	n := Normalize(raw)
	if len(n) >= b.minLen && len(n) <= b.maxLen {
		if b.frozen {
			if _, ok := b.counts[n]; ok {
				b.counts[n]++
			}
		} else {
			b.counts[n]++
		}
	}
}

// Tighten bounds memory on very large corpora: lines seen only once
// are dropped and new lines stop accumulating, so the map converges to
// the set of repeated lines. Rebuild rather than delete so bucket
// memory is actually freed.
func (b *Builder) Tighten() {
	nm := make(map[string]int32, len(b.counts)/4)
	for k, c := range b.counts {
		if c >= 2 {
			nm[k] = c
		}
	}
	b.counts = nm
	b.frozen = true
}

// Compact produces the sorted immutable index.
func (b *Builder) Compact() *Index {
	keys := make([]string, 0, len(b.counts))
	for k := range b.counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	idx := &Index{Off: make([]uint32, len(keys)+1), Cnts: make([]int32, len(keys))}
	var blob []byte
	for i, k := range keys {
		idx.Off[i] = uint32(len(blob))
		blob = append(blob, k...)
		idx.Cnts[i] = b.counts[k]
	}
	idx.Off[len(keys)] = uint32(len(blob))
	idx.Blob = blob
	return idx
}

// Continuation is a proposed rest-of-line with its corpus frequency.
type Continuation struct {
	Text  string
	Count int
}

// Complete finds continuations for the current line prefix. prefix should
// be the raw text before the cursor on the current line. Returns at most
// limit continuations ranked by count. scanCap bounds the number of index
// entries examined (a broad prefix like "return " can match millions).
func (idx *Index) Complete(prefix string, limit, scanCap int) []Continuation {
	norm := Normalize(prefix)
	if len(norm) < 2 || limit <= 0 {
		return nil
	}
	lo := sort.Search(len(idx.Cnts), func(i int) bool { return idx.Key(i) >= norm })
	if lo >= len(idx.Cnts) {
		return nil
	}
	// Keys sharing the prefix sort by their remainder, so identical
	// rests are adjacent: group them in one pass and keep only the
	// best `limit` (count desc, rest asc). No map, no full sort.
	tops := make([]Continuation, 0, limit+1)
	push := func(rest string, cnt int) {
		// Insertion into a small descending slice. Rests arrive in
		// ascending order so equal counts never need to move earlier
		// entries.
		i := len(tops)
		for i > 0 && tops[i-1].Count < cnt {
			i--
		}
		if i == limit {
			return
		}
		tops = append(tops, Continuation{})
		copy(tops[i+1:], tops[i:])
		tops[i] = Continuation{Text: rest, Count: cnt}
		if len(tops) > limit {
			tops = tops[:limit]
		}
	}
	var last string
	var lastCnt int
	examined := 0
	for i := lo; i < len(idx.Cnts) && examined < scanCap; i++ {
		k := idx.Key(i)
		if !strings.HasPrefix(k, norm) {
			break
		}
		examined++
		rest := k[len(norm):]
		if rest == last {
			lastCnt += int(idx.Cnts[i])
			continue
		}
		if len(last) >= 2 {
			push(last, lastCnt)
		}
		last, lastCnt = rest, int(idx.Cnts[i])
	}
	if len(last) >= 2 {
		push(last, lastCnt)
	}
	return tops
}

// Lookup returns the index of the exact normalized line, or false.
// Used to bind gram-index rows to line keys.
func (idx *Index) Lookup(norm string) (int, bool) {
	i := sort.Search(len(idx.Cnts), func(i int) bool { return idx.Key(i) >= norm })
	return i, i < len(idx.Cnts) && idx.Key(i) == norm
}

// HasPrefix reports whether any indexed line starts with p. p must
// already be normalized. Used by the fill-in-the-middle check to verify
// that a suggestion splices cleanly into existing text after the cursor.
func (idx *Index) HasPrefix(p string) bool {
	if p == "" {
		return false
	}
	i := sort.Search(len(idx.Cnts), func(i int) bool { return idx.Key(i) >= p })
	return i < len(idx.Cnts) && strings.HasPrefix(idx.Key(i), p)
}

// Len returns the number of unique lines.
func (idx *Index) Len() int { return len(idx.Cnts) }

// Len returns the number of distinct lines seen. Used by diagnostics.
func (b *Builder) Len() int { return len(b.counts) }
