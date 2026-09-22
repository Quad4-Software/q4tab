// Package lines implements the verbatim-retrieval half of the engine: a
// sorted index of normalized source lines with prefix search. Given the
// text the user has typed on the current line, it finds corpus lines that
// begin the same way and proposes their continuations, ranked by how
// often each continuation was seen.
package lines

import (
	"sort"
	"strings"
)

// Index is a static sorted line index.
type Index struct {
	Keys []string // normalized lines, sorted
	Cnts []int32  // occurrence count per key
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
		if c == ' ' || c == '\t' {
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
				b.counts[n]++
			}
		}
	}
}

// AddLine adds a single already-normalized or raw line.
func (b *Builder) AddLine(raw string) {
	n := Normalize(raw)
	if len(n) >= b.minLen && len(n) <= b.maxLen {
		b.counts[n]++
	}
}

// Compact produces the sorted immutable index.
func (b *Builder) Compact() *Index {
	keys := make([]string, 0, len(b.counts))
	for k := range b.counts {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	idx := &Index{Keys: keys, Cnts: make([]int32, len(keys))}
	for i, k := range keys {
		idx.Cnts[i] = b.counts[k]
	}
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
	if len(norm) < 2 {
		return nil
	}
	lo := sort.SearchStrings(idx.Keys, norm)
	if lo >= len(idx.Keys) {
		return nil
	}
	conts := make(map[string]int)
	examined := 0
	for i := lo; i < len(idx.Keys) && examined < scanCap; i++ {
		k := idx.Keys[i]
		if !strings.HasPrefix(k, norm) {
			break
		}
		examined++
		rest := k[len(norm):]
		if rest == "" {
			continue
		}
		// Only propose continuations that extend the prefix by a
		// meaningful amount.
		if len(rest) < 2 {
			continue
		}
		conts[rest] += int(idx.Cnts[i])
	}
	if len(conts) == 0 {
		return nil
	}
	out := make([]Continuation, 0, len(conts))
	for t, c := range conts {
		out = append(out, Continuation{Text: t, Count: c})
	}
	sort.Slice(out, func(i, j int) bool {
		if out[i].Count != out[j].Count {
			return out[i].Count > out[j].Count
		}
		return out[i].Text < out[j].Text
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

// Len returns the number of unique lines.
func (idx *Index) Len() int { return len(idx.Keys) }
