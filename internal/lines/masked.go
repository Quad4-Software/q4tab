package lines

import (
	"sort"
	"strings"

	"q4tab/internal/tokenize"
)

// masked.go implements identifier-insensitive line retrieval. The
// verbatim index only matches lines whose identifiers are byte-equal,
// so "if err := decode(&in)" never retrieves the stored
// "if err := decode(&req)". The masked index keys lines by their shape
// with identifiers replaced by a placeholder, then Adapt rewrites the
// retrieved continuation with the identifiers the caller actually
// typed. This is lexical adaptation: retrieve by structure, rebind by
// position.

// identMark is the placeholder byte for a masked identifier token.
// It cannot collide with real source: the lexer never emits it.
const identMark = "\x02"

// maskToks returns toks with every non-keyword identifier replaced by
// identMark. Keywords and literals stay: they carry the line's shape.
func maskToks(toks []string) []string {
	out := make([]string, len(toks))
	for i, t := range toks {
		if isMaskable(t) {
			out[i] = identMark
		} else {
			out[i] = t
		}
	}
	return out
}

// isMaskable reports whether t is an identifier that should be masked:
// identifier-shaped, any length, and not a keyword. Keywords like if
// and return stay unmasked so structural matching keeps meaning.
func isMaskable(t string) bool {
	if t == "" || t == identMark {
		return false
	}
	c := t[0]
	if !(c == '_' || c == '$' ||
		(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) {
		return false
	}
	for i := 1; i < len(t); i++ {
		c := t[i]
		if !(c == '_' || c == '$' || c == '-' ||
			(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c >= 0x80) {
			return false
		}
	}
	return !tokenize.IsKeywordish(t)
}

// MaskedLine returns the identifier-masked form of a normalized line:
// lexed, non-keyword idents replaced by the mark, rejoined exactly.
func MaskedLine(norm string) string {
	toks := tokenize.LexLine([]byte(norm))
	if len(toks) == 0 {
		return ""
	}
	return strings.Join(maskToks(toks), "")
}

// MaskedIndex maps masked line shapes to entries in the verbatim
// index. Build once per model load; it is a few percent of the line
// index's size.
type MaskedIndex struct {
	src  *Index
	moff []uint32 // CSR into mids, one group per masked key
	mids []int32  // orig line indices, grouped per masked key
	blob []byte   // joined masked keys
	off  []uint32 // offsets into blob
}

// BuildMaskedIndex groups the verbatim index by masked shape.
func BuildMaskedIndex(idx *Index) *MaskedIndex {
	if idx == nil {
		return nil
	}
	n := len(idx.Cnts)
	type ent struct {
		mk  string
		idx int32
	}
	ents := make([]ent, 0, n)
	for i := 0; i < n; i++ {
		mk := MaskedLine(idx.Key(i))
		if mk == "" {
			continue
		}
		ents = append(ents, ent{mk, int32(i)})
	}
	sort.Slice(ents, func(i, j int) bool { return ents[i].mk < ents[j].mk })
	mi := &MaskedIndex{src: idx}
	var blob []byte
	var prev string
	groupStart := 0
	for i, e := range ents {
		if i == 0 || e.mk != prev {
			// Rank each key's original lines by frequency so the
			// Adapt budget keeps the most attested variants.
			sort.Slice(mi.mids[groupStart:], func(a, b int) bool {
				return idx.Cnts[mi.mids[groupStart+a]] > idx.Cnts[mi.mids[groupStart+b]]
			})
			groupStart = len(mi.mids)
			mi.off = append(mi.off, uint32(len(blob)))
			blob = append(blob, e.mk...)
			mi.moff = append(mi.moff, uint32(len(mi.mids)))
			prev = e.mk
		}
		mi.mids = append(mi.mids, e.idx)
	}
	sort.Slice(mi.mids[groupStart:], func(a, b int) bool {
		return idx.Cnts[mi.mids[groupStart+a]] > idx.Cnts[mi.mids[groupStart+b]]
	})
	mi.off = append(mi.off, uint32(len(blob)))
	mi.moff = append(mi.moff, uint32(len(mi.mids)))
	mi.blob = blob
	return mi
}

// mkey returns masked key i.
func (mi *MaskedIndex) mkey(i int) string {
	return string(mi.blob[mi.off[i]:mi.off[i+1]])
}

// Adapt finds continuations for a raw line prefix by masked shape:
// lex the prefix, mask it, prefix-match against masked keys, then for
// each original line rebind its identifiers to the ones the caller
// typed and emit the renamed remainder.
func (mi *MaskedIndex) Adapt(prefix string, limit, scanCap int) []Continuation {
	if mi == nil || limit <= 0 {
		return nil
	}
	norm := Normalize(prefix)
	pt := tokenize.LexLine([]byte(norm))
	// Shape-only matching needs enough tokens to be distinctive. A
	// short masked prefix like "_ . _" matches thousands of lines and
	// the rebound continuations are noise, not signal.
	nw := 0
	for _, t := range pt {
		if strings.TrimSpace(t) != "" {
			nw++
		}
	}
	if nw < 5 {
		return nil
	}
	var out []Continuation
	seen := map[string]bool{}
	// budget bounds total original lines rebound across all masked
	// keys: hot shapes like "if _ != nil {" group thousands of lines
	// and each rebind lexes. Latency matters more than recall here.
	budget := 256
	// A keyword-shaped token in the query may be a masked identifier
	// in storage (Go code happily names variables "in", which the
	// masker treats as a Python keyword). Expand each such position.
	for _, mp := range maskedPrefixes(pt) {
		if len(mp) < 2 || budget <= 0 {
			continue
		}
		lo := sort.Search(len(mi.moff)-1, func(i int) bool { return mi.mkey(i) >= mp })
		for i := lo; i < len(mi.moff)-1 && examined(scanCap, i-lo) && budget > 0; i++ {
			mk := mi.mkey(i)
			if !strings.HasPrefix(mk, mp) {
				break
			}
			for _, oi := range mi.mids[mi.moff[i]:mi.moff[i+1]] {
				if budget <= 0 {
					break
				}
				budget--
				rest, compl, renamed, nRen := rebind(mi.src.Key(int(oi)), pt)
				// A match with no renamed identifier is a verbatim
				// hit the exact index already covers.
				if !renamed {
					continue
				}
				// Match quality: longer shape prefixes are more
				// specific; heavy renaming is weaker evidence than a
				// near-verbatim match. q lands in roughly (0, 0.85].
				q := float64(nw) / (float64(nw) + 4) / (1 + 0.15*float64(nRen))
				if len(compl) >= 2 && !seen[compl] {
					seen[compl] = true
					out = append(out, Continuation{Text: compl, Count: int(mi.src.Cnts[oi]), Qual: q})
				}
				if len(rest) >= 2 && !seen[rest] {
					seen[rest] = true
					out = append(out, Continuation{Text: rest, Count: int(mi.src.Cnts[oi]), Qual: q})
				}
			}
		}
	}
	// Rank by count descending; renames from higher-frequency lines win.
	sort.SliceStable(out, func(i, j int) bool { return out[i].Count > out[j].Count })
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

// maskedPrefixes returns the distinct masked forms of pt to probe
// with: the base mask plus variants where keyword tokens are also
// masked, capped at 16 combinations.
func maskedPrefixes(pt []string) []string {
	base := maskToks(pt)
	var amb []int
	for i, t := range pt {
		if tokenize.IsKeywordish(t) {
			amb = append(amb, i)
		}
	}
	if len(amb) == 0 {
		return []string{strings.Join(base, "")}
	}
	if len(amb) > 4 {
		amb = amb[:4]
	}
	out := make([]string, 0, 1<<len(amb))
	seen := map[string]bool{}
	for m := 0; m < (1 << len(amb)); m++ {
		v := make([]string, len(base))
		copy(v, base)
		for j, pos := range amb {
			if m&(1<<j) != 0 {
				v[pos] = identMark
			}
		}
		s := strings.Join(v, "")
		if !seen[s] {
			seen[s] = true
			out = append(out, s)
		}
	}
	return out
}

// Rebind applies positional identifier renaming to one normalized
// line: align the typed prefix's tokens with the line's, map stored
// identifiers to the typed ones, and return the renamed remainder.
// "" means the shapes diverge or nothing remains.
func Rebind(origNorm, prefixNorm string) string {
	pt := tokenize.LexLine([]byte(Normalize(prefixNorm)))
	if len(pt) == 0 {
		return ""
	}
	rest, _, _, _ := rebind(origNorm, pt)
	return rest
}

// MatchesMasked reports whether the normalized line's masked shape
// starts with any masked variant of prefixNorm.
func MatchesMasked(origNorm, prefixNorm string) bool {
	pt := tokenize.LexLine([]byte(Normalize(prefixNorm)))
	if len(pt) == 0 {
		return false
	}
	mo := MaskedLine(origNorm)
	for _, mp := range maskedPrefixes(pt) {
		if mp != "" && strings.HasPrefix(mo, mp) {
			return true
		}
	}
	return false
}

func examined(cap, n int) bool { return cap <= 0 || n < cap }

// rebind aligns the typed prefix tokens with the original line's
// tokens, maps each stored identifier to the typed identifier at the
// same position, and returns the renamed remainder of the line.
//
// The second return handles a trailing partial identifier: when the
// last typed token is a strict prefix of the stored one ("Ge" over
// "Get"), compl completes the stored name before the remainder.
// "" means the alignment fails. renamed reports whether any stored
// identifier actually changed: a match that renames nothing adds no
// information over verbatim retrieval. nRen counts the distinct
// stored identifiers remapped in the prefix.
func rebind(origLine string, pt []string) (rest, compl string, renamed bool, nRen int) {
	ot := tokenize.LexLine([]byte(origLine))
	if len(ot) <= len(pt) {
		return "", "", false, 0
	}
	rename := map[string]string{}
	for i, t := range pt {
		o := ot[i]
		if isMaskable(o) {
			if looksIdent(t) {
				if t != o {
					rename[o] = t
					renamed = true
				}
				continue
			}
			// Typed a non-identifier where the stored line had one:
			// shapes diverge, skip this line.
			return "", "", false, 0
		}
		if o != t {
			// Non-identifier tokens must match exactly: a different
			// operator or literal means a different line shape.
			return "", "", false, 0
		}
	}
	var b strings.Builder
	renamedInRest := 0
	for _, o := range ot[len(pt):] {
		if r, ok := rename[o]; ok {
			b.WriteString(r)
			renamedInRest++
		} else {
			b.WriteString(o)
		}
	}
	rest = b.String()
	// Partial tail: typed "Ge", stored "Get". Offer the stored name
	// completed, with the remainder still rebound.
	if last := pt[len(pt)-1]; looksIdent(last) {
		if o := ot[len(pt)-1]; isMaskable(o) && len(o) > len(last) &&
			strings.HasPrefix(o, last) {
			compl = o[len(last):] + rest
		}
	}
	// Adaptation earns its place only when the rebound name actually
	// appears in the continuation. A rest with no renamed identifier
	// is a shape continuation the token model already covers; emitting
	// it as a retrieval hit just competes with that.
	if renamedInRest == 0 {
		rest = ""
	}
	return rest, compl, renamed, len(rename)
}

// looksIdent reports whether t is identifier-shaped, including
// partial trailing tokens (the cursor may sit mid-identifier).
func looksIdent(t string) bool {
	if t == "" {
		return false
	}
	c := t[0]
	if !(c == '_' || c == '$' ||
		(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80) {
		return false
	}
	for i := 1; i < len(t); i++ {
		c := t[i]
		if !(c == '_' || c == '$' ||
			(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
			(c >= '0' && c <= '9') || c >= 0x80) {
			return false
		}
	}
	return true
}
