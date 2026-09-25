package engine

import (
	"strings"
	"time"

	"q4tab/internal/tokenize"
)

// edits.go: edit-history reasoning. Every accepted-completion engine
// watches keystrokes; q4tab watches what they changed. A didChange
// diff yields rewrite rules ("items" became "jobs"), and those rules
// fire two ways at completion time:
//
//   - propagation: a candidate still using the old side of a recent
//     edit gets a variant rewritten to the new side. Rename a field
//     once and every stale call site suggestion follows.
//   - working set: identifiers the user just touched are likelier
//     to appear in the next line than cold ones.
//
// Rules are token sequences, so application never rewrites inside a
// larger identifier.

// editRule is one observed rewrite: the token window "from" was
// replaced by "to" in a recent document edit.
type editRule struct {
	from, to   []string
	fromS, toS string // joined, for a cheap Contains pre-check
	at         time.Time
}

// diffEdits compares the previous and current document text and
// extracts rewrite rules from changed line pairs. Positional pairing
// covers the common case (edits rarely change line count much); when
// counts differ, the shared prefix/suffix is aligned and the interior
// pairs by position within the delta window.
func diffEdits(oldText, newText string, maxRules int) []editRule {
	if oldText == newText || oldText == "" || newText == "" {
		return nil
	}
	ol := strings.Split(oldText, "\n")
	nl := strings.Split(newText, "\n")

	// Trim shared prefix and suffix so pairs align inside the
	// changed window.
	lo := 0
	for lo < len(ol) && lo < len(nl) && ol[lo] == nl[lo] {
		lo++
	}
	hi := 0
	for hi < len(ol)-lo && hi < len(nl)-lo &&
		ol[len(ol)-1-hi] == nl[len(nl)-1-hi] {
		hi++
	}
	om := ol[lo : len(ol)-hi]
	nm := nl[lo : len(nl)-hi]
	var out []editRule
	n := len(om)
	if len(nm) < n {
		n = len(nm)
	}
	for i := 0; i < n && len(out) < maxRules; i++ {
		if r := lineRule(om[i], nm[i]); r != nil {
			out = append(out, *r)
		}
	}
	return out
}

// lineRule extracts a single rewrite rule from a changed line pair:
// the maximal common token prefix and suffix are stripped, and the
// remaining middle spans become the rule. Pure inserts (empty from)
// and deletes (empty to) are kept; both sides are capped so rules
// stay small substitutions, not whole-line rewrites.
func lineRule(o, n string) *editRule {
	if o == n {
		return nil
	}
	ot := compactToks(tokenize.LexLine([]byte(o)))
	nt := compactToks(tokenize.LexLine([]byte(n)))
	if len(ot) == 0 && len(nt) == 0 {
		return nil
	}
	pre := 0
	for pre < len(ot) && pre < len(nt) && ot[pre] == nt[pre] {
		pre++
	}
	suf := 0
	for suf < len(ot)-pre && suf < len(nt)-pre &&
		ot[len(ot)-1-suf] == nt[len(nt)-1-suf] {
		suf++
	}
	from := ot[pre : len(ot)-suf]
	to := nt[pre : len(nt)-suf]
	// Whole-line rewrites carry no transferable pattern.
	if len(from) > 8 || len(to) > 8 || (len(from) == 0 && len(to) == 0) {
		return nil
	}
	if len(from) == 0 && len(to) > 4 {
		return nil // long inserts are context, not a rewrite
	}
	fs, ts := strings.Join(from, ""), strings.Join(to, "")
	if fs == ts {
		return nil
	}
	return &editRule{from: from, to: to, fromS: fs, toS: ts, at: time.Now()}
}

// compactToks drops whitespace tokens from a lexed line.
func compactToks(toks []string) []string {
	out := toks[:0]
	for _, t := range toks {
		if strings.TrimSpace(t) != "" {
			out = append(out, t)
		}
	}
	return out
}

// apply returns cand with the rule's "from" text swapped for "to" at
// the first token-bounded occurrence, or false when absent. Splices
// the raw text so candidate formatting is preserved.
func (r *editRule) apply(cand string) (string, bool) {
	if len(r.from) == 0 || r.fromS == "" {
		return "", false
	}
	i := indexTokBounded(cand, r.fromS)
	if i < 0 {
		return "", false
	}
	return cand[:i] + r.toS + cand[i+len(r.fromS):], true
}

// indexTokBounded finds needle in s such that it does not sit inside
// a larger identifier.
func indexTokBounded(s, needle string) int {
	for off := 0; off <= len(s)-len(needle); {
		i := strings.Index(s[off:], needle)
		if i < 0 {
			return -1
		}
		i += off
		if (i == 0 || !isIdentByte(s[i-1])) &&
			(i+len(needle) >= len(s) || !isIdentByte(s[i+len(needle)])) {
			return i
		}
		off = i + 1
	}
	return -1
}

// touches reports whether cand mentions the rule's new side: the
// working-set signal.
func (r *editRule) touches(cand string) bool {
	return r.toS != "" && strings.Contains(cand, r.toS)
}

// NextEditHint is a predicted next edit site: a line still carrying
// the old side of a recent rewrite rule, plus the replacement text.
type NextEditHint struct {
	URI  string `json:"uri"`
	Line int    `json:"line"`
	Char int    `json:"char"`
	Old  string `json:"old"`
	New  string `json:"new"`
}

// NextEdit returns the most likely next edit sites: locations where a
// recent rewrite rule's old side still appears. This is the
// next-edit half of the edit-rule machinery: propagation rewrites
// suggestions, prediction locates the places worth visiting. Rules
// fire newest-first across every open document.
func (e *Engine) NextEdit(limit int) []NextEditHint {
	if limit <= 0 {
		limit = 8
	}
	e.emu.Lock()
	rules := append([]editRule(nil), e.edits...)
	e.emu.Unlock()
	var out []NextEditHint
	type seen struct {
		uri  string
		line int
	}
	dedup := map[seen]bool{}
	e.docs.Range(func(k, v any) bool {
		d := v.(*doc)
		d.mu.Lock()
		text := d.prev
		d.mu.Unlock()
		if text == "" {
			return true
		}
		docLines := strings.Split(text, "\n")
		// Newest rules are the freshest intent; walk back to front.
		for ri := len(rules) - 1; ri >= 0 && len(out) < limit; ri-- {
			r := rules[ri]
			if r.fromS == "" || r.fromS == r.toS {
				continue
			}
			for li, l := range docLines {
				col := strings.Index(l, r.fromS)
				if col < 0 {
					continue
				}
				// A line already carrying the new side is done.
				if r.toS != "" && strings.Contains(l, r.toS) {
					continue
				}
				key := seen{k.(string), li}
				if dedup[key] {
					continue
				}
				dedup[key] = true
				out = append(out, NextEditHint{
					URI:  key.uri,
					Line: li,
					Char: col,
					Old:  r.fromS,
					New:  strings.Replace(l, r.fromS, r.toS, 1),
				})
				if len(out) >= limit {
					break
				}
			}
		}
		return len(out) < limit
	})
	return out
}
