package engine

import (
	"strings"

	"q4tab/internal/tokenize"
)

// filter.go: candidate plausibility. Retrieval layers occasionally
// emit text that cannot sit at the cursor: prose that leaked into the
// line index, struct-tag fragments, a second "{" right after the
// opener. These checks reject what the token stream alone cannot.

// scalarTypes cannot begin an expression in operand position unless
// they are conversions (immediately followed by "(").
var scalarTypes = map[string]bool{
	"int": true, "int8": true, "int16": true, "int32": true, "int64": true,
	"uint": true, "uint8": true, "uint16": true, "uint32": true, "uint64": true,
	"uintptr": true, "float32": true, "float64": true,
	"complex64": true, "complex128": true,
	"string": true, "bool": true, "byte": true, "rune": true,
	"error": true, "any": true, "comparable": true,
}

// firstNonWS returns the first non-space byte of s and its index.
func firstNonWS(s string) (byte, int) {
	for i := 0; i < len(s); i++ {
		if c := s[i]; c != ' ' && c != '\t' && c != '\n' {
			return c, i
		}
	}
	return 0, -1
}

// exprTail reports the last significant byte of the line prefix: the
// context the candidate lands in.
func exprTail(linePrefix string) byte {
	t := strings.TrimRight(linePrefix, " \t")
	if t == "" {
		return 0
	}
	return t[len(t)-1]
}

// langNever lists whole-token markers that cannot appear in the
// given language's code. The corpus is multi-language and retrieval
// is lang-agnostic, so a Go idiom can surface inside a Python file:
// these markers veto what is unambiguously foreign. Kept minimal on
// purpose: under-rejecting is better than killing a legit line.
var langNever = map[string][]string{
	"python":     {"func", "defer", "chan"},
	"go":         {"def", "elif", "None", "self."},
	"javascript": {"elif", "def", "func", "chan", "defer"},
	"typescript": {"elif", "def", "func", "chan", "defer"},
	"java":       {"def", "elif", "self."},
	"c":          {"def", "elif", "self."},
	"cpp":        {"def", "elif"},
}

// langSubNever lists substrings (not whole tokens) that are invalid
// in the language.
var langSubNever = map[string][]string{
	"python":     {":=", "};"},
	"go":         {"=>", "->"},
	"javascript": {":=", "<-"},
	"typescript": {":=", "<-"},
	"rust":       {":="},
	"java":       {":=", "<-"},
	"c":          {":="},
	"cpp":        {":="},
}

// foreignLine reports whether the first line of cand carries a marker
// that cannot be valid in lang. lang "" or an unlisted language
// passes everything.
func foreignLine(cand, lang string) bool {
	first := cand
	if i := strings.IndexByte(first, '\n'); i >= 0 {
		first = first[:i]
	}
	for _, m := range langNever[lang] {
		if strings.HasSuffix(m, ".") {
			if strings.Contains(first, m) {
				return true
			}
			continue
		}
		if indexTokBounded(first, m) >= 0 {
			return true
		}
	}
	for _, s := range langSubNever[lang] {
		idx := strings.Index(first, s)
		if idx < 0 {
			continue
		}
		if s == ":=" && lang == "python" {
			// Walrus is legal Python only inside an enclosing
			// expression. The := is foreign when the text before it
			// opens no bracket and starts no comprehension or
			// conditional.
			head, tl := first[:idx], strings.TrimLeft(first, " \t")
			if !strings.ContainsAny(head, "([{") &&
				!strings.HasPrefix(tl, "if ") && !strings.HasPrefix(tl, "while ") &&
				!strings.HasPrefix(tl, "elif ") && !strings.HasPrefix(tl, "for ") &&
				!strings.HasPrefix(tl, "return ") && !strings.HasPrefix(tl, "assert ") {
				return true
			}
			continue
		}
		return true
	}
	return false
}

// plausible rejects candidates that cannot extend the line at the
// cursor. linePrefix is the text on the current line before it; lang
// scopes the cross-language veto.
func plausible(cand, linePrefix, lang string) bool {
	// Mid-token cursor: a candidate extending the fragment may not
	// land a second bare identifier right after it. lis + "a i := 0"
	// produces the invalid "lisa i"; urn + " nil" produces the legal
	// "return nil" because the completed token is a keyword. Inside a
	// string literal the fragment is content, not code.
	if frag := trailingIdent(linePrefix); frag != "" && !inString(linePrefix) &&
		len(cand) > 0 && isIdentStart(cand[0]) {
		j := 0
		for j < len(cand) && isIdentByte(cand[j]) {
			j++
		}
		if !tokenize.IsKeywordish(frag+cand[:j]) &&
			j+1 < len(cand) && cand[j] == ' ' && isIdentByte(cand[j+1]) {
			return false
		}
	}
	c, ci := firstNonWS(cand)
	if ci < 0 {
		return false
	}
	last := exprTail(linePrefix)

	// Cross-language bleed: a candidate that cannot exist in this
	// file's language is dropped regardless of retrieval strength.
	if foreignLine(cand, lang) {
		return false
	}

	// After a dot only a member name (or a type assertion paren) is
	// legal. This is the strongest filter: dot-position junk was the
	// loudest failure mode in driving tests.
	if last == '.' {
		if c == '(' {
			// Type assertion x.(T): the paren must hold a type.
			ci++
			for ci < len(cand) && (cand[ci] == ' ' || cand[ci] == '\t') {
				ci++
			}
			if ci >= len(cand) || !isIdentStart(cand[ci]) {
				return false
			}
			return true
		}
		if !isIdentStart(c) {
			return false
		}
		// A keyword cannot be a member name.
		j := ci
		for j < len(cand) && isIdentByte(cand[j]) {
			j++
		}
		if tokenize.IsKeywordish(cand[ci:j]) {
			return false
		}
		// A member continuation is one expression: statement
		// separators and bare block braces on the first line mean
		// retrieval glue, not a member.
		first := cand
		if nl := strings.IndexByte(first, '\n'); nl >= 0 {
			first = first[:nl]
		}
		if strings.ContainsAny(first, ";{}") {
			return false
		}
	}

	// An opening brace right after an opening brace is retrieval
	// noise: the corpus line continued with a nested block or the
	// mask collapsed a literal. A bare "{" interior line is the same
	// shape one level down.
	if last == '{' && c == '{' {
		return false
	}
	if strings.IndexByte(cand, '\n') >= 0 {
		for _, l := range strings.Split(cand, "\n")[1:] {
			// A lone opener inside the suggestion is an unclosed
			// block glued by the chain decode; a lone closer is how
			// blocks legitimately end.
			if strings.TrimSpace(l) == "{" {
				return false
			}
		}
	}

	// Trailing comment prose inside a multi-line candidate:
	// "} // The completed {" glued two corpus artifacts together.
	// Only mid-line comments count; a comment-first line is legit.
	for _, marker := range []string{"//", "#"} {
		k := strings.Index(cand, marker+" ")
		if k <= 0 {
			continue
		}
		rest := strings.TrimLeft(cand[k+len(marker):], " \t")
		if rest == "" || rest[0] < 'A' || rest[0] > 'Z' {
			continue
		}
		w := 0
		for w < len(rest) && isIdentByte(rest[w]) {
			w++
		}
		if w+1 < len(rest) && rest[w] == ' ' && rest[w+1] >= 'a' && rest[w+1] <= 'z' {
			return false
		}
	}

	// Prose leak: a capitalized word followed by a lowercase word is
	// a sentence fragment, not code. Ident-start capital then space
	// then lowercase covers "The completed", "No such" etc.
	if c >= 'A' && c <= 'Z' {
		j := ci
		for j < len(cand) && isIdentByte(cand[j]) {
			j++
		}
		if j < len(cand) && (cand[j] == ' ' || cand[j] == '\t') {
			k := j
			for k < len(cand) && (cand[k] == ' ' || cand[k] == '\t') {
				k++
			}
			if k < len(cand) && cand[k] >= 'a' && cand[k] <= 'z' {
				return false
			}
		}
	}

	// Expression contexts (after ( , . = : operators or keywords like
	// return): scalar type keywords and struct-tag fragments are
	// declaration material, not operands.
	switch last {
	case '(', ',', '.', '=', ':', '[', '!', '&', '|', '+', '-', '*', '/', '%', '^', '<', '>', '~':
		if strings.IndexByte(cand, '`') >= 0 {
			return false
		}
		// A leading dot is member-access residue, never an operand.
		if c == '.' {
			return false
		}
		if isIdentStart(c) {
			j := ci
			for j < len(cand) && isIdentByte(cand[j]) {
				j++
			}
			// "x :=" is a declaration, never an operand: kills
			// loop-header junk landing after <- or inside calls.
			if strings.HasPrefix(strings.TrimLeft(cand[j:], " \t"), ":=") {
				return false
			}
			if scalarTypes[cand[ci:j]] {
				// Allowed only as a conversion: int(x) not int `tag`.
				k, _ := firstNonWS(cand[j:])
				if k != '(' {
					return false
				}
			}
		}
	}
	return true
}

// argSynthesis produces fallback candidates for argument position:
// scoped error sentinels inside errors.Is/As-style calls, element-typed
// vars for append(coll, and string literals already present in the
// document for calls that take one. All fire only in call-arg position
// (line ends with "(" or ",").
func argSynthesis(linePrefix string, scope, sessDecls, docDecls map[string]bool, lits []string, pf, sf *facts) []string {
	last := exprTail(linePrefix)
	if last != '(' && last != ',' {
		return nil
	}
	var out []string

	// Find the call head: the ident before the enclosing "(", plus the
	// text of the first argument when the cursor sits at the second.
	toks := tokenize.LexLine([]byte(strings.TrimRight(linePrefix, " \t")))
	head := ""
	depth := 0
	parenAt := -1
	for i := len(toks) - 1; i >= 0; i-- {
		t := toks[i]
		switch t {
		case ")":
			depth++
		case "(":
			if depth == 0 {
				parenAt = i
				j := i - 1
				for j >= 0 && strings.TrimSpace(toks[j]) == "" {
					j--
				}
				if j >= 0 {
					head = toks[j]
				}
				i = -1
			} else {
				depth--
			}
		}
	}
	var firstArg string
	if parenAt >= 0 && last == ',' {
		var b strings.Builder
		for _, t := range toks[parenAt+1:] {
			if t == "," {
				break
			}
			if strings.TrimSpace(t) != "" {
				b.WriteString(t)
			}
		}
		firstArg = b.String()
	}

	// errors.Is(err, / errors.As(err, / a.Equal( style predicates
	// want a sentinel. Offer doc-declared error names. Head must end
	// in a predicate-ish suffix; Error itself is a printer name.
	if strings.HasSuffix(head, "Is") || strings.HasSuffix(head, "As") ||
		strings.HasSuffix(head, "Equal") {
		n := 0
		for _, set := range []map[string]bool{scope, docDecls, sessDecls} {
			for id := range set {
				if len(id) < 3 || id == head {
					continue
				}
				if strings.HasPrefix(id, "Err") || strings.HasSuffix(id, "Error") {
					c := " " + id + ")"
					dup := false
					for _, o := range out {
						if o == c {
							dup = true
						}
					}
					if !dup {
						out = append(out, c)
						n++
					}
					if n >= 3 {
						break
					}
				}
			}
		}
	}

	// append(coll, wants an element of coll's type: a scoped var of
	// that element type is the likeliest second argument.
	if head == "append" && firstArg != "" {
		if elemT := argElemType(firstArg, pf, sf); elemT != "" {
			n := 0
			for _, f := range []*facts{pf, sf} {
				if f == nil {
					continue
				}
				for v, t := range f.recv {
					if t != elemT || v == firstArg {
						continue
					}
					// Pointer vars go in as *v: an []Entry wants the
					// element, and e *Entry dereferences to it.
					c := " " + v + ")"
					if f.ptr[v] {
						c = " *" + v + ")"
					}
					dup := false
					for _, o := range out {
						if o == c {
							dup = true
						}
					}
					if !dup {
						out = append(out, c)
						n++
					}
					if n >= 2 {
						break
					}
				}
			}
		}
	}

	// String-taking calls get the file's own literals: a path or a
	// format string written earlier is the likeliest argument.
	for i, l := range lits {
		if i >= 2 {
			break
		}
		out = append(out, " "+l+")")
	}
	return out
}

// docLiterals collects distinct quoted literals from the document
// prefix, nearest the cursor first: the path or format string just
// above the line being written is the likeliest argument.
func docLiterals(prefix string, limit int) []string {
	var out []string
	seen := map[string]bool{}
	for i := len(prefix) - 1; i >= 0; i-- {
		q := prefix[i]
		if q != '"' && q != '\'' {
			continue
		}
		// Found a closing quote: walk back to its opener, skipping
		// backslash-escaped quotes inside the literal.
		j := i - 1
		for j >= 0 && prefix[j] != '\n' {
			if prefix[j] == q && (j == 0 || prefix[j-1] != '\\') {
				break
			}
			j--
		}
		if j < 0 || prefix[j] != q {
			continue
		}
		lit := prefix[j : i+1]
		i = j
		if len(lit) > 2 && len(lit) <= 60 && !seen[lit] {
			seen[lit] = true
			out = append(out, lit)
			if len(out) >= limit {
				return out
			}
		}
	}
	return out
}

// trailingIdent returns the identifier fragment ending the line
// prefix: the partial word the cursor sits inside.
func trailingIdent(linePrefix string) string {
	i := len(linePrefix)
	for i > 0 && isIdentByte(linePrefix[i-1]) {
		i--
	}
	return linePrefix[i:]
}

// inString reports whether the cursor sits inside a quoted literal:
// an unclosed quote on the line toggles string state.
func inString(linePrefix string) bool {
	open := byte(0)
	for i := 0; i < len(linePrefix); i++ {
		c := linePrefix[i]
		if open == 0 {
			if c == '"' || c == '\'' || c == '`' {
				open = c
			}
			continue
		}
		if c == '\\' && open != '`' {
			i++
			continue
		}
		if c == open {
			open = 0
		}
	}
	return open != 0
}
