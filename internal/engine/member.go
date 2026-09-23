package engine

import (
	"sort"
	"strings"

	"q4tab/internal/tokenize"
)

// member.go: member memory. The n-gram knows text, not symbols, so a
// bare "x." at the cursor falls back to whatever receivers were
// popular in the corpus. This layer extracts a lightweight fact set
// per document and per corpus file - which types have which members,
// which variables hold which types, which function results get which
// methods called on them - and resolves the receiver expression at
// the cursor against those facts. It is textual inference, not a
// typechecker: wrong guesses are a ranking nuisance, never a crash.

// facts is the per-source extraction result.
type facts struct {
	recv  map[string]string            // var or receiver ident -> type name
	tyMem map[string]map[string]bool   // type -> member ("Name(" = method, "Name" = field)
	tyFld map[string]map[string]string // type -> field name -> field type
	callM map[string]map[string]int    // func name -> member invoked on its result
	decls map[string]bool              // top-level declared names (var/const/type/func)
}

func newFacts() *facts {
	return &facts{
		recv:  map[string]string{},
		tyMem: map[string]map[string]bool{},
		tyFld: map[string]map[string]string{},
		callM: map[string]map[string]int{},
		decls: map[string]bool{},
	}
}

// merge folds o into f. Corpus counts carry over; identical member
// sets just stay true.
func (f *facts) merge(o *facts) {
	if o == nil {
		return
	}
	for k, v := range o.recv {
		f.recv[k] = v
	}
	for t, ms := range o.tyMem {
		m := f.tyMem[t]
		if m == nil {
			m = map[string]bool{}
			f.tyMem[t] = m
		}
		for mem := range ms {
			m[mem] = true
		}
	}
	for t, fs := range o.tyFld {
		m := f.tyFld[t]
		if m == nil {
			m = map[string]string{}
			f.tyFld[t] = m
		}
		for fld, ft := range fs {
			m[fld] = ft
		}
	}
	for fn, ms := range o.callM {
		m := f.callM[fn]
		if m == nil {
			m = map[string]int{}
			f.callM[fn] = m
		}
		for mem, n := range ms {
			m[mem] += n
		}
	}
	for d := range o.decls {
		f.decls[d] = true
	}
}

// addMem records member m on type t. meth marks method syntax.
func (f *facts) addMem(t, m string, meth bool) {
	if t == "" || m == "" || tokenize.IsKeywordish(m) {
		return
	}
	if meth {
		m += "("
	}
	row := f.tyMem[t]
	if row == nil {
		row = map[string]bool{}
		f.tyMem[t] = row
	}
	row[m] = true
}

// identTok is IsIdentTok without the length floor: receiver and
// variable names are routinely a single byte (s, r, w, x).
func identTok(t string) bool {
	return len(t) > 0 && isIdentStart(t[0])
}

// nonWS returns the index of the first non-whitespace token at or
// after i, or len(toks).
func nonWS(toks []string, i int) int {
	for i < len(toks) && strings.TrimSpace(toks[i]) == "" {
		i++
	}
	return i
}

func atTok(toks []string, i int) string {
	if i < len(toks) {
		return toks[i]
	}
	return ""
}

// extractFactsToks runs one pass over a token stream and collects the
// member-memory facts. Patterns handled:
//
//	func (r *T) M(            -> methods[T]+=M, recv[r]=T
//	func f(a *T, b T2, ...)   -> recv[param]=type
//	type T struct { f U }     -> field f of T has type U, member f
//	x := NewT( / x = pkg.NewT(-> recv[x]=T
//	var x T / var x *T        -> recv[x]=T
//	Name( ... ).M             -> callM[Name]+=M
func extractFactsToks(toks []string) *facts {
	f := newFacts()
	// callStk tracks the ident before each open paren so a ") . M"
	// sequence can be attributed to the callee.
	var callStk []string
	for i := 0; i < len(toks); i++ {
		t := toks[i]
		switch t {
		case tokenize.NL, tokenize.WS, tokenize.EOF:
			continue
		case "(":
			// Ident immediately before the opener, skipping past
			// whitespace.
			j := i - 1
			for j >= 0 && strings.TrimSpace(toks[j]) == "" {
				j--
			}
			if j >= 0 && identTok(toks[j]) {
				callStk = append(callStk, toks[j])
			} else {
				callStk = append(callStk, "")
			}
			continue
		case ")":
			var fn string
			if len(callStk) > 0 {
				fn = callStk[len(callStk)-1]
				callStk = callStk[:len(callStk)-1]
			}
			if fn != "" {
				j := nonWS(toks, i+1)
				if atTok(toks, j) == "." {
					if m := atTok(toks, nonWS(toks, j+1)); identTok(m) {
						row := f.callM[fn]
						if row == nil {
							row = map[string]int{}
							f.callM[fn] = row
						}
						row[m]++
					}
				}
			}
			continue
		case "[", "]", "{", "}", ",", ";":
			continue
		}

		switch t {
		case "func":
			j := nonWS(toks, i+1)
			if atTok(toks, j) == "(" {
				// Method: func (r *T) M(
				rname, rtype, end := recvDecl(toks, j)
				if rtype != "" {
					if rname != "" {
						f.recv[rname] = rtype
					}
					j = nonWS(toks, end+1)
					if m := atTok(toks, j); identTok(m) &&
						atTok(toks, nonWS(toks, j+1)) == "(" {
						f.addMem(rtype, m, true)
						i = j // let the "(" case record the call frame
						continue
					}
				}
			} else {
				if n := atTok(toks, j); identTok(n) {
					f.decls[n] = true
				}
				// Plain func: scan its parameter list for typed
				// names so "w http.ResponseWriter" teaches w.
				scanParams(toks, i, f.recv)
			}
		case "type":
			j := nonWS(toks, i+1)
			name := atTok(toks, j)
			if !identTok(name) {
				continue
			}
			f.decls[name] = true
			k := nonWS(toks, j+1)
			if atTok(toks, k) != "struct" {
				continue
			}
			k = nonWS(toks, k+1)
			if atTok(toks, k) != "{" {
				continue
			}
			i = scanStructFields(toks, k, name, f)
		case "var", "const":
			j := nonWS(toks, i+1)
			v := atTok(toks, j)
			if !identTok(v) {
				continue
			}
			f.decls[v] = true
			k := nonWS(toks, j+1)
			if atTok(toks, k) == "=" {
				// var x = NewT(...)
				k = nonWS(toks, k+1)
				cand := atTok(toks, k)
				if k2 := nonWS(toks, k+1); atTok(toks, k2) == "." {
					k = nonWS(toks, k2+1)
					cand = atTok(toks, k)
				}
				if strings.HasPrefix(cand, "New") && len(cand) > 3 &&
					atTok(toks, nonWS(toks, k+1)) == "(" {
					f.recv[v] = cand[3:]
				}
				continue
			}
			if atTok(toks, k) == "*" {
				k = nonWS(toks, k+1)
			}
			if ty := atTok(toks, k); identTok(ty) && !tokenize.IsKeywordish(ty) {
				f.recv[v] = ty
			}
		default:
			// x := NewT( / x = pkg.NewT( / var x = NewT(
			if identTok(t) {
				j := nonWS(toks, i+1)
				op := atTok(toks, j)
				if op != ":=" && op != "=" {
					continue
				}
				k := nonWS(toks, j+1)
				cand := atTok(toks, k)
				if k2 := nonWS(toks, k+1); atTok(toks, k2) == "." {
					k = nonWS(toks, k2+1)
					cand = atTok(toks, k)
				}
				if strings.HasPrefix(cand, "New") && len(cand) > 3 &&
					atTok(toks, nonWS(toks, k+1)) == "(" {
					f.recv[t] = cand[3:]
				}
			}
		}
	}
	return f
}

// recvDecl parses "( r * T )" or "( r T )" starting at the "(" index.
// Returns the receiver name, its type, and the index of ")".
func recvDecl(toks []string, open int) (name, typ string, end int) {
	var ids []string
	for i := open + 1; i < len(toks); i++ {
		t := toks[i]
		if t == ")" {
			end = i
			break
		}
		if t == tokenize.NL {
			return "", "", 0
		}
		if identTok(t) {
			ids = append(ids, t)
		}
	}
	if end == 0 {
		return "", "", 0
	}
	switch len(ids) {
	case 1:
		return "", ids[0], end // unnamed receiver: func (T) M(
	case 2:
		return ids[0], ids[1], end
	}
	return "", "", 0
}

// scanParams walks the parameter list of the func decl starting at
// the "func" token and records ident -> type pairs like "a *T" and
// "a, b *T". Best effort: stops at the first "{" or newline-ish
// boundary after the closing paren.
func scanParams(toks []string, fi int, recv map[string]string) {
	i := nonWS(toks, fi+1)
	if atTok(toks, i) == "(" {
		return // method decl: recvDecl handles it
	}
	// Skip the name (and generics bracket) to the parameter opener.
	for i < len(toks) && atTok(toks, i) != "(" && toks[i] != tokenize.NL {
		i++
	}
	if i >= len(toks) || atTok(toks, i) != "(" {
		return
	}
	var pending []string
	depth := 0
	for ; i < len(toks); i++ {
		t := toks[i]
		switch t {
		case "(":
			depth++
			pending = pending[:0]
		case ")":
			depth--
			if depth <= 0 {
				return
			}
			pending = pending[:0]
		case ",":
			pending = pending[:0]
		case tokenize.NL, tokenize.WS:
		case "*", "[":
			// Pointer or slice marker before the type ident.
		default:
			if !identTok(t) {
				pending = pending[:0]
				continue
			}
			// Ident followed by another ident or a *ident is a
			// name; an ident followed by , or ) joins the pending
			// name list waiting for a type.
			j := nonWS(toks, i+1)
			nt := atTok(toks, j)
			if nt == "," || nt == ")" {
				pending = append(pending, t)
				continue
			}
			if nt == "*" {
				j = nonWS(toks, j+1)
				nt = atTok(toks, j)
			}
			if identTok(nt) && !tokenize.IsKeywordish(nt) {
				pending = append(pending, t)
				for _, p := range pending {
					recv[p] = nt
				}
				pending = pending[:0]
			} else {
				pending = pending[:0]
			}
		}
	}
}

// scanStructFields consumes the body of "type T struct { ... }"
// starting at the "{" index and records each field: member name on
// T plus field -> type for chain resolution. Returns the index of
// the closing "}".
func scanStructFields(toks []string, open int, typ string, f *facts) int {
	depth := 1
	var line []string // idents on the current field line
	flush := func() {
		if len(line) == 0 {
			return
		}
		var name, ft string
		switch {
		case len(line) == 1:
			// Embedded field: "Store" or "sync.Mutex" already
			// reduced to its idents.
			name, ft = line[0], line[0]
		default:
			name, ft = line[0], line[len(line)-1]
		}
		if name != "" && !tokenize.IsKeywordish(name) {
			f.addMem(typ, name, false)
			row := f.tyFld[typ]
			if row == nil {
				row = map[string]string{}
				f.tyFld[typ] = row
			}
			row[name] = ft
		}
		line = line[:0]
	}
	for i := open + 1; i < len(toks); i++ {
		t := toks[i]
		switch t {
		case "{":
			depth++
		case "}":
			depth--
			if depth <= 0 {
				flush()
				return i
			}
		case tokenize.NL, ",", ";":
			flush()
		case tokenize.WS, tokenize.EOF:
		default:
			if identTok(t) {
				line = append(line, t)
			} else if len(t) > 0 && t[0] == '`' {
				// Struct tag: ignore; the flush already took name+type.
			}
		}
	}
	return len(toks)
}

// membersFor resolves the receiver chain ending at the cursor and
// returns member continuations for it, session facts first.
// chain is the dotted path before the trailing dot: "s.st." gives
// ["s","st"], "st." gives ["st"]. Resolution order per step: doc
// facts, then session facts, then corpus tables.
func membersFor(chain []string, doc, sess *facts, tyMem, callM map[string][]string) (mems []string, corpusOnly bool) {
	if len(chain) == 0 {
		return nil, false
	}
	typ := lookupRecv(chain[0], doc, sess)
	if typ == "" && len(chain) == 1 {
		// Bare "x.": x may be a field of some type the file or
		// session declares ("st" inside a Server method).
		typ = fieldOwnerType(chain[0], doc, sess)
	}
	for _, link := range chain[1:] {
		if typ == "" {
			return nil, false
		}
		nt := ""
		if doc != nil && doc.tyFld[typ] != nil {
			nt = doc.tyFld[typ][link]
		}
		if nt == "" && sess != nil && sess.tyFld[typ] != nil {
			nt = sess.tyFld[typ][link]
		}
		typ = nt
	}
	if typ == "" {
		return nil, false
	}
	seen := map[string]bool{}
	var out []string
	add := func(m string) {
		if !seen[m] {
			seen[m] = true
			out = append(out, m)
		}
	}
	if doc != nil {
		for m := range doc.tyMem[typ] {
			add(m)
		}
	}
	if sess != nil {
		for m := range sess.tyMem[typ] {
			add(m)
		}
	}
	nSess := len(out)
	for _, m := range tyMem[typ] {
		add(m)
	}
	sortMembers(out)
	return out, nSess == 0 && len(out) > 0
}

// sortMembers orders member continuations: method calls first (they
// dominate dot completion), then fields, alphabetical within each.
func sortMembers(mems []string) {
	sort.Slice(mems, func(i, j int) bool {
		mi, mj := strings.HasSuffix(mems[i], "("), strings.HasSuffix(mems[j], "(")
		if mi != mj {
			return mi
		}
		return mems[i] < mems[j]
	})
}

// callMembers returns members observed on the result of function
// name: "NewDecoder(...)." -> Decode(. Doc and session facts rank
// ahead of corpus counts.
func callMembers(name string, doc, sess *facts, callM map[string][]string) []string {
	seen := map[string]bool{}
	var out []string
	add := func(m string) {
		if !seen[m] {
			seen[m] = true
			out = append(out, m+"(")
		}
	}
	if doc != nil {
		for m := range doc.callM[name] {
			add(m)
		}
	}
	if sess != nil {
		for m := range sess.callM[name] {
			add(m)
		}
	}
	for _, m := range callM[name] {
		add(strings.TrimSuffix(m, "("))
	}
	return out
}

func lookupRecv(name string, doc, sess *facts) string {
	if doc != nil {
		if t := doc.recv[name]; t != "" {
			return t
		}
	}
	if sess != nil {
		return sess.recv[name]
	}
	return ""
}

// fieldOwnerType finds the type of a field named ident on any type
// declared in doc or session facts. Used for bare "st." when the
// receiver chain has no explicit head type.
func fieldOwnerType(ident string, doc, sess *facts) string {
	for _, f := range []*facts{doc, sess} {
		if f == nil {
			continue
		}
		for _, fields := range f.tyFld {
			if t := fields[ident]; t != "" {
				return t
			}
		}
	}
	return ""
}

// dotChain parses the receiver expression ending at the trailing dot
// of linePrefix. Returns the ident chain for "a.b." or, for a call
// receiver "f(...).", the callee name with called=true.
func dotChain(linePrefix string) (chain []string, call string, ok bool) {
	tail := strings.TrimRight(linePrefix, " \t")
	if !strings.HasSuffix(tail, ".") {
		return nil, "", false
	}
	toks := tokenize.LexLine([]byte(tail))
	// Walk the token list backwards from the trailing dot.
	i := len(toks) - 1
	for i >= 0 && strings.TrimSpace(toks[i]) == "" {
		i--
	}
	if i < 0 || toks[i] != "." {
		return nil, "", false
	}
	i--
	for i >= 0 && strings.TrimSpace(toks[i]) == "" {
		i--
	}
	if i < 0 {
		return nil, "", false
	}
	if toks[i] == ")" {
		// Call receiver: find the matching "(" and the ident before it.
		depth := 0
		open := -1
		for j := i; j >= 0; j-- {
			switch toks[j] {
			case ")":
				depth++
			case "(":
				depth--
				if depth == 0 {
					open = j
					j = -1
				}
			}
		}
		if open <= 0 {
			return nil, "", false
		}
		j := open - 1
		for j >= 0 && strings.TrimSpace(toks[j]) == "" {
			j--
		}
		if j < 0 || !identTok(toks[j]) {
			return nil, "", false
		}
		return nil, toks[j], true
	}
	// Ident chain: a.b.c.
	var rev []string
	for i >= 0 {
		t := toks[i]
		if strings.TrimSpace(t) == "" {
			i--
			continue
		}
		if identTok(t) {
			rev = append(rev, t)
			i--
			for i >= 0 && strings.TrimSpace(toks[i]) == "" {
				i--
			}
			if i >= 0 && toks[i] == "." {
				i--
				continue
			}
		}
		break
	}
	if len(rev) == 0 {
		return nil, "", false
	}
	for l, r := 0, len(rev)-1; l < r; l, r = l+1, r-1 {
		rev[l], rev[r] = rev[r], rev[l]
	}
	return rev, "", true
}

// factBuilder aggregates member facts across the whole corpus at
// index time. Counts decide which members survive compaction.
type factBuilder struct {
	tyMem map[string]map[string]int // type -> member -> count
	callM map[string]map[string]int // func -> member -> count
}

func newFactBuilder() *factBuilder {
	return &factBuilder{
		tyMem: map[string]map[string]int{},
		callM: map[string]map[string]int{},
	}
}

func (fb *factBuilder) Add(toks []string) {
	f := extractFactsToks(toks)
	for t, ms := range f.tyMem {
		row := fb.tyMem[t]
		if row == nil {
			row = map[string]int{}
			fb.tyMem[t] = row
		}
		for m := range ms {
			row[m]++
		}
	}
	for fn, ms := range f.callM {
		row := fb.callM[fn]
		if row == nil {
			row = map[string]int{}
			fb.callM[fn] = row
		}
		for m, n := range ms {
			row[m] += n
		}
	}
}

// Compact emits type -> members and func -> members tables, keeping
// members seen often enough to be conventions and capping each row.
func (fb *factBuilder) Compact() (map[string][]string, map[string][]string) {
	top := func(m map[string]map[string]int, minCnt, rowCap int) map[string][]string {
		out := make(map[string][]string, len(m))
		for k, row := range m {
			type ent struct {
				s string
				n int
			}
			var es []ent
			for s, n := range row {
				if n >= minCnt {
					es = append(es, ent{s, n})
				}
			}
			sort.Slice(es, func(i, j int) bool {
				if es[i].n != es[j].n {
					return es[i].n > es[j].n
				}
				return es[i].s < es[j].s
			})
			if len(es) > rowCap {
				es = es[:rowCap]
			}
			if len(es) == 0 {
				continue
			}
			ls := make([]string, len(es))
			for i, e := range es {
				ls[i] = e.s
			}
			out[k] = ls
		}
		return out
	}
	// Members are declarations: each appears once per codebase, so the
	// df floor is 1. Call-site members are usage counts: a single
	// observation is noise, two is a habit.
	return top(fb.tyMem, 1, 48), top(fb.callM, 2, 12)
}
