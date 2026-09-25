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
	elem  map[string]string            // var -> element type of a map/slice/array decl
	tyMem map[string]map[string]bool   // type -> member ("Name(" = method, "Name" = field)
	tyFld map[string]map[string]string // type -> field name -> field type
	callM map[string]map[string]int    // func name -> member invoked on its result
	retT  map[string]string            // func or method name -> first non-builtin result type
	alias map[string]string            // type A = B / type A B where B is a named type
	ptr   map[string]bool              // vars whose declared type is a pointer: recv strips *
	decls map[string]bool              // top-level declared names (var/const/type/func)
	imps  map[string]bool              // imported module/package last segments
	sigT  map[string][]string          // func name -> ordered param types ("" when unknown)
}

func newFacts() *facts {
	return &facts{
		recv:  map[string]string{},
		elem:  map[string]string{},
		tyMem: map[string]map[string]bool{},
		tyFld: map[string]map[string]string{},
		callM: map[string]map[string]int{},
		retT:  map[string]string{},
		alias: map[string]string{},
		ptr:   map[string]bool{},
		decls: map[string]bool{},
		imps:  map[string]bool{},
		sigT:  map[string][]string{},
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
	for k, v := range o.elem {
		f.elem[k] = v
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
	for fn, t := range o.retT {
		f.retT[fn] = t
	}
	for k, v := range o.alias {
		f.alias[k] = v
	}
	for k := range o.ptr {
		f.ptr[k] = true
	}
	for d := range o.decls {
		f.decls[d] = true
	}
	for k := range o.imps {
		f.imps[k] = true
	}
	for k, v := range o.sigT {
		f.sigT[k] = v
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
						// (r *T) receivers are pointers.
						for ri := j + 1; ri < end; ri++ {
							if toks[ri] == "*" {
								f.ptr[rname] = true
							}
						}
					}
					j = nonWS(toks, end+1)
					if m := atTok(toks, j); identTok(m) &&
						atTok(toks, nonWS(toks, j+1)) == "(" {
						f.addMem(rtype, m, true)
						if pe := matchParen(toks, nonWS(toks, j+1)); pe > 0 {
							if rt := resultType(toks, pe); rt != "" {
								f.retT[m] = rt
							}
						}
						// Method params teach types the same as
						// plain func params: (s *T) m(w io.Writer).
						// sigT keys on Type.Method so the arg-slot
						// goal at s.m(a,| resolves through recv decls;
						// the bare name is kept as a fallback.
						if po := nonWS(toks, j+1); atTok(toks, po) == "(" {
							scanParamList(toks, po, f, rtype+"."+m)
						}
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
				scanParams(toks, i, f)
			}
		case "import", "use", "require", "from":
			// Record the last segment of each imported path: a file
			// that imports encoding/json should rank json.* lines
			// higher. Covers import "a/b", the import ( ... ) block
			// spanning lines, from a.b import c, and use a::b::C.
			depth := 0
			for j := i + 1; j < len(toks); j++ {
				s := toks[j]
				if s == tokenize.NL && depth == 0 {
					break
				}
				if s == "(" {
					depth++
					continue
				}
				if s == ")" {
					if depth == 0 {
						break
					}
					depth--
					continue
				}
				if strings.TrimSpace(s) == "" {
					continue
				}
				if strings.HasPrefix(s, "\"") || strings.HasPrefix(s, "'") {
					seg := strings.Trim(s, "\"'`")
					if k := strings.LastIndexAny(seg, "/\\."); k >= 0 {
						seg = seg[k+1:]
					}
					if identTok(seg) {
						f.imps[seg] = true
					}
					continue
				}
				// Dotted/qualified path: keep the trailing segment.
				if identTok(s) && !tokenize.IsKeywordish(s) {
					last := s
					for strings.HasPrefix(atTok(toks, j+1), ".") ||
						atTok(toks, j+1) == "::" {
						j += 2
						if nx := atTok(toks, j); identTok(nx) {
							last = nx
						}
					}
					f.imps[last] = true
					continue
				}
			}
		case "type":
			j := nonWS(toks, i+1)
			name := atTok(toks, j)
			if !identTok(name) {
				continue
			}
			f.decls[name] = true
			k := nonWS(toks, j+1)
			if atTok(toks, k) == "[" {
				// Generic decl: type Box[T any] struct { ... }
				if e := matchBracket(toks, k); e > 0 {
					k = nonWS(toks, e+1)
				}
			}
			switch atTok(toks, k) {
			case "struct":
				k = nonWS(toks, k+1)
				if atTok(toks, k) != "{" {
					continue
				}
				i = scanStructFields(toks, k, name, f)
			case "interface":
				// type T interface { M(...) }: methods are members.
				k = nonWS(toks, k+1)
				if atTok(toks, k) == "{" {
					scanTypeBody(toks, k+1, name, f)
				}
			case "=":
				// type A = pkg.B: alias resolution lets a. see B's
				// members through the alias.
				if a := lastTypeIdent(toks, nonWS(toks, k+1)); a != "" {
					f.alias[name] = a
				}
			}
		case "class", "interface":
			// TS/Python/Java shape: class C { m() } / class C: def m.
			j := nonWS(toks, i+1)
			if name := atTok(toks, j); identTok(name) {
				f.decls[name] = true
				scanTypeBody(toks, j+1, name, f)
			}
		case "def", "function":
			// Python def / JS-TS function: name is a decl, params may
			// carry annotations, "-> T" is the result type.
			j := nonWS(toks, i+1)
			if n := atTok(toks, j); identTok(n) {
				f.decls[n] = true
			}
			scanParams(toks, i, f)
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
				f.assignedType(v, toks, k)
				continue
			}
			isPtr := false
			if atTok(toks, k) == "*" {
				k = nonWS(toks, k+1)
				isPtr = true
			}
			ty := atTok(toks, k)
			if isPtr && identTok(ty) && !tokenize.IsKeywordish(ty) {
				f.recv[v] = ty
				f.ptr[v] = true
				continue
			}
			switch {
			case ty == "[":
				// var s []T: container, record the element type.
				if e := matchBracket(toks, k); e > 0 {
					if el := atTok(toks, nonWS(toks, e+1)); identTok(el) && !builtinType(el) {
						f.elem[v] = el
					}
				}
			case ty == "map" || ty == "chan":
				// Container decl: the element or value type is the
				// last ident on the line.
				if el := lastTypeIdent(toks, k); el != "" {
					f.elem[v] = el
				}
			case identTok(ty) && !tokenize.IsKeywordish(ty):
				f.recv[v] = ty
			default:
				// Builtin-typed vars are invisible to member lookup
				// but the arg filler needs them: var timeout int
				// feeds NewStore(ttl int).
				if identTok(ty) && builtinType(ty) {
					f.recv[v] = ty
				}
			}
		case "for":
			scanRange(toks, i, f)
		default:
			// x := NewT( / x = pkg.NewT( / x: T annotation
			if !identTok(t) {
				continue
			}
			j := nonWS(toks, i+1)
			op := atTok(toks, j)
			if op == ":" {
				// TS/Python annotation: "u: User", "x: int".
				if ty := atTok(toks, nonWS(toks, j+1)); identTok(ty) && !builtinType(ty) {
					f.recv[t] = ty
				}
				continue
			}
			if op != ":=" && op != "=" {
				continue
			}
			f.assignedType(t, toks, nonWS(toks, j+1))
		}
	}
	return f
}

// recvDecl parses "( r * T )" or "( r T )" starting at the "(" index.
// Returns the receiver name, its type, and the index of ")". Generic
// receivers like "(b *Box[T])" skip the instantiation bracket.
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
		if t == "[" {
			if cl := matchBracket(toks, i); cl > 0 {
				i = cl
				continue
			}
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

// matchBracket returns the index of the "]" matching the "[" at
// open, or -1 within a short bound.
func matchBracket(toks []string, open int) int {
	depth := 0
	for i := open; i < len(toks) && i < open+64; i++ {
		switch toks[i] {
		case "[":
			depth++
		case "]":
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// lastTypeIdent returns the last non-builtin ident before the decl
// ends (newline, "=", "{", ",", ";"): for "map[string]*Session" it
// is Session, for "chan error" it is empty.
func lastTypeIdent(toks []string, i int) string {
	s, _ := lastTypeIdentAt(toks, i)
	return s
}

// lastTypeIdentAt is lastTypeIdent plus the index where it stopped,
// so callers can advance the outer scan past the consumed type.
func lastTypeIdentAt(toks []string, i int) (string, int) {
	last := ""
	lastIdx := i
	for ; i < len(toks); i++ {
		t := toks[i]
		if t == tokenize.NL || t == "{" || t == ";" || t == "=" || t == "," {
			return last, lastIdx
		}
		if identTok(t) && !builtinType(t) {
			last = t
			lastIdx = i
		}
	}
	return last, lastIdx
}

// assignedType records the type of "v := expr" / "v = expr" starting
// at the expression token index k: NewT and other capitalized calls,
// &T{ and T{ literals, make/map/slice containers (element type),
// and JS "new T(".
func (f *facts) assignedType(v string, toks []string, k int) {
	cand := atTok(toks, k)
	switch cand {
	case "&", "new":
		if ty := atTok(toks, nonWS(toks, k+1)); identTok(ty) && !builtinType(ty) {
			f.recv[v] = ty
			f.ptr[v] = true
		}
		return
	case "[":
		if e := matchBracket(toks, k); e > 0 {
			if el := atTok(toks, nonWS(toks, e+1)); identTok(el) && !builtinType(el) {
				f.elem[v] = el
			}
		}
		return
	case "map", "make":
		if el := lastTypeIdent(toks, k); el != "" {
			f.elem[v] = el
		}
		return
	}
	// pkg.Ctor(...): skip the package qualifier.
	if k2 := nonWS(toks, k+1); atTok(toks, k2) == "." {
		k = nonWS(toks, k2+1)
		cand = atTok(toks, k)
	}
	// Generic instantiation: Ctor[T](...) / T[K]{...} — the bracket
	// holds type args, skip it to reach the call or literal.
	next := atTok(toks, nonWS(toks, k+1))
	if next == "[" {
		if e := matchBracket(toks, nonWS(toks, k+1)); e > 0 {
			next = atTok(toks, nonWS(toks, e+1))
		}
	}
	if strings.HasPrefix(cand, "New") && len(cand) > 3 && next == "(" {
		f.recv[v] = cand[3:]
		return
	}
	if cand == "" {
		return
	}
	// Any capitalized call is a constructor in most languages:
	// Session(...), Queue(), ReadCloser(...
	if cand[0] >= 'A' && cand[0] <= 'Z' && next == "(" {
		f.recv[v] = cand
		return
	}
	// x := T{ composite literal.
	if identTok(cand) && !builtinType(cand) && next == "{" {
		f.recv[v] = cand
	}
}

// resolveChain resolves a dotted ident chain against this fact set:
// head via recv/elem, each link via tyFld. Field types already hold
// the element type of container fields.
func (f *facts) resolveChain(expr []string) string {
	if len(expr) == 0 {
		return ""
	}
	typ := f.recv[expr[0]]
	if typ == "" {
		typ = f.elem[expr[0]]
	}
	for _, link := range expr[1:] {
		if typ == "" {
			return ""
		}
		nt := ""
		if row := f.tyFld[typ]; row != nil {
			nt = row[link]
		}
		if nt == "" {
			return ""
		}
		typ = nt
	}
	return typ
}

// rangeExpr resolves the ranged-over expression starting at token i
// and binds the value name to its element type.
func (f *facts) rangeExpr(toks []string, i int, names []string) {
	if len(names) == 0 {
		return
	}
	var expr []string
	lastClose := false
	for ; i < len(toks); i++ {
		t := toks[i]
		if t == tokenize.NL || t == "{" || t == ":" || t == ";" {
			break
		}
		if identTok(t) {
			expr = append(expr, t)
		} else if t == ")" {
			lastClose = true
		}
	}
	typ := ""
	if lastClose && len(expr) > 0 {
		// range f(): the result type already unwraps containers.
		typ = f.retT[expr[len(expr)-1]]
	} else {
		typ = f.resolveChain(expr)
	}
	if typ != "" {
		f.recv[names[len(names)-1]] = typ
	}
}

// scanRange handles "for k, v := range expr" and python "for x in
// expr": the value name gets the container's element type.
func scanRange(toks []string, fi int, f *facts) {
	var names []string
	for j := nonWS(toks, fi+1); j < len(toks); j++ {
		t := toks[j]
		if t == tokenize.NL || t == "{" {
			return
		}
		if t == "in" && len(names) > 0 {
			f.rangeExpr(toks, nonWS(toks, j+1), names[len(names)-1:])
			return
		}
		if t == ":=" || t == "=" {
			j = nonWS(toks, j+1)
			if atTok(toks, j) == "range" {
				j = nonWS(toks, j+1)
			}
			f.rangeExpr(toks, j, names)
			return
		}
		if identTok(t) {
			names = append(names, t)
		}
	}
}

// scanTypeBody collects member names inside a class/interface body:
// "def m(", "m(", and "name :" / "name ;" fields, for both brace
// bodies and indent bodies (stops at the next top-level class).
func scanTypeBody(toks []string, i int, name string, f *facts) {
	depth := 0
	for ; i < len(toks); i++ {
		t := toks[i]
		switch t {
		case "{":
			depth++
		case "}":
			depth--
			if depth <= 0 {
				return
			}
		case "class", "interface":
			if depth == 0 {
				return // indent-style body ended at the next decl
			}
		case "def":
			if n := atTok(toks, nonWS(toks, i+1)); identTok(n) {
				f.addMem(name, n, true)
			}
		default:
			if !identTok(t) || tokenize.IsKeywordish(t) {
				continue
			}
			switch atTok(toks, nonWS(toks, i+1)) {
			case "(":
				f.addMem(name, t, true)
			case ":", ";", ",", "?", "=":
				f.addMem(name, t, false)
			}
		}
	}
}

// matchParen returns the index of the ")" matching the "(" at open,
// or -1 when the list does not close within a sane bound. Signatures
// may span lines, so newlines do not terminate the search; the bound
// keeps incomplete code from running to EOF per decl.
func matchParen(toks []string, open int) int {
	depth := 0
	for i := open; i < len(toks) && i < open+512; i++ {
		switch toks[i] {
		case "(":
			depth++
		case ")":
			depth--
			if depth == 0 {
				return i
			}
		}
	}
	return -1
}

// builtinType reports whether s is a result ident carrying no member
// information: scalars, error, and keywords.
func builtinType(s string) bool {
	switch s {
	case "error", "string", "bool", "any", "comparable",
		"int", "int8", "int16", "int32", "int64",
		"uint", "uint8", "uint16", "uint32", "uint64", "uintptr",
		"byte", "rune", "float32", "float64", "complex64", "complex128":
		return true
	}
	return tokenize.IsKeywordish(s)
}

// resultType scans the tokens after a signature's closing ")" for
// the result list and returns the last non-builtin type ident:
// "(*Job, error)" -> Job, "*Queue" -> Queue, "error" -> "". Type
// parameters inside brackets are instantiation args, not the type.
func resultType(toks []string, close int) string {
	last := ""
	depth := 0
	for i := close + 1; i < len(toks); i++ {
		t := toks[i]
		switch t {
		case "[":
			depth++
		case "]":
			depth--
		case "{", tokenize.NL, ";":
			return last
		}
		if depth == 0 && identTok(t) && !builtinType(t) {
			last = t
		}
	}
	return last
}

// scanParams walks the parameter list of the func decl starting at
// the "func" token and records ident -> type pairs like "a *T" and
// "a, b *T", plus the function's result type into retT. Best effort:
// stops at the signature's closing paren.
func scanParams(toks []string, fi int, f *facts) {
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
	scanParamList(toks, i, f, atTok(toks, nonWS(toks, fi+1)))
}

// scanParamList walks the parameter tokens starting at the opening
// paren index i. Used for plain funcs and methods alike; name is the
// declaring function for retT, or empty to skip result recording.
func scanParamList(toks []string, i int, f *facts, name string) {
	var pending []string
	var sig []string
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
				// Untyped trailing names count as unknown slots.
				for range pending {
					sig = append(sig, "")
				}
				if name != "" {
					// Bounded: real signatures stay small, anything
					// past 32 is a misparse and just wastes space.
					if len(sig) > 32 {
						sig = sig[:32]
					}
					f.sigT[name] = sig
				}
				if rt := resultType(toks, i); rt != "" && identTok(name) {
					f.retT[name] = rt
				}
				return
			}
			pending = pending[:0]
		case ",":
			pending = pending[:0]
		case tokenize.NL, tokenize.WS:
		case "*":
			// Pointer marker before the type ident.
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
			if nt == ":" {
				// Annotation style: "u: User", "x: int". Advance i
				// past the consumed type or it re-enters as a name.
				tyi := nonWS(toks, j+1)
				ty := atTok(toks, tyi)
				if identTok(ty) && !builtinType(ty) {
					pending = append(pending, t)
					for _, p := range pending {
						f.recv[p] = ty
						sig = append(sig, ty)
					}
				} else {
					for range pending {
						sig = append(sig, "")
					}
					sig = append(sig, "")
				}
				pending = pending[:0]
				i = tyi
				continue
			}
			isElem := false
			if nt == "[" {
				// "a []T": the element type follows the bracket.
				if e := matchBracket(toks, j); e > 0 {
					j = nonWS(toks, e+1)
					nt = atTok(toks, j)
					isElem = true
				}
			}
			isPtr := false
			if nt == "*" {
				j = nonWS(toks, j+1)
				nt = atTok(toks, j)
				isPtr = true
			}
			if nt == "map" || nt == "chan" {
				// Container param: element type is the last ident
				// before the next comma or close paren.
				pending = append(pending, t)
				if el, eli := lastTypeIdentAt(toks, j); el != "" {
					for _, p := range pending {
						f.elem[p] = el
						sig = append(sig, nt+":"+el)
					}
					i = eli
				} else {
					for range pending {
						sig = append(sig, "")
					}
				}
				pending = pending[:0]
				continue
			}
			if identTok(nt) && (!tokenize.IsKeywordish(nt) || builtinType(nt)) {
				// Builtin param types count too: NewStore(ttl int)
				// must know its slot wants an int.
				pending = append(pending, t)
				for _, p := range pending {
					if isElem {
						f.elem[p] = nt
						sig = append(sig, "[]"+nt)
					} else {
						f.recv[p] = nt
						if isPtr {
							f.ptr[p] = true
							sig = append(sig, "*"+nt)
						} else {
							sig = append(sig, nt)
						}
					}
				}
				pending = pending[:0]
				i = j // consume the type token so it is not re-read
			} else {
				for range pending {
					sig = append(sig, "")
				}
				sig = append(sig, "")
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
// facts, then session facts, then corpus tables. tyMems carries the
// static tables: the base model plus any aux overlay, oldest first.
func membersFor(chain []string, indexed bool, doc, sess *facts, tyMems []map[string][]string, auxAlias map[string]string) (mems []string, local map[string]bool, corpusOnly bool) {
	if len(chain) == 0 {
		return nil, nil, false
	}
	typ := ""
	if indexed {
		// "m[k].": the head is a container; resolve the element
		// type rather than the container variable.
		typ = lookupElem(chain[0], doc, sess)
	}
	if typ == "" {
		typ = lookupRecv(chain[0], doc, sess)
	}
	if typ == "" && len(chain) == 1 {
		// Bare "x.": x may be a field of some type the file or
		// session declares ("st" inside a Server method).
		typ = fieldOwnerType(chain[0], doc, sess)
	}
	for _, link := range chain[1:] {
		if typ == "" {
			return nil, nil, false
		}
		nt := ""
		for _, f := range []*facts{doc, sess} {
			if f == nil {
				continue
			}
			if row := f.tyFld[typ]; row != nil && row[link] != "" {
				nt = row[link]
				break
			}
			// Field lookup through an alias: type A = B stores
			// B's fields under B.
			if at := f.alias[typ]; at != "" && f.tyFld[at] != nil && f.tyFld[at][link] != "" {
				nt = f.tyFld[at][link]
				break
			}
		}
		typ = nt
	}
	if typ == "" {
		return nil, nil, false
	}
	// Alias chase: collect the type AND its alias targets - an alias
	// widens the member set, it never replaces it. A corpus
	// "type Entry = X" must not redirect a session type named Entry
	// into X's members.
	types := []string{typ}
	seenT := map[string]bool{typ: true}
	for hop := 0; hop < 4 && len(types) < 8; hop++ {
		cur := types[len(types)-1]
		nt := auxAlias[cur]
		for _, f := range []*facts{doc, sess} {
			if f != nil && f.alias[cur] != "" {
				nt = f.alias[cur]
				break
			}
		}
		if nt == "" || seenT[nt] {
			break
		}
		seenT[nt] = true
		types = append(types, nt)
	}
	seen := map[string]bool{}
	var out []string
	localSet := map[string]bool{} // doc/session members: provenance for scoring
	add := func(m string) {
		if !seen[m] {
			seen[m] = true
			out = append(out, m)
		}
	}
	addLocal := func(m string) {
		localSet[m] = true
		add(m)
	}
	collect := func(t string) {
		if doc != nil {
			for m := range doc.tyMem[t] {
				addLocal(m)
			}
		}
		if sess != nil {
			for m := range sess.tyMem[t] {
				addLocal(m)
			}
		}
		for _, tab := range tyMems {
			for _, m := range tab[t] {
				add(m)
			}
		}
	}
	for _, t := range types {
		collect(t)
	}
	// Embedded fields promote: "type A struct { B }" offers B's
	// methods on a. directly. One hop is enough for ranking.
	for _, f := range []*facts{doc, sess} {
		if f == nil {
			continue
		}
		for fld, ft := range f.tyFld[typ] {
			if fld != ft {
				continue
			}
			for _, f2 := range []*facts{doc, sess} {
				if f2 == nil {
					continue
				}
				for m := range f2.tyMem[ft] {
					add(m)
				}
			}
			for _, tab := range tyMems {
				for _, m := range tab[ft] {
					add(m)
				}
			}
		}
	}
	// corpusOnly means no session facts contributed the base type.
	corpusOnly = true
	for _, f := range []*facts{doc, sess} {
		if f == nil {
			continue
		}
		for _, t := range types {
			if len(f.tyMem[t]) > 0 {
				corpusOnly = false
			}
		}
	}
	// Sort session members first among themselves, corpus after.
	var loc, corp []string
	for _, m := range out {
		if localSet[m] {
			loc = append(loc, m)
		} else {
			corp = append(corp, m)
		}
	}
	sortMembers(loc)
	sortMembers(corp)
	mems = append(loc, corp...)
	return mems, localSet, corpusOnly
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
// ahead of corpus counts. Declared result types fill the gap when
// nothing in the corpus chains a member off the call: "NewQueue()."
// resolves through "func NewQueue() *Queue" into Queue's members.
func callMembers(name string, doc, sess *facts, tyMems, callMs []map[string][]string) []string {
	seen := map[string]bool{}
	var out []string
	addRaw := func(m string) {
		if !seen[m] {
			seen[m] = true
			out = append(out, m)
		}
	}
	add := func(m string) { addRaw(m + "(") }
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
	for _, tab := range callMs {
		for _, m := range tab[name] {
			add(strings.TrimSuffix(m, "("))
		}
	}
	for _, f := range []*facts{doc, sess} {
		if f == nil {
			continue
		}
		rt := f.retT[name]
		if rt == "" {
			continue
		}
		for m := range f.tyMem[rt] {
			addRaw(m)
		}
		for _, tab := range tyMems {
			for _, m := range tab[rt] {
				addRaw(m)
			}
		}
	}
	return out
}

// argElemType resolves the element type of the container expression
// at append(coll,: a bare var goes through the elem table, a dotted
// chain through recv and field types (container fields store their
// element type already).
func argElemType(argExpr string, doc, sess *facts) string {
	argExpr = strings.TrimSpace(argExpr)
	if argExpr == "" {
		return ""
	}
	if i := strings.IndexByte(argExpr, '.'); i > 0 {
		parts := strings.Split(argExpr, ".")
		typ := lookupRecv(strings.TrimSpace(parts[0]), doc, sess)
		for _, link := range parts[1:] {
			if typ == "" {
				return ""
			}
			nt := ""
			for _, f := range []*facts{doc, sess} {
				if f != nil && f.tyFld[typ] != nil && f.tyFld[typ][strings.TrimSpace(link)] != "" {
					nt = f.tyFld[typ][strings.TrimSpace(link)]
					break
				}
			}
			typ = nt
		}
		return typ
	}
	if t := lookupElem(argExpr, doc, sess); t != "" {
		return t
	}
	return ""
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

func lookupElem(name string, doc, sess *facts) string {
	if doc != nil {
		if t := doc.elem[name]; t != "" {
			return t
		}
	}
	if sess != nil {
		return sess.elem[name]
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
// of linePrefix. Returns the ident chain for "a.b.", or for a call
// receiver "f(...)." the callee name. indexed reports that the last
// segment was an index expression ("m[k].", "jobs[0]."), which makes
// the head resolve to a container's element type.
func dotChain(linePrefix string) (chain []string, call string, indexed, ok bool) {
	tail := strings.TrimRight(linePrefix, " \t")
	if !strings.HasSuffix(tail, ".") {
		return nil, "", false, false
	}
	toks := tokenize.LexLine([]byte(tail))
	// Walk the token list backwards from the trailing dot.
	i := len(toks) - 1
	for i >= 0 && strings.TrimSpace(toks[i]) == "" {
		i--
	}
	if i < 0 || toks[i] != "." {
		return nil, "", false, false
	}
	i--
	for i >= 0 && strings.TrimSpace(toks[i]) == "" {
		i--
	}
	if i < 0 {
		return nil, "", false, false
	}
	if toks[i] == ")" {
		// Call receiver: find the matching "(" and the ident before it.
		depth := 0
		open := -1
		for j := i; j >= 0; j-- {
			if toks[j] == ")" {
				depth++
			} else if toks[j] == "(" {
				depth--
				if depth == 0 {
					open = j
					break
				}
			}
		}
		if open <= 0 {
			return nil, "", false, false
		}
		j := open - 1
		for j >= 0 && strings.TrimSpace(toks[j]) == "" {
			j--
		}
		if j < 0 || !identTok(toks[j]) {
			return nil, "", false, false
		}
		return nil, toks[j], false, true
	}
	// Ident chain: a.b.c. Index expressions ("jobs[0].") collapse to
	// their container link: field types already store the element
	// type, so "w.q.jobs[0]." resolves as w -> q -> jobs -> element.
	var rev []string
	for i >= 0 {
		t := toks[i]
		if strings.TrimSpace(t) == "" {
			i--
			continue
		}
		if t == "]" {
			if len(rev) == 0 {
				indexed = true
			}
			depth := 0
			for ; i >= 0; i-- {
				if toks[i] == "]" {
					depth++
				} else if toks[i] == "[" {
					depth--
					if depth == 0 {
						break
					}
				}
			}
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
		return nil, "", false, false
	}
	for l, r := 0, len(rev)-1; l < r; l, r = l+1, r-1 {
		rev[l], rev[r] = rev[r], rev[l]
	}
	return rev, "", indexed, true
}

// factBuilder aggregates member facts across the whole corpus at
// index time. Counts decide which members survive compaction.
type factBuilder struct {
	tyMem map[string]map[string]int // type -> member -> count
	callM map[string]map[string]int // func -> member -> count
	retT  map[string]string         // func -> declared result type
	sigT  map[string][]string       // func -> ordered param types
	alias map[string]string         // type alias -> target type
}

func newFactBuilder() *factBuilder {
	return &factBuilder{
		tyMem: map[string]map[string]int{},
		callM: map[string]map[string]int{},
		retT:  map[string]string{},
		sigT:  map[string][]string{},
		alias: map[string]string{},
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
	for fn, t := range f.retT {
		fb.retT[fn] = t
	}
	for fn, sig := range f.sigT {
		fb.sigT[fn] = sig
	}
	for a, t := range f.alias {
		fb.alias[a] = t
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
	// Constructors and accessors return a type: teach callM their
	// members at the survival floor so "NewT()." works corpus-wide
	// even where no call site chains a member directly. Observed
	// usage keeps its higher count and ranks above.
	for fn, t := range fb.retT {
		row := fb.callM[fn]
		if row == nil {
			row = map[string]int{}
			fb.callM[fn] = row
		}
		for m := range fb.tyMem[t] {
			if row[m] == 0 {
				row[m] = 2
			}
		}
	}
	// Members are declarations: each appears once per codebase, so the
	// df floor is 1. Call-site members are usage counts: a single
	// observation is noise, two is a habit.
	return top(fb.tyMem, 1, 48), top(fb.callM, 2, 12)
}
