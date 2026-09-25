package engine

// synth.go: type-directed expression assembly. When the cursor sits at
// a position whose required type is knowable - var decl, assignment,
// return, call argument, struct field - we compose the expression
// from available producers instead of guessing tokens: in-scope vars
// of the type, constructor calls whose result is the type, field
// paths that reach it, pointer adapts, interface satisfaction, and
// zero values. It is assemblage, not invention: it only ever emits
// code that typechecks against facts the extraction recorded.
//
// Depth is one producer hop plus arg filling from scope vars. The
// search never leaves the session/corpus fact tables, so worst case
// is a bounded scan over types, not a solver.

import (
	"strings"

	"q4tab/internal/tokenize"
)

// goalT is the type a cursor position demands.
type goalT struct {
	name string // base ident: Store, error, io.Reader's Reader
	ptr  bool   // *T
	slc  bool   // []T
	lhs  string // the var being assigned; never offered to itself
}

// funcOf scans prefix text backward for the innermost enclosing func
// decl and returns its name, so "return |" can ask retT for the goal.
func funcOf(toks []string) string {
	name := ""
	for i := 0; i < len(toks); i++ {
		if toks[i] != "func" {
			continue
		}
		j := nonWS(toks, i+1)
		if atTok(toks, j) == "(" {
			// Method: skip the receiver, take the name.
			if e := matchParen(toks, j); e > 0 {
				j = nonWS(toks, e+1)
			}
		}
		if n := atTok(toks, j); identTok(n) && n != "" && !tokenize.IsKeywordish(n) {
			name = n
		}
	}
	return name
}

// goalAt determines what type the cursor position requires.
// toks is the tokenized prefix, line the current line text.
func goalAt(line string, toks []string, pf, sf *facts) goalT {
	g := goalT{}
	lp := strings.TrimRight(line, " \t")
	if lp == "" {
		return g
	}

	// Statement-tail dispatch on the trimmed line.
	switch {
	case strings.HasSuffix(lp, "return"), strings.HasSuffix(lp, "return "):
		// Goal = enclosing function's declared result.
		if fn := funcOf(toks); fn != "" {
			for _, f := range []*facts{pf, sf} {
				if f != nil && f.retT[fn] != "" {
					g.name = f.retT[fn]
					break
				}
			}
		}
	case strings.HasSuffix(lp, "="), strings.HasSuffix(lp, ":="):
		lhs := strings.TrimSpace(strings.TrimRight(lp, "=:"))
		if strings.HasPrefix(lhs, "var ") {
			// "var x T =": the trailing ident IS the goal type; the
			// name is the ident before any * marker.
			rest := strings.TrimSpace(lhs[3:])
			ty := lastIdent(rest)
			if ty != "" && !tokenize.IsKeywordish(ty) {
				g.name = ty
				head := strings.TrimRight(rest[:len(rest)-len(ty)], "* \t")
				g.lhs = lastIdent(head)
				g.ptr = strings.HasSuffix(head, "*") ||
					strings.HasSuffix(rest[:len(rest)-len(ty)], "*")
			}
		} else if id := lastIdent(strings.TrimRight(lhs, "* \t")); id != "" {
			g.lhs = id
			// "x =": look up x's recorded type.
			for _, f := range []*facts{pf, sf} {
				if f == nil {
					continue
				}
				if t := f.recv[id]; t != "" {
					g.name = t
					g.ptr = f.ptr[id]
					break
				}
				if el := f.elem[id]; el != "" {
					g.name, g.slc = el, true
					break
				}
			}
		}
	case strings.HasSuffix(lp, ":"):
		// Struct field in a composite literal: T{F: |}
		before := strings.TrimSpace(lp[:len(lp)-1])
		fld := lastIdent(before)
		ty := lastIdent(strings.TrimSpace(before[:len(before)-len(fld)]))
		// before ends with "{" after T.
		if strings.HasSuffix(before[:len(before)-len(fld)], "{") || fld != "" {
			for _, f := range []*facts{pf, sf} {
				if f == nil {
					continue
				}
				if row := f.tyFld[ty]; row != nil && row[fld] != "" {
					g.name = row[fld]
					break
				}
			}
		}
	}

	// Call argument position: f(a, | or f(|. Methods key their
	// signature as Type.M: resolve the receiver chain for the key,
	// then fall back to the bare name.
	if g.name == "" && (strings.HasSuffix(lp, "(") || strings.HasSuffix(lp, ",")) {
		if fn, argIdx, recv := callSite(lp); fn != "" {
			keys := []string{fn}
			if recv != "" {
				rt := ""
				for _, f := range []*facts{pf, sf} {
					if f != nil {
						if t := f.resolveChain(strings.Split(recv, ".")); t != "" {
							rt = t
							break
						}
					}
				}
				if rt != "" {
					keys = append([]string{rt + "." + fn}, keys...)
				}
			}
			for _, k := range keys {
				for _, f := range []*facts{pf, sf} {
					if f == nil {
						continue
					}
					if sig := f.sigT[k]; argIdx < len(sig) && sig[argIdx] != "" {
						g = parseSigType(sig[argIdx])
						break
					}
				}
				if g.name != "" {
					break
				}
			}
		}
	}
	return g
}

// parseSigType turns a recorded signature slot ("T", "*T", "[]T",
// "map:K:V") into a goal. Container kinds keep the element type.
func parseSigType(s string) goalT {
	switch {
	case strings.HasPrefix(s, "*"):
		return goalT{name: s[1:], ptr: true}
	case strings.HasPrefix(s, "[]"):
		return goalT{name: s[2:], slc: true}
	case strings.HasPrefix(s, "map:") || strings.HasPrefix(s, "chan:"):
		parts := strings.SplitN(s, ":", 3)
		return goalT{name: parts[len(parts)-1]}
	default:
		return goalT{name: s}
	}
}

// callSite finds the innermost open call and the arg index at the
// cursor by walking the trimmed line backward. It also returns the
// dotted receiver chain before the callee, so method signatures can
// resolve through their receiver type.
func callSite(lp string) (fn string, args int, recv string) {
	depth := 0
	for i := len(lp) - 1; i >= 0; i-- {
		switch lp[i] {
		case ')':
			depth++
		case '(':
			if depth == 0 {
				j := i - 1
				for j >= 0 && (lp[j] == ' ' || lp[j] == '\t') {
					j--
				}
				end := j
				for j >= 0 && isIdentByte(lp[j]) {
					j--
				}
				fn = lp[j+1 : end+1]
				args := 0
				depth2 := 0
				for k := i + 1; k < len(lp); k++ {
					switch lp[k] {
					case '(':
						depth2++
					case ')':
						depth2--
					case ',':
						if depth2 == 0 {
							args++
						}
					}
				}
				// The dotted receiver before fn: "s.st.Set2(" gives
				// recv "s.st".
				if j >= 0 && lp[j] == '.' {
					e2 := j
					for e2 > 0 && (isIdentByte(lp[e2-1]) || lp[e2-1] == '.') {
						e2--
					}
					recv = lp[e2:j]
				}
				return fn, args, recv
			}
			depth--
		case ',':
			if depth == 0 {
				args++
			}
		}
	}
	return "", 0, ""
}

// lastIdent returns the trailing identifier of a trimmed line.
func lastIdent(s string) string {
	i := len(s)
	for i > 0 && isIdentByte(s[i-1]) {
		i--
	}
	return s[i:]
}

// isZeroable reports whether a goal can be satisfied by nil or a
// zero literal: pointers, interfaces, slices, chans, maps, funcs.
func (g goalT) zeroable(f *facts) bool {
	if g.ptr || g.slc {
		return true
	}
	if f == nil {
		return false
	}
	// An interface has members that are all methods and no fields.
	ms, ok := f.tyMem[g.name]
	if !ok || len(ms) == 0 {
		return false
	}
	for m := range ms {
		if !strings.HasSuffix(m, "(") {
			return false // has fields -> struct, T{} not nil
		}
	}
	return true
}

// methodsOnly reports whether t's recorded members are all methods -
// the interface-shape check.
func methodsOnly(ms map[string]bool) bool {
	if len(ms) == 0 || len(ms) > 8 {
		return false
	}
	for m := range ms {
		if !strings.HasSuffix(m, "(") {
			return false
		}
	}
	return true
}

// satisfies reports whether T's member set covers every method of the
// interface type goal. Covers Session-facts and corpus tables.
func satisfies(t, iface string, sf *facts, tyMems []map[string][]string) bool {
	var need map[string]bool
	if sf != nil {
		need = sf.tyMem[iface]
	}
	if need == nil {
		for _, tab := range tyMems {
			if row := tab[iface]; len(row) > 0 {
				need = map[string]bool{}
				for _, m := range row {
					need[m] = true
				}
				break
			}
		}
	}
	if !methodsOnly(need) {
		return false
	}
	var have map[string]bool
	if sf != nil {
		have = sf.tyMem[t]
	}
	if have == nil {
		for _, tab := range tyMems {
			if row := tab[t]; len(row) > 0 {
				have = map[string]bool{}
				for _, m := range row {
					have[m] = true
				}
				break
			}
		}
	}
	for m := range need {
		if !have[m] {
			return false
		}
	}
	return true
}

// synthExpr assembles candidate expressions that produce goal.
// pf/sf carry session facts; tyMems are the corpus member tables;
// lines from the session pool supply attested arg text for producers
// whose parameters we cannot fill from scope.
// Producers split by provenance: doc/session factories outrank corpus
// producers since name collisions make every same-named type claim
// every ctor. cRet/cSig are the corpus tables (aux sidecar).
func synthExpr(g goalT, pf, sf *facts, tyMems []map[string][]string, pool []string,
	cRet map[string]string, cSig map[string][]string) (local, corpus []string) {
	if g.name == "" {
		return nil, nil
	}
	seen := map[string]bool{}
	cSeen := map[string]bool{}
	add := func(s string) {
		if s != "" && !seen[s] {
			seen[s] = true
			local = append(local, s)
		}
	}
	addC := func(s string) {
		if s != "" && !seen[s] && !cSeen[s] {
			cSeen[s] = true
			corpus = append(corpus, s)
		}
	}

	// Direct vars of the goal type, pointer adapts, and field paths.
	for _, f := range []*facts{pf, sf} {
		if f == nil {
			continue
		}
		for v, t := range f.recv {
			if t != g.name || v == g.lhs {
				continue
			}
			switch {
			case f.ptr[v] == g.ptr:
				add(v)
			case f.ptr[v] && !g.ptr:
				add("*" + v)
			case !f.ptr[v] && g.ptr:
				add("&" + v)
			}
		}
		if g.slc {
			for v, el := range f.elem {
				if el == g.name && v != g.lhs {
					add(v)
				}
			}
		}
		// Field paths: a var whose field has the goal type.
		for v, vt := range f.recv {
			if v == g.lhs {
				continue
			}
			row := f.tyFld[vt]
			if row == nil {
				continue
			}
			for fld, ft := range row {
				if ft == g.name {
					add(v + "." + fld)
				}
			}
		}
		// Interface satisfaction: v's type covers the goal's methods.
		for v, vt := range f.recv {
			if v != g.lhs && vt != g.name && satisfies(vt, g.name, sf, tyMems) {
				add(v)
			}
		}
	}

	// Producer calls: functions whose declared result is the goal.
	// Args fill from scope vars by type; unresolved params stay an
	// open paren so the cursor lands inside for arg synthesis.
	prod := map[string]bool{}
	for _, f := range []*facts{pf, sf} {
		if f == nil {
			continue
		}
		for fn, rt := range f.retT {
			if rt == g.name {
				prod[fn] = true
			}
		}
	}
	cProd := map[string]bool{}
	for fn, rt := range cRet {
		if rt == g.name && !prod[fn] {
			cProd[fn] = true
		}
	}
	for fn := range cProd {
		sig := cSig[fn]
		args := ""
		ok := true
		for i, pt := range sig {
			av := ""
			for _, f := range []*facts{pf, sf} {
				if f == nil {
					continue
				}
				want := parseSigType(pt)
				for v, t := range f.recv {
					if t == want.name && f.ptr[v] == want.ptr {
						av = v
						break
					}
				}
				if av != "" {
					break
				}
			}
			if i > 0 {
				args += ", "
			}
			if av == "" {
				ok = false
				break
			}
			args += av
		}
		switch {
		case ok && len(sig) > 0:
			addC(fn + "(" + args + ")")
		case len(sig) == 0:
			addC(fn + "(")
		default:
			addC(fn + "(")
		}
	}
	for fn := range prod {
		var sig []string
		for _, f := range []*facts{pf, sf} {
			if f != nil && len(f.sigT[fn]) > 0 {
				sig = f.sigT[fn]
				break
			}
		}
		if len(sig) == 0 {
			sig = cSig[fn]
		}
		args := ""
		ok := true
		for i, pt := range sig {
			av := ""
			for _, f := range []*facts{pf, sf} {
				if f == nil {
					continue
				}
				want := parseSigType(pt)
				for v, t := range f.recv {
					if t == want.name && f.ptr[v] == want.ptr {
						av = v
						break
					}
				}
				if av != "" {
					break
				}
			}
			if i > 0 {
				args += ", "
			}
			if av == "" {
				ok = false
				break
			}
			args += av
		}
		if ok && len(sig) > 0 {
			add(fn + "(" + args + ")")
		} else if len(sig) == 0 {
			add(fn + "()")
		} else {
			// Attested call text from the session pool fills gaps.
			// Declarations are skipped: "func NewStore(ttl int)" is
			// a signature, not a call shape worth copying.
			attested := false
			for _, l := range pool {
				if strings.HasPrefix(l, "func ") || strings.HasPrefix(l, "def ") {
					continue
				}
				k := strings.Index(l, fn+"(")
				if k < 0 {
					continue
				}
				rest := l[k:]
				if e := strings.Index(rest, ")"); e > len(fn) {
					add(rest[:e+1])
					attested = true
					break
				}
			}
			if !attested {
				add(fn + "(")
			}
		}
	}

	// Zero value is the honest answer for pointer/slice/interface
	// goals when no producer exists.
	if len(local) == 0 && (g.zeroable(pf) || g.zeroable(sf)) {
		add("nil")
	}
	// Composite literal for a struct-shaped goal.
	if len(local) == 0 && g.name != "" && !g.ptr && !g.slc {
		for _, f := range []*facts{pf, sf} {
			if f == nil {
				continue
			}
			if len(f.tyFld[g.name]) > 0 {
				add(g.name + "{}")
			}
		}
	}
	return local, corpus
}

// isIdentText reports whether s is a bare identifier - short-var
// synthesis uses it to bypass the junk length floor safely.
func isIdentText(s string) bool {
	if s == "" {
		return false
	}
	for i := 0; i < len(s); i++ {
		if !isIdentByte(s[i]) {
			return false
		}
	}
	return true
}
