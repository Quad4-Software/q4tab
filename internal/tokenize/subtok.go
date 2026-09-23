package tokenize

// subtok.go: identifier subtoken splitting. A name like parseHTTPResponse
// carries more signal as parts (parse|HTTP|Response) than as one opaque
// token: the model can score a name it has never seen whole, and typed
// prefixes that cross a case boundary still match. Splits keep the
// original bytes contiguous so joining the parts reproduces the ident.

// IsIdentTok reports whether a token is an identifier (letter, _ or $
// start, len >= 2). Keywords count: splitting them is harmless.
func IsIdentTok(t string) bool {
	return len(t) >= 2 && isIdentStart(t[0])
}

// Subtoks splits an identifier into its parts. Boundaries:
//
//   - separators: "_" and "-" start a new part that keeps the separator
//     (snake_case, SCREAMING_CASE, kebab-ish names)
//   - lowercase/digit to uppercase: parseHTTP -> parse|HTTP
//   - acronym to camel: HTTPResponse -> HTTP|Response
//   - letter to digit or digit to letter: utf8 -> utf|8
//
// Joining the result reproduces the input exactly.
func Subtoks(ident string) []string {
	if !IsIdentTok(ident) {
		return []string{ident}
	}
	var out []string
	start := 0
	for i := 1; i < len(ident); i++ {
		c, p := ident[i], ident[i-1]
		var n byte
		if i+1 < len(ident) {
			n = ident[i+1]
		}
		boundary := false
		switch {
		case c == '_' || c == '-':
			boundary = true // separator begins the next part
		case isDigit(c) != isDigit(p) && p != '_' && p != '-':
			boundary = true // letter<->digit transition
		case isLower(p) && isUpper(c):
			boundary = true // camel hump
		case isUpper(p) && isUpper(c) && isLower(n):
			// Last capital of an acronym run starts the camel word:
			// HTTPResponse splits before the R, not before the P.
			boundary = true
		}
		if boundary && i > start {
			out = append(out, ident[start:i])
			start = i
		}
	}
	out = append(out, ident[start:])
	return out
}

func isLower(c byte) bool { return c >= 'a' && c <= 'z' }
func isUpper(c byte) bool { return c >= 'A' && c <= 'Z' }

// StructState tracks cheap lexical structure while walking a token
// stream: brace depth, whether the position is inside parens or
// brackets, and the last significant keyword class. It needs no parser
// and degrades gracefully on unbalanced edits or unknown languages.
type StructState struct {
	depth int
	par   int
	kw    uint8
}

// Keyword classes. Language-agnostic: a keyword only shifts the class,
// never changes what the surrounding tokens mean.
const (
	kwNone  = 0
	kwRet   = 1 // return, raise, yield
	kwCtrl  = 2 // if, else, switch, case, match, when, catch
	kwLoop  = 3 // for, while, each, loop, repeat, do
	kwDecl  = 4 // func, def, fn, class, struct, type, interface, enum, impl, trait
	kwImp   = 5 // import, package, from, use, include, require, mod
	kwVar   = 6 // var, let, const, local, new
	kwOther = 7 // any other keyword-shaped token
)

var kwClass = map[string]uint8{
	"return": kwRet, "raise": kwRet, "yield": kwRet, "throw": kwRet,
	"if": kwCtrl, "else": kwCtrl, "elif": kwCtrl, "switch": kwCtrl,
	"case": kwCtrl, "match": kwCtrl, "when": kwCtrl, "catch": kwCtrl,
	"except": kwCtrl, "finally": kwCtrl, "default": kwCtrl,
	"for": kwLoop, "while": kwLoop, "each": kwLoop, "loop": kwLoop,
	"repeat": kwLoop, "do": kwLoop, "until": kwLoop,
	"func": kwDecl, "function": kwDecl, "def": kwDecl, "fn": kwDecl,
	"class": kwDecl, "struct": kwDecl, "type": kwDecl, "interface": kwDecl,
	"enum": kwDecl, "impl": kwDecl, "trait": kwDecl, "record": kwDecl,
	"import": kwImp, "package": kwImp, "from": kwImp, "use": kwImp,
	"include": kwImp, "require": kwImp, "mod": kwImp, "module": kwImp,
	"var": kwVar, "let": kwVar, "const": kwVar, "local": kwVar,
	"new": kwVar, "final": kwVar, "static": kwVar, "pub": kwVar,
	"try": kwOther, "in": kwOther, "of": kwOther, "go": kwOther,
	"defer": kwOther, "async": kwOther, "await": kwOther, "then": kwOther,
	"end": kwOther, "begin": kwOther, "nil": kwOther, "null": kwOther,
	"none": kwOther, "true": kwOther, "false": kwOther, "this": kwOther,
	"self": kwOther, "super": kwOther, "and": kwOther, "or": kwOther,
	"not": kwOther, "with": kwOther, "as": kwOther, "pass": kwOther,
	"break": kwOther, "continue": kwOther, "void": kwOther, "int": kwOther,
	"string": kwOther, "bool": kwOther, "float": kwOther,
}

// IsKeywordish reports whether t is a recognized keyword: the mask
// used for identifier-insensitive retrieval keeps keywords literal so
// structural matching keeps the line's meaning.
func IsKeywordish(t string) bool {
	_, ok := kwClass[t]
	return ok
}

// Key packs the state into a table key: 4 bits brace depth (capped),
// 1 bit in-parens/brackets, 3 bits last keyword class.
func (s *StructState) Key() uint16 {
	d := s.depth
	if d > 15 {
		d = 15
	}
	k := uint16(d) << 4
	if s.par > 0 {
		k |= 1 << 3
	}
	return k | uint16(s.kw)
}

// Advance folds one token into the state. Call with tokens in stream
// order; Key() is meaningful before each Advance (the state under which
// the token was produced).
func (s *StructState) Advance(tok string) {
	if len(tok) == 1 {
		switch tok[0] {
		case '{':
			s.depth++
		case '}':
			s.depth--
		case '(', '[':
			s.par++
		case ')', ']':
			s.par--
		}
		return
	}
	if IsIdentTok(tok) && len(tok) < 16 {
		if c, ok := kwClass[tok]; ok {
			s.kw = c
		}
	}
}
