// Package tokenize turns source text into a flat token stream for the
// statistical model. It is language-agnostic: it understands identifiers,
// numbers, string literals, comments, operators, indentation, and newlines.
//
// Output tokens are literal strings. In-line whitespace is normalized to a
// single " " token, line-leading whitespace is emitted verbatim, and a "\n"
// token separates lines. Detokenizing a token slice with Detok reproduces
// the (whitespace-normalized) source.
package tokenize

// Special tokens.
const (
	NL  = "\n"
	EOF = "<eof>"
	UNK = "<unk>"
	WS  = " "
	BOL = "" // never emitted; zero value guard
)

func isIdentStart(c byte) bool {
	return c == '_' || c == '$' ||
		(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80
}

func isIdentPart(c byte) bool {
	return isIdentStart(c) || (c >= '0' && c <= '9')
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }

func isSpace(c byte) bool { return c == ' ' || c == '\t' }

var multiOps = []string{
	"...", "<<=", ">>=", "**=",
	"==", "!=", "<=", ">=", "->", "=>", "::", "&&", "||",
	"+=", "-=", "*=", "/=", "%=", "++", "--", "<<", ">>",
	"<-", ":=", "**", "&=", "|=", "^=", "??", "?.", "=>",
}

func matchOp(src []byte, i int) string {
	for _, op := range multiOps {
		if i+len(op) <= len(src) && string(src[i:i+len(op)]) == op {
			return op
		}
	}
	return ""
}

// Lex converts source bytes into a token slice. It always terminates the
// stream with a single EOF token. Blank lines produce only an NL token.
func Lex(src []byte) []string {
	toks := make([]string, 0, len(src)/4)
	n := len(src)
	i := 0
	lineStart := true
	for i < n {
		c := src[i]
		if c == '\r' {
			i++
			continue
		}
		if c == '\n' {
			toks = append(toks, NL)
			i++
			lineStart = true
			continue
		}
		if isSpace(c) {
			j := i
			for j < n && isSpace(src[j]) {
				j++
			}
			if lineStart {
				// Peek: if only whitespace until EOL, skip indent token.
				k := j
				for k < n && src[k] == '\r' {
					k++
				}
				if k < n && src[k] != '\n' {
					toks = append(toks, string(src[i:j]))
				}
			} else {
				toks = append(toks, WS)
			}
			i = j
			continue
		}
		lineStart = false
		// Line comments: // and # (not #! handled, shebang is fine as comment)
		if c == '/' && i+1 < n && src[i+1] == '/' {
			toks = append(toks, "//")
			i += 2
			continue
		}
		if c == '#' {
			toks = append(toks, "#")
			i++
			continue
		}
		if c == '/' && i+1 < n && src[i+1] == '*' {
			toks = append(toks, "/*")
			i += 2
			continue
		}
		if c == '*' && i+1 < n && src[i+1] == '/' {
			toks = append(toks, "*/")
			i += 2
			continue
		}
		// String literals: ", ', ` with escape handling.
		if c == '"' || c == '\'' || c == '`' {
			j := i + 1
			for j < n {
				if src[j] == '\\' {
					j += 2
					continue
				}
				if src[j] == c {
					j++
					break
				}
				if src[j] == '\n' && c != '`' {
					break // unterminated string ends at EOL
				}
				j++
			}
			toks = append(toks, string(src[i:j]))
			i = j
			continue
		}
		if isIdentStart(c) {
			j := i + 1
			for j < n && isIdentPart(src[j]) {
				j++
			}
			toks = append(toks, string(src[i:j]))
			i = j
			continue
		}
		if isDigit(c) {
			j := i + 1
			for j < n && (isIdentPart(src[j]) || src[j] == '.') {
				j++
			}
			toks = append(toks, string(src[i:j]))
			i = j
			continue
		}
		if op := matchOp(src, i); op != "" {
			toks = append(toks, op)
			i += len(op)
			continue
		}
		toks = append(toks, string(src[i:i+1]))
		i++
	}
	toks = append(toks, EOF)
	return toks
}

// Detok concatenates tokens back into text.
func Detok(toks []string) string {
	total := 0
	for _, t := range toks {
		total += len(t)
	}
	b := make([]byte, 0, total)
	for _, t := range toks {
		b = append(b, t...)
	}
	return string(b)
}

// LineTokens returns the tokens of a single logical line (no NL, no indent),
// for indexing lines in the retrieval index.
func LexLine(line []byte) []string {
	toks := Lex(append(append([]byte{}, line...), '\n'))
	out := toks[:0]
	for _, t := range toks {
		if t == NL || t == EOF {
			break
		}
		out = append(out, t)
	}
	// Drop a leading indent token for normalized matching.
	if len(out) > 0 && (out[0][0] == ' ' || out[0][0] == '\t') {
		out = out[1:]
	}
	return out
}
