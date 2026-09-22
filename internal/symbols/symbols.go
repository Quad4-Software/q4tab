// Package symbols extracts identifier definitions (funcs, types,
// classes, vars) from source files with a language-agnostic token
// scan, and indexes them for exact and prefix lookup. It is the
// engine's answer to "where is X defined" without ctags or a parser.
package symbols

import (
	"bufio"
	"encoding/json"
	"os"
	"sort"
	"strings"
	"sync"

	"q4complete/internal/tokenize"
)

// Sym is one definition site.
type Sym struct {
	Kind string `json:"k"` // func, method, type, var
	Name string `json:"n"`
	Path string `json:"p"`
	Line int    `json:"l"` // 1-based
	Sig  string `json:"s"` // trimmed source line
}

// Index maps names to their definition sites. Safe for concurrent use.
type Index struct {
	mu     sync.RWMutex
	byName map[string][]Sym
	names  []string // sorted, for prefix lookup. Nil = stale
}

func NewIndex() *Index { return &Index{byName: map[string][]Sym{}} }

func (ix *Index) Add(s Sym) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	ix.names = nil // invalidate sorted cache
	ix.byName[s.Name] = append(ix.byName[s.Name], s)
}

// RemovePath drops every symbol defined in path (file changed).
func (ix *Index) RemovePath(path string) {
	ix.mu.Lock()
	defer ix.mu.Unlock()
	for n, ss := range ix.byName {
		keep := ss[:0]
		for _, s := range ss {
			if s.Path != path {
				keep = append(keep, s)
			}
		}
		if len(keep) == 0 {
			delete(ix.byName, n)
		} else {
			ix.byName[n] = keep
		}
	}
	ix.names = nil
}

func (ix *Index) Lookup(name string) []Sym {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return ix.byName[name]
}

// LookupPrefix returns symbols whose name starts with prefix, exact
// matches first, then by name.
func (ix *Index) LookupPrefix(prefix string, limit int) []Sym {
	ix.mu.Lock() // writes the sorted-names cache
	defer ix.mu.Unlock()
	if ix.names == nil {
		ix.names = make([]string, 0, len(ix.byName))
		for n := range ix.byName {
			ix.names = append(ix.names, n)
		}
		sort.Strings(ix.names)
	}
	var out []Sym
	lo := sort.SearchStrings(ix.names, prefix)
	for i := lo; i < len(ix.names) && strings.HasPrefix(ix.names[i], prefix); i++ {
		out = append(out, ix.byName[ix.names[i]]...)
		if len(out) >= limit*2 {
			break
		}
	}
	sort.SliceStable(out, func(i, j int) bool {
		li, lj := len(out[i].Name), len(out[j].Name)
		if li != lj {
			return li < lj // shortest name first (exact match wins)
		}
		return out[i].Name < out[j].Name
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

func (ix *Index) Len() int {
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	return len(ix.byName)
}

// Save writes the index as JSONL (one Sym per line).
func (ix *Index) Save(path string) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	w := bufio.NewWriterSize(f, 1<<20)
	enc := json.NewEncoder(w)
	ix.mu.RLock()
	defer ix.mu.RUnlock()
	for _, ss := range ix.byName {
		for _, s := range ss {
			if err := enc.Encode(s); err != nil {
				return err
			}
		}
	}
	return w.Flush()
}

// Load reads a JSONL symbol index. Missing file is not an error.
func Load(path string) (*Index, error) {
	f, err := os.Open(path)
	if err != nil {
		if os.IsNotExist(err) {
			return NewIndex(), nil
		}
		return nil, err
	}
	defer f.Close()
	ix := NewIndex()
	dec := json.NewDecoder(bufio.NewReaderSize(f, 1<<20))
	for {
		var s Sym
		if err := dec.Decode(&s); err != nil {
			break // EOF or corrupt tail: keep what loaded
		}
		if s.Name != "" {
			ix.Add(s)
		}
	}
	return ix, nil
}

// defKind is one token-pattern: the trigger word, then where the name
// sits relative to it.
var defWords = map[string]string{
	"func":      "func",
	"def":       "func",
	"fn":        "func",
	"function":  "func",
	"class":     "type",
	"interface": "type",
	"struct":    "type",
	"enum":      "type",
	"trait":     "type",
	"type":      "type",
	"const":     "var",
	"var":       "var",
	"let":       "var",
}

func isNameTok(t string) bool {
	if t == "" || !isIdentStart(t[0]) {
		return false
	}
	switch t {
	case "func", "def", "fn", "function", "class", "interface",
		"struct", "enum", "trait", "type", "const", "var", "let",
		"if", "for", "while", "switch", "return", "else", "do",
		"in", "of", "new", "this", "true", "false", "nil", "null",
		"None", "impl", "pub", "private", "public", "static",
		"async", "await", "where", "package", "import":
		return false
	}
	return true
}

func isIdentStart(c byte) bool {
	return c == '_' || c == '$' ||
		(c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c >= 0x80
}

// Extract scans data for definition patterns and returns the symbols
// with their source line. It intentionally keeps to a small grammar so
// it stays language-agnostic and fast (one lex pass per file).
func Extract(path string, data []byte) []Sym {
	toks := tokenize.Lex(data)
	src := string(data)
	srcLines := strings.Split(src, "\n")
	lineOf := func(nlCount int) string {
		if nlCount < len(srcLines) {
			return strings.TrimSpace(srcLines[nlCount])
		}
		return ""
	}
	var out []Sym
	line := 0
	// next returns the index of the first non-WS token at or after i.
	next := func(i int) int {
		for i < len(toks) && toks[i] == tokenize.WS {
			i++
		}
		return i
	}
	at := func(i int) string {
		if i < len(toks) {
			return toks[i]
		}
		return ""
	}
	for i, t := range toks {
		switch t {
		case tokenize.NL:
			line++
			continue
		case tokenize.WS:
			continue
		}
		kind, isDef := defWords[t]
		if !isDef {
			continue
		}
		j := next(i + 1)
		// Go method: func (recv) name(
		if t == "func" && at(j) == "(" {
			for j < len(toks) && toks[j] != ")" && toks[j] != tokenize.NL {
				j++
			}
			if at(j) == ")" {
				j = next(j + 1)
			}
			kind = "method"
		}
		name := at(j)
		if !isNameTok(name) {
			continue
		}
		// Require the token after the name to look like an opener or
		// qualifier for funcs, so "return funcValue" does not fire.
		if kind == "func" || kind == "method" {
			nx := at(next(j + 1))
			if nx != "(" && nx != "[" && nx != "<" {
				continue
			}
		}
		out = append(out, Sym{
			Kind: kind, Name: name, Path: path,
			Line: line + 1, Sig: lineOf(line),
		})
	}
	return out
}
