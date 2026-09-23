package engine

import (
	"bytes"
	"path/filepath"
	"sort"
	"strings"

	"q4tab/internal/lines"
	"q4tab/internal/model"
	"q4tab/internal/tokenize"
)

// aux.go: the auxiliary model layers that ship inside the model file.
//
// Beyond the token n-gram, a trained bundle carries:
//
//   - LineBi: previous normalized line(s) -> following line. Retrieval
//     for "what comes after this line", which the token model answers
//     only token at a time.
//   - Struct: cheap lexical-context key -> token counts. Conditions
//     candidates on brace depth / in-parens / last keyword class.
//   - Langs: per-language unigram+bigram tables so Go habits do not
//     bleed into Python ranking.
//   - Sub/Idents: an order-3 model over identifier SUBTOKENS plus a
//     subtoken-path -> ident index. This is the open-vocabulary piece:
//     it can score names the main model has never seen whole.
//
// All of them are optional: a v3 file without a section leaves the
// field nil and the engine silently skips that layer.

// Bundle is everything a trained model file contains.
type Bundle struct {
	M      *model.Model
	Lines  *lines.Index
	LineBi *lines.GramIndex
	Struct *model.Order
	Langs  map[string]*model.LangTable
	Sub    *model.Model
	Idents *model.IdentIndex
	// FileStarts maps a language to its most common first non-blank
	// lines: the prior for what a new file in that language opens
	// with. Powers thin-context proposals.
	FileStarts map[string][]string
	// DirIdents maps a directory path to its most frequent
	// non-keyword identifiers: the names a package or directory is
	// built around. Powers the directory-level cache and the
	// import-adjacency boost.
	DirIdents map[string][]string
}

// LangOf classifies a path or URI into a language bucket. Extensions
// are matched lowercase; well-known extensionless names are handled.
func LangOf(path string) string {
	if i := strings.Index(path, "://"); i >= 0 {
		path = path[i+3:]
	}
	if i := strings.IndexAny(path, "?#"); i >= 0 {
		path = path[:i]
	}
	base := filepath.Base(path)
	switch strings.ToLower(base) {
	case "dockerfile", "containerfile":
		return "dockerfile"
	case "makefile", "gnumakefile":
		return "makefile"
	}
	switch strings.ToLower(filepath.Ext(base)) {
	case ".go":
		return "go"
	case ".py", ".pyi":
		return "python"
	case ".ts", ".tsx", ".mts", ".cts":
		return "typescript"
	case ".js", ".jsx", ".mjs", ".cjs":
		return "javascript"
	case ".rs":
		return "rust"
	case ".c", ".h":
		return "c"
	case ".cpp", ".cc", ".cxx", ".hpp", ".hh":
		return "cpp"
	case ".java":
		return "java"
	case ".kt", ".kts":
		return "kotlin"
	case ".rb":
		return "ruby"
	case ".php":
		return "php"
	case ".sh", ".bash", ".zsh":
		return "shell"
	case ".lua":
		return "lua"
	case ".vim":
		return "vim"
	case ".json", ".jsonc":
		return "json"
	case ".yaml", ".yml":
		return "yaml"
	case ".toml":
		return "toml"
	case ".md", ".markdown":
		return "markdown"
	case ".css", ".scss", ".less":
		return "css"
	case ".html", ".htm":
		return "html"
	case ".xml":
		return "xml"
	case ".sql":
		return "sql"
	case ".vue":
		return "vue"
	case ".svelte":
		return "svelte"
	case ".ex", ".exs":
		return "elixir"
	case ".erl", ".hrl":
		return "erlang"
	case ".hs":
		return "haskell"
	case ".ml", ".mli":
		return "ocaml"
	case ".scala", ".sc":
		return "scala"
	case ".swift":
		return "swift"
	case ".r":
		return "r"
	case ".pl", ".pm":
		return "perl"
	case ".proto":
		return "proto"
	case ".tf", ".tfvars":
		return "terraform"
	case ".nix":
		return "nix"
	case ".dart":
		return "dart"
	case ".zig":
		return "zig"
	case ".cs":
		return "csharp"
	case ".fs", ".fsx":
		return "fsharp"
	case ".clj", ".cljs":
		return "clojure"
	case ".dockerfile":
		return "dockerfile"
	case ".txt", ".text":
		return "text"
	}
	return "other"
}

// rowsToOrder compiles a counted (ctxKey -> tok -> count) map into a
// packed in-memory Order: sorted keys, rows sorted by count desc and
// capped. Used by the struct and language tables, which are built by
// counting rather than by the n-gram Builder.
func rowsToOrder(rows map[uint64]map[uint32]uint32, minCnt uint32, rowCap int) *model.Order {
	keys := make([]uint64, 0, len(rows))
	for k := range rows {
		keys = append(keys, k)
	}
	sort.Slice(keys, func(i, j int) bool { return keys[i] < keys[j] })
	o := &model.Order{Off: make([]int64, 0, len(keys)+1)}
	o.Off = append(o.Off, 0)
	for _, k := range keys {
		m := rows[k]
		type tc struct {
			t uint32
			c uint32
		}
		row := make([]tc, 0, len(m))
		for t, c := range m {
			if c >= minCnt {
				row = append(row, tc{t, c})
			}
		}
		if len(row) == 0 {
			continue
		}
		sort.Slice(row, func(i, j int) bool {
			if row[i].c != row[j].c {
				return row[i].c > row[j].c
			}
			return row[i].t < row[j].t // deterministic builds
		})
		if len(row) > rowCap {
			row = row[:rowCap]
		}
		var tot int64
		for _, e := range row {
			o.Toks = append(o.Toks, int32(e.t))
			o.Cnts = append(o.Cnts, int32(e.c))
			tot += int64(e.c)
		}
		o.Keys = append(o.Keys, k)
		o.Off = append(o.Off, int64(len(o.Toks)))
		o.Totals = append(o.Totals, tot)
	}
	o.NToks = int64(len(o.Toks))
	if len(o.Keys) == 0 {
		return nil
	}
	return o
}

// langRows are per-language low-order counts gathered during indexing.
type langRows struct {
	uni map[uint32]uint32
	bi  map[uint64]map[uint32]uint32 // prev token -> next -> count
}

// startBuilder counts the first non-blank line of each file under its
// language bucket. In a thin context (a fresh or nearly-empty file)
// these priors seed plausible openings: package decls, imports,
// shebangs.
type startBuilder struct {
	m map[string]map[string]int // lang -> normalized line -> file count
}

func newStartBuilder() *startBuilder {
	return &startBuilder{m: map[string]map[string]int{}}
}

// Add records the file's first non-blank normalized line.
func (sb *startBuilder) Add(path string, data []byte) {
	var l string
	for len(data) > 0 {
		line := data
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			line, data = data[:i], data[i+1:]
		} else {
			data = nil
		}
		if l = lines.Normalize(string(line)); l != "" {
			break
		}
	}
	if l == "" || len(l) > 96 {
		return // empty file, or a minified/generated blob
	}
	lang := LangOf(path)
	t := sb.m[lang]
	if t == nil {
		t = map[string]int{}
		sb.m[lang] = t
	}
	t[l]++
}

// Compact keeps each language's top lines seen in enough files to be
// a convention rather than a one-off.
func (sb *startBuilder) Compact(top, minFiles int) map[string][]string {
	out := make(map[string][]string, len(sb.m))
	for lang, t := range sb.m {
		type ent struct {
			l string
			n int
		}
		var es []ent
		for l, n := range t {
			if n >= minFiles {
				es = append(es, ent{l, n})
			}
		}
		sort.Slice(es, func(i, j int) bool {
			if es[i].n != es[j].n {
				return es[i].n > es[j].n
			}
			return es[i].l < es[j].l
		})
		if len(es) > top {
			es = es[:top]
		}
		if len(es) == 0 {
			continue
		}
		ls := make([]string, len(es))
		for i, e := range es {
			ls[i] = e.l
		}
		out[lang] = ls
	}
	return out
}

// dirBuilder counts non-keyword identifier occurrences per directory.
// The table answers "what names does this package center on", which
// is both a directory-level cache signal (files under the same dir
// share vocabulary) and an import-adjacency signal (a file importing
// pkg/foo probably wants foo's names).
type dirBuilder struct {
	m map[string]map[string]int // dir -> ident -> count
}

func newDirBuilder() *dirBuilder {
	return &dirBuilder{m: map[string]map[string]int{}}
}

// Add counts the file's identifiers under its directory. The byte
// scan is cheaper than a full lex and good enough for a top-N table.
func (db *dirBuilder) Add(path string, data []byte) {
	dir := filepath.Dir(path)
	if dir == "" || dir == "." {
		return
	}
	t := db.m[dir]
	if t == nil {
		t = map[string]int{}
		db.m[dir] = t
	}
	start := -1
	for i := 0; i <= len(data); i++ {
		if i < len(data) && isIdentByte(data[i]) {
			if start < 0 {
				start = i
			}
			continue
		}
		if start >= 0 {
			if s := string(data[start:i]); len(s) >= 3 && isIdentStart(s[0]) &&
				!tokenize.IsKeywordish(s) {
				t[s]++
			}
			start = -1
		}
	}
}

// Compact keeps each directory's top idents by count.
func (db *dirBuilder) Compact(top, minCnt int) map[string][]string {
	out := make(map[string][]string, len(db.m))
	for dir, t := range db.m {
		type ent struct {
			s string
			n int
		}
		var es []ent
		for s, n := range t {
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
		if len(es) > top {
			es = es[:top]
		}
		if len(es) == 0 {
			continue
		}
		ls := make([]string, len(es))
		for i, e := range es {
			ls[i] = e.s
		}
		out[dir] = ls
	}
	return out
}

// langBuilder accumulates per-language token statistics.
type langBuilder struct {
	tabs   map[string]*langRows
	frozen bool // Tighten was called: only known (prev,next) pairs count
}

func newLangBuilder() *langBuilder {
	return &langBuilder{tabs: map[string]*langRows{}}
}

// Add counts the token stream under the file's language bucket.
func (lb *langBuilder) Add(path string, ids []uint32, eof uint32) {
	lang := LangOf(path)
	t := lb.tabs[lang]
	if t == nil {
		t = &langRows{uni: map[uint32]uint32{}, bi: map[uint64]map[uint32]uint32{}}
		lb.tabs[lang] = t
	}
	var prev uint32 = eof
	for _, id := range ids {
		if id == eof {
			prev = eof
			continue
		}
		t.uni[id]++
		if prev != eof {
			key := model.LangCtxKey(prev)
			row := t.bi[key]
			if row == nil {
				if !lb.frozen {
					row = map[uint32]uint32{}
					t.bi[key] = row
				}
			}
			if row != nil {
				row[id]++
			}
		}
		prev = id
	}
}

// Tighten bounds the bigram tables on very large corpora: rows are
// capped at their current top-64 pairs and new (prev,next) pairs stop
// accumulating. The unigram rows keep counting either way.
func (lb *langBuilder) Tighten() {
	for _, t := range lb.tabs {
		for key, row := range t.bi {
			if len(row) <= 64 {
				continue
			}
			type tc struct {
				t uint32
				c uint32
			}
			top := make([]tc, 0, 64)
			for tok, c := range row {
				top = append(top, tc{tok, c})
			}
			sort.Slice(top, func(i, j int) bool { return top[i].c > top[j].c })
			nr := make(map[uint32]uint32, 64)
			for _, e := range top[:64] {
				nr[e.t] = e.c
			}
			t.bi[key] = nr
		}
	}
	lb.frozen = true
}

// Compact folds the per-language counts into LangTables. Languages
// with too little data are dropped: a 200-token table adds noise, not
// signal.
func (lb *langBuilder) Compact() map[string]*model.LangTable {
	out := make(map[string]*model.LangTable, len(lb.tabs))
	for lang, t := range lb.tabs {
		var n int64
		for _, c := range t.uni {
			n += int64(c)
		}
		if n < 2048 {
			continue
		}
		uni := rowsToOrder(map[uint64]map[uint32]uint32{0: t.uni}, 1, 64)
		bi := rowsToOrder(t.bi, 2, 32)
		if uni == nil {
			continue
		}
		lt := &model.LangTable{}
		lt.Uni = *uni
		if bi != nil {
			lt.Bi = *bi
		}
		out[lang] = lt
	}
	return out
}

// structBuilder counts tokens under their lexical-context key.
type structBuilder struct {
	rows map[uint64]map[uint32]uint32
}

func newStructBuilder() *structBuilder {
	return &structBuilder{rows: map[uint64]map[uint32]uint32{}}
}

// Add walks the token stream tracking StructState and counts each token
// under the state in which it appeared.
func (sb *structBuilder) Add(toks []string, ids []uint32) {
	var st tokenize.StructState
	for i, t := range toks {
		if t == tokenize.NL || t == tokenize.EOF {
			continue
		}
		key := uint64(st.Key())
		row := sb.rows[key]
		if row == nil {
			row = map[uint32]uint32{}
			sb.rows[key] = row
		}
		row[ids[i]]++
		st.Advance(t)
	}
}

// Tighten caps each context row at its current top-64 tokens, bounding
// memory on very large corpora where common contexts accumulate the
// whole vocabulary.
func (sb *structBuilder) Tighten() {
	for key, row := range sb.rows {
		if len(row) <= 64 {
			continue
		}
		type tc struct {
			t uint32
			c uint32
		}
		top := make([]tc, 0, len(row))
		for tok, c := range row {
			top = append(top, tc{tok, c})
		}
		sort.Slice(top, func(i, j int) bool { return top[i].c > top[j].c })
		nr := make(map[uint32]uint32, 64)
		for _, e := range top[:64] {
			nr[e.t] = e.c
		}
		sb.rows[key] = nr
	}
}

// Compact emits the struct-context table. Rare contexts are pruned;
// each row keeps its most useful tokens.
func (sb *structBuilder) Compact() *model.Order {
	return rowsToOrder(sb.rows, 2, 24)
}

// subMinCnt is the pruning floor for the order-3 subtoken model,
// shared by the in-memory and spill build paths.
var subMinCnt = []uint32{0, 1, 1, 2}

// subtokIDs maps a token stream to the subtoken stream for the
// identifier model: identifiers split into parts, continuation parts
// carry a \x01 marker so the model can tell starts from middles, and
// every other token maps to itself in the sub-vocab.
func subtokIDs(toks []string, sv *model.Vocab) []uint32 {
	out := make([]uint32, 0, len(toks)+len(toks)/4)
	for _, t := range toks {
		if tokenize.IsIdentTok(t) && len(t) >= 3 {
			parts := tokenize.Subtoks(t)
			if len(parts) > 1 {
				out = append(out, sv.ID(parts[0]))
				for _, p := range parts[1:] {
					out = append(out, sv.ID("\x01"+p))
				}
				continue
			}
		}
		out = append(out, sv.ID(t))
	}
	return out
}
