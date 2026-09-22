// Package engine merges the two completion layers: verbatim line
// retrieval over the indexed corpus, and the n-gram model interpolated
// with dynamic file/session caches. It owns the open-document store used
// by the LSP server.
package engine

import (
	"bytes"
	"encoding/gob"
	"os"
	"strings"
	"sync"

	"q4complete/internal/lines"
	"q4complete/internal/model"
	"q4complete/internal/tokenize"
)

type Config struct {
	Lam       float64 // interpolation weight per order step
	Gamma     float64 // cache mixing denominator
	MaxItems  int
	MaxLines  int // max lines per inline completion
	ScanCap   int // line-index scan cap
	MaxToks   int // max generated tokens
	CacheOrdr int
	CacheCap  int // session cache token budget
}

func DefaultConfig() Config {
	return Config{
		Lam:       0.8,
		Gamma:     4.0,
		MaxItems:  4,
		MaxLines:  1,
		ScanCap:   8192,
		MaxToks:   64,
		CacheOrdr: 3,
		CacheCap:  2 << 20,
	}
}

type doc struct {
	text  string
	cache *model.Cache
}

// Engine is the live completion state: static model + line index plus
// dynamic caches for open documents and the editing session.
type Engine struct {
	cfg     Config
	m       *model.Model
	li      *lines.Index
	eofID   uint32
	nlID    uint32
	mu      sync.Mutex
	docs    map[string]*doc
	session *model.Cache
	sessTok int
}

func New(cfg Config) *Engine {
	return &Engine{
		cfg:     cfg,
		docs:    make(map[string]*doc),
		session: model.NewCache(cfg.CacheOrdr),
	}
}

// SetModel installs the trained artifacts.
func (e *Engine) SetModel(m *model.Model, li *lines.Index) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.m = m
	e.li = li
	if m != nil {
		e.eofID, _ = m.Vocab.Lookup(tokenize.EOF)
		e.nlID, _ = m.Vocab.Lookup(tokenize.NL)
	}
}

// Stats reports index sizes for the status command.
func (e *Engine) Stats() map[string]any {
	e.mu.Lock()
	defer e.mu.Unlock()
	out := map[string]any{"docs": len(e.docs)}
	if e.li != nil {
		out["lines"] = e.li.Len()
	}
	if e.m != nil {
		out["vocab"] = e.m.Vocab.Len()
		out["order"] = e.m.N
		var rows int64
		for k := 1; k <= e.m.N; k++ {
			rows += int64(len(e.m.Orders[k].Keys))
		}
		out["contexts"] = rows
	}
	return out
}

// UpdateDoc records the full text of an open document and rebuilds its
// file-scoped cache.
func (e *Engine) UpdateDoc(uri, text string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	c := model.NewCache(e.cfg.CacheOrdr)
	ids := e.intern(text)
	c.Add(ids, e.eofID)
	e.docs[uri] = &doc{text: text, cache: c}
	e.session.Add(ids, e.eofID)
	e.sessTok += len(ids)
	if e.sessTok > e.cfg.CacheCap {
		e.session = model.NewCache(e.cfg.CacheOrdr)
		e.sessTok = 0
	}
}

// CloseDoc drops the per-file cache.
func (e *Engine) CloseDoc(uri string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	delete(e.docs, uri)
}

func (e *Engine) intern(text string) []uint32 {
	if e.m == nil {
		return nil
	}
	toks := tokenize.Lex([]byte(text))
	ids := make([]uint32, len(toks))
	for i, t := range toks {
		if id, ok := e.m.Vocab.Lookup(t); ok {
			ids[i] = id
		} else {
			ids[i] = e.m.Vocab.ID(t) // extend vocab for cache use
		}
	}
	return ids
}

// Item is one completion candidate.
type Item struct {
	Text   string
	Source string // "corpus", "file", "model"
	Score  float64
}

func isIdentByte(c byte) bool {
	return c == '_' || c == '$' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80
}

// Complete returns ranked inline-completion texts for (text, offset).
// uri may be empty for anonymous buffers.
func (e *Engine) Complete(uri, text string, offset int) []Item {
	e.mu.Lock()
	defer e.mu.Unlock()
	if offset > len(text) {
		offset = len(text)
	}
	prefix := text[:offset]
	lineStart := strings.LastIndexByte(prefix, '\n') + 1
	linePrefix := prefix[lineStart:]

	var items []Item
	seen := map[string]bool{}

	// Layer 1: verbatim retrieval from the corpus line index.
	if e.li != nil {
		for _, c := range e.li.Complete(linePrefix, e.cfg.MaxItems*2, e.cfg.ScanCap) {
			if !seen[c.Text] {
				seen[c.Text] = true
				items = append(items, Item{Text: c.Text, Source: "corpus", Score: float64(c.Count)})
			}
		}
	}

	// Layer 1b: same-file verbatim matches, excluding the current line.
	curLine := linePrefix
	if nl := strings.IndexByte(text[offset:], '\n'); nl >= 0 {
		curLine = text[lineStart : offset+nl]
	}
	exclude := lines.Normalize(curLine)
	if d, ok := e.docs[uri]; ok && d.text != "" {
		for _, c := range fileLines(d.text, linePrefix, exclude, 3) {
			if !seen[c] {
				seen[c] = true
				items = append(items, Item{Text: c, Source: "file", Score: 1e6})
			}
		}
	}

	// Layer 2: model generation with cache mixing.
	if e.m != nil {
		for _, it := range e.generate(uri, prefix, linePrefix) {
			if !seen[it.Text] {
				seen[it.Text] = true
				items = append(items, it)
			}
		}
	}

	if len(items) > e.cfg.MaxItems {
		items = items[:e.cfg.MaxItems]
	}
	return items
}

// fileLines scans the open document for lines matching the prefix and
// returns their continuations. Same-file repeats deserve top billing.
func fileLines(text, prefix, exclude string, limit int) []string {
	norm := lines.Normalize(prefix)
	if len(norm) < 3 {
		return nil
	}
	var out []string
	start := 0
	for i := 0; i <= len(text) && len(out) < limit*4; i++ {
		if i == len(text) || text[i] == '\n' {
			l := lines.Normalize(text[start:i])
			start = i + 1
			if l != exclude && strings.HasPrefix(l, norm) && len(l) > len(norm)+1 {
				rest := l[len(norm):]
				dup := false
				for _, o := range out {
					if o == rest {
						dup = true
						break
					}
				}
				if !dup {
					out = append(out, rest)
				}
			}
		}
	}
	if len(out) > limit {
		out = out[:limit]
	}
	return out
}

// generate produces model-driven continuations.
func (e *Engine) generate(uri, prefix, linePrefix string) []Item {
	// Tokenize the text before the cursor.
	toks := tokenize.Lex([]byte(prefix))
	// Drop the trailing EOF; the last real token may be a partial ident.
	if len(toks) > 0 && toks[len(toks)-1] == tokenize.EOF {
		toks = toks[:len(toks)-1]
	}
	partial := ""
	if len(toks) > 0 && len(prefix) > 0 && isIdentByte(prefix[len(prefix)-1]) {
		partial = toks[len(toks)-1]
		toks = toks[:len(toks)-1]
	}
	ids := make([]uint32, 0, len(toks))
	for _, t := range toks {
		if id, ok := e.m.Vocab.Lookup(t); ok {
			ids = append(ids, id)
		}
	}
	// Trim context to model order-1.
	ctx := ids
	if len(ctx) > e.m.N-1 {
		ctx = ctx[len(ctx)-(e.m.N-1):]
	}

	multi := e.allowMultiline(linePrefix)

	var caches []*model.Cache
	if d, ok := e.docs[uri]; ok {
		caches = append(caches, d.cache)
	}
	caches = append(caches, e.session)

	var out []Item
	for _, cand := range e.chain(ctx, caches, partial, multi) {
		if cand == "" {
			continue
		}
		out = append(out, Item{Text: cand, Source: "model", Score: 1})
	}
	return out
}

func (e *Engine) allowMultiline(linePrefix string) bool {
	if e.cfg.MaxLines <= 1 {
		return false
	}
	p := strings.TrimRight(linePrefix, " \t")
	if p == "" {
		return true
	}
	last := p[len(p)-1]
	return last == '{' || last == '(' || last == '[' || last == ':' || last == ','
}

// chain runs greedy generation, returning up to two variants (the argmax
// path and, if distinct, the path from the runner-up first token).
func (e *Engine) chain(ctx []uint32, caches []*model.Cache, partial string, multi bool) []string {
	first := e.candidates(ctx, caches)
	if partial != "" {
		var filt []model.Cand
		for _, c := range first {
			if strings.HasPrefix(e.m.Vocab.Str(c.Tok), partial) && len(e.m.Vocab.Str(c.Tok)) > len(partial) {
				filt = append(filt, c)
			}
		}
		first = filt
	}
	if len(first) == 0 {
		return nil
	}
	variants := 1
	if len(first) > 1 {
		variants = 2
	}
	var out []string
	for v := 0; v < variants; v++ {
		var gen []string
		cur := ctx
		lines := 1
		for step := 0; step < e.cfg.MaxToks; step++ {
			var cands []model.Cand
			if step == 0 {
				cands = first
				if v >= len(cands) {
					break
				}
				cands = cands[v : v+1]
			} else {
				cands = e.candidates(cur, caches)
			}
			if len(cands) == 0 || cands[0].P < 0.02 {
				break
			}
			t := cands[0].Tok
			if t == 0 || t == e.eofID {
				break
			}
			if t == e.nlID {
				if !multi || lines >= e.cfg.MaxLines {
					break
				}
				lines++
			}
			s := e.m.Vocab.Str(t)
			if s == "" {
				break
			}
			gen = append(gen, s)
			cur = append(cur, t)
			if len(cur) > e.m.N-1 {
				cur = cur[len(cur)-(e.m.N-1):]
			}
		}
		if len(gen) == 0 {
			continue
		}
		// Strip the partial prefix from the first generated token.
		if partial != "" {
			if !strings.HasPrefix(gen[0], partial) {
				continue
			}
			gen[0] = gen[0][len(partial):]
		}
		text := tokenize.Detok(gen)
		if len(text) < 2 {
			continue
		}
		dup := false
		for _, o := range out {
			if o == text {
				dup = true
			}
		}
		if !dup {
			out = append(out, text)
		}
	}
	return out
}

// candidates merges static-model candidates with all live caches.
func (e *Engine) candidates(ctx []uint32, caches []*model.Cache) []model.Cand {
	static := e.m.TopUnion(ctx, e.cfg.Lam, 8)
	merged := static
	for _, c := range caches {
		if c == nil {
			continue
		}
		dist, tot := c.Dist(ctx)
		merged = model.MixCandidates(merged, dist, tot, e.cfg.Gamma)
	}
	return merged
}

// --- persistence ---

type storeFile struct {
	Vocab    []string
	N        int
	Orders   []model.Order
	LineKeys []string
	LineCnts []int32
}

// Save writes model + line index to path.
func Save(path string, m *model.Model, li *lines.Index) error {
	sf := storeFile{N: m.N, Vocab: m.Vocab.Strings()}
	sf.Orders = m.Orders[1:]
	if li != nil {
		sf.LineKeys = li.Keys
		sf.LineCnts = li.Cnts
	}
	var buf bytes.Buffer
	if err := gob.NewEncoder(&buf).Encode(sf); err != nil {
		return err
	}
	return os.WriteFile(path, buf.Bytes(), 0o644)
}

// Load reads model + line index from path.
func Load(path string) (*model.Model, *lines.Index, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, nil, err
	}
	var sf storeFile
	if err := gob.NewDecoder(bytes.NewReader(data)).Decode(&sf); err != nil {
		return nil, nil, err
	}
	v := model.NewVocab()
	for _, s := range sf.Vocab {
		v.ID(s)
	}
	m := &model.Model{Vocab: v, N: sf.N}
	m.Orders = make([]model.Order, sf.N+1)
	for i, o := range sf.Orders {
		m.Orders[i+1] = o
	}
	var li *lines.Index
	if len(sf.LineKeys) > 0 {
		li = &lines.Index{Keys: sf.LineKeys, Cnts: sf.LineCnts}
	}
	return m, li, nil
}
