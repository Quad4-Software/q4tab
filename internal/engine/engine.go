// Package engine merges the completion layers: verbatim line retrieval
// over the indexed corpus, a dynamic index of open/edited documents and
// learned completions, and the n-gram model interpolated with scoped
// caches. It owns the open-document store used by the LSP server.
package engine

import (
	"bytes"
	"encoding/json"
	"math"
	"os"
	"runtime"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"q4complete/internal/lines"
	"q4complete/internal/model"
	"q4complete/internal/symbols"
	"q4complete/internal/tokenize"
)

type Config struct {
	Lam        float64 // interpolation weight per order step
	Gamma      float64 // cache mixing denominator
	MaxItems   int
	MaxLines   int // max lines per inline completion
	ScanCap    int // line-index scan cap
	MaxToks    int // max generated tokens
	CacheOrdr  int
	CacheCap   int           // session cache token budget
	Budget     time.Duration // per-request latency budget
	MinProb    float64       // stop generation below this
	MinProbML  float64       // stricter floor for lines after the first
	MinLen     int           // minimum useful suggestion length
	Journal    string        // learned-completions journal path
	MaxDocSize int           // skip caches for documents larger than this
}

// Weights parameterizes candidate scoring so a tune pass can shift the
// balance between sources without code changes. Values are scores in
// the same units the scorer has always used.
type Weights struct {
	File   float64 `json:"file"`   // same-file verbatim repeat
	Dyn    float64 `json:"dyn"`    // dynamic index (open docs, learned, delta lines)
	Learn  float64 `json:"learn"`  // per-accept-count boost on exact learned lines
	FIM    float64 `json:"fim"`    // verified fill-in-the-middle bonus
	Model  float64 `json:"model"`  // model chain rank unit
	FIMIdx float64 `json:"fimIdx"` // index-verified (weaker) FIM bonus
}

func DefaultWeights() Weights {
	return Weights{
		File: 1e6, Dyn: 1e5, Learn: 2e5,
		FIM: 4e6, FIMIdx: 3e6, Model: 1,
	}
}

func DefaultConfig() Config {
	return Config{
		Lam:        0.8,
		Gamma:      4.0,
		MaxItems:   4,
		MaxLines:   4,
		ScanCap:    8192,
		MaxToks:    96,
		CacheOrdr:  3,
		CacheCap:   2 << 20,
		Budget:     30 * time.Millisecond,
		MinProb:    0.02,
		MinProbML:  0.05,
		MinLen:     2,
		MaxDocSize: 4 << 20,
	}
}

type doc struct {
	mu    sync.Mutex
	gen   int // bumped per UpdateDoc. Stale builds are dropped
	cache *model.Cache
	lines []string // normalized nonblank lines, for the dynamic index
}

// Engine is the live completion state: static model + line index plus
// dynamic caches for open documents, the editing session, and learned
// (accepted) completions.
// Locking: mu is a RWMutex. Complete holds RLock for the whole gather
// and generate path so many clients complete in parallel. Writers
// (UpdateDoc, Learn, InstallDelta) take Lock. Caches and the vocab
// carry their own internal locks, so they are safe to touch under the
// engine's read lock. recordShown runs under Lock at the end.
type Engine struct {
	cfg        Config
	m          *model.Model
	li         *lines.Index
	eofID      uint32
	nlID       uint32
	mu         sync.RWMutex
	docs       sync.Map // uri -> *doc. Never under e.mu so didOpen is not a writer
	docsN      atomic.Int64
	dirs       map[string]*model.Cache // per-directory caches: sibling-file idioms
	dirTok     map[string]int
	session    *model.Cache
	sessTok    int
	learned    *model.Cache
	learnTok   int            // tokens fed to the learned cache since last reset
	learnLines []string       // normalized accepted lines, merged into dyn index
	learnSet   map[string]int // completed line -> times accepted
	learnIdx   *lines.Index   // sorted learnLines for prefix lookups
	pendLearn  []string       // learned texts seen before the model loaded
	journal    string
	journalF   *os.File // lazily opened append handle for the journal
	learnN     int
	vocabCap   int                         // interning stops growing the vocab past this
	delta      *model.Cache                // incremental-index overlay: new/changed files
	deltaLines []string                    // delta file lines, merged into dyn index
	deltaFiles int                         // files in the current delta overlay
	deltaTok   int                         // tokens fed into the delta cache
	dyn        atomic.Pointer[lines.Index] // rebuilt lazily from open docs + learned
	// Builder queue state, all under dmu: pend maps uri -> queued doc
	// task, dynDirt marks a pending dyn rebuild, building marks a live
	// buildWorker. Kept off e.mu so didOpen bursts never starve
	// completion readers.
	dmu      sync.Mutex
	pend     map[string]docTask
	dynDirt  bool
	building bool
	bmu      sync.Mutex     // serializes drainOnce runs (worker and Flush)
	syms     *symbols.Index // definition sites for lookup_symbol

	// shown tracks the most recent suggestions per (doc, line) so a
	// Learn can be correlated with what was on screen: matched items
	// are accepts, the rest are implicit rejects. The accept rate per
	// source drives the adaptive probability floor.
	// shmu is a leaf lock for the shown table and accept/reject
	// accounting: CompleteFor finishes by taking it briefly instead of
	// the engine write lock, so shown bookkeeping never serializes
	// completion work. jmu guards journalF for the same reason.
	shmu     sync.Mutex
	jmu      sync.Mutex
	shown    map[uint64]shownEnt
	shownN   map[string]int
	accN     map[string]int
	rejN     map[string]int
	minProb  float64 // adaptive floor. Starts at cfg.MinProb
	winShown int     // model-source items shown since last tune
	winAcc   int     // model-source items accepted since last tune
	w        Weights // scoring weights. Tuned offline by `q4complete tune`

	// users holds per-tenant overlays for hosted mode: each tenant gets
	// a private cache + learned-line set so one server can learn each
	// caller separately without leaking idioms across users. Bounded by
	// maxUsrs. Least-recently-used entries are evicted. sync.Map because
	// overlays are created inside read-locked paths.
	users   sync.Map // string -> *userOverlay
	usersN  atomic.Int64
	maxUsrs int
}

type userOverlay struct {
	mu       sync.Mutex // guards learnSet and toks
	cache    atomic.Pointer[model.Cache]
	learnSet map[string]int
	toks     int
	lastUse  atomic.Int64 // unix nano, for LRU eviction
}

// shownEnt records what was displayed for a (doc, line) context.
type shownEnt struct {
	linePrefix string
	items      []Item
	at         time.Time
}

func New(cfg Config) *Engine {
	return &Engine{
		cfg:      cfg,
		pend:     make(map[string]docTask),
		dirs:     make(map[string]*model.Cache),
		dirTok:   make(map[string]int),
		learnSet: make(map[string]int),
		session:  model.NewCache(cfg.CacheOrdr),
		learned:  model.NewCache(cfg.CacheOrdr),
		shown:    make(map[uint64]shownEnt),
		shownN:   make(map[string]int),
		accN:     make(map[string]int),
		rejN:     make(map[string]int),
		minProb:  cfg.MinProb,
		w:        DefaultWeights(),
		maxUsrs:  512,
	}
}

// overlayFor returns the tenant overlay for user, creating it if
// needed. Empty user means the shared/global path.
func (e *Engine) overlayFor(user string) *userOverlay {
	if user == "" {
		return nil
	}
	nov := &userOverlay{learnSet: map[string]int{}}
	nov.cache.Store(model.NewCache(e.cfg.CacheOrdr))
	v, loaded := e.users.LoadOrStore(user, nov)
	o := v.(*userOverlay)
	o.lastUse.Store(time.Now().UnixNano())
	if !loaded && e.usersN.Add(1) > int64(e.maxUsrs) {
		e.evictUsers()
	}
	return o
}

// evictUsers drops the least recently used overlays past the cap.
// CompareAndDelete keeps the count honest when a racing overlayFor
// recreated or still holds an entry we picked.
func (e *Engine) evictUsers() {
	type ent struct {
		k string
		t int64
		v *userOverlay
	}
	var oldest []ent
	e.users.Range(func(k, v any) bool {
		o := v.(*userOverlay)
		oldest = append(oldest, ent{k.(string), o.lastUse.Load(), o})
		return true
	})
	sort.Slice(oldest, func(i, j int) bool { return oldest[i].t < oldest[j].t })
	for _, o := range oldest[:len(oldest)/4+1] {
		// Delete only if the entry is still the same one we measured:
		// a racing overlayFor may have already touched or recreated it.
		if e.users.CompareAndDelete(o.k, o.v) {
			e.usersN.Add(-1)
		}
	}
}

// SetWeights swaps the scoring weights (see `q4complete tune`).
func (e *Engine) SetWeights(w Weights) {
	e.mu.Lock()
	e.w = w
	e.mu.Unlock()
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
		// Runtime interning (open docs, learned text) extends the
		// vocab, but unbounded growth leaks memory over long sessions.
		// Cap at load size plus headroom. Beyond it, unseen tokens are
		// dropped from cache contexts.
		e.vocabCap = m.Vocab.Len() + 1<<20
		// Drain texts learned before the model was ready.
		for _, t := range e.pendLearn {
			e.learned.Add(e.intern(t), e.eofID)
		}
		e.pendLearn = nil
	}
}

// DeltaFile is one changed file recorded by the incremental indexer.
type DeltaFile struct {
	Path    string
	ModTime int64
	Data    []byte
}

// InstallDelta feeds the incremental-index overlay: files that changed
// since the base model was built. Their tokens go into a dedicated cache
// (base-vocab ids) and their lines into the dynamic index sources.
// Requires a model so tokens intern against the base vocab. Without one
// the delta is dropped.
func (e *Engine) InstallDelta(files []DeltaFile) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.m == nil {
		return
	}
	// An empty delta still clears the old overlay: a full rebuild may
	// have folded the changes into the base model.
	e.delta = model.NewCache(e.cfg.CacheOrdr)
	e.deltaLines = e.deltaLines[:0]
	e.deltaFiles = 0
	e.deltaTok = 0
	for _, f := range files {
		if len(f.Data) > e.cfg.MaxDocSize {
			continue
		}
		ids := e.intern(string(f.Data))
		e.delta.Add(ids, e.eofID)
		e.deltaTok += len(ids)
		e.deltaFiles++
		e.deltaLines = append(e.deltaLines, docLines(string(f.Data))...)
		if e.syms != nil {
			e.syms.RemovePath(f.Path)
			for _, s := range symbols.Extract(f.Path, f.Data) {
				e.syms.Add(s)
			}
		}
	}
	e.dmu.Lock()
	e.dynDirt = true
	e.kickBuilderLocked()
	e.dmu.Unlock()
}

// SetJournal points the learned-completions journal at path and loads it.
// Call after SetModel so learned tokens intern into the model vocab.
func (e *Engine) SetJournal(path string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.journal = path
	e.loadJournalLocked()
}

func (e *Engine) loadJournalLocked() {
	if e.journal == "" || e.m == nil {
		return
	}
	data, err := os.ReadFile(e.journal)
	if err != nil {
		return
	}
	// Cap replay size: keep the newest records.
	const maxReplay = 4 << 20
	if len(data) > maxReplay {
		data = data[len(data)-maxReplay:]
		if i := bytes.IndexByte(data, '\n'); i >= 0 {
			data = data[i+1:]
		}
	}
	for _, line := range bytes.Split(data, []byte("\n")) {
		if len(line) == 0 {
			continue
		}
		var rec struct {
			Text string `json:"t"`
		}
		if json.Unmarshal(line, &rec) != nil || rec.Text == "" {
			continue
		}
		if e.m != nil {
			e.learned.Add(e.intern(rec.Text), e.eofID)
		} else {
			e.pendLearn = append(e.pendLearn, rec.Text)
		}
		for _, l := range docLines(rec.Text) {
			e.learnLines = append(e.learnLines, l)
			e.learnSet[l]++
		}
		e.learnN++
	}
	e.trimLearnedLocked()
	e.dmu.Lock()
	e.dynDirt = true
	e.kickBuilderLocked()
	e.dmu.Unlock()
}

// trimLearnedLocked bounds the learned line memory. Call under e.mu.
func (e *Engine) trimLearnedLocked() {
	if len(e.learnLines) > 8192 {
		e.learnLines = e.learnLines[len(e.learnLines)-8192:]
	}
	if len(e.learnSet) > 16384 {
		// Rebuild from the surviving window rather than evicting
		// arbitrarily: counts drift but stay recent.
		e.learnSet = make(map[string]int, len(e.learnLines))
		for _, l := range e.learnLines {
			e.learnSet[l]++
		}
	}
	lb := lines.NewBuilder()
	for _, l := range e.learnLines {
		lb.AddLine(l)
	}
	e.learnIdx = lb.Compact()
}

// shownKey maps a (uri, line) pair to a compact shown-table key.
func shownKey(uri string, line int) uint64 {
	h := uint64(1469598103934665603)
	for i := 0; i < len(uri); i++ {
		h = h*1099511628211 ^ uint64(uri[i])
	}
	return h ^ uint64(line)*0x9e3779b97f4a7c15
}

// recordShown stores what was last displayed so a later Learn or
// Reject can be correlated. Takes only the leaf lock shmu, so it is
// safe under e.mu or with no engine lock at all.
func (e *Engine) recordShown(uri string, line int, linePrefix string, items []Item) {
	if len(items) == 0 {
		return
	}
	e.shmu.Lock()
	defer e.shmu.Unlock()
	if len(e.shown) > 1024 {
		e.shown = make(map[uint64]shownEnt) // bound memory. Keys churn fast
	}
	cp := make([]Item, len(items))
	copy(cp, items)
	e.shown[shownKey(uri, line)] = shownEnt{linePrefix, cp, time.Now()}
	for _, it := range items {
		src := baseSource(it.Source)
		e.shownN[src]++
		if src == "model" {
			e.winShown++
		}
	}
	e.tuneProbLocked()
}

// baseSource strips the "+learn" marker for accept-rate accounting.
func baseSource(s string) string {
	if i := strings.IndexByte(s, '+'); i >= 0 {
		return s[:i]
	}
	return s
}

// recordAccept matches the learned text against the shown table:
// the matched item is an accept, unmatched siblings are rejects.
// Takes shmu itself. Safe under e.mu or standalone.
func (e *Engine) recordAccept(uri, text string, line int) {
	key := shownKey(uri, line)
	e.shmu.Lock()
	se, ok := e.shown[key]
	if ok {
		delete(e.shown, key)
	}
	e.shmu.Unlock()
	if !ok || time.Since(se.at) > 2*time.Minute {
		return
	}
	completed := lines.Normalize(firstLineOf(text))
	for i, it := range se.items {
		if lines.Normalize(se.linePrefix+firstLineOf(it.Text)) == completed {
			e.shmu.Lock()
			src := baseSource(it.Source)
			e.accN[src]++
			if src == "model" {
				e.winAcc++
			}
			e.shmu.Unlock()
			e.journalEventLocked("a", it, i)
		} else {
			e.shmu.Lock()
			e.rejN[baseSource(it.Source)]++
			e.shmu.Unlock()
			e.journalEventLocked("r", it, i)
		}
	}
}

// Reject marks the suggestions shown at (uri, line) as not accepted.
// Clients without a native reject event may skip calling this. The
// accept correlation in Learn already covers implicit rejects.
func (e *Engine) Reject(uri string, line int) {
	key := shownKey(uri, line)
	e.shmu.Lock()
	se, ok := e.shown[key]
	if ok {
		delete(e.shown, key)
		for i := range se.items {
			e.rejN[baseSource(se.items[i].Source)]++
		}
	}
	e.shmu.Unlock()
	if !ok {
		return
	}
	for i, it := range se.items {
		e.journalEventLocked("r", it, i)
	}
}

func firstLineOf(s string) string {
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		return s[:i]
	}
	return s
}

// tuneProbLocked adapts the model probability floor to the observed
// accept rate: too many ignored model suggestions means the floor is
// too low, a high accept rate means it can afford to be looser. Called
// under e.mu whenever shown items are recorded.
func (e *Engine) tuneProbLocked() {
	if e.winShown < 256 {
		return
	}
	rate := float64(e.winAcc) / float64(e.winShown)
	if rate < 0.10 {
		e.minProb *= 1.2
		if e.minProb > 0.08 {
			e.minProb = 0.08
		}
	} else if rate > 0.50 {
		e.minProb /= 1.2
		if e.minProb < 0.008 {
			e.minProb = 0.008
		}
	}
	e.winShown = 0
	e.winAcc = 0
}

// Learn records an accepted completion. text should be the completed
// line(s): the line prefix plus the accepted text, so the learned entry
// is retrievable by the context that produced it. uri and line locate
// the accept for shown-suggestion correlation. Pass them as -1/"" when
// the caller does not know. Learned lines feed the n-gram learned cache
// and the dynamic line index, and persist to the journal.
func (e *Engine) Learn(uri, text string, line int) {
	e.LearnFor("", uri, text, line)
}

// LearnFor is Learn scoped to a tenant overlay. Shared accept/reject
// counters still update (they are aggregates, not content), but the
// accepted text itself goes only to the user's private overlay and
// never touches the shared learned cache, learned-line set, or journal:
// hosted tenants must not leak their code to each other.
func (e *Engine) LearnFor(user, uri, text string, line int) {
	e.mu.Lock()
	defer e.mu.Unlock()
	if text == "" {
		return
	}
	e.recordAccept(uri, text, line)
	if ov := e.overlayFor(user); ov != nil {
		if e.m != nil {
			ids := e.intern(text)
			ov.mu.Lock()
			ov.cache.Load().Add(ids, e.eofID)
			ov.toks += len(ids)
			for _, l := range docLines(text) {
				ov.learnSet[l]++
			}
			if ov.toks > e.cfg.CacheCap/4 {
				ov.cache.Store(model.NewCache(e.cfg.CacheOrdr))
				ov.learnSet = map[string]int{}
				ov.toks = 0
			}
			ov.mu.Unlock()
		}
		return // tenant-isolated: nothing below runs for hosted users
	}
	if e.m != nil {
		ids := e.intern(text)
		e.learned.Add(ids, e.eofID)
		e.learnTok += len(ids)
		if e.learnTok > e.cfg.CacheCap {
			// Bound the learned cache like the session cache. The
			// journal keeps everything regardless.
			e.learned = model.NewCache(e.cfg.CacheOrdr)
			e.learnTok = 0
		}
	} else {
		// No model yet: queue for cache replay when SetModel runs.
		e.pendLearn = append(e.pendLearn, text)
	}
	e.learnN++
	for _, l := range docLines(text) {
		e.learnLines = append(e.learnLines, l)
		e.learnSet[l]++
	}
	e.trimLearnedLocked()
	e.dmu.Lock()
	e.dynDirt = true
	e.kickBuilderLocked()
	e.dmu.Unlock()
	if e.journal != "" {
		rec, _ := json.Marshal(struct {
			Text string `json:"t"`
		}{text})
		// Keep an append handle open rather than open/write/close per
		// accepted completion. jmu serializes with journalEventLocked.
		e.jmu.Lock()
		if e.journalF == nil {
			if f, err := os.OpenFile(e.journal, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
				e.journalF = f
			}
		}
		if e.journalF != nil {
			e.journalF.Write(append(rec, '\n'))
		}
		e.jmu.Unlock()
		// Keep the journal bounded.
		if e.learnN%512 == 0 {
			if st, err := os.Stat(e.journal); err == nil && st.Size() > 4<<20 {
				e.compactJournalLocked()
			}
		}
	}
}

// journalEventLocked appends a labeled accept/reject record carrying
// the features the offline reranker trains on: the item's source,
// score, and shown rank. Serializes on jmu only. O_APPEND keeps each
// record a single atomic line.
func (e *Engine) journalEventLocked(kind string, it Item, rank int) {
	if e.journal == "" {
		return
	}
	rec, _ := json.Marshal(struct {
		E   string  `json:"e"`
		Src string  `json:"s"`
		Scr float64 `json:"v"`
		R   int     `json:"r"`
	}{kind, baseSource(it.Source), it.Score, rank})
	e.jmu.Lock()
	defer e.jmu.Unlock()
	if e.journalF == nil {
		if f, err := os.OpenFile(e.journal, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644); err == nil {
			e.journalF = f
		}
	}
	if e.journalF == nil {
		return
	}
	e.journalF.Write(append(rec, '\n'))
}

// compactJournalLocked truncates the journal to its newest records.
// Call under e.mu. Jmu serializes the file swap against event writers.
func (e *Engine) compactJournalLocked() {
	data, err := os.ReadFile(e.journal)
	if err != nil {
		return
	}
	keep := data
	if len(data) > 2<<20 {
		keep = data[len(data)-(2<<20):]
		if i := bytes.IndexByte(keep, '\n'); i >= 0 {
			keep = keep[i+1:]
		}
	}
	tmp := e.journal + ".tmp"
	if err := os.WriteFile(tmp, keep, 0o644); err == nil {
		e.jmu.Lock()
		os.Rename(tmp, e.journal)
		// The open append handle now points at the replaced inode.
		// drop it so the next Learn reopens the compacted file.
		if e.journalF != nil {
			e.journalF.Close()
			e.journalF = nil
		}
		e.jmu.Unlock()
	}
}

// LookupLines returns corpus and dynamic-index lines that start with
// the given prefix, most frequent first. Used by the MCP lookup tool.
func (e *Engine) LookupLines(prefix string, limit int) []string {
	e.mu.RLock()
	defer e.mu.RUnlock()
	seen := map[string]bool{}
	var out []string
	for _, idx := range []*lines.Index{e.dynIndex(), e.li} {
		if idx == nil {
			continue
		}
		for _, c := range idx.Complete(prefix, limit*2, 4096) {
			if !seen[c.Text] {
				seen[c.Text] = true
				out = append(out, c.Text)
				if len(out) >= limit {
					return out
				}
			}
		}
	}
	return out
}

// SetSymbols installs the definition index used by LookupSymbol.
func (e *Engine) SetSymbols(ix *symbols.Index) {
	e.mu.Lock()
	e.syms = ix
	e.mu.Unlock()
}

// LookupSymbol returns definition sites for an exact or prefix name.
func (e *Engine) LookupSymbol(name string, limit int) []symbols.Sym {
	e.mu.RLock()
	defer e.mu.RUnlock()
	if e.syms == nil {
		return nil
	}
	if exact := e.syms.Lookup(name); len(exact) > 0 {
		if len(exact) > limit {
			return exact[:limit]
		}
		return exact
	}
	return e.syms.LookupPrefix(name, limit)
}

// Stats reports index sizes and memory usage for the status command.
func (e *Engine) Stats() map[string]any {
	e.mu.RLock()
	defer e.mu.RUnlock()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	out := map[string]any{
		"docs":     e.docsN.Load(),
		"learned":  e.learnN,
		"heapMB":   ms.HeapAlloc >> 20,
		"sysMB":    ms.Sys >> 20,
		"sessToks": e.sessTok,
		"learnTok": e.learnTok,
		"delta":    e.deltaFiles,
		"deltaTok": e.deltaTok,
	}
	e.shmu.Lock()
	out["shownN"] = e.shownN
	out["acceptN"] = e.accN
	out["rejectN"] = e.rejN
	out["minProb"] = e.minProb
	out["users"] = e.usersN.Load()
	e.shmu.Unlock()
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

// UpdateDoc records the full text of an open document. All indexing
// goes through the background worker: it lexes and builds outside the
// engine lock, dedupes by generation, and batches the dynamic-index
// rebuild, so a burst of edits (or hundreds of didOpens) never stalls
// concurrent completions behind the write lock.
func (e *Engine) UpdateDoc(uri, text string) {
	dv, loaded := e.docs.LoadOrStore(uri, &doc{})
	d := dv.(*doc)
	if !loaded {
		e.docsN.Add(1)
	}
	d.mu.Lock()
	d.gen++
	t := docTask{uri: uri, d: d, gen: d.gen}
	if len(text) > e.cfg.MaxDocSize {
		d.cache = nil
		d.lines = nil
	} else {
		t.text = text
	}
	d.mu.Unlock()
	e.dmu.Lock()
	if t.text != "" {
		e.pend[uri] = t
	} else {
		delete(e.pend, uri)
	}
	e.dynDirt = true
	e.kickBuilderLocked()
	e.dmu.Unlock()
}

// dirOfURI maps a document URI to a directory key for the dir cache.
func dirOfURI(uri string) string {
	p := strings.TrimPrefix(uri, "file://")
	i := strings.LastIndexByte(p, '/')
	if i <= 0 {
		return ""
	}
	return p[:i]
}

// CloseDoc drops the per-file cache and its dynamic-index entries.
func (e *Engine) CloseDoc(uri string) {
	if _, ok := e.docs.LoadAndDelete(uri); ok {
		e.docsN.Add(-1)
	}
	e.dmu.Lock()
	delete(e.pend, uri)
	e.dynDirt = true
	e.kickBuilderLocked()
	e.dmu.Unlock()
}

// kickBuilderLocked starts the background builder if none is running.
// Callers must hold e.dmu.
func (e *Engine) kickBuilderLocked() {
	if e.building {
		return
	}
	e.building = true
	go e.buildWorker()
}

// docTask is a snapshot of one document awaiting indexing.
type docTask struct {
	uri  string
	d    *doc
	text string
	gen  int
}

// buildWorker drains pending document updates and dynamic-index rebuilds
// until no work remains, then exits. Lexing and index building run
// outside the lock. Interning and pointer swaps happen inside it, so
// completions never wait on document processing.
func (e *Engine) buildWorker() {
	defer func() {
		e.dmu.Lock()
		e.building = false
		// Work may have arrived after the last drain.
		if e.dynDirt || len(e.pend) > 0 {
			e.kickBuilderLocked()
		}
		e.dmu.Unlock()
	}()
	// bmu serializes drains with any concurrent Flush.
	e.bmu.Lock()
	defer e.bmu.Unlock()
	// Settle briefly so a burst of didOpens/didChanges coalesces into
	// one drain instead of hundreds of writer acquisitions starving
	// readers behind the writer-preferring RWMutex.
	time.Sleep(2 * time.Millisecond)
	for e.drainOnce() {
		// Yield between drains so queued readers get in before the
		// next write lock acquisition.
		time.Sleep(time.Millisecond)
	}
}

// drainOnce performs one snapshot/lex/install/rebuild cycle and
// reports whether it did any work. Callers serialize on e.bmu.
func (e *Engine) drainOnce() bool {
	e.dmu.Lock()
	tasks := make([]docTask, 0, len(e.pend))
	for uri, t := range e.pend {
		tasks = append(tasks, t)
		delete(e.pend, uri)
	}
	rebuild := e.dynDirt || len(tasks) > 0
	e.dynDirt = false
	e.dmu.Unlock()
	if len(tasks) == 0 && !rebuild {
		return false
	}

	// Pure work outside the lock: lex, intern, and build the per-doc
	// caches. Vocab and Cache carry their own locking, so the write lock
	// below only covers map writes and counter bumps. Readers never
	// stall behind tokenization or cache construction.
	type built struct {
		docTask
		ids   []uint32
		cache *model.Cache
		lines []string
	}
	var bs []built
	for _, t := range tasks {
		if len(t.text) > e.cfg.MaxDocSize {
			continue // oversized docs get no cache or line index
		}
		var ids []uint32
		var c *model.Cache
		if e.m != nil {
			ids = e.internToks(tokenize.Lex([]byte(t.text)))
			c = model.NewCache(e.cfg.CacheOrdr)
			c.Add(ids, e.eofID)
		}
		bs = append(bs, built{t, ids, c, docLines(t.text)})
	}

	// Install per-doc results under each doc's own lock. Docs updated
	// since the snapshot are dropped.
	var live []*built
	for i := range bs {
		b := &bs[i]
		b.d.mu.Lock()
		if b.d.gen == b.gen {
			b.d.cache = b.cache
			b.d.lines = b.lines
			live = append(live, b)
		}
		b.d.mu.Unlock()
	}

	// e.mu section: shared counters and get-or-create of session/dir
	// caches only. Cache.Add runs after release.
	type cacheAdd struct {
		c   *model.Cache
		ids []uint32
	}
	var adds []cacheAdd
	e.mu.Lock()
	for _, b := range live {
		e.sessTok += len(b.ids)
		if e.sessTok > e.cfg.CacheCap {
			e.session = model.NewCache(e.cfg.CacheOrdr)
			e.sessTok = 0
		}
		adds = append(adds, cacheAdd{e.session, b.ids})
		if dir := dirOfURI(b.uri); dir != "" && (len(e.dirs) < 64 || e.dirs[dir] != nil) {
			c := e.dirs[dir]
			if c == nil {
				c = model.NewCache(e.cfg.CacheOrdr)
				e.dirs[dir] = c
			}
			e.dirTok[dir] += len(b.ids)
			if e.dirTok[dir] > e.cfg.CacheCap/4 {
				e.dirs[dir] = model.NewCache(e.cfg.CacheOrdr)
				e.dirTok[dir] = 0
				c = e.dirs[dir]
			}
			adds = append(adds, cacheAdd{c, b.ids})
		}
	}
	// Snapshot for the dynamic index rebuild: docs map is lock-free,
	// learnLines/deltaLines need e.mu.
	var src [][]string
	if rebuild {
		e.docs.Range(func(_, v any) bool {
			d := v.(*doc)
			d.mu.Lock()
			if len(d.lines) > 0 {
				src = append(src, d.lines)
			}
			d.mu.Unlock()
			return true
		})
		src = append(src, e.learnLines, e.deltaLines)
	}
	e.mu.Unlock()

	for _, a := range adds {
		a.c.Add(a.ids, e.eofID)
	}

	if rebuild {
		b := lines.NewBuilder()
		for _, ls := range src {
			for _, l := range ls {
				b.AddLine(l)
			}
		}
		e.dyn.Store(b.Compact())
	}
	return true
}

// Flush drains pending document work synchronously. Tests and one-shot
// tools call it after UpdateDoc when they need the doc cache effective
// immediately. Serving paths rely on the background worker.
func (e *Engine) Flush() {
	e.bmu.Lock()
	defer e.bmu.Unlock()
	for e.drainOnce() {
	}
}

// hasPending reports whether any doc is waiting on the builder.
// Callers must hold e.dmu.
func (e *Engine) hasPending() bool {
	return len(e.pend) > 0
}

// docLines extracts normalized lines from document text.
func docLines(text string) []string {
	var out []string
	start := 0
	for i := 0; i <= len(text); i++ {
		if i == len(text) || text[i] == '\n' {
			if n := lines.Normalize(text[start:i]); len(n) >= 4 {
				out = append(out, n)
			}
			start = i + 1
		}
	}
	return out
}

// intern interns text to token ids. Tokens already in the vocab
// map back to their ids. Unseen tokens extend the vocab up to vocabCap,
// past which they are dropped (returned sequence omits them).
func (e *Engine) intern(text string) []uint32 {
	return e.internToks(tokenize.Lex([]byte(text)))
}

func (e *Engine) internToks(toks []string) []uint32 {
	ids := make([]uint32, 0, len(toks))
	for _, t := range toks {
		if id, ok := e.m.Vocab.Lookup(t); ok {
			ids = append(ids, id)
			continue
		}
		if e.vocabCap > 0 && e.m.Vocab.Len() >= e.vocabCap {
			continue // vocab full: drop rather than grow forever
		}
		ids = append(ids, e.m.Vocab.ID(t))
	}
	return ids
}

// dynIndex returns the dynamic line overlay. Rebuilds happen in the
// background worker, so the index may lag the latest edit by one build
// cycle. That is fine for a hint layer.
func (e *Engine) dynIndex() *lines.Index {
	return e.dyn.Load()
}

// Item is one completion candidate.
type Item struct {
	Text   string
	Source string // "corpus", "file", "model"
	Score  float64
	// ReplaceToEOL is true when the suggestion diverges from text
	// already after the cursor on this line. Clients should replace to
	// end-of-line rather than insert at the cursor.
	ReplaceToEOL bool
}

func isIdentByte(c byte) bool {
	return c == '_' || c == '$' || (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c >= 0x80
}

// Complete returns ranked inline-completion texts for (text, offset).
// uri may be empty for anonymous buffers. A panic anywhere in the
// pipeline (corrupt model data, an unexercised tokenizer edge) must not
// kill the language server. It degrades to no suggestions for that
// request instead.
func (e *Engine) Complete(uri, text string, offset int) []Item {
	return e.CompleteFor("", uri, text, offset)
}

// CompleteFor is Complete with a tenant overlay mixed in: the user's
// private cache and learned lines rank alongside the shared sources.
// Used by hosted mode so each caller's accepts only help that caller.
func (e *Engine) CompleteFor(user, uri, text string, offset int) (items []Item) {
	defer func() {
		if r := recover(); r != nil {
			items = nil
		}
	}()
	if offset > len(text) {
		offset = len(text)
	}
	if offset < 0 {
		offset = 0
	}
	prefix := text[:offset]
	lineStart := strings.LastIndexByte(prefix, '\n') + 1
	linePrefix := prefix[lineStart:]
	curLineIdx := strings.Count(prefix, "\n")

	// Normalized document tail, computed once for the accept checks.
	// Only a bounded window is needed: suggestions are at most MaxLines
	// lines long, so a few KB of tail covers every prefix check.
	suffix := text[offset:]
	if len(suffix) > 1<<13 {
		if i := strings.IndexByte(suffix[1<<13:], '\n'); i >= 0 {
			suffix = suffix[:1<<13+i]
		} else {
			suffix = suffix[:1<<13]
		}
	}
	ns := lines.Normalize(suffix)
	rest := suffix
	if eol := strings.IndexByte(suffix, '\n'); eol >= 0 {
		rest = suffix[:eol]
	}
	restN := lines.Normalize(rest)

	seen := map[string]bool{}

	// Same-file verbatim matches first: repeating yourself in this file
	// is the strongest signal of all.
	curLine := linePrefix
	if nl := strings.IndexByte(text[offset:], '\n'); nl >= 0 {
		curLine = text[lineStart : offset+nl]
	} else {
		curLine = text[lineStart:]
	}
	// Items that survived accept() with a non-empty rest-of-line
	// diverge from the existing tail: they must replace to EOL rather
	// than splice into it.
	replace := restN != ""
	mk := func(text, source string, score float64) Item {
		return Item{Text: text, Source: source, Score: score, ReplaceToEOL: replace}
	}

	exclude := lines.Normalize(curLine)
	// The gather/generate/post-pass path only reads engine state, so
	// it runs under the read lock: many clients complete in parallel.
	// Caches and the vocab carry their own internal locks.
	func() {
		e.mu.RLock()
		defer e.mu.RUnlock()
		w := e.w

		if len(text) <= e.cfg.MaxDocSize {
			for _, c := range fileLines(text, linePrefix, exclude, curLineIdx, 3) {
				if ok := e.accept(c, ns, restN); ok && !seen[c] {
					seen[c] = true
					items = append(items, mk(c, "file", w.File))
				}
			}
		}

		// Dynamic overlay: lines from other open documents and learned
		// completions.
		if di := e.dynIndex(); di != nil {
			for _, c := range di.Complete(linePrefix, e.cfg.MaxItems*2, 4096) {
				if ok := e.accept(c.Text, ns, restN); ok && !seen[c.Text] {
					seen[c.Text] = true
					items = append(items, mk(c.Text, "file", w.Dyn+float64(c.Count)))
				}
			}
		}

		// Corpus verbatim index.
		if e.li != nil {
			for _, c := range e.li.Complete(linePrefix, e.cfg.MaxItems*2, e.cfg.ScanCap) {
				if ok := e.accept(c.Text, ns, restN); ok && !seen[c.Text] {
					seen[c.Text] = true
					items = append(items, mk(c.Text, "corpus", float64(c.Count)))
				}
			}
		}

		// Model generation with cache mixing.
		if e.m != nil {
			for _, it := range e.generate(user, uri, text, offset, linePrefix) {
				if ok := e.accept(it.Text, ns, restN); ok && !seen[it.Text] {
					seen[it.Text] = true
					it.ReplaceToEOL = replace
					items = append(items, it)
				}
			}
		}

		// Post-pass per item: learned-line boost and fill-in-the-middle
		// verification.
		for i := range items {
			it := &items[i]
			// Learned completions get stickier the more often the same
			// completed line was accepted.
			first := it.Text
			if nl := strings.IndexByte(first, '\n'); nl >= 0 {
				first = first[:nl]
			}
			n := e.learnSet[lines.Normalize(linePrefix+first)]
			if ov := e.overlayFor(user); ov != nil {
				ov.mu.Lock()
				n += ov.learnSet[lines.Normalize(linePrefix+first)]
				ov.mu.Unlock()
			}
			if n > 0 {
				it.Score += float64(min(n, 10)) * w.Learn
				it.Source = it.Source + "+learn"
			} else if e.learnIdx != nil && len(first) >= 4 &&
				e.learnIdx.HasPrefix(lines.Normalize(linePrefix+first)) {
				// The suggestion is a strict prefix of a line that was
				// accepted before: weaker but still useful signal.
				it.Score += w.Learn / 2
				it.Source = it.Source + "+learn"
			}
			// Fill-in-the-middle, two forms. Both turn a replace-to-EOL
			// suggestion into a zero-width insert mid-line.
			if restN != "" && it.ReplaceToEOL && strings.IndexByte(it.Text, '\n') < 0 {
				// 1. The suggestion already ends with the existing tail:
				// the model regenerated what follows the cursor. Keep the
				// tail and insert only the novel middle. No index lookup
				// needed: the tail match is proof enough.
				// f(a, |b) suggesting "x, b)" inserts "x, " before "b)".
				if len(restN) >= 2 && len(it.Text) > len(rest) && strings.HasSuffix(it.Text, rest) {
					it.Text = strings.TrimSuffix(it.Text, rest)
					it.ReplaceToEOL = false
					it.Score += w.FIM
					it.Source = it.Source + "+fim"
					continue
				}
				// 2. A shorter suggestion that does not include the tail:
				// verify that prefix+suggestion+tail is attested in an
				// index before allowing a plain insert.
				joined := lines.Normalize(linePrefix + it.Text + rest)
				if len(joined) >= 4 {
					verified := e.li != nil && e.li.HasPrefix(joined)
					if !verified {
						if di := e.dynIndex(); di != nil {
							verified = di.HasPrefix(joined)
						}
					}
					if verified {
						it.ReplaceToEOL = false
						it.Score += w.FIMIdx
						it.Source = it.Source + "+fim"
					}
				}
			}
		}
	}()

	sort.SliceStable(items, func(i, j int) bool { return items[i].Score > items[j].Score })
	if len(items) > e.cfg.MaxItems {
		items = items[:e.cfg.MaxItems]
	}
	// Shown bookkeeping runs on its own leaf lock: taking the engine
	// write lock here would convoy every concurrent reader.
	e.recordShown(uri, curLineIdx, linePrefix, items)
	return items
}

// accept filters out suggestions that duplicate text already after the
// cursor or are degenerate (too short, whitespace-only). ns is the
// normalized document tail and restN the normalized rest of the current
// line, both computed once by the caller.
func (e *Engine) accept(sug, ns, restN string) bool {
	if len(sug) < e.cfg.MinLen {
		return false
	}
	ng := lines.Normalize(sug)
	if ng == "" {
		return false
	}
	if ns == "" {
		return true
	}
	// If the text right after the cursor already begins with this
	// suggestion, the code is already there.
	if strings.HasPrefix(ns, ng) {
		return false
	}
	// Rest-of-line overlap. A suggestion equal to the tail duplicates
	// it. A suggestion containing the tail mid-way diverges and then
	// re-includes it. But a suggestion that ENDS with the tail is the
	// fill-in-the-middle case: it regenerates what follows the cursor
	// and the post-pass trims it to a zero-width insert.
	if restN == "" {
		return true
	}
	if len(restN) >= 2 {
		if ng == restN || strings.HasSuffix(ng, restN) {
			return len(ng) > len(restN)
		}
		return !strings.Contains(ng, restN)
	}
	return !strings.HasSuffix(ng, restN)
}

// fileLines scans text for lines matching the prefix and returns their
// continuations, ranked by proximity to the cursor line. Repetitive
// files (import blocks, sibling test cases, numbered URLs) usually
// repeat the needed line near the cursor, so distance is the best
// ranking signal available without a second pass.
func fileLines(text, prefix, exclude string, curLine, limit int) []string {
	norm := lines.Normalize(prefix)
	if len(norm) < 3 {
		return nil
	}
	// Collect distinct continuations with their closest occurrence.
	best := map[string]int{} // continuation -> min line distance
	var order []string       // first-seen order, for stable output
	lineNo := 0
	start := 0
	for i := 0; i <= len(text); i++ {
		if i == len(text) || text[i] == '\n' {
			l := lines.Normalize(text[start:i])
			d := lineNo - curLine
			if d < 0 {
				d = -d
			}
			lineNo++
			start = i + 1
			if l != exclude && strings.HasPrefix(l, norm) && len(l) > len(norm)+1 {
				rest := l[len(norm):]
				if bd, ok := best[rest]; !ok {
					best[rest] = d
					order = append(order, rest)
				} else if d < bd {
					best[rest] = d
				}
			}
		}
	}
	sort.SliceStable(order, func(i, j int) bool {
		return best[order[i]] < best[order[j]]
	})
	if len(order) > limit {
		order = order[:limit]
	}
	return order
}

// generate produces model-driven continuations.
func (e *Engine) generate(user, uri, text string, offset int, linePrefix string) []Item {
	prefix := text[:offset]
	toks := tokenize.Lex([]byte(prefix))
	if len(toks) > 0 && toks[len(toks)-1] == tokenize.EOF {
		toks = toks[:len(toks)-1]
	}
	partial := ""
	if len(toks) > 0 && len(prefix) > 0 && isIdentByte(prefix[len(prefix)-1]) {
		partial = toks[len(toks)-1]
		toks = toks[:len(toks)-1]
	}
	// Intern (not only look up) so context tokens match the ids that
	// UpdateDoc/Learn put into the caches. Otherwise a context ending in
	// a novel identifier would hash differently and miss the doc cache
	// at exactly the orders where it is strongest. Interning is capped
	// at vocabCap so a long session cannot grow the vocab forever.
	ids := make([]uint32, 0, len(toks))
	for _, t := range toks {
		if id, ok := e.m.Vocab.Lookup(t); ok {
			ids = append(ids, id)
		} else if e.vocabCap == 0 || e.m.Vocab.Len() < e.vocabCap {
			ids = append(ids, e.m.Vocab.ID(t))
		}
	}
	ctx := ids
	if len(ctx) > e.m.N-1 {
		ctx = ctx[len(ctx)-(e.m.N-1):]
	}

	// Full text of the line being completed, for indent context.
	lineStart := strings.LastIndexByte(prefix, '\n') + 1
	curLine := text[lineStart:]
	if nl := strings.IndexByte(curLine, '\n'); nl >= 0 {
		curLine = curLine[:nl]
	}

	var caches []*model.Cache
	if dv, ok := e.docs.Load(uri); ok {
		d := dv.(*doc)
		d.mu.Lock()
		c := d.cache
		d.mu.Unlock()
		if c != nil {
			caches = append(caches, c)
		}
	}
	if dc := e.dirs[dirOfURI(uri)]; dc != nil {
		caches = append(caches, dc)
	}
	if ov := e.overlayFor(user); ov != nil {
		caches = append(caches, ov.cache.Load())
	}
	caches = append(caches, e.delta, e.session, e.learned)

	o := genOpts{
		multi:      e.allowMultiline(text, offset),
		indentUnit: detectIndent(text, uri),
		nextLines:  nextLinesNorm(text, offset, e.cfg.MaxLines),
		curLineRaw: curLine,
		openExpr:   endsInOperator(linePrefix),
	}
	if e.cfg.Budget > 0 {
		o.deadline = time.Now().Add(e.cfg.Budget)
	}

	// Per-request decode memo for the static model: the scorer revisits
	// each context row per candidate, so memoizing within this request
	// avoids repeated unpacks without a shared cache lock.
	q := e.m.NewQuery()
	var out []Item
	gen := e.chain(ctx, caches, partial, o, q)
	n := len(gen)
	for i, cand := range gen {
		if cand == "" {
			continue
		}
		// Rank-scaled so a confident chain outranks a weak one.
		out = append(out, Item{Text: cand, Source: "model",
			Score: e.w.Model * float64(n-i)})
	}
	return out
}

// nextLinesNorm returns the normalized content of the lines following
// the one containing offset, used to stop generation when it reproduces
// code that is already there.
func nextLinesNorm(text string, offset, n int) []string {
	rest := text[offset:]
	// Drop the remainder of the current line.
	i := strings.IndexByte(rest, '\n')
	if i < 0 {
		return nil
	}
	rest = rest[i+1:]
	var out []string
	start := 0
	for j := 0; j <= len(rest) && len(out) < n; j++ {
		if j == len(rest) || rest[j] == '\n' {
			if l := lines.Normalize(rest[start:j]); l != "" {
				out = append(out, l)
			}
			start = j + 1
		}
	}
	return out
}

// detectIndent finds the document's indentation unit.
func detectIndent(text, uri string) string {
	start := 0
	for i := 0; i <= len(text); i++ {
		if i == len(text) || text[i] == '\n' {
			line := text[start:i]
			start = i + 1
			j := 0
			for j < len(line) && (line[j] == ' ' || line[j] == '\t') {
				j++
			}
			if j > 0 && j < len(line) {
				if line[0] == '\t' {
					return "\t"
				}
				// Smallest space indent is the unit.
				n := j
				if n > 8 {
					n = 8
				}
				if n < 1 {
					n = 4
				}
				return strings.Repeat(" ", n)
			}
		}
	}
	if strings.HasSuffix(uri, ".go") {
		return "\t"
	}
	return "    "
}

// reindent rewrites a generated indent token in the document's unit,
// preserving visual depth (tabs count as one level, spaces are measured
// against the space unit width).
func reindent(genIndent, unit string) string {
	if unit == "" {
		return genIndent
	}
	depth := 0
	col := 0
	spaceW := len(unit)
	if spaceW == 0 {
		spaceW = 4
	}
	for i := 0; i < len(genIndent); i++ {
		if genIndent[i] == '\t' {
			depth++
			col = 0
		} else {
			col++
		}
	}
	depth += col / spaceW
	if unit == "\t" {
		return strings.Repeat("\t", depth)
	}
	return strings.Repeat(unit, depth)
}

// allowMultiline permits block completion only when the cursor is at the
// end of the line (nothing but whitespace after it). Completing
// mid-line with a multi-line ghost block reads badly and fights the
// existing suffix.
func (e *Engine) allowMultiline(text string, offset int) bool {
	if e.cfg.MaxLines <= 1 {
		return false
	}
	rest := text[offset:]
	if i := strings.IndexByte(rest, '\n'); i >= 0 {
		rest = rest[:i]
	}
	return strings.TrimSpace(rest) == ""
}

type genOpts struct {
	multi      bool
	indentUnit string
	nextLines  []string
	curLineRaw string // full text of the line being completed
	openExpr   bool   // line ends mid-expression. Suppress leading newline
	deadline   time.Time
}

// endsInOperator reports whether the line ends with an operator-like
// character, meaning a completion that starts with a newline would
// strand an incomplete expression.
func endsInOperator(linePrefix string) bool {
	p := strings.TrimRight(linePrefix, " \t")
	if p == "" {
		return false
	}
	switch p[len(p)-1] {
	case '=', '!', '<', '>', '&', '|', '+', '-', '*', '/', '%', '.', ':', '?',
		',', '(', '[', '{':
		return true
	}
	return strings.HasSuffix(p, "==") || strings.HasSuffix(p, "!=")
}

// probFloor reads the adaptive probability floor under its leaf lock.
func (e *Engine) probFloor() float64 {
	e.shmu.Lock()
	defer e.shmu.Unlock()
	return e.minProb
}

// chain runs generation for the top first-token variants and returns
// them best-first by mean log probability, so a confident chain outranks
// a lucky one regardless of which first token it sprang from.
func (e *Engine) chain(ctx []uint32, caches []*model.Cache, partial string, o genOpts, q *model.Query) []string {
	first := e.candidates(ctx, caches, q)
	var filt []model.Cand
	floor := e.probFloor()
	for _, c := range first {
		// The first token must clear the floor on its own merit.
		// generateBlock forces p=1.0 at step 0, so gate it here.
		if c.P < floor {
			continue
		}
		if partial != "" {
			if s := e.m.Vocab.Str(c.Tok); !strings.HasPrefix(s, partial) || len(s) <= len(partial) {
				continue
			}
		}
		filt = append(filt, c)
	}
	first = filt
	if len(first) == 0 {
		return nil
	}
	variants := 4
	if len(first) < variants {
		variants = len(first)
	}
	type scored struct {
		text string
		logp float64
	}
	var out []scored
	for v := 0; v < variants; v++ {
		// Later variants are refinement. Stop spending once we have
		// a decent chain or the request budget is gone.
		if !o.deadline.IsZero() && v > 0 && len(out) > 0 && time.Now().After(o.deadline) {
			break
		}
		text, lp := e.generateBlock(ctx, caches, first[v].Tok, first[v].P, partial, o, q)
		if text == "" {
			continue
		}
		dup := false
		for _, s := range out {
			if s.text == text {
				dup = true
			}
		}
		if !dup {
			out = append(out, scored{text, lp})
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].logp > out[j].logp })
	res := make([]string, 0, len(out))
	for _, s := range out {
		res = append(res, s.text)
	}
	return res
}

// generateBlock decodes one continuation greedily. It emits the rest of
// the current line, then optionally whole new lines until a stop rule:
// max lines, bracket dedent below the cursor's depth, a generated line
// that equals code already after the cursor, or low confidence.
func (e *Engine) generateBlock(ctx []uint32, caches []*model.Cache, firstTok uint32, firstP float64, partial string, o genOpts, q *model.Query) (string, float64) {
	cur := make([]uint32, len(ctx), len(ctx)+e.cfg.MaxToks)
	copy(cur, ctx)

	// Mean log probability over emitted tokens, seeded with the real
	// probability of the forced first token. Chains are ranked on this.
	logSum := math.Log(max(firstP, 1e-9))
	logN := 1

	var b strings.Builder    // committed output (finished lines + current)
	var line strings.Builder // in-progress new line, held for the dup check
	freshLine := false       // generating a line after a newline token
	linesDone := 0           // new lines committed (current line excluded)
	depth := 0               // bracket depth relative to the cursor
	curIndent := leadingWS(o.curLineRaw)

	emit := func(s string) {
		if freshLine {
			line.WriteString(s)
		} else {
			b.WriteString(s)
		}
	}
	flushLine := func() (dup bool) {
		nl := lines.Normalize(line.String())
		for _, ex := range o.nextLines {
			if nl != "" && nl == ex {
				return true
			}
		}
		b.WriteString(line.String())
		line.Reset()
		return false
	}

	// seen4 tracks 4-gram hashes emitted so far. Greedy decode on a
	// low-entropy context otherwise loops forever (e.g. "in a .mjs
	// file in a .mjs file ...").
	seen4 := map[uint64]bool{}

	for step := 0; step < e.cfg.MaxToks; step++ {
		if !o.deadline.IsZero() && step&7 == 0 && time.Now().After(o.deadline) {
			break
		}
		var t uint32
		var p float64
		if step == 0 {
			t, p = firstTok, 1.0
		} else {
			cands := e.candidates(cur, caches, q)
			if len(cands) == 0 {
				break
			}
			// Pick the first candidate that does not repeat a
			// generated 4-gram.
			ci := 0
			if len(cur) >= 3 {
				for ; ci < len(cands); ci++ {
					h := hash4(cur[len(cur)-3], cur[len(cur)-2], cur[len(cur)-1], cands[ci].Tok)
					if !seen4[h] {
						break
					}
				}
				if ci == len(cands) {
					break // every candidate loops. Stop the block
				}
			}
			t, p = cands[ci].Tok, cands[ci].P
			if len(cur) >= 3 {
				seen4[hash4(cur[len(cur)-3], cur[len(cur)-2], cur[len(cur)-1], t)] = true
			}
			logSum += math.Log(max(p, 1e-9))
			logN++
		}
		floor := e.probFloor()
		if linesDone > 0 {
			floor = e.cfg.MinProbML
		}
		if p < floor {
			break
		}
		if t == 0 || t == e.eofID {
			break
		}
		s := e.m.Vocab.Str(t)
		if s == "" {
			break
		}

		if t == e.nlID {
			if step == 0 && o.openExpr {
				goto done // leading newline would strand an open expression
			}
			if freshLine && flushLine() {
				goto done
			}
			b.WriteByte('\n')
			freshLine = true
			linesDone++
			if !o.multi || linesDone >= e.cfg.MaxLines {
				goto done
			}
			goto next
		}

		if s == "{" || s == "(" || s == "[" {
			depth++
		}
		if s == "}" || s == ")" || s == "]" {
			depth--
			if depth < 0 {
				// Closed a block that was open at the cursor: emit the
				// closer at the block's indent and stop.
				if freshLine && line.Len() == 0 {
					line.WriteString(curIndent)
				}
				emit(s)
				if freshLine {
					flushLine()
				}
				goto done
			}
		}

		if freshLine && line.Len() == 0 && isIndentToken(s) {
			line.WriteString(reindent(s, o.indentUnit))
			goto next
		}
		emit(s)

	next:
		cur = append(cur, t)
		if len(cur) > e.m.N-1 {
			cur = cur[len(cur)-(e.m.N-1):]
		}
	}
done:
	if freshLine {
		flushLine()
	}
	out := strings.TrimRight(b.String(), "\n")
	if partial != "" {
		if !strings.HasPrefix(out, partial) {
			return "", 0
		}
		out = out[len(partial):]
	}
	if len(out) < e.cfg.MinLen {
		return "", 0
	}
	return out, logSum / float64(logN)
}

func isIndentToken(s string) bool {
	return len(s) > 1 && (s[0] == ' ' || s[0] == '\t')
}

// hash4 hashes the last 3 context ids plus a candidate into a loop
// detector key.
func hash4(a, b, c, d uint32) uint64 {
	h := uint64(a)*1099511628211 ^ uint64(b)
	h = h*1099511628211 ^ uint64(c)
	h = h*1099511628211 ^ uint64(d)
	return h
}

func leadingWS(line string) string {
	i := 0
	for i < len(line) && (line[i] == ' ' || line[i] == '\t') {
		i++
	}
	return line[:i]
}

// candidates merges static-model candidates with all live caches.
func (e *Engine) candidates(ctx []uint32, caches []*model.Cache, q *model.Query) []model.Cand {
	static := e.m.TopUnion(ctx, e.cfg.Lam, 8, q)
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

// Persistence lives in store.go: a packed little-endian format that
// Load mmaps read-only so only touched pages become resident.
