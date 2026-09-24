package engine

import (
	"strings"
	"testing"

	"q4tab/internal/tokenize"
)

// hasText reports whether any item contains sub.
func hasText(items []Item, sub string) bool {
	for _, it := range items {
		if strings.Contains(it.Text, sub) {
			return true
		}
	}
	return false
}

const storeSrc = `package demo

import "sync"

var ErrNotFound = errors.New("not found")

type Store struct {
	mu    sync.RWMutex
	items map[string]Item
}

func (s *Store) Get(id string) (Item, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.items[id], nil
}

func (s *Store) Put(it Item) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.items[it.ID] = it
}

func (s *Store) List() []Item {
	return nil
}
`

const serverHead = `package demo

import "net/http"

type Server struct {
	st *Store
}

`

func TestMemberSessionCrossFile(t *testing.T) {
	// store.go is open in the session; server.go is being typed. The
	// methods Store declares in the sibling file must complete at
	// s.st. even though server.go never mentions them.
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///p/store.go", storeSrc)
	e.Flush()

	text := serverHead + "func (s *Server) handleItems(w http.ResponseWriter, r *http.Request) {\n\ts.st."
	items := e.Complete("file:///p/server.go", text, len(text))
	for _, want := range []string{"Get(", "Put(", "List("} {
		if !hasText(items, want) {
			t.Fatalf("member %q missing at s.st.: %v", want, texts(items))
		}
	}

	// Killswitch.
	t.Setenv("Q4TAB_DISABLE", "mem")
	items = e.Complete("file:///p/server.go", text, len(text))
	if hasText(items, "Get(") {
		t.Fatalf("member completion fired despite Q4TAB_DISABLE=mem: %v", texts(items))
	}
}

func TestMemberChainResolution(t *testing.T) {
	// it, err := s.st.Get already used once, bare "st." should still
	// resolve: st is a field of Server whose type is Store.
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///p/store.go", storeSrc)
	e.Flush()

	text := serverHead + "func (s *Server) handleItem(w http.ResponseWriter, r *http.Request) {\n\ts.st."
	items := e.Complete("file:///p/server.go", text, len(text))
	if !hasText(items, "Get(") {
		t.Fatalf("chained s.st. did not resolve to Store members: %v", texts(items))
	}
}

func TestMemberCorpusTable(t *testing.T) {
	// The type's methods live in the trained corpus, not the session.
	// Two occurrences clear the compaction floor.
	var b strings.Builder
	for i := 0; i < 3; i++ {
		b.WriteString("func (v *Vault) Seal() error { return nil }\n")
		b.WriteString("func (v *Vault) Open() error { return nil }\n")
	}
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{"f.go": b.String()})
	if len(e.tyMem["Vault"]) == 0 {
		t.Fatal("corpus TypeMem not built")
	}
	// v is a *Vault receiver in a doc that never declares the methods.
	text := "package p\nfunc (v *Vault) Rotate() error {\n\tv."
	items := e.Complete("file:///p/x.go", text, len(text))
	if !hasText(items, "Seal(") || !hasText(items, "Open(") {
		t.Fatalf("corpus member table missed: %v", texts(items))
	}
}

func TestCallMemberCompletion(t *testing.T) {
	// Corpus habit: NewDecoder results get .Decode called on them.
	var b strings.Builder
	for i := 0; i < 3; i++ {
		b.WriteString("v := json.NewDecoder(r.Body).Decode(&x)\n")
	}
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{"f.go": b.String()})
	text := "package p\nfunc f() {\n\tjson.NewDecoder(r.Body)."
	items := e.Complete("file:///p/x.go", text, len(text))
	if !hasText(items, "Decode(") {
		t.Fatalf("call-result member missed: %v", texts(items))
	}
}

func TestErrorSentinelSynthesis(t *testing.T) {
	// ErrNotFound is declared in a sibling session doc; the
	// errors.Is(err, position should offer it.
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///p/store.go", storeSrc)
	e.Flush()

	text := serverHead + "func (s *Server) handleItem(w http.ResponseWriter, r *http.Request) {\n\tit, err := s.st.Get(id)\n\tif errors.Is(err,"
	items := e.Complete("file:///p/server.go", text, len(text))
	if !hasText(items, "ErrNotFound") {
		t.Fatalf("scoped sentinel not offered at errors.Is: %v", texts(items))
	}
}

func TestDocLiteralFallback(t *testing.T) {
	// The path literal the file already routes on should be offered
	// inside a string-taking call.
	e := buildEngine(t, DefaultConfig())
	text := serverHead + `func (s *Server) routes() {
	http.HandleFunc("/items/", s.handleItem)
}
func (s *Server) handleItem(w http.ResponseWriter, r *http.Request) {
	id := strings.TrimPrefix(r.URL.Path,`
	items := e.Complete("file:///p/server.go", text, len(text))
	if !hasText(items, `"/items/"`) {
		t.Fatalf("doc literal not offered in call args: %v", texts(items))
	}
}

func TestPlausibleFilters(t *testing.T) {
	cases := []struct {
		name, cand, line string
		want             bool
	}{
		{"member after dot", "Get(", "s.st.", true},
		{"prose after dot", " The completed {", "s.st.", false},
		{"scalar after dot", " int `json:\"x\"`", "s.st.", false},
		{"brace after brace", " {\n\t\t}", "switch x {", false},
		{"block after brace ok", "\n\tcase 1:", "switch x {", true},
		{"prose line", " The completed {", "x := f(", false},
		{"scalar nonconversion", " int `json:\"x\"`", "f(", false},
		{"conversion ok", "int(v)", "f(", true},
		{"composite lit ok", "map[string]any{", "x := f(", true},
		{"plain arg", ` "x")`, "f(", true},
		{"decl after send", " i := 0; i < n", "ch <-", false},
		{"decl in call arg", " i := 0", "f(", false},
		{"decl at stmt start ok", " i := 0", "x = f()\n\t", true},
		{"dot junk in operand", ".N; i := 0", "ch <-", false},
		{"comment prose midline", "i++ {\n\t}// The completed {", "for x", false},
		{"comment lowercase ok", "i++ // bump it", "for x", true},
	}
	for _, c := range cases {
		if got := plausible(c.cand, c.line); got != c.want {
			t.Errorf("%s: plausible(%q, %q) = %v, want %v", c.name, c.cand, c.line, got, c.want)
		}
	}
}

func TestExtractFactsBasics(t *testing.T) {
	f := extractFactsToks(tokenize.Lex([]byte(storeSrc)))
	if !f.tyMem["Store"]["Get("] || !f.tyMem["Store"]["Put("] {
		t.Fatalf("methods not extracted: %v", f.tyMem["Store"])
	}
	if f.tyFld["Store"]["items"] != "Item" {
		t.Fatalf("field type wrong: %v", f.tyFld["Store"])
	}
	if f.recv["s"] != "Store" {
		t.Fatalf("receiver type wrong: %v", f.recv["s"])
	}
	if !f.decls["ErrNotFound"] || !f.decls["Store"] {
		t.Fatalf("decls missing: %v", f.decls)
	}
}

func TestMemberIndexChain(t *testing.T) {
	// "w.q.jobs[0]." collapses the index to its container link:
	// Worker -> Queue -> []Job -> Job fields.
	chain, call, indexed, ok := dotChain("\tw.q.jobs[0].")
	if !ok || call != "" {
		t.Fatalf("dotChain: %v %q %v", chain, call, ok)
	}
	if !indexed {
		t.Fatal("jobs[0] should mark the tail segment indexed")
	}
	if len(chain) != 3 || chain[0] != "w" || chain[2] != "jobs" {
		t.Fatalf("chain wrong: %v", chain)
	}

	f := extractFactsToks(tokenize.Lex([]byte(`package q

type Job struct {
	ID string
}

type Queue struct {
	jobs []Job
}

type Worker struct {
	q *Queue
}

func (w *Worker) tick() {
}
`)))
	mems, _ := membersFor(chain, true, f, nil, nil, nil)
	var got []string
	for _, m := range mems {
		got = append(got, m)
	}
	if len(mems) == 0 || mems[0] != "ID" {
		t.Fatalf("index chain members wrong: %v", got)
	}
}

func TestReturnTypeCompletion(t *testing.T) {
	// "NewQueue()." has no observed chained-member usage; the
	// declared result type carries it.
	f := extractFactsToks(tokenize.Lex([]byte(`package q

type Queue struct {
	jobs []int
}

func NewQueue() *Queue { return &Queue{} }

func (q *Queue) Push(j int) {}
func (q *Queue) Len() int { return 0 }

func bad() (int, error) { return 0, nil }
`)))
	if f.retT["NewQueue"] != "Queue" {
		t.Fatalf("retT wrong: %v", f.retT)
	}
	if f.retT["bad"] != "" {
		t.Fatalf("builtin results should not record: %v", f.retT)
	}
	mems := callMembers("NewQueue", f, nil, nil, nil)
	var flat []string
	for _, m := range mems {
		flat = append(flat, m)
	}
	want := map[string]bool{"Push(": true, "Len(": true, "jobs": true}
	for _, w := range []string{"Push(", "Len(", "jobs"} {
		found := false
		for _, m := range flat {
			if m == w {
				found = true
			}
		}
		if !found {
			t.Fatalf("want %q in %v", w, flat)
		}
	}
	_ = want
}

func TestCallMemCorpusInjection(t *testing.T) {
	// A function whose result type has corpus members gets them in
	// CallMem even with zero observed chained uses.
	fb := newFactBuilder()
	fb.Add(tokenize.Lex([]byte(`package q

type Sess struct{}

func (s *Sess) Close() {}
func (s *Sess) Flush() {}

func Dial() (*Sess, error) { return nil, nil }
`)))
	tyMem, callM := fb.Compact()
	if len(tyMem["Sess"]) == 0 {
		t.Fatal("no Sess members")
	}
	var flat string
	for _, m := range callM["Dial"] {
		flat += m + " "
	}
	if !strings.Contains(flat, "Close(") {
		t.Fatalf("Dial should offer Sess members via retT: %v", flat)
	}
}

func TestMapVarIndexCompletion(t *testing.T) {
	// "var m map[string]*Session" then "m[k]." resolves to the value
	// type's fields, while bare "m." stays unresolved (maps have no
	// members worth completing).
	f := extractFactsToks(tokenize.Lex([]byte(`package q

type Session struct {
	Token string
}

func x() {
	var m map[string]*Session
	m["k"].`)))
	if f.elem["m"] != "Session" {
		t.Fatalf("elem wrong: %v", f.elem)
	}
	if f.recv["m"] != "" {
		t.Fatalf("map var should not record recv type: %v", f.recv)
	}
	chain, _, indexed, ok := dotChain("\tm[\"k\"].")
	if !ok || !indexed || len(chain) != 1 {
		t.Fatalf("dotChain: %v %v %v", chain, indexed, ok)
	}
	mems, _ := membersFor(chain, indexed, f, nil, nil, nil)
	if len(mems) == 0 || mems[0] != "Token" {
		t.Fatalf("map index members wrong: %v", mems)
	}
}

func TestRangeVarElem(t *testing.T) {
	f := extractFactsToks(tokenize.Lex([]byte(`package q

type Job struct{ ID string }

type Store struct {
	jobs []Job
}

func (s *Store) all() {
	for _, j := range s.jobs {
		_ = j
	}
}
`)))
	if f.recv["j"] != "Job" {
		t.Fatalf("range var type wrong: %v", f.recv)
	}
}

func TestSliceParamElem(t *testing.T) {
	// "func f(ss []Session)" binds ss's element type, not a type.
	f := extractFactsToks(tokenize.Lex([]byte(`package q

type Session struct{ Token string }

func f(ss []Session) {
}
`)))
	if f.elem["ss"] != "Session" {
		t.Fatalf("slice param elem wrong: %v", f.elem)
	}
	if f.recv["ss"] != "" {
		t.Fatalf("slice param should not record recv: %v", f.recv)
	}
}

func TestPythonAndTSFacts(t *testing.T) {
	f := extractFactsToks(tokenize.Lex([]byte(`class Queue:
    def push(self, job):
        pass
    def pop(self):
        pass

def make_queue() -> Queue:
    return Queue()

q = Queue()
`)))
	if !f.tyMem["Queue"]["push("] || !f.tyMem["Queue"]["pop("] {
		t.Fatalf("class methods missing: %v", f.tyMem)
	}
	if f.retT["make_queue"] != "Queue" {
		t.Fatalf("py retT wrong: %v", f.retT)
	}
	if f.recv["q"] != "Queue" {
		t.Fatalf("ctor call recv wrong: %v", f.recv)
	}

	ts := extractFactsToks(tokenize.Lex([]byte(`interface User {
	id: string;
	name: string;
	save(): void;
}

const u: User = getUser();
u.`)))
	if !ts.tyMem["User"]["id"] || !ts.tyMem["User"]["save("] {
		t.Fatalf("interface members missing: %v", ts.tyMem)
	}
	if ts.recv["u"] != "User" {
		t.Fatalf("annotation recv wrong: %v", ts.recv)
	}
	chain, _, _, ok := dotChain("u.")
	if !ok {
		t.Fatal("dotChain failed")
	}
	mems, _ := membersFor(chain, false, ts, nil, nil, nil)
	if len(mems) == 0 {
		t.Fatal("no members for annotated u")
	}
}

func TestDiffEditsRename(t *testing.T) {
	old := "package x\n\nfunc f() {\n\ts.items[id] = it\n\t_ = s.items\n}"
	new := "package x\n\nfunc f() {\n\ts.jobs[id] = it\n\t_ = s.jobs\n}"
	rules := diffEdits(old, new, 8)
	found := false
	for _, r := range rules {
		if r.fromS == "items" && r.toS == "jobs" {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected items->jobs rule, got %+v", rules)
	}
	// apply splices at token boundaries only
	r := editRule{from: []string{"items"}, to: []string{"jobs"}, fromS: "items", toS: "jobs"}
	if nv, ok := r.apply("s.items[id] = it"); !ok || nv != "s.jobs[id] = it" {
		t.Fatalf("apply: %q %v", nv, ok)
	}
	if _, ok := r.apply("s.itemsX = 1"); ok {
		t.Fatal("must not rewrite inside a larger identifier")
	}
	if _, ok := r.apply("no match here"); ok {
		t.Fatal("absent window must not apply")
	}
}

func TestEnclosingBlockLines(t *testing.T) {
	doc := "package x\n\nfunc a() {\n\tx := 1\n}\n\nfunc b() {\n\ty := 2\n\tz := "
	set := enclosingBlockLines(doc)
	if set == nil {
		t.Fatal("no block found")
	}
	if !set["y := 2"] && !set["\ty := 2"] {
		t.Fatalf("enclosing block missing its lines: %v", set)
	}
	joined := ""
	for k := range set {
		joined += k + "|"
	}
	if !strings.Contains(joined, "y := 2") {
		t.Fatal("block should contain b's body")
	}
	if strings.Contains(joined, "x := 1") {
		t.Fatal("block should not contain a's body")
	}
}

func TestRankerLearns(t *testing.T) {
	r := NewRanker()
	it := Item{Text: "x := f()", Source: "file", Score: 10}
	feat := itemFeat(&it, 2, false, false, false, false, false)
	before := r.mult(feat)
	for i := 0; i < 400; i++ {
		r.update(feat, true)
	}
	after := r.mult(feat)
	if after <= before {
		t.Fatalf("accepts should raise the multiplier: %v -> %v", before, after)
	}
	for i := 0; i < 400; i++ {
		r.update(feat, false)
	}
	if r.mult(feat) >= after {
		t.Fatal("rejects should lower the multiplier")
	}
}

func TestEditPropagationEndToEnd(t *testing.T) {
	e := New(DefaultConfig())
	// User just renamed items -> jobs; a stale call site now suggests
	// the new name even though the file text still says items.
	e.emu.Lock()
	e.edits = append(e.edits, editRule{
		from: []string{"items"}, to: []string{"jobs"},
		fromS: "items", toS: "jobs",
	})
	e.emu.Unlock()
	// Cursor sits after "s.": file repeats offer "items[...]" and the
	// rule rewrites them to "jobs".
	doc := "package x\n\nfunc f(s *Store) {\n\t_ = s.items[\"a\"]\n\t_ = s.items[\"b\"]\n\t_ = s."
	got := e.Complete("file:///x.go", doc, len(doc))
	var saw bool
	for _, it := range got {
		if strings.Contains(it.Text, "jobs") {
			saw = true
		}
	}
	if !saw {
		t.Fatalf("no jobs variant propagated; items=%v", got)
	}
}
