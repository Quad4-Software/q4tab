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
