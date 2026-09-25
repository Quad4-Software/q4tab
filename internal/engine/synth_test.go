package engine

import (
	"testing"

	"q4tab/internal/tokenize"
)

func TestGoalAtVarDecl(t *testing.T) {
	pf := extractFactsToks(tokenize.Lex([]byte("var st *Store = ")))
	g := goalAt("var st *Store = ", nil, pf, nil)
	if g.name != "Store" || !g.ptr {
		t.Fatalf("goal = %+v, want *Store", g)
	}
}

func TestGoalAtAssign(t *testing.T) {
	f := newFacts()
	f.recv["s"] = "Session"
	f.ptr["s"] = true
	g := goalAt("s = ", nil, f, nil)
	if g.name != "Session" || !g.ptr {
		t.Fatalf("goal = %+v, want *Session", g)
	}
}

func TestGoalAtCallArg(t *testing.T) {
	f := newFacts()
	f.sigT["Set"] = []string{"string", "*Entry"}
	g := goalAt("s.Set(k, ", nil, f, nil)
	if g.name != "Entry" || !g.ptr {
		t.Fatalf("goal = %+v, want *Entry", g)
	}
}

func TestGoalAtReturn(t *testing.T) {
	f := newFacts()
	f.retT["loadCfg"] = "Config"
	toks := tokenize.Lex([]byte("func loadCfg() *Config {\n\treturn "))
	g := goalAt("	return ", toks, f, nil)
	if g.name != "Config" {
		t.Fatalf("goal = %+v, want Config", g)
	}
}

func TestSynthCtor(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///syn.go", `package q

type Store struct{ ttl int }

func NewStore(ttl int) *Store { return &Store{} }

func boot() {
	var st *Store =
}`)
	e.Flush()
	text := "package q\n\nfunc boot() {\n\tvar st *Store = "
	items := e.Complete("file:///syn.go", text, len(text))
	if !hasText(items, "NewStore(") && !hasText(items, "NewStore()") {
		t.Fatalf("ctor missing: %v", texts(items))
	}
}

func TestSynthCtorArgs(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///syn2.go", `package q

type Store struct{ ttl int }

func NewStore(ttl int) *Store { return &Store{} }

func boot() {
	var timeout int
	var st *Store =
}`)
	e.Flush()
	text := "package q\n\nfunc boot() {\n\tvar timeout int\n\tvar st *Store = "
	items := e.Complete("file:///syn2.go", text, len(text))
	got := texts(items)
	found := false
	for _, s := range got {
		if s == "NewStore(timeout)" {
			found = true
		}
	}
	if !found {
		t.Fatalf("ctor with scope args missing: %v", got)
	}
}

func TestSynthFieldPath(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///syn3.go", `package q

type Config struct{ Mode string }
type Server struct{ cfg *Config }

func f(s *Server) {
	var c *Config =
}`)
	e.Flush()
	text := "package q\n\ntype Config struct{ Mode string }\ntype Server struct{ cfg *Config }\n\nfunc f(s *Server) {\n\tvar c *Config = "
	items := e.Complete("file:///syn3.go", text, len(text))
	if !hasText(items, "s.cfg") {
		t.Fatalf("field path missing: %v", texts(items))
	}
}

func TestSynthNilForPtr(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///syn4.go", `package q

type Cfg struct{ X int }

func f(c *Cfg) {
	c =
}`)
	e.Flush()
	text := "package q\n\ntype Cfg struct{ X int }\n\nfunc f(c *Cfg) {\n\tc = "
	items := e.Complete("file:///syn4.go", text, len(text))
	if !hasText(items, "nil") {
		t.Fatalf("nil for ptr goal missing: %v", texts(items))
	}
}

func TestSynthInterfaceSatisfies(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///syn5.go", `package q

type Source interface {
	Read(buf []byte) (int, error)
}

type Conn struct{}

func (c *Conn) Read(buf []byte) (int, error) { return 0, nil }

func pipe(c *Conn) {
	var src Source =
}`)
	e.Flush()
	text := "package q\n\nfunc pipe(c *Conn) {\n\tvar src Source = "
	items := e.Complete("file:///syn5.go", text, len(text))
	if !hasText(items, "c") {
		t.Fatalf("interface satisfaction missing c: %v", texts(items))
	}
}

func TestSynthErrorVar(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///syn6.go", `package q

func load() error {
	var err error
	err =
}`)
	e.Flush()
	text := "package q\n\nfunc load() error {\n\tvar err error\n\terr = "
	items := e.Complete("file:///syn6.go", text, len(text))
	if !hasText(items, "nil") && !hasText(items, "err") {
		t.Fatalf("error goal missing nil/err: %v", texts(items))
	}
}
