package symbols

import (
	"path/filepath"
	"testing"
)

const fixture = `package demo

import "fmt"

// Greeter greets.
type Greeter struct {
	name string
}

func NewGreeter(n string) *Greeter {
	return &Greeter{name: n}
}

func (g *Greeter) Hello() string {
	return fmt.Sprintf("hi %s", g.name)
}

var defaultGreeter = NewGreeter("x")

const maxNameLen = 64

type Sink interface {
	Write([]byte) (int, error)
}
`

func TestExtract(t *testing.T) {
	syms := Extract("demo.go", []byte(fixture))
	got := map[string]Sym{}
	for _, s := range syms {
		got[s.Name] = s
	}
	cases := map[string]string{
		"Greeter":        "type",
		"NewGreeter":     "func",
		"Hello":          "method",
		"defaultGreeter": "var",
		"maxNameLen":     "var",
		"Sink":           "type",
	}
	for name, kind := range cases {
		s, ok := got[name]
		if !ok {
			t.Errorf("missing symbol %s (got %v)", name, syms)
			continue
		}
		if s.Kind != kind {
			t.Errorf("%s: kind %s, want %s", name, s.Kind, kind)
		}
		if s.Line <= 0 || s.Sig == "" {
			t.Errorf("%s: bad loc %+v", name, s)
		}
	}
	if _, bad := got["fmt"]; bad {
		t.Error("extracted import name")
	}
	if _, bad := got["name"]; bad {
		t.Error("extracted field/var from struct body as def?")
	}
}

func TestExtractPythonJS(t *testing.T) {
	py := `def handle_request(req):
    return req

class Handler:
    pass
`
	got := map[string]string{}
	for _, s := range Extract("h.py", []byte(py)) {
		got[s.Name] = s.Kind
	}
	if got["handle_request"] != "func" {
		t.Errorf("py def: %v", got)
	}
	if got["Handler"] != "type" {
		t.Errorf("py class: %v", got)
	}
	js := `function renderPage(doc) {
  return doc;
}
const API_URL = "https://x";
`
	got = map[string]string{}
	for _, s := range Extract("a.js", []byte(js)) {
		got[s.Name] = s.Kind
	}
	if got["renderPage"] != "func" {
		t.Errorf("js function: %v", got)
	}
	if got["API_URL"] != "var" {
		t.Errorf("js const: %v", got)
	}
}

func TestIndexLookupAndPersist(t *testing.T) {
	ix := NewIndex()
	for _, s := range Extract("demo.go", []byte(fixture)) {
		ix.Add(s)
	}
	if len(ix.Lookup("NewGreeter")) != 1 {
		t.Fatal("exact lookup failed")
	}
	if len(ix.LookupPrefix("Greet", 10)) == 0 {
		t.Fatal("prefix lookup failed")
	}
	ix.RemovePath("demo.go")
	if len(ix.Lookup("NewGreeter")) != 0 {
		t.Fatal("RemovePath failed")
	}
	// roundtrip
	for _, s := range Extract("demo.go", []byte(fixture)) {
		ix.Add(s)
	}
	p := filepath.Join(t.TempDir(), "s.symbols")
	if err := ix.Save(p); err != nil {
		t.Fatal(err)
	}
	ix2, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(ix2.Lookup("NewGreeter")) != 1 {
		t.Fatal("roundtrip lookup failed")
	}
}
