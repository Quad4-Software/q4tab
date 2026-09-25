package engine

import (
	"os"
	"path/filepath"
	"testing"
)

// Aux sidecar round-trip: build over a tiny corpus, save, load, and
// the overlay must serve members without touching the base model.
func TestAuxSidecarOverlay(t *testing.T) {
	dir := t.TempDir()
	src := `package q

type Widget struct {
	Name string
}

func (w *Widget) Render() {}
func (w *Widget) Resize() {}

type Gadget = Widget
`
	if err := os.WriteFile(filepath.Join(dir, "w.go"), []byte(src), 0o644); err != nil {
		t.Fatal(err)
	}

	a, err := BuildAux([]string{dir}, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(a.TypeMem["Widget"]) == 0 {
		t.Fatalf("no Widget members: %v", a.TypeMem)
	}
	if a.Aliases["Gadget"] != "Widget" {
		t.Fatalf("alias missing: %v", a.Aliases)
	}
	if a.Files == 0 || a.BuiltAt == 0 {
		t.Fatal("header fields missing")
	}
	path := filepath.Join(dir, "m.aux")
	if err := SaveAux(path, a); err != nil {
		t.Fatal(err)
	}
	back := LoadAux(path)
	if back == nil {
		t.Fatal("LoadAux nil")
	}
	if len(back.TypeMem["Widget"]) == 0 || back.Aliases["Gadget"] != "Widget" {
		t.Fatalf("round-trip lost tables: %+v", back)
	}
	if back.Files != a.Files {
		t.Fatalf("files %d != %d", back.Files, a.Files)
	}

	// Overlay consults aux even with an empty base model.
	e := buildEngine(t, DefaultConfig())
	e.SetAux(back)
	text := "package q\n\nfunc f(w *Widget) {\n\tw."
	if it := e.Complete("file:///x.go", text, len(text)); !hasText(it, "Render(") {
		t.Fatalf("aux members missing: %v", texts(it))
	}
	// Corpus alias resolves through the overlay.
	e2 := buildEngine(t, DefaultConfig())
	e2.SetAux(back)
	e2.UpdateDoc("file:///y.go", "package q\n\nvar g Gadget\n")
	e2.Flush()
	q := "package q\n\nvar g Gadget\n\nfunc h() {\n\tg."
	if it := e2.Complete("file:///y.go", q, len(q)); !hasText(it, "Render(") {
		t.Fatalf("alias member missing: %v", texts(it))
	}
	// Swap to empty overlay: members gone.
	e.SetAux(&AuxData{})
	if it := e.Complete("file:///x.go", text, len(text)); hasText(it, "Render(") {
		t.Fatalf("cleared aux still serves: %v", texts(it))
	}
}
