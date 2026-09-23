package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// buildFixture writes files and builds an engine with default config.
func buildFixture(t *testing.T, files map[string]string) *Engine {
	t.Helper()
	return buildEngineFrom(t, DefaultConfig(), files)
}

func findItem(items []Item, substr string) *Item {
	for i := range items {
		if strings.Contains(items[i].Text, substr) {
			return &items[i]
		}
	}
	return nil
}

func TestAdaptedCorpusRetrieval(t *testing.T) {
	// Corpus uses "req"; the buffer uses "in". Identifier-masked
	// retrieval should surface the line with the caller's name,
	// rebound through the rest of the line.
	e := buildFixture(t, map[string]string{
		"a.go": `package demo

func a() {
	if err := decode(&req); err != nil { report(req) }
	if err := decode(&req); err != nil { report(req) }
	if err := decode(&req); err != nil { report(req) }
}
`,
	})
	text := "package demo\n\nfunc b() {\n\tvar in Request\n\tif err := decode(&in"
	items := e.Complete("file:///proj/b.go", text, len(text))
	it := findItem(items, "); err != nil")
	if it == nil {
		var dbg []string
		for _, i := range items {
			dbg = append(dbg, i.Source+":"+i.Text)
		}
		t.Fatalf("no adapted completion, got %v", dbg)
	}
	if strings.Contains(it.Text, "req") {
		t.Fatalf("adapted item still carries original ident: %q", it.Text)
	}
	if !strings.HasPrefix(it.Source, "adapt") && !strings.HasPrefix(it.Source, "file") {
		t.Fatalf("adapted item source = %q", it.Source)
	}
}

func TestFileAdaptRename(t *testing.T) {
	// No corpus match needed: the same file repeats a shape with a
	// different variable name, and the continuation uses the rebound
	// name.
	e := buildFixture(t, map[string]string{"a.go": "package demo\n"})
	text := `package demo

func first() {
	if err := load(&cfg); err != nil { use(&cfg) }
}

func second() {
	var st State
	if err := load(&st`
	items := e.Complete("file:///proj/x.go", text, len(text))
	it := findItem(items, "); err != nil")
	if it == nil {
		var dbg []string
		for _, i := range items {
			dbg = append(dbg, i.Source+":"+i.Text)
		}
		t.Fatalf("no same-file adapted completion, got %v", dbg)
	}
	if strings.Contains(it.Text, "cfg") {
		t.Fatalf("adapted item still carries cfg: %q", it.Text)
	}
	if !strings.Contains(it.Text, "use(&st)") {
		t.Fatalf("adapted item did not rebind the rest: %q", it.Text)
	}
}

func TestThinContextFileStarts(t *testing.T) {
	// Three files share the same first line so it becomes a prior.
	e := buildFixture(t, map[string]string{
		"a.go": "package demo\n\nfunc a() {}\n",
		"b.go": "package demo\n\nfunc b() {}\n",
		"c.go": "package demo\n\nfunc c() {}\n",
	})
	if len(e.fstarts["go"]) == 0 {
		t.Fatal("no file-start priors collected")
	}
	items := e.Complete("file:///proj/util/d.go", "", 0)
	it := findItem(items, "package")
	if it == nil {
		var dbg []string
		for _, i := range items {
			dbg = append(dbg, i.Source+":"+i.Text)
		}
		t.Fatalf("no package prior on empty file, got %v", dbg)
	}
	// The directory-name package clause should rank first.
	if items[0].Text != "package util" {
		t.Fatalf("top = %q, want directory-derived package clause", items[0].Text)
	}
}

func TestFileStartsRoundTrip(t *testing.T) {
	files := map[string]string{
		"a.go": "package demo\n\nfunc a() {}\n",
		"b.go": "package demo\n\nfunc b() {}\n",
		"c.go": "package demo\n\nfunc c() {}\n",
	}
	dir := t.TempDir()
	for name, src := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	bun, _, err := BuildIndex([]string{dir}, 6, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(bun.FileStarts["go"]) == 0 {
		t.Fatal("no file-start priors built")
	}
	p := filepath.Join(t.TempDir(), "m.q4")
	if err := Save(p, bun); err != nil {
		t.Fatal(err)
	}
	bun2, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(bun2.FileStarts["go"]) == 0 || bun2.FileStarts["go"][0] != "package demo" {
		t.Fatalf("file-starts lost on round trip: %v", bun2.FileStarts)
	}
}

func TestThinContextNoPriofFalse(t *testing.T) {
	e := buildFixture(t, map[string]string{
		"a.go": "package demo\n\nfunc a() {}\n",
	})
	// A rich context is not thin: no priors injected.
	text := "package demo\n\nfunc b() int {\n\tx := compute(a, b)\n\tif x > 0 {\n\t\t"
	items := e.Complete("file:///proj/e.go", text, len(text))
	for _, it := range items {
		if it.Source == "prior" {
			t.Fatalf("prior surfaced on non-thin context: %q", it.Text)
		}
	}
}

func TestFirstUnit(t *testing.T) {
	cases := []struct {
		in   string
		want string
		ok   bool
	}{
		{`k ` + "`json:\"line\"`", "k", true},
		{"foo(); cleanup()", "foo();", true},
		{" err", "", false},
		{" != nil {", "", false},
		{"(a, b) string {", "", false},
	}
	for _, c := range cases {
		got, ok := firstUnit(c.in)
		if ok != c.ok || got != c.want {
			t.Errorf("firstUnit(%q) = %q,%v want %q,%v", c.in, got, ok, c.want, c.ok)
		}
	}
}

func TestUnitHead(t *testing.T) {
	for _, s := range []string{"name", ")", "]", ";", "\"x\"", "'a'", "`tag`"} {
		if !unitHead(s) {
			t.Errorf("unitHead(%q) = false, want true", s)
		}
	}
	for _, s := range []string{"=", ".", ",", "(", "return", "if", "for"} {
		if unitHead(s) {
			t.Errorf("unitHead(%q) = true, want false", s)
		}
	}
}

func TestGoPkgName(t *testing.T) {
	if got := goPkgName("/home/u/proj/myutil"); got != "myutil" {
		t.Fatalf("goPkgName = %q", got)
	}
	if got := goPkgName("/home/u/proj/my-util"); got != "myutil" {
		t.Fatalf("goPkgName dash = %q", got)
	}
	if got := goPkgName("/x/internal/engine"); got != "engine" {
		t.Fatalf("goPkgName engine = %q", got)
	}
	if got := goPkgName("/x/internal"); got != "" {
		t.Fatalf("goPkgName internal = %q, want empty", got)
	}
}
