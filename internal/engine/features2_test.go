package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestScopeFreqBoost(t *testing.T) {
	// Nested cache: an identifier seen many times in the document is a
	// stronger reuse signal than one seen once, so candidates reusing
	// it rank higher.
	var b strings.Builder
	b.WriteString("call(zzBusy)\n")
	b.WriteString("call(zzOnce)\n")
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{"f.txt": b.String()})

	var doc strings.Builder
	doc.WriteString("package p\n")
	for i := 0; i < 12; i++ {
		doc.WriteString("zzBusy++\n")
	}
	doc.WriteString("zzOnce++\ncall(")
	text := doc.String()
	items := e.Complete("file:///p/y.go", text, len(text))
	var busy, once float64
	for _, it := range items {
		if strings.Contains(it.Text, "zzBusy") {
			busy = it.Score
		}
		if strings.Contains(it.Text, "zzOnce") {
			once = it.Score
		}
	}
	if busy == 0 || once == 0 {
		t.Fatalf("expected both candidates, got %v", items)
	}
	if busy <= once {
		t.Fatalf("freq-scaled boost lost: busy %v vs once %v", busy, once)
	}
	if t.Failed() {
		return
	}
	// Killswitch: with scope off the freq signal must not apply.
	t.Setenv("Q4_DISABLE", "scope")
	items = e.Complete("file:///p/y.go", text, len(text))
	for _, it := range items {
		if strings.Contains(it.Source, "scope") {
			t.Fatalf("scope boost applied despite Q4_DISABLE=scope: %v", it.Source)
		}
	}
}

func TestOpHealColon(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 10; i++ {
		b.WriteString("total += compute()\n")
		b.WriteString("name := resolve(x)\n")
	}
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{"f.go": b.String()})
	text := "package p\nfunc f() {\n\tname :"
	items := e.Complete("file:///p/y.go", text, len(text))
	found := false
	for _, it := range items {
		if strings.HasPrefix(it.Text, "=") || strings.HasPrefix(it.Text, ":=") {
			found = true
		}
	}
	if !found {
		t.Fatalf("healed := variant missing, got %v", items)
	}
}

func TestIterRetrieval(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 6; i++ {
		b.WriteString("db, err := openDB(dsn)\n")
		b.WriteString("defer db.Close()\n")
	}
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{"f.go": b.String()})
	text := "package p\nfunc f() {\n\tdb, err := open"
	items := e.Complete("file:///p/y.go", text, len(text))
	var iter *Item
	for i := range items {
		if strings.Contains(items[i].Source, "+iter") {
			iter = &items[i]
			break
		}
	}
	if iter == nil {
		t.Fatalf("no iterative item, got %v", items)
	}
	if !strings.Contains(iter.Text, "db.Close") {
		t.Fatalf("iter line = %q, want second attested line", iter.Text)
	}
}

func TestImportScopeBoost(t *testing.T) {
	// A library directory with a frequent identifier, and a consumer
	// whose import path resolves to it.
	var lib strings.Builder
	for i := 0; i < 9; i++ {
		lib.WriteString("zzHelperFn()\n")
	}
	lib.WriteString("use(zzHelperFn)\n")
	// The app dir deliberately lacks zzHelperFn: otherwise the
	// own-directory union would mask the import signal.
	var app strings.Builder
	app.WriteString("use(zzOtherFn)\n")
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{
		"proj/lib/x.go": lib.String(),
		"proj/app/a.go": app.String(),
	})
	if len(e.dirIds) == 0 {
		t.Fatal("DirIdents not built")
	}
	if got := e.resolveDirs([]string{"x", "proj/lib"}); len(got) == 0 {
		t.Fatalf("resolveDirs missed proj/lib")
	}
	// With the import present, the dir union puts zzHelperFn in scope
	// and the retrieved line earns +scope. Without it, no boost.
	withImp := "package app\nimport \"proj/lib\"\n\nfunc f() {\n\tuse("
	without := "package app\n\nfunc f() {\n\tuse("
	hasScope := func(txt string) bool {
		for _, it := range e.Complete("file:///proj/app/b.go", txt, len(txt)) {
			if strings.Contains(it.Text, "zzHelperFn") && strings.Contains(it.Source, "+scope") {
				return true
			}
		}
		return false
	}
	if !hasScope(withImp) {
		t.Fatalf("import-adjacent ident got no scope boost")
	}
	if hasScope(without) {
		t.Fatalf("scope boost fired without the import adjacency")
	}
}

func TestMMRDedupe(t *testing.T) {
	items := []Item{
		{Text: "use(alpha)", Score: 10},
		{Text: "use(beta)", Score: 9},
		{Text: "use(alpha)\nclose(alpha)", Score: 8},
		{Text: "other()", Score: 7},
	}
	out := dedupeShapes(items, nil)
	if len(out) != 3 {
		t.Fatalf("dedupeShapes kept %d, want 3: %v", len(out), out)
	}
	if out[1].Text != "use(alpha)\nclose(alpha)" {
		t.Fatalf("multi-line variant lost: %v", out)
	}
	// A same-shape variant whose identifier is in scope is real
	// diversity, not a dup.
	scope := map[string]bool{"beta": true}
	out = dedupeShapes(items, scope)
	if len(out) != 4 {
		t.Fatalf("scoped variant wrongly deduped: %v", out)
	}
}

func TestSrcMultJournal(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	dir := t.TempDir()
	jp := filepath.Join(dir, "learned.jsonl")
	var j strings.Builder
	for i := 0; i < 12; i++ {
		j.WriteString(`{"e":"a","s":"corpus","i":0}` + "\n")
		j.WriteString(`{"e":"r","s":"model","i":1}` + "\n")
	}
	if err := os.WriteFile(jp, []byte(j.String()), 0o644); err != nil {
		t.Fatal(err)
	}
	e.SetJournal(jp)
	mc := e.srcMult("corpus")
	mm := e.srcMult("model")
	if mc <= mm {
		t.Fatalf("srcMult corpus %v <= model %v", mc, mm)
	}
}

func TestDocLangEmbed(t *testing.T) {
	cases := []struct {
		base, prefix, want string
	}{
		{"markdown", "text\n```go\nif err != nil {\n", "go"},
		{"markdown", "text\n```go\nx := 1\n```\nmore text\n", "markdown"},
		{"md", "a\n```python\nx = ", "python"},
		{"markdown", "a\n```\nx", "markdown"},
		{"svelte", "<p>hi</p>\n<script>\nconst x = ", "javascript"},
		{"svelte", "<script lang=\"ts\">\nconst x: number = ", "typescript"},
		{"html", "<script>\nf()\n</script>\n<div>", "html"},
		{"php", "<html>\n<?php\n$x = ", "php"},
		{"php", "<?php echo 1; ?>\nplain", "php"},
		{"go", "package p\nfunc f() {\n", "go"},
	}
	for _, c := range cases {
		if got := docLang(c.base, c.prefix); got != c.want {
			t.Errorf("docLang(%q, ...) = %q, want %q", c.base, got, c.want)
		}
	}
}

func TestMarkdownFenceUsesGoModel(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 8; i++ {
		b.WriteString("if err != nil {\n\treturn fmt.Errorf(\"w: %w\", err)\n}\n")
	}
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{"f.go": b.String()})
	text := "# doc\n\n```go\nfunc f() error {\n\terr := g()\n\tif err !"
	items := e.Complete("file:///d/readme.md", text, len(text))
	for _, it := range items {
		if strings.Contains(it.Text, "= nil") {
			return
		}
	}
	t.Fatalf("go idioms unavailable inside a go fence: %v", items)
}
