package engine

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"q4tab/internal/model"
)

// TestStoreRoundTrip saves and reloads a built model and verifies the
// completion behavior is identical.
func TestStoreRoundTrip(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	p := filepath.Join(t.TempDir(), "m.bin")
	if err := Save(p, e.bundle()); err != nil {
		t.Fatal(err)
	}
	bun2, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	m2 := bun2.M
	if m2.N != e.m.N || m2.Vocab.Len() != e.m.Vocab.Len() || m2.KN != e.m.KN {
		t.Fatalf("roundtrip mismatch: n=%d/%d vocab=%d/%d kn=%v/%v",
			m2.N, e.m.N, m2.Vocab.Len(), e.m.Vocab.Len(), m2.KN, e.m.KN)
	}
	if bun2.Lines.Len() != e.li.Len() {
		t.Fatalf("line index mismatch: %d/%d", bun2.Lines.Len(), e.li.Len())
	}
	// Spot-check vocab and a completion.
	if got := m2.Vocab.Str(0); got != "<pad>" {
		t.Fatalf("vocab[0] = %q", got)
	}
	e2 := New(DefaultConfig())
	e2.SetBundle(bun2)
	text := "package demo\n\nfunc d() error {\n\terr := work()\n\tif err !="
	a := texts(e.Complete("file:///d.go", text, len(text)))
	b := texts(e2.Complete("file:///d.go", text, len(text)))
	if strings.Join(a, "|") != strings.Join(b, "|") {
		t.Fatalf("completions differ after roundtrip:\n%v\n%v", a, b)
	}
	// Interning into a loaded (frozen) vocab must work.
	_ = m2.Vocab.ID("brandNewTokenXYZ")
	if _, ok := m2.Vocab.Lookup("brandNewTokenXYZ"); !ok {
		t.Fatal("intern into mapped vocab failed")
	}
	if _, err := os.Stat(p + ".tmp"); err == nil {
		t.Fatal("save left tmp file behind")
	}
}

// TestAsyncDocBuild verifies documents over the inline threshold are
// indexed by the background worker and that the doc cache comes up.
func TestAsyncDocBuild(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	var sb strings.Builder
	sb.WriteString("package big\n\n")
	for i := 0; i < 20000; i++ {
		fmt.Fprintf(&sb, "func fn%d() int { return %d }\n", i, i)
	}
	text := sb.String() // >128KB triggers the async path
	e.UpdateDoc("file:///big.go", text)
	e.Flush()
	deadline := time.Now().Add(10 * time.Second)
	for {
		var ready bool
		if dv, ok := e.docs.Load("file:///big.go"); ok {
			d := dv.(*doc)
			d.mu.Lock()
			ready = d.cache != nil && d.lines != nil
			d.mu.Unlock()
		}
		if ready {
			break
		}
		if time.Now().After(deadline) {
			t.Fatal("async doc build never completed")
		}
		time.Sleep(5 * time.Millisecond)
	}
	// Completion should work and file-layer matches exist.
	off := len(text)
	items := e.Complete("file:///big.go", text+"\nfunc fn1", off+9)
	if len(items) == 0 {
		t.Fatal("no completions after async build")
	}
}

// TestNoGreedyLoop ensures repetition of a generated 4-gram stops the
// block rather than producing "in a .mjs file in a .mjs file ...".
func TestNoGreedyLoop(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	text := "package demo\n\n// loop bait\n\nfunc d() error {\n\terr := work()\n\tif err !="
	for _, it := range e.Complete("file:///d.go", text, len(text)) {
		// A normalized 4-word window may repeat at most once.
		words := strings.Fields(it.Text)
		seen := map[string]int{}
		for i := 0; i+4 <= len(words); i++ {
			k := strings.Join(words[i:i+4], " ")
			seen[k]++
			if seen[k] > 2 {
				t.Fatalf("greedy loop in %q", it.Text)
			}
		}
	}
}

// TestDeltaRoundTrip verifies the incremental overlay survives
// save/load and reaches completions through InstallDelta.
func TestDeltaRoundTrip(t *testing.T) {
	dir := t.TempDir()
	dp := filepath.Join(dir, "model.delta")
	files := []DeltaFile{
		{Path: "/src/new.go", ModTime: 42, Data: []byte("package src\nfunc zzDeltaFn() int { return 7 }\n")},
		{Path: "/src/other.go", ModTime: 43, Data: []byte("package src\nvar zzDeltaVar = \"hello\"\n")},
	}
	if err := SaveDelta(dp, files); err != nil {
		t.Fatal(err)
	}
	got, err := LoadDelta(dp)
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || string(got[0].Data)+string(got[1].Data) == "" {
		t.Fatalf("delta roundtrip lost files: %v", got)
	}
	byPath := map[string]DeltaFile{}
	for _, d := range got {
		byPath[d.Path] = d
	}
	if string(byPath["/src/new.go"].Data) != string(files[0].Data) {
		t.Fatal("delta data mismatch")
	}

	// Corrupt delta must error, not panic.
	os.WriteFile(dp, []byte("Q4D1garbage"), 0o644)
	if _, err := LoadDelta(dp); err == nil {
		t.Fatal("corrupt delta loaded")
	}
}

// TestDeltaCompletion: a file absent from the base model surfaces via
// the delta overlay once installed.
func TestDeltaCompletion(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.InstallDelta([]DeltaFile{
		{Path: "/src/new.go", Data: []byte("package src\n\tzzDeltaMarker(alpha, beta)\n")},
	})
	waitDyn(t, e, "zzDeltaMarker")
	items := e.Complete("file:///x.go", "package demo\n\n\tzzDelta", 21)
	found := false
	for _, it := range items {
		if strings.Contains(it.Text, "Marker") {
			found = true
		}
	}
	if !found {
		t.Fatalf("delta file line did not surface, got %q", texts(items))
	}
}

// TestStoreEmptyOrderRoundTrip: a build that gated or froze an order to
// zero rows writes a section with nk=0. Loading it must not panic on
// the zero-length typed slices.
func TestStoreEmptyOrderRoundTrip(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	bun := e.bundle()
	// Empty the top order: zero keys, zero rows.
	top := &bun.M.Orders[bun.M.N]
	top.Keys = nil
	top.Off = nil
	top.Toks = nil
	top.Cnts = nil
	top.Totals = nil
	top.NToks = 0
	// Empty a lang bigram table too.
	for _, lt := range bun.Langs {
		lt.Bi = model.Order{}
		break
	}
	p := filepath.Join(t.TempDir(), "m.bin")
	if err := Save(p, bun); err != nil {
		t.Fatal(err)
	}
	bun2, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if len(bun2.M.Orders[bun2.M.N].Keys) != 0 {
		t.Fatal("expected empty top order")
	}
}
