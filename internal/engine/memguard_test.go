package engine

import (
	"fmt"
	"runtime"
	"strings"
	"testing"
)

// TestMemoryBounds simulates a long editing session and asserts the
// engine's growth is bounded: vocab capped, caches reset, line lists
// capped.
func TestMemoryBounds(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	baseVocab := e.m.Vocab.Len()

	// Churn 200 documents full of unique identifiers.
	var b strings.Builder
	for i := 0; i < 200; i++ {
		b.Reset()
		fmt.Fprintf(&b, "package p%d\n\n", i)
		for j := 0; j < 50; j++ {
			fmt.Fprintf(&b, "func uniq_%d_%d() { x%d := call%d()\n", i, j, j, j)
		}
		text := b.String()
		uri := fmt.Sprintf("file:///doc%d.go", i)
		e.UpdateDoc(uri, text)
		e.Flush()
		e.Complete(uri, text, len(text))
		if i%10 == 9 {
			e.CloseDoc(uri)
		}
	}

	if got := e.m.Vocab.Len(); got > e.vocabCap {
		t.Fatalf("vocab grew past cap: %d > %d", got, e.vocabCap)
	}
	_ = baseVocab
	if e.sessTok > e.cfg.CacheCap+e.cfg.MaxDocSize {
		t.Fatalf("session cache unbounded: %d", e.sessTok)
	}
}

// TestLearnedCacheBound drives Learn past the cache cap and checks the
// learned cache resets rather than growing forever.
func TestLearnedCacheBound(t *testing.T) {
	cfg := DefaultConfig()
	cfg.CacheCap = 4096 // tiny cap to exercise the reset path
	e := buildEngine(t, cfg)
	for i := 0; i < 500; i++ {
		e.Learn("", fmt.Sprintf("learnedCall%d(arg%d, opts%d)", i, i, i), -1)
	}
	if e.learnTok > cfg.CacheCap {
		t.Fatalf("learned cache exceeded cap: %d > %d", e.learnTok, cfg.CacheCap)
	}
	if len(e.learnLines) > 8192 {
		t.Fatalf("learnLines unbounded: %d", len(e.learnLines))
	}
}

// TestVocabCapStopsInterning forces the cap and verifies unseen tokens
// stop extending the vocabulary.
func TestVocabCapStopsInterning(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.vocabCap = e.m.Vocab.Len() // at cap already
	n0 := e.m.Vocab.Len()
	e.UpdateDoc("file:///x.go", "package x\n\nvar neverSeenBeforeZZZ = 1")
	e.Flush()
	if e.m.Vocab.Len() != n0 {
		t.Fatalf("vocab grew past cap: %d -> %d", n0, e.m.Vocab.Len())
	}
	// Completion must still work and not panic on dropped tokens.
	items := e.Complete("file:///x.go", "package x\n\nvar neverSeenBefore", len("package x\n\nvar neverSeenBefore"))
	_ = items
}

// TestCompleteRSSSoak runs many completions over churned docs and
// asserts heap stays within a generous bound of the model size.
func TestCompleteRSSSoak(t *testing.T) {
	if testing.Short() {
		t.Skip("soak test")
	}
	e := buildEngine(t, DefaultConfig())
	runtime.GC()
	var m0 runtime.MemStats
	runtime.ReadMemStats(&m0)

	for i := 0; i < 300; i++ {
		text := fmt.Sprintf("package s\n\nfunc f%d() error {\n\terr := work()\n\tif err !=", i)
		uri := fmt.Sprintf("file:///s%d.go", i%20)
		e.UpdateDoc(uri, text)
		e.Flush()
		e.Complete(uri, text, len(text))
		e.Learn("", fmt.Sprintf("\tif err != nil { return wrap%d(err) }", i), -1)
	}
	runtime.GC()
	var m1 runtime.MemStats
	runtime.ReadMemStats(&m1)
	growth := int64(m1.HeapAlloc) - int64(m0.HeapAlloc)
	if growth > 64<<20 {
		t.Fatalf("heap grew %d MB over soak", growth>>20)
	}
	t.Logf("heap growth over soak: %d KB", growth>>10)
}
