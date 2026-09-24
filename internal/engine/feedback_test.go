package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"q4tab/internal/lines"
)

// buildEngineFrom indexes arbitrary fixture files.
func buildEngineFrom(t *testing.T, cfg Config, files map[string]string) *Engine {
	t.Helper()
	dir := t.TempDir()
	for name, src := range files {
		p := filepath.Join(dir, name)
		if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(p, []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	bun, _, err := BuildIndex([]string{dir}, 6, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	e := New(cfg)
	e.SetBundle(bun)
	return e
}

func TestFIMTailSuffix(t *testing.T) {
	// Corpus line: call(alpha, beta, gamma) repeated so the model knows it.
	var b strings.Builder
	for i := 0; i < 8; i++ {
		b.WriteString("call(alpha, beta, gamma)\n")
	}
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{"f.txt": b.String()})

	// Cursor mid-line: call(|beta, gamma). The model suggests
	// "alpha, beta, gamma)" which ends with the existing tail, so the
	// correct edit is inserting "alpha, " before the cursor, not
	// replacing "beta, gamma)".
	text := "call(beta, gamma)"
	off := strings.Index(text, "(") + 1
	items := e.Complete("file:///d.txt", text, off)
	var fim *Item
	for i := range items {
		if strings.Contains(items[i].Source, "fim") {
			fim = &items[i]
			break
		}
	}
	if fim == nil {
		t.Fatalf("no FIM suggestion; got %v", texts(items))
	}
	if fim.ReplaceToEOL {
		t.Fatalf("FIM item still replaces EOL: %q", fim.Text)
	}
	if !strings.Contains(fim.Text, "alpha") {
		t.Fatalf("FIM insert should contain the missing arg, got %q", fim.Text)
	}
	if strings.Contains(fim.Text, "gamma") {
		t.Fatalf("FIM insert must not duplicate the tail: %q", fim.Text)
	}
}

func TestFIMRejectsUnverified(t *testing.T) {
	var b strings.Builder
	for i := 0; i < 8; i++ {
		b.WriteString("call(alpha, beta, gamma)\n")
	}
	e := buildEngineFrom(t, DefaultConfig(), map[string]string{"f.txt": b.String()})

	// Tail that does not match any known line suffix: the suggestion
	// must keep ReplaceToEOL so the client splices safely.
	text := "call(other_thing)"
	off := strings.Index(text, "(") + 1
	items := e.Complete("file:///d.txt", text, off)
	for _, it := range items {
		if !it.ReplaceToEOL && strings.Contains(it.Source, "fim") &&
			strings.Contains(it.Text, "other") {
			t.Fatalf("FIM verified an impossible splice: %q", it.Text)
		}
	}
}

func TestDirCacheFeedsSiblings(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.UpdateDoc("file:///proj/a/x.go", "package a\nvar zzSiblingToken = 1\n")
	e.Flush()
	if e.dirs["/proj/a"] == nil {
		t.Fatal("dir cache not populated for /proj/a")
	}
	// A sibling in the same directory picks up the idiom.
	items := e.Complete("file:///proj/a/y.go", "package a\nvar zzSib", 20)
	found := false
	for _, it := range items {
		if strings.Contains(it.Text, "SiblingToken") {
			found = true
		}
	}
	if !found {
		t.Fatalf("sibling-file token did not surface, got %q", texts(items))
	}
	// A file in another directory keys a different cache.
	if dirOfURI("file:///other/y.go") == dirOfURI("file:///proj/a/y.go") {
		t.Fatal("different directories share a cache key")
	}
}

func TestDirOfURIEdgeCases(t *testing.T) {
	if dirOfURI("file:///a/b/c.go") != "/a/b" {
		t.Fatal("unix path")
	}
	if dirOfURI("file:///a%20b/c.go") != "/a%20b" {
		t.Fatal("encoded path should still group consistently")
	}
	if dirOfURI("untitled:foo") != "" {
		t.Fatal("non-file uri must not get a dir cache")
	}
	if dirOfURI("file:///only.go") != "" {
		t.Fatal("root file has no directory")
	}
}

// waitDyn polls until the dynamic index knows prefix or the deadline
// passes. The background builder is asynchronous.
func waitDyn(t *testing.T, e *Engine, prefix string) {
	t.Helper()
	for i := 0; i < 200; i++ {
		for _, di := range e.dynFor("go") {
			if di != nil && di.HasPrefix(prefix) {
				return
			}
		}
		time.Sleep(5 * time.Millisecond)
	}
	t.Fatalf("dyn index never learned %q", prefix)
}

func TestLearnRerankBoost(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	// Accept the same completed line repeatedly. Its learnSet count
	// grows and the matching suggestion gains score.
	for i := 0; i < 3; i++ {
		e.Learn("", "var zzFreqToken = computeFreq(ctx)", -1)
	}
	n := e.learnSet["var zzFreqToken = computeFreq(ctx)"]
	if n != 3 {
		t.Fatalf("learnSet count = %d, want 3", n)
	}
	waitDyn(t, e, "var zzFreqToken")
	text := "package demo\n\nvar zzFreq"
	items := e.Complete("file:///n.go", text, len(text))
	if len(items) == 0 {
		t.Fatal("no items")
	}
	var learned *Item
	for i := range items {
		if strings.Contains(items[i].Source, "learn") {
			learned = &items[i]
		}
	}
	if learned == nil {
		t.Fatalf("no learn-boosted item in %v", texts(items))
	}
	if !strings.Contains(learned.Text, "computeFreq") {
		t.Fatalf("learn-boosted item does not continue the learned line: %q", learned.Text)
	}
}

func TestRejectAndAcceptAccounting(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	text := "package demo\n\nfunc d() error {\n\terr := work()\n\tif err !="
	uri := "file:///d.go"
	items := e.Complete(uri, text, len(text))
	if len(items) == 0 {
		t.Fatal("no items shown")
	}
	line := strings.Count(text, "\n")

	// Accept the first shown item: every shown item whose first line
	// normalizes to the same completed line counts as an accept, the
	// rest as implicit rejects.
	matched := 0
	want := lines.Normalize("\tif err !=" + firstLineOf(items[0].Text))
	for _, it := range items {
		if lines.Normalize("\tif err !="+firstLineOf(it.Text)) == want {
			matched++
		}
	}
	acc0 := totalMap(e.accN)
	e.Learn(uri, "\tif err !="+items[0].Text, line)
	if totalMap(e.accN) != acc0+matched {
		t.Fatalf("accepts = %d, want %d: acc=%v", totalMap(e.accN)-acc0, matched, e.accN)
	}
	if totalMap(e.rejN) != len(items)-matched {
		t.Fatalf("implicit rejects = %d, want %d", totalMap(e.rejN), len(items)-matched)
	}

	// Explicit reject of a fresh show counts every item.
	rej0 := totalMap(e.rejN)
	items = e.Complete(uri, text, len(text))
	e.Reject(uri, line)
	if totalMap(e.rejN) != rej0+len(items) {
		t.Fatalf("explicit rejects = %d, want %d", totalMap(e.rejN), rej0+len(items))
	}

	// Reject of an unknown context is a no-op.
	e.Reject("file:///never.go", 99)
}

func TestAdaptiveMinProb(t *testing.T) {
	cfg := DefaultConfig()
	e := New(cfg)
	base := e.minProb
	// Simulate a window of model suggestions with zero accepts.
	e.winShown = 300
	e.winAcc = 0
	e.tuneProbLocked()
	if e.minProb <= base {
		t.Fatal("floor did not rise on low accept rate")
	}
	if e.minProb > 0.08 {
		t.Fatalf("floor escaped upper bound: %f", e.minProb)
	}
	// And a high accept rate loosens it, bounded below.
	for i := 0; i < 30; i++ {
		e.winShown = 300
		e.winAcc = 200
		e.tuneProbLocked()
	}
	if e.minProb < 0.008 {
		t.Fatalf("floor escaped lower bound: %f", e.minProb)
	}
}

func TestStatsExposeFeedback(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	e.Complete("file:///d.go", "package demo\n\nvar x", 17)
	st := e.Stats()
	for _, k := range []string{"shownN", "acceptN", "rejectN", "minProb"} {
		if _, ok := st[k]; !ok {
			t.Fatalf("stats missing %q", k)
		}
	}
}

func totalMap(m map[string]int) int {
	n := 0
	for _, v := range m {
		n += v
	}
	return n
}
