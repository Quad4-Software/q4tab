package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"

	"q4tab/internal/engine"
)

func TestLangOf(t *testing.T) {
	for p, want := range map[string]string{
		"/a/b/c.go":        "go",
		"/a/b/c.py":        "python",
		"/a/b/C.TS":        "typescript",
		"/a/x/Dockerfile":  "dockerfile",
		"/a/x/Makefile":    "makefile",
		"/a/x/unknown.qqq": "other",
		"/a/x/noext":       "other",
		"/a/x/readme.MD":   "markdown",
	} {
		if got := langOf(p); got != want {
			t.Errorf("langOf(%q) = %q, want %q", p, got, want)
		}
	}
}

func quietLog(string, ...any) {}

func TestIncrementalIndex(t *testing.T) {
	root := t.TempDir()
	src := filepath.Join(root, "a.go")
	os.WriteFile(src, []byte("package a\nvar first = 1\n"), 0o644)
	modelPath := filepath.Join(t.TempDir(), "model.bin")

	// Full build writes a manifest.
	bun, st, err := engine.BuildIndex([]string{root}, 6, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := engine.Save(modelPath, bun); err != nil {
		t.Fatal(err)
	}
	if err := writeManifest(modelPath+".manifest", []string{root}, st.Meta); err != nil {
		t.Fatal(err)
	}

	// Unchanged run: no delta produced.
	if !runIncremental(modelPath, []string{root}, quietLog) {
		t.Fatal("incremental run rejected valid manifest")
	}
	if _, err := os.Stat(modelPath + ".delta"); !os.IsNotExist(err) {
		t.Fatal("unchanged run wrote a delta")
	}

	// Modify one file. The delta must contain only it.
	time.Sleep(1100 * time.Millisecond) // mtime granularity
	os.WriteFile(src, []byte("package a\nvar second = 2\n"), 0o644)
	added := filepath.Join(root, "b.go")
	os.WriteFile(added, []byte("package a\nvar brandNew = 3\n"), 0o644)
	if !runIncremental(modelPath, []string{root}, quietLog) {
		t.Fatal("incremental run failed")
	}
	delta, err := engine.LoadDelta(modelPath + ".delta")
	if err != nil {
		t.Fatal(err)
	}
	if len(delta) != 2 {
		t.Fatalf("delta has %d files, want 2", len(delta))
	}

	// Delete the added file. Next run drops it from the delta.
	os.Remove(added)
	time.Sleep(1100 * time.Millisecond)
	if !runIncremental(modelPath, []string{root}, quietLog) {
		t.Fatal("incremental run after delete failed")
	}
	delta, _ = engine.LoadDelta(modelPath + ".delta")
	if len(delta) != 1 {
		t.Fatalf("delta has %d files after delete, want 1", len(delta))
	}

	// Corrupt manifest falls back to a full build (returns false).
	os.WriteFile(modelPath+".manifest", []byte("garbage"), 0o644)
	if runIncremental(modelPath, []string{root}, quietLog) {
		t.Fatal("corrupt manifest did not trigger full-build fallback")
	}
}
