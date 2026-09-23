package corpus

import (
	"bytes"
	"os"
	"path/filepath"
	"testing"
)

func writeFile(t *testing.T, root, rel string, data []byte) {
	t.Helper()
	full := filepath.Join(root, rel)
	if err := os.MkdirAll(filepath.Dir(full), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(full, data, 0o644); err != nil {
		t.Fatal(err)
	}
}

func collect(t *testing.T, roots ...string) []File {
	t.Helper()
	ch := make(chan File, 64)
	go Collect(roots, ch)
	var out []File
	for f := range ch {
		out = append(out, f)
	}
	return out
}

func TestCollectSkipsGeneratedAndVendored(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "main.go", []byte("package main\n\nfunc main() {}\n"))
	body := []byte("package x\n\nvar x = 1\n")
	for _, rel := range []string{
		"foo.pb.go", "foo.gen.go", "app.min.js", "app.min.css",
		"thing_generated.py", "Form.Designer.cs", "util.g.dart",
		"package-lock.json", "yarn.lock", "pnpm-lock.yaml",
		"Cargo.lock", "go.sum", "composer.lock", "Gemfile.lock",
		"poetry.lock", "vendor/x/y.go", "third_party/z/z.go",
	} {
		writeFile(t, root, rel, body)
	}
	writeFile(t, root, "big.go", bytes.Repeat([]byte("x"), maxFileSize+1))

	got := collect(t, root)
	if len(got) != 1 {
		t.Fatalf("expected 1 file, got %d", len(got))
	}
	if filepath.Base(got[0].Path) != "main.go" {
		t.Fatalf("expected main.go, got %s", got[0].Path)
	}
}

func TestCollectDupContent(t *testing.T) {
	root := t.TempDir()
	body := []byte("package dup\n\nfunc Helper() int {\n\treturn 42\n}\n")
	writeFile(t, root, "a/dup.go", body)
	writeFile(t, root, "b/dup.go", body)
	writeFile(t, root, "c/dup.go", body)
	writeFile(t, root, "d/uniq.go", []byte("package uniq\n\nfunc Other() int {\n\treturn 7\n}\n"))

	got := collect(t, root)
	if len(got) != 4 {
		t.Fatalf("expected 4 files, got %d", len(got))
	}
	// Sorted walk order puts a/dup.go first; it owns the content.
	first := got[0]
	if filepath.Base(first.Path) != "dup.go" || first.DupOf != "" || first.Data == nil {
		t.Fatalf("first emission should be unique with data: %+v", first)
	}
	for _, f := range got[1:3] {
		if f.DupOf != first.Path {
			t.Fatalf("expected DupOf %q, got %q", first.Path, f.DupOf)
		}
		if f.Data != nil {
			t.Fatalf("dup %s should not carry data", f.Path)
		}
		if f.Size == 0 || f.ModTime == 0 {
			t.Fatalf("dup %s lost its metadata: %+v", f.Path, f)
		}
	}
	if got[3].DupOf != "" || got[3].Data == nil {
		t.Fatalf("distinct file should be unaffected: %+v", got[3])
	}
}

func TestCollectSameNameDifferentContent(t *testing.T) {
	root := t.TempDir()
	writeFile(t, root, "a/util.go", []byte("package a\n\nfunc A() {}\n"))
	writeFile(t, root, "b/util.go", []byte("package b\n\nfunc B() {}\n"))

	got := collect(t, root)
	if len(got) != 2 {
		t.Fatalf("expected 2 files, got %d", len(got))
	}
	for _, f := range got {
		if f.DupOf != "" || f.Data == nil {
			t.Fatalf("distinct content should not dedup: %+v", f)
		}
	}
}

func TestCollectDupScopePerCall(t *testing.T) {
	root := t.TempDir()
	body := []byte("package dup\n\nfunc Helper() int {\n\treturn 42\n}\n")
	writeFile(t, root, "a/dup.go", body)
	writeFile(t, root, "b/dup.go", body)

	for call := 0; call < 2; call++ {
		got := collect(t, root)
		if len(got) != 2 {
			t.Fatalf("call %d: expected 2 files, got %d", call, len(got))
		}
		if got[0].Data == nil || got[0].DupOf != "" {
			t.Fatalf("call %d: first file should carry data: %+v", call, got[0])
		}
		if got[1].Data != nil || got[1].DupOf != got[0].Path {
			t.Fatalf("call %d: second file should be a dup of the first: %+v", call, got[1])
		}
	}
}
