package commitmsg

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestPrompt(t *testing.T) {
	diff := "diff --git a/x.go b/x.go\n" +
		"index 111..222 100644\n" +
		"--- a/x.go\n" +
		"+++ b/x.go\n" +
		"@@ -1,2 +1,3 @@\n" +
		" ctx\n" +
		"+add\n" +
		"-del\n"
	got := Prompt([]string{"x.go"}, diff)
	want := "commit staged\nfile x.go\n+add\n-del\nmsg:"
	if got != want {
		t.Errorf("Prompt =\n%s\nwant\n%s", got, want)
	}
}

func TestPromptFallbackToFileList(t *testing.T) {
	// No diff body: the file list still produces file lines.
	got := Prompt([]string{"a.go", "dir/b.go"}, "")
	want := "commit staged\nfile a.go\nfile dir/b.go\nmsg:"
	if got != want {
		t.Errorf("Prompt =\n%s\nwant\n%s", got, want)
	}
}

func TestType(t *testing.T) {
	modDiff := "diff --git a/x.go b/x.go\n+line\n-line\n"
	newDiff := "diff --git a/x.go b/x.go\nnew file mode 100644\n+line\n"
	for _, tc := range []struct {
		files []string
		diff  string
		want  string
	}{
		{[]string{"a_test.go"}, "", "test"},
		{[]string{"x_test.go", "sub/y_test.go"}, "", "test"},
		{[]string{"README.md", "docs/a.md"}, "", "docs"},
		{[]string{".github/workflows/ci.yml"}, "", "ci"},
		{[]string{"go.mod", "go.sum"}, "", "build"},
		{[]string{"internal/engine/x.go"}, modDiff, "fix"},
		{[]string{"internal/engine/x.go"}, newDiff, "feat"},
		{[]string{"internal/engine/x.go"}, "-gone\n", "refactor"},
		{nil, "", "chore"},
	} {
		if got := Type(tc.files, tc.diff); got != tc.want {
			t.Errorf("Type(%v) = %q, want %q", tc.files, got, tc.want)
		}
	}
}

func TestScope(t *testing.T) {
	for _, tc := range []struct {
		files []string
		want  string
	}{
		{[]string{"internal/engine/x.go"}, "engine"},
		{[]string{"cmd/foo/main.go"}, "foo"},
		{[]string{"lib/a.go", "lib/b.go"}, "lib"},
		{[]string{"main.go", "go.mod"}, ""},
		{[]string{"internal/engine/a.go", "internal/model/b.go"}, "internal"},
		{[]string{"a/b/c/d.go", "a/b/e/f.go"}, "b"},
		{nil, ""},
	} {
		if got := Scope(tc.files); got != tc.want {
			t.Errorf("Scope(%v) = %q, want %q", tc.files, got, tc.want)
		}
	}
}

var suggestFiles = []string{"internal/engine/x.go"}

var suggestDiff = "diff --git a/internal/engine/x.go b/internal/engine/x.go\n" +
	"+line\n-line\n"

func TestSuggestModelFirst(t *testing.T) {
	gen := func(ctx string) []string {
		if !strings.HasSuffix(ctx, "msg:") {
			t.Errorf("gen ctx does not end in msg: %q", ctx[len(ctx)-8:])
		}
		return []string{"add helper\nsecond line", "add helper"}
	}
	got := Suggest(gen, suggestFiles, suggestDiff)
	if len(got) != 2 {
		t.Fatalf("Suggest = %v, want 2 entries", got)
	}
	if got[0] != "fix(engine): add helper" {
		t.Errorf("got[0] = %q, want fix(engine): add helper", got[0])
	}
	if got[1] != "fix(engine): update internal/engine" {
		t.Errorf("got[1] = %q, want fallback", got[1])
	}
}

func TestSuggestKeepsPrefix(t *testing.T) {
	gen := func(string) []string { return []string{"feat(cli): wire flag"} }
	got := Suggest(gen, suggestFiles, suggestDiff)
	if got[0] != "feat(cli): wire flag" {
		t.Errorf("got[0] = %q, want unprefixed passthrough", got[0])
	}
}

func TestSuggestFallbackOnly(t *testing.T) {
	got := Suggest(func(string) []string { return nil }, suggestFiles, suggestDiff)
	if len(got) != 1 || got[0] != "fix(engine): update internal/engine" {
		t.Fatalf("Suggest = %v, want one fallback", got)
	}
}

func TestSuggestCap(t *testing.T) {
	gen := func(string) []string {
		return []string{"a", "b", "c", "d", "e", "f", "g"}
	}
	got := Suggest(gen, suggestFiles, suggestDiff)
	if len(got) != 5 {
		t.Fatalf("Suggest = %v, want cap of 5", got)
	}
	if got[4] != "fix(engine): update internal/engine" {
		t.Errorf("last = %q, want fallback reserved in last slot", got[4])
	}
}

func git(t *testing.T, dir string, args ...string) {
	t.Helper()
	cmd := exec.Command("git", append([]string{"-c", "commit.gpgsign=false", "-C", dir}, args...)...)
	cmd.Env = append(os.Environ(),
		"GIT_AUTHOR_NAME=t", "GIT_AUTHOR_EMAIL=t@t",
		"GIT_COMMITTER_NAME=t", "GIT_COMMITTER_EMAIL=t@t")
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
}

func TestExtract(t *testing.T) {
	if _, err := exec.LookPath("git"); err != nil {
		t.Skip("git not found")
	}
	dir := t.TempDir()
	git(t, dir, "init", "-q")
	if err := os.WriteFile(filepath.Join(dir, "a.go"), []byte("package a\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git(t, dir, "add", "a.go")
	git(t, dir, "commit", "-qm", "feat: add a")
	// An empty commit produces no diff and must be skipped.
	git(t, dir, "commit", "-qm", "chore: nothing", "--allow-empty")
	if err := os.WriteFile(filepath.Join(dir, "a.go"), []byte("package a\n\nvar x = 1\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	git(t, dir, "add", "a.go")
	git(t, dir, "commit", "-qm", "fix: bump x")

	out := t.TempDir()
	n, err := Extract(dir, out, 100)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("Extract wrote %d docs, want 2", n)
	}
	entries, err := os.ReadDir(out)
	if err != nil {
		t.Fatal(err)
	}
	seen := map[string]bool{}
	for _, e := range entries {
		if !strings.HasSuffix(e.Name(), ".commit") {
			t.Errorf("unexpected file %s", e.Name())
			continue
		}
		data, err := os.ReadFile(filepath.Join(out, e.Name()))
		if err != nil {
			t.Fatal(err)
		}
		s := string(data)
		if !strings.HasPrefix(s, "commit ") {
			t.Errorf("%s: missing commit line\n%s", e.Name(), s)
		}
		if !strings.Contains(s, "\nfile a.go\n") {
			t.Errorf("%s: missing file line\n%s", e.Name(), s)
		}
		for _, bad := range []string{"+++", "--- a/", "@@", " ctx"} {
			if strings.Contains(s, bad) {
				t.Errorf("%s: contains %q\n%s", e.Name(), bad, s)
			}
		}
		lines := strings.Split(strings.TrimRight(s, "\n"), "\n")
		last := lines[len(lines)-1]
		if !strings.HasPrefix(last, "msg: ") {
			t.Errorf("%s: last line = %q, want msg:", e.Name(), last)
		}
		seen[last] = true
	}
	if !seen["msg: feat: add a"] || !seen["msg: fix: bump x"] {
		t.Errorf("subjects = %v, want both commit messages", seen)
	}
}
