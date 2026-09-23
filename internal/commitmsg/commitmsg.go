// Package commitmsg turns git history and staged diffs into
// conventional-commit message suggestions.
//
// Extract writes one pseudo-document per commit, in the same shape the
// indexer already trains on:
//
//	commit <sha>
//	file <path>
//	+ <added line>
//	- <removed line>
//	msg: <first line of the commit message>
//
// Prompt builds the same shape for a staged diff, ending at "msg:" so
// the model completes the message text. Suggest merges model output
// with file/diff heuristics so a candidate exists even when the model
// returns nothing.
package commitmsg

import (
	"path"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
)

// maxDocBytes caps one pseudo-document. The diff body is truncated at
// this size; the msg line is always written.
const maxDocBytes = 64 << 10

// maxSuggestions caps Suggest output.
const maxSuggestions = 5

// convRe detects an existing conventional-commit prefix.
var convRe = regexp.MustCompile(`^(feat|fix|refactor|test|docs|chore|build|ci|perf|style|revert)(\(|:)`)

// diffParser classifies raw unified-diff lines into pseudo-document
// lines: file headers plus added and removed lines. Context lines,
// hunk headers, and binary payloads are dropped.
type diffParser struct {
	binary bool // inside a Binary files / GIT binary patch block
}

// feed maps one raw diff line to a pseudo-document line, or "" to skip.
func (p *diffParser) feed(line string) string {
	if strings.HasPrefix(line, "diff --git ") {
		p.binary = false
		if name := bPath(line); name != "" {
			return "file " + name
		}
		return ""
	}
	if strings.HasPrefix(line, "Binary files ") || line == "GIT binary patch" {
		p.binary = true
		return ""
	}
	if p.binary || line == "" {
		return ""
	}
	switch line[0] {
	case '+':
		if !strings.HasPrefix(line, "+++") {
			return line
		}
	case '-':
		if !strings.HasPrefix(line, "---") {
			return line
		}
	}
	return ""
}

// bPath pulls the b/ path out of a "diff --git a/x b/y" header. Paths
// with spaces or escapes arrive quoted; the b/ name is the second
// quoted string.
func bPath(line string) string {
	rest := strings.TrimPrefix(line, "diff --git ")
	if rest == "" {
		return ""
	}
	var b string
	if rest[0] == '"' {
		i := strings.LastIndex(rest, `" "`)
		if i < 0 {
			return ""
		}
		s, err := strconv.Unquote(rest[i+2:])
		if err != nil {
			return ""
		}
		b = s
	} else {
		i := strings.LastIndex(rest, " b/")
		if i < 0 {
			return ""
		}
		b = rest[i+3:]
	}
	return strings.TrimPrefix(b, "b/")
}

// diffPaths lists the b/ paths in a unified diff.
func diffPaths(diff string) []string {
	var p diffParser
	var out []string
	for _, line := range strings.Split(diff, "\n") {
		if s := p.feed(line); strings.HasPrefix(s, "file ") {
			out = append(out, s[len("file "):])
		}
	}
	return out
}

// diffStats counts added and removed body lines in a unified diff and
// reports whether the diff creates a file.
func diffStats(diff string) (added, removed int, newFile bool) {
	for _, line := range strings.Split(diff, "\n") {
		switch {
		case strings.HasPrefix(line, "new file mode"):
			newFile = true
		case strings.HasPrefix(line, "+++"), strings.HasPrefix(line, "---"):
		case strings.HasPrefix(line, "+"):
			added++
		case strings.HasPrefix(line, "-"):
			removed++
		}
	}
	return
}

// Prompt builds the query document for a staged diff: the Extract
// format ending at "msg:" with no message text, so the model completes
// the message.
func Prompt(files []string, diff string) string {
	var b strings.Builder
	b.WriteString("commit staged\n")
	var p diffParser
	wrote := false
	for _, line := range strings.Split(diff, "\n") {
		s := p.feed(line)
		if s == "" {
			continue
		}
		wrote = true
		if b.Len() < maxDocBytes {
			b.WriteString(s)
			b.WriteByte('\n')
		}
	}
	if !wrote {
		for _, f := range files {
			b.WriteString("file ")
			b.WriteString(filepath.ToSlash(f))
			b.WriteByte('\n')
		}
	}
	b.WriteString("msg:")
	return b.String()
}

// Type suggests the conventional commit type from the changed file set
// plus light diff signals. A plain modification to existing code
// defaults to fix.
func Type(files []string, diff string) string {
	if len(files) == 0 {
		files = diffPaths(diff)
	}
	all := func(pred func(string) bool) bool {
		if len(files) == 0 {
			return false
		}
		for _, f := range files {
			if !pred(f) {
				return false
			}
		}
		return true
	}
	switch {
	case all(isDocFile):
		return "docs"
	case all(isTestFile):
		return "test"
	case all(isCIFile):
		return "ci"
	case all(isBuildFile):
		return "build"
	}
	added, removed, newFile := diffStats(diff)
	switch {
	case newFile:
		return "feat"
	case added == 0 && removed > 0:
		return "refactor"
	case len(files) == 0:
		return "chore"
	default:
		return "fix"
	}
}

// Scope suggests the parenthesized scope from the common directory
// prefix: internal/engine -> engine, cmd/x -> x, other shared dirs ->
// the dir name. Files spread across roots give "".
func Scope(files []string) string {
	common := commonDir(files)
	if common == "" {
		return ""
	}
	parts := strings.Split(common, "/")
	if len(parts) >= 2 {
		switch parts[0] {
		case "internal", "cmd", "pkg":
			return parts[1]
		}
	}
	return parts[len(parts)-1]
}

// Suggest merges model completions for the Prompt context with a
// heuristic fallback. Model candidates come first, trimmed to their
// first line and given a type(scope): prefix when they lack one. One
// fallback built from the diff is always present. Output is deduped,
// deterministic, and capped at 5.
func Suggest(gen func(ctx string) []string, files []string, diff string) []string {
	prefix := Type(files, diff)
	if scope := Scope(files); scope != "" {
		prefix += "(" + scope + ")"
	}
	prefix += ": "

	seen := map[string]bool{}
	var out []string
	push := func(s string) {
		if s == "" || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
	}
	if gen != nil {
		for _, cand := range gen(Prompt(files, diff)) {
			first, _, _ := strings.Cut(cand, "\n")
			first = strings.TrimSpace(first)
			if first == "" {
				continue
			}
			if !convRe.MatchString(first) {
				first = prefix + first
			}
			push(first)
			if len(out) == maxSuggestions-1 {
				break // keep one slot for the fallback
			}
		}
	}
	push(prefix + fallbackSummary(files, diff))
	if len(out) > maxSuggestions {
		out = out[:maxSuggestions]
	}
	return out
}

// fallbackSummary builds the heuristic message body: a verb from the
// diff shape plus the shared path prefix.
func fallbackSummary(files []string, diff string) string {
	added, removed, newFile := diffStats(diff)
	verb := "update"
	switch {
	case newFile:
		verb = "add"
	case added == 0 && removed > 0:
		verb = "remove"
	}
	top := commonDir(files)
	if top == "" && len(files) > 0 {
		top = filepath.ToSlash(files[0])
	}
	if top == "" {
		top = "tree"
	}
	return verb + " " + top
}

// commonDir returns the longest directory prefix shared by all files,
// slash-separated. "" when files sit at the root or share no directory.
func commonDir(files []string) string {
	common := ""
	first := true
	for _, f := range files {
		d := path.Dir(filepath.ToSlash(f))
		if d == "." {
			d = ""
		}
		if first {
			common = d
			first = false
			continue
		}
		common = commonPath(common, d)
	}
	return common
}

// commonPath returns the shared leading path components of a and b.
func commonPath(a, b string) string {
	ap := strings.Split(a, "/")
	bp := strings.Split(b, "/")
	n := 0
	for n < len(ap) && n < len(bp) && ap[n] == bp[n] {
		n++
	}
	if n == 0 {
		return ""
	}
	return strings.Join(ap[:n], "/")
}

func isDocFile(f string) bool {
	f = strings.ToLower(filepath.ToSlash(f))
	switch path.Ext(f) {
	case ".md", ".markdown", ".rst", ".txt", ".adoc":
		return true
	}
	if strings.HasPrefix(f, "docs/") || strings.HasPrefix(f, "doc/") {
		return true
	}
	switch path.Base(f) {
	case "readme", "license", "copying", "changelog", "contributing", "authors", "notice":
		return true
	}
	return false
}

func isTestFile(f string) bool {
	f = strings.ToLower(filepath.ToSlash(f))
	base := path.Base(f)
	if strings.HasPrefix(base, "test_") ||
		strings.HasSuffix(base, "_test.go") || strings.HasSuffix(base, "_test.py") ||
		strings.Contains(base, ".test.") || strings.Contains(base, ".spec.") {
		return true
	}
	return strings.HasPrefix(f, "test/") || strings.HasPrefix(f, "tests/") ||
		strings.HasPrefix(f, "testdata/") ||
		strings.Contains(f, "/test/") || strings.Contains(f, "/tests/") ||
		strings.Contains(f, "/testdata/")
}

func isCIFile(f string) bool {
	f = strings.ToLower(filepath.ToSlash(f))
	if strings.HasPrefix(f, ".github/workflows/") ||
		strings.HasPrefix(f, ".circleci/") || strings.HasPrefix(f, ".buildkite/") {
		return true
	}
	switch path.Base(f) {
	case ".gitlab-ci.yml", ".travis.yml", "jenkinsfile",
		"azure-pipelines.yml", "bitbucket-pipelines.yml",
		"appveyor.yml", ".drone.yml":
		return true
	}
	return false
}

func isBuildFile(f string) bool {
	f = strings.ToLower(filepath.ToSlash(f))
	base := path.Base(f)
	switch base {
	case "makefile", "gnumakefile", "dockerfile", "go.mod", "go.sum",
		"package.json", "package-lock.json", "cargo.toml", "cargo.lock",
		"cmakelists.txt", "pyproject.toml", "setup.py", "setup.cfg",
		"pom.xml", "build.gradle", "meson.build", "build", "workspace",
		"docker-compose.yml", "docker-compose.yaml":
		return true
	}
	return strings.HasSuffix(base, ".mk") || strings.HasPrefix(base, "dockerfile.")
}
