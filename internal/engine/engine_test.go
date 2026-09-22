package engine

import (
	"bytes"
	"encoding/binary"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

var fixtureFiles = map[string]string{
	"errcheck.go": `package demo

func a() error {
	err := work()
	if err != nil {
		return err
	}
	return nil
}

func b() error {
	err := work()
	if err != nil {
		return err
	}
	return nil
}

func c() error {
	err := work()
	if err != nil {
		return err
	}
	return nil
}
`,
	"handlers.go": `package demo

func HandleAlpha() error {
	return nil
}

func HandleBeta() error {
	return nil
}
`,
}

func buildEngine(t *testing.T, cfg Config) *Engine {
	dir := t.TempDir()
	for name, src := range fixtureFiles {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	m, li, _, err := BuildIndex([]string{dir}, 6, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	e := New(cfg)
	e.SetModel(m, li)
	return e
}

func texts(items []Item) []string {
	out := make([]string, len(items))
	for i, it := range items {
		out[i] = it.Text
	}
	return out
}

func TestMultilineCompletion(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxLines = 4
	e := buildEngine(t, cfg)
	text := "package demo\n\nfunc d() error {\n\terr := work()\n\tif err !="
	items := e.Complete("file:///d.go", text, len(text))
	for _, it := range items {
		t.Logf("[%s] %q", it.Source, it.Text)
	}
	found := false
	for _, it := range items {
		if strings.Contains(it.Text, "return err") {
			found = true
		}
	}
	if !found {
		t.Fatalf("expected multi-line completion containing 'return err', got %q", texts(items))
	}
}

func TestMultilineNoLeadingNewline(t *testing.T) {
	cfg := DefaultConfig()
	cfg.MaxLines = 8
	e := buildEngine(t, cfg)
	text := "package demo\n\nfunc d() error {\n\terr := work()\n\tif err !="
	items := e.Complete("file:///d.go", text, len(text))
	for _, it := range items {
		if strings.HasPrefix(it.Text, "\n") {
			t.Fatalf("leading-newline completion on open expression: %q", it.Text)
		}
		// And braces must be balanced relative to the open block.
		if strings.Count(it.Text, "{") < strings.Count(it.Text, "}")-1 {
			t.Fatalf("unbalanced braces in %q", it.Text)
		}
	}
}

func TestAcceptSuffixRules(t *testing.T) {
	e := New(DefaultConfig())
	// Cursor before a 1-char suffix: suggestion ending with it would
	// double the character.
	if e.accept("x)", ")", ")") {
		t.Fatal("1-char suffix hole: 'x)' accepted before ')'")
	}
	if !e.accept("x, y", ")", ")") {
		t.Fatal("over-rejected: 'x, y' before ')'")
	}
	// CRLF rest-of-line must still be detected.
	if e.accept("x)", ")\r\n}", ")") {
		t.Fatal("CRLF suffix not detected")
	}
	// A suggestion ending with the tail is the fill-in-the-middle
	// case: it is kept so the post-pass can trim it to an insert.
	if !e.accept("a != nil {", "x != nil {", "!= nil {") {
		t.Fatal("FIM tail-suffix case rejected")
	}
	// Tail appearing mid-suggestion still rejects.
	if e.accept("a != nil { b }", "x != nil {", "!= nil {") {
		t.Fatal("mid-suggestion tail should reject")
	}
	// A suggestion equal to the tail is a pure duplicate.
	if e.accept("!= nil {", "x != nil {", "!= nil {") {
		t.Fatal("tail duplicate accepted")
	}
	// Empty tail accepts anything non-degenerate.
	if !e.accept("foo(bar)", "", "") {
		t.Fatal("empty-suffix accept failed")
	}
}

func TestLoadRejectsCorrupt(t *testing.T) {
	dir := t.TempDir()
	p := filepath.Join(dir, "bad.bin")
	os.WriteFile(p, []byte("not a model file at all, just text"), 0o644)
	if _, _, err := Load(p); err == nil {
		t.Fatal("corrupt model loaded without error")
	}
	// Valid magic but truncated before sections complete.
	var buf bytes.Buffer
	binary.Write(&buf, binary.LittleEndian, uint32(storeMagic))
	binary.Write(&buf, binary.LittleEndian, uint32(1))
	binary.Write(&buf, binary.LittleEndian, uint32(6))
	binary.Write(&buf, binary.LittleEndian, uint32(10))
	p2 := filepath.Join(dir, "bad2.bin")
	os.WriteFile(p2, buf.Bytes(), 0o644)
	if _, _, err := Load(p2); err == nil {
		t.Fatal("truncated model loaded without error")
	}
	// Valid structure but corrupted vocab offsets: must be rejected,
	// not panic later inside Vocab.Str.
	e := buildEngine(t, DefaultConfig())
	good := filepath.Join(dir, "good.bin")
	if err := Save(good, e.m, e.li); err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile(good)
	if err != nil {
		t.Fatal(err)
	}
	// Vocab offsets start at byte 16. Set the second offset backwards.
	data[20] = 0xFF
	data[21] = 0xFF
	data[22] = 0xFF
	bad := filepath.Join(dir, "badoffs.bin")
	os.WriteFile(bad, data, 0o644)
	if _, _, err := Load(bad); err == nil {
		t.Fatal("model with corrupt offsets loaded without error")
	}
	// A tail-truncated v2 file must fail cleanly at load.
	data, _ = os.ReadFile(good)
	trunc := filepath.Join(dir, "trunc.bin")
	os.WriteFile(trunc, data[:len(data)-64], 0o644)
	if _, _, err := Load(trunc); err == nil {
		t.Fatal("truncated v2 model loaded without error")
	}
}

func TestLearnWithoutModel(t *testing.T) {
	e := New(DefaultConfig())
	e.SetJournal(filepath.Join(t.TempDir(), "j.jsonl"))
	e.Learn("", "var pendingThing = 1", -1)
	if e.learnN != 1 || len(e.pendLearn) != 1 {
		t.Fatalf("learn before model: learnN=%d pend=%d", e.learnN, len(e.pendLearn))
	}
	// Once a model arrives the pending text must reach the cache.
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "f.go"), []byte(fixtureFiles["errcheck.go"]), 0o644)
	m, li, _, err := BuildIndex([]string{dir}, 6, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	e.SetModel(m, li)
	if len(e.pendLearn) != 0 {
		t.Fatal("pending learns not drained")
	}
}

func TestSuffixDedupe(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	// The continuation after the cursor is already the right code.
	text := "package demo\n\nfunc d() error {\n\tif err != nil {\n\t\treturn err\n\t}\n"
	offset := strings.Index(text, "!=") + 2
	items := e.Complete("file:///d.go", text, offset)
	for _, it := range items {
		if strings.Contains(it.Text, "nil") {
			t.Fatalf("suggested text already present after cursor: %q", it.Text)
		}
	}
}

func TestLearnPersists(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	journal := filepath.Join(t.TempDir(), "learned.jsonl")
	e.SetJournal(journal)
	e.Learn("", "customInternalCall(ctx, payload)", -1)
	if _, err := os.Stat(journal); err != nil {
		t.Fatal("journal not written")
	}
	// Fresh engine over same corpus loads the journal.
	e2 := buildEngine(t, DefaultConfig())
	e2.SetJournal(journal)
	if e2.learnN == 0 {
		t.Fatal("journal not replayed")
	}
}

func TestLearnedBoostsCompletion(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	journal := filepath.Join(t.TempDir(), "learned.jsonl")
	e.SetJournal(journal)
	// Teach a novel line the corpus never saw, as it would arrive from
	// an accept event (line prefix + accepted text).
	for i := 0; i < 5; i++ {
		e.Learn("", "var zzzCallSite = acquireZeta(ctx)", -1)
	}
	text := "package demo\n\nvar zzzCal"
	items := e.Complete("file:///n.go", text, len(text))
	found := false
	for _, it := range items {
		if strings.Contains(it.Text, "lSite") {
			found = true
		}
	}
	if !found {
		t.Fatalf("learned completion did not surface, got %q", texts(items))
	}
}

func TestEmptyAndEdgeInputs(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	for _, tc := range []struct {
		text string
		off  int
	}{
		{"", 0},
		{"package demo\n", 13},
		{"x", 0},
		{"x", 5}, // offset past end
	} {
		e.Complete("file:///e.go", tc.text, tc.off) // must not panic
	}
}

func TestNoModelEngine(t *testing.T) {
	e := New(DefaultConfig())
	e.UpdateDoc("file:///x.go", "package x\nvar alphaBeta = 1\n")
	e.Flush()
	items := e.Complete("file:///x.go", "package x\nvar alphaBeta = 1\nvar alphaB", 37)
	if len(items) == 0 {
		t.Fatal("expected file-local completion without model")
	}
}
