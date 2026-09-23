package esort

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// out is one aggregated record yielded by Iter.
type out struct {
	a, b, cnt uint64
}

// drain collects every record from it and closes it.
func drain(t *testing.T, it *Iter) []out {
	t.Helper()
	var res []out
	for {
		a, b, c, ok := it.Next()
		if !ok {
			break
		}
		res = append(res, out{a, b, c})
	}
	if err := it.Close(); err != nil {
		t.Fatalf("Iter.Close: %v", err)
	}
	return res
}

// runFiles lists the stream's run files still present in dir.
func runFiles(t *testing.T, dir, tag string) []string {
	t.Helper()
	ents, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	var names []string
	for _, e := range ents {
		n := e.Name()
		if strings.HasPrefix(n, tag+"-") && strings.HasSuffix(n, runExt) {
			names = append(names, n)
		}
	}
	return names
}

func TestEmptyStream(t *testing.T) {
	dir := t.TempDir()
	s, err := NewStream(dir, "empty", 8)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	it, err := s.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if it.Len() != 0 {
		t.Fatalf("Len = %d, want 0 runs", it.Len())
	}
	if _, _, _, ok := it.Next(); ok {
		t.Fatal("Next on empty stream returned a record")
	}
	if err := it.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

func TestSingleRunOrderAndCounts(t *testing.T) {
	dir := t.TempDir()
	s, err := NewStream(dir, "one", 64)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	// Out-of-order keys with repeats; all fit in one buffer.
	adds := []Rec{{3, 1}, {1, 2}, {1, 1}, {3, 1}, {2, 9}, {1, 1}, {0, 0}}
	for _, r := range adds {
		s.Add(r.A, r.B)
	}
	it, err := s.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if it.Len() != 1 {
		t.Fatalf("Len = %d, want 1 run", it.Len())
	}
	got := drain(t, it)
	want := []out{{0, 0, 1}, {1, 1, 2}, {1, 2, 1}, {2, 9, 1}, {3, 1, 2}}
	if len(got) != len(want) {
		t.Fatalf("got %d records, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("record %d = %+v, want %+v", i, got[i], want[i])
		}
	}
}

// TestMultiRunMerge interleaves keys so no run holds the full key range
// and several runs share keys.
func TestMultiRunMerge(t *testing.T) {
	dir := t.TempDir()
	const bufRecs = 4
	s, err := NewStream(dir, "merge", bufRecs)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	// 13 sightings -> 4 runs. Keys zigzag across the range so runs
	// interleave badly: run 0 sees keys 4,3,2,1 and so on.
	adds := []Rec{
		{4, 1}, {3, 2}, {2, 3}, {1, 4},
		{4, 2}, {3, 1}, {2, 4}, {1, 3},
		{4, 3}, {3, 4}, {2, 1}, {1, 2},
		{2, 2},
	}
	for _, r := range adds {
		s.Add(r.A, r.B)
	}
	it, err := s.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if it.Len() != 4 {
		t.Fatalf("Len = %d, want 4 runs", it.Len())
	}
	got := drain(t, it)
	want := []out{
		{1, 2, 1}, {1, 3, 1}, {1, 4, 1},
		{2, 1, 1}, {2, 2, 1}, {2, 3, 1}, {2, 4, 1},
		{3, 1, 1}, {3, 2, 1}, {3, 4, 1},
		{4, 1, 1}, {4, 2, 1}, {4, 3, 1},
	}
	if len(got) != len(want) {
		t.Fatalf("got %d records, want %d: %v", len(got), len(want), got)
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("record %d = %+v, want %+v", i, got[i], want[i])
		}
	}
	// Output must be strictly increasing in (a, b).
	for i := 1; i < len(got); i++ {
		if got[i].a < got[i-1].a || (got[i].a == got[i-1].a && got[i].b <= got[i-1].b) {
			t.Fatalf("records %d and %d out of order: %+v then %+v", i-1, i, got[i-1], got[i])
		}
	}
}

// TestCountsAcrossRuns repeats one key across several flushes so its
// count must be summed at merge time.
func TestCountsAcrossRuns(t *testing.T) {
	dir := t.TempDir()
	s, err := NewStream(dir, "sum", 2)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	// Key (7,7) lands in every run alongside a filler key.
	for i := uint64(0); i < 5; i++ {
		s.Add(7, 7)
		s.Add(i, i)
	}
	it, err := s.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	got := drain(t, it)
	var cnt uint64
	var seen int
	for _, o := range got {
		if o.a == 7 && o.b == 7 {
			cnt = o.cnt
			seen++
		}
	}
	if seen != 1 || cnt != 5 {
		t.Fatalf("key (7,7) seen %d times with cnt %d, want once with cnt 5", seen, cnt)
	}
}

// TestManyRuns forces many runs from a small buffer and checks total
// ordering over a larger random-ish input.
func TestManyRuns(t *testing.T) {
	dir := t.TempDir()
	s, err := NewStream(dir, "many", 3)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	const n = 500
	for i := uint64(0); i < n; i++ {
		// LCG-style key spread so consecutive sightings rarely share a run.
		s.Add((i*2654435761)%97, (i*40503)%53)
	}
	it, err := s.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if it.Len() < n/3 {
		t.Fatalf("Len = %d, want at least %d runs", it.Len(), n/3)
	}
	var total uint64
	var prev out
	first := true
	for {
		a, b, c, ok := it.Next()
		if !ok {
			break
		}
		if !first && (a < prev.a || (a == prev.a && b <= prev.b)) {
			t.Fatalf("key (%d,%d) out of order after (%d,%d)", a, b, prev.a, prev.b)
		}
		if c == 0 {
			t.Fatalf("key (%d,%d) yielded zero count", a, b)
		}
		total += c
		prev = out{a, b, c}
		first = false
	}
	if total != n {
		t.Fatalf("summed counts = %d, want %d", total, n)
	}
	if err := it.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
}

// TestCloseRemovesRuns checks run files exist after Finish and are
// gone after Close.
func TestCloseRemovesRuns(t *testing.T) {
	dir := t.TempDir()
	s, err := NewStream(dir, "clean", 2)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	for i := uint64(0); i < 10; i++ {
		s.Add(i, i)
	}
	it, err := s.Finish()
	if err != nil {
		t.Fatalf("Finish: %v", err)
	}
	if n := len(runFiles(t, dir, "clean")); n != it.Len() {
		t.Fatalf("%d run files on disk, want %d", n, it.Len())
	}
	if err := it.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if n := len(runFiles(t, dir, "clean")); n != 0 {
		t.Fatalf("%d run files left after Close", n)
	}
}

// TestCleanup verifies the standalone leftover-file remover.
func TestCleanup(t *testing.T) {
	dir := t.TempDir()
	s, err := NewStream(dir, "stale", 2)
	if err != nil {
		t.Fatalf("NewStream: %v", err)
	}
	for i := uint64(0); i < 6; i++ {
		s.Add(i, i)
	}
	// Simulate a crash: runs on disk, Finish never called.
	if len(runFiles(t, dir, "stale")) == 0 {
		t.Fatal("no run files to clean up")
	}
	if err := Cleanup(dir, "stale"); err != nil {
		t.Fatalf("Cleanup: %v", err)
	}
	if n := len(runFiles(t, dir, "stale")); n != 0 {
		t.Fatalf("%d run files left after Cleanup", n)
	}
}

// TestNewStreamBadDir fails fast when dir does not exist.
func TestNewStreamBadDir(t *testing.T) {
	_, err := NewStream(filepath.Join(t.TempDir(), "nope"), "x", 8)
	if err == nil {
		t.Fatal("NewStream on missing dir succeeded")
	}
}
