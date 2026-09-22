package lines

import (
	"fmt"
	"testing"
)

func build(t *testing.T, lines []string) *Index {
	t.Helper()
	b := NewBuilder()
	for _, l := range lines {
		b.AddLine(l)
	}
	return b.Compact()
}

func TestCompleteRanksByCount(t *testing.T) {
	var ls []string
	for i := 0; i < 5; i++ {
		ls = append(ls, "if err != nil {")
	}
	for i := 0; i < 3; i++ {
		ls = append(ls, "if x == 1 {")
	}
	ls = append(ls, "if y > 0 {")
	idx := build(t, ls)
	got := idx.Complete("if ", 8, 1024)
	if len(got) != 3 {
		t.Fatalf("got %d continuations, want 3: %+v", len(got), got)
	}
	if got[0].Text != " err != nil {" || got[0].Count != 5 {
		t.Fatalf("top = %+v, want err != nil { count 5", got[0])
	}
	if got[1].Text != " x == 1 {" || got[1].Count != 3 {
		t.Fatalf("second = %+v", got[1])
	}
	if got[2].Text != " y > 0 {" || got[2].Count != 1 {
		t.Fatalf("third = %+v", got[2])
	}
}

func TestCompleteLimitAndTies(t *testing.T) {
	// Equal counts must keep lexicographic order (rests arrive sorted).
	var ls []string
	for _, rest := range []string{" b()", " a()", "c()"} {
		ls = append(ls, "call "+rest)
	}
	idx := build(t, ls)
	got := idx.Complete("call ", 2, 1024)
	if len(got) != 2 || got[0].Text != " a()" || got[1].Text != " b()" {
		t.Fatalf("ties = %+v, want a() then b()", got)
	}
}

func TestCompleteMergesDuplicateRests(t *testing.T) {
	// Same rest via different full lines must merge counts. Rests from
	// different keys sharing the prefix stay grouped.
	idx := build(t, []string{
		"x := a.b()", "x := a.b()", "x := a.b()",
		"y := a.b()", // different prefix, must not appear
		"x := a.c()",
	})
	got := idx.Complete("x := a.", 8, 1024)
	if len(got) != 2 {
		t.Fatalf("got %+v, want 2 groups", got)
	}
	if got[0].Text != "b()" || got[0].Count != 3 {
		t.Fatalf("top = %+v", got[0])
	}
}

func TestCompleteScanCap(t *testing.T) {
	var ls []string
	for i := 0; i < 100; i++ {
		ls = append(ls, fmt.Sprintf("pp %03d rest%d", i, i))
	}
	idx := build(t, ls)
	full := idx.Complete("pp ", 8, 1024)
	capped := idx.Complete("pp ", 8, 10)
	if len(full) != 8 {
		t.Fatalf("full = %d, want 8", len(full))
	}
	if len(capped) == 0 {
		t.Fatal("capped returned nothing")
	}
}

func TestCompleteSkipsShortRests(t *testing.T) {
	// rests shorter than 2 chars (after the normalized prefix) are not
	// meaningful continuations and must be skipped.
	idx := build(t, []string{"gofa()", "gox", "goy"})
	got := idx.Complete("go", 8, 1024)
	if len(got) != 1 || got[0].Text != "fa()" {
		t.Fatalf("got %+v, want only fa()", got)
	}
}
