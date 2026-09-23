package tokenize

import (
	"strings"
	"testing"
)

func TestSubtoks(t *testing.T) {
	cases := []struct {
		in   string
		want []string
	}{
		{"parseHTTPResponse", []string{"parse", "HTTP", "Response"}},
		{"parseHTTP", []string{"parse", "HTTP"}},
		{"snake_case_name", []string{"snake", "_case", "_name"}},
		{"SCREAMING_CASE", []string{"SCREAMING", "_CASE"}},
		{"utf8", []string{"utf", "8"}},
		{"utf8Decoder", []string{"utf", "8", "Decoder"}},
		{"sha256Sum", []string{"sha", "256", "Sum"}},
		{"lowercase", []string{"lowercase"}},
		{"PascalCase", []string{"Pascal", "Case"}},
		{"A", []string{"A"}},
		{"xy", []string{"xy"}},
		{"a2b2", []string{"a", "2", "b", "2"}},
		{"kebab-name", []string{"kebab", "-name"}},
		{"IOReader", []string{"IO", "Reader"}},
		{"getURL2Path", []string{"get", "URL", "2", "Path"}},
	}
	for _, c := range cases {
		got := Subtoks(c.in)
		if strings.Join(got, "|") != strings.Join(c.want, "|") {
			t.Errorf("Subtoks(%q) = %v, want %v", c.in, got, c.want)
			continue
		}
		// Invariant: parts always rejoin to the input exactly.
		if strings.Join(got, "") != c.in {
			t.Errorf("Subtoks(%q) does not rejoin: %v", c.in, got)
		}
	}
}

func TestStructState(t *testing.T) {
	var st StructState
	base := st.Key()
	st.Advance("func")
	kw := st.Key() & 7
	if kw != kwDecl {
		t.Fatalf("func keyword class = %d want %d", kw, kwDecl)
	}
	st.Advance("{")
	d := st.Key() >> 4
	if d != 1 {
		t.Fatalf("depth after { = %d want 1", d)
	}
	st.Advance("(")
	if st.Key()&8 == 0 {
		t.Fatal("paren bit not set after (")
	}
	st.Advance(")")
	st.Advance("}")
	if st.Key() != base|uint16(kwDecl) {
		t.Fatalf("state did not return to baseline: %x vs %x", st.Key(), base)
	}
	// Depth caps at 15 and never goes negative enough to underflow.
	var s2 StructState
	for i := 0; i < 40; i++ {
		s2.Advance("{")
	}
	if s2.Key()>>4 != 15 {
		t.Fatalf("depth cap = %d want 15", s2.Key()>>4)
	}
	s2.Advance("}")
	if s2.Key()>>4 != 15 {
		t.Fatalf("depth after one close = %d want 15 (capped, not wrapped)", s2.Key()>>4)
	}
}
