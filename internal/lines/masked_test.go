package lines

import "testing"

func TestMaskedAdaptRenames(t *testing.T) {
	idx := build(t, []string{
		"if err := decode(&req); err != nil { report(req) }",
		"if err := decode(&req); err != nil { report(req) }",
		"if err := decode(&req); err != nil { report(req) }",
		"if err := decode(&req); err != nil { report(req) }",
	})
	mi := BuildMaskedIndex(idx)
	got := mi.Adapt("if err := decode(&in", 8, 64)
	if len(got) == 0 {
		t.Fatal("no adapted continuations")
	}
	if got[0].Text != "); err != nil { report(in) }" {
		t.Fatalf("top = %q, want %q", got[0].Text, "); err != nil { report(in) }")
	}
	// The verbatim index misses this entirely: identifiers differ.
	if vb := idx.Complete("if err := decode(&in", 8, 64); len(vb) != 0 {
		t.Fatalf("verbatim should not match renamed idents, got %+v", vb)
	}
}

func TestMaskedAdaptSkipsUnreboundRest(t *testing.T) {
	// When the continuation uses none of the renamed identifiers, the
	// adapt layer stays silent: the token model covers shape-only
	// continuations.
	idx := build(t, []string{
		"if err := decode(&req); err != nil {",
		"if err := decode(&req); err != nil {",
		"if err := decode(&req); err != nil {",
	})
	mi := BuildMaskedIndex(idx)
	if got := mi.Adapt("if err := decode(&in", 8, 64); len(got) != 0 {
		t.Fatalf("adapt emitted a rest with no rebound names: %+v", got)
	}
}

func TestMaskedAdaptReceiverRename(t *testing.T) {
	idx := build(t, []string{
		"func (s *Store) Get(key string) (Value, error) {",
		"func (s *Store) Get(key string) (Value, error) {",
		"func (s *Store) Get(key string) (Value, error) {",
	})
	mi := BuildMaskedIndex(idx)
	got := mi.Adapt("func (c *Cache) Ge", 8, 64)
	if len(got) == 0 {
		t.Fatal("no adapted continuations")
	}
	// "Ge" is a strict prefix of the stored "Get": the completing
	// variant finishes the stored name with the rebound rest.
	want := "t(key string) (Value, error) {"
	if got[0].Text != want {
		t.Fatalf("top = %q, want %q", got[0].Text, want)
	}
}

func TestMaskedAdaptNoMatchOnShapeDivergence(t *testing.T) {
	idx := build(t, []string{
		"if err := decode(&req); err != nil {",
		"if err := decode(&req); err != nil {",
		"if err := decode(&req); err != nil {",
	})
	mi := BuildMaskedIndex(idx)
	// A literal where the stored line has an identifier is a
	// different shape: no match.
	if got := mi.Adapt("if err := decode(&42", 8, 64); len(got) != 0 {
		t.Fatalf("unexpected hits: %+v", got)
	}
	// Different operator is a different shape.
	if got := mi.Adapt("if err := decode(&req) == nil", 8, 64); len(got) != 0 {
		t.Fatalf("unexpected hits: %+v", got)
	}
}

func TestMaskedAdaptKeepsKeywords(t *testing.T) {
	idx := build(t, []string{
		"for i := 0; i < n; i++ { use(i) }",
		"for i := 0; i < n; i++ { use(i) }",
		"for i := 0; i < n; i++ { use(i) }",
	})
	mi := BuildMaskedIndex(idx)
	got := mi.Adapt("for j := 0; j < total; j++", 8, 64)
	if len(got) == 0 || got[0].Text != " { use(j) }" {
		t.Fatalf("got %+v, want rebind i->j n->total", got)
	}
}

func TestRebindDirect(t *testing.T) {
	got := Rebind("if err := decode(&req); err != nil { use(req) }", "if err := decode(&in")
	if got != "); err != nil { use(in) }" {
		t.Fatalf("rebind = %q", got)
	}
	// Identical prefix: whole line consumed, nothing to return.
	if got := Rebind("return nil, err", "return nil, err"); got != "" {
		t.Fatalf("full-line rebind = %q, want empty", got)
	}
	// Divergent shape.
	if got := Rebind("x := a + b", "x := a -"); got != "" {
		t.Fatalf("divergent rebind = %q, want empty", got)
	}
}

func TestMaskedLineShape(t *testing.T) {
	a := MaskedLine("if err := f(&req); err != nil {")
	b := MaskedLine("if err := g(&res); err != nil {")
	if a != b {
		t.Fatalf("masked shapes differ:\n%q\n%q", a, b)
	}
	c := MaskedLine("if err := f(&req) {")
	if a == c {
		t.Fatal("divergent line shares masked shape")
	}
	// A keyword-shaped query token still matches through expansion:
	// "in" is a Python keyword but a common Go identifier.
	if !MatchesMasked("if err := f(&req); err != nil {", "if err := g(&in") {
		t.Fatal("ambiguous keyword token did not expand to masked form")
	}
}
