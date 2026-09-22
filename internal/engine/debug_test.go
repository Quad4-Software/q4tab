package engine

import (
	"fmt"
	"testing"

	"q4complete/internal/tokenize"
)

func TestDebugCandidates(t *testing.T) {
	m, li, err := Load("../../bin/model.bin")
	if err != nil {
		t.Skip("no model:", err)
	}
	e := New(DefaultConfig())
	e.SetModel(m, li)

	text := "package main\n\nfunc check() error {\n\tif err := doThing(); e"
	uri := "file:///t.go"
	e.UpdateDoc(uri, text)

	prefix := text
	toks := tokenize.Lex([]byte(prefix))
	if toks[len(toks)-1] == tokenize.EOF {
		toks = toks[:len(toks)-1]
	}
	ids := make([]uint32, 0, len(toks))
	for _, tk := range toks[:len(toks)-1] { // drop partial "e"
		if id, ok := m.Vocab.Lookup(tk); ok {
			ids = append(ids, id)
		}
	}
	ctx := ids
	if len(ctx) > m.N-1 {
		ctx = ctx[len(ctx)-(m.N-1):]
	}
	fmt.Printf("ctx tokens: %v\n", toks[len(toks)-min(len(toks), 5):len(toks)-1])
	for i, c := range e.m.TopUnion(ctx, 0.8, 40) {
		fmt.Printf("  cand %d: id=%d %q p=%.4f\n", i, c.Tok, m.Vocab.Str(c.Tok), c.P)
	}
	// also check the raw order-3 row for [";"," "]
	semID, _ := m.Vocab.Lookup(";")
	spID, _ := m.Vocab.Lookup(" ")
	fmt.Printf("semID=%d spID=%d\n", semID, spID)
	items := e.Complete(uri, text, len(text))
	for _, it := range items {
		fmt.Printf("item [%s] %q\n", it.Source, it.Text)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
