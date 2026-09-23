package engine

import (
	"fmt"
	"os"
	"runtime"
	"testing"

	"q4tab/internal/corpus"
	"q4tab/internal/lines"
	"q4tab/internal/model"
	"q4tab/internal/tokenize"
)

// TestMemDiag attributes build memory to structures. Diagnostic only:
// go test ./internal/engine -run TestMemDiag -v with Q4TAB_MEMDIAG set to
// a corpus root. Reports entry counts per structure plus process RSS
// at a token checkpoint.
func TestMemDiag(t *testing.T) {
	root := os.Getenv("Q4TAB_MEMDIAG")
	if root == "" {
		t.Skip("Q4TAB_MEMDIAG unset")
	}
	v := model.NewVocab()
	mb := model.NewBuilder(v, 6, nil)
	lb := lines.NewBuilder()
	gb := lines.NewGramBuilder()
	sb := newStructBuilder()
	langs := newLangBuilder()
	sv := model.NewVocab()
	subB := model.NewBuilderPlain(sv, 3, []uint32{0, 1, 1, 2})
	eof := v.ID(tokenize.EOF)

	var toks int
	var files int
	ch := make(chan corpus.File, 64)
	go corpus.Collect([]string{root}, ch)
	for f := range ch {
		files++
		lb.AddFile(f.Data)
		gb.AddFile(f.Data)
		ts := tokenize.Lex(f.Data)
		ids := v.Intern(ts)
		mb.Add(ids)
		sb.Add(ts, ids)
		langs.Add(f.Path, ids, eof)
		subB.Add(subtokIDs(ts, sv))
		toks += len(ids)
		if toks > 48_000_000 {
			break
		}
	}
	runtime.GC()
	var ms runtime.MemStats
	runtime.ReadMemStats(&ms)
	fmt.Printf("files=%d tokens=%dM heapMB=%d sysMB=%d\n", files, toks/1_000_000, ms.HeapAlloc>>20, ms.Sys>>20)
	fmt.Printf("vocab=%d subvocab=%d\n", v.Len(), sv.Len())
	for k := 1; k <= 8; k++ {
		fmt.Printf("  order%d entries=%d\n", k, mb.OrderLen(k))
	}
	for k := 1; k <= 3; k++ {
		fmt.Printf("  sub%d entries=%d\n", k, subB.OrderLen(k))
	}
	fmt.Printf("  lines=%d grams=%d structRows=%d\n", lb.Len(), gb.Len(), len(sb.rows))
	for lang, t := range langs.tabs {
		n := 0
		for _, row := range t.bi {
			n += len(row)
		}
		fmt.Printf("  lang %s uni=%d bipairs=%d\n", lang, len(t.uni), n)
	}
}
