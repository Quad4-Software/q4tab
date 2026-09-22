package engine

import (
	"fmt"
	"io"

	"q4complete/internal/corpus"
	"q4complete/internal/lines"
	"q4complete/internal/model"
	"q4complete/internal/tokenize"
)

// BuildStats reports what went into a trained model.
type BuildStats struct {
	Files  int
	Bytes  int
	Tokens int
	Lines  int
	Vocab  int
}

// BuildIndex trains a model over the given corpus roots. progress, if
// non-nil, receives a line of status per 1000 files.
func BuildIndex(roots []string, order int, minCnt []uint32, progress io.Writer) (*model.Model, *lines.Index, BuildStats, error) {
	v := model.NewVocab()
	mb := model.NewBuilder(v, order, minCnt)
	lb := lines.NewBuilder()
	var st BuildStats
	ch := make(chan corpus.File, 64)
	go corpus.Collect(roots, ch)
	for f := range ch {
		st.Files++
		st.Bytes += len(f.Data)
		lb.AddFile(f.Data)
		toks := tokenize.Lex(f.Data)
		ids := v.Intern(toks)
		mb.Add(ids)
		st.Tokens += len(ids)
		if progress != nil && st.Files%1000 == 0 {
			fmt.Fprintf(progress, "indexed %d files, %d MB, %dM tokens\n", st.Files, st.Bytes>>20, st.Tokens/1_000_000)
		}
	}
	m := mb.Compact()
	li := lb.Compact()
	st.Vocab = v.Len()
	st.Lines = li.Len()
	return m, li, st, nil
}
