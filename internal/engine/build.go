package engine

import (
	"fmt"
	"io"
	"os"
	"runtime/debug"
	"strconv"
	"strings"

	"q4complete/internal/corpus"
	"q4complete/internal/lines"
	"q4complete/internal/model"
	"q4complete/internal/symbols"
	"q4complete/internal/tokenize"
)

// FileMeta is the manifest record for one indexed file: enough to detect
// a change on the next incremental run without reading the file.
type FileMeta struct {
	Path    string
	Size    int64
	ModTime int64
}

// BuildStats reports what went into a trained model.
type BuildStats struct {
	Files   int
	Bytes   int
	Tokens  int
	Lines   int
	Vocab   int
	Meta    []FileMeta     // per-file records for the incremental manifest
	Symbols *symbols.Index // definition sites for lookup_symbol
}

// BuildIndex trains a model bundle over the given corpus roots: the
// Kneser-Ney n-gram, the line index, the line-ngram next-line table,
// the structural-context table, per-language low-order tables, and the
// identifier-subtoken model. progress, if non-nil, receives a line of
// status per 1000 files.
//
// When the corpus exceeds tightenToks the builder retroactively gates
// orders >= 2 (singleton n-grams stop accumulating and repeat contexts
// continue under the bloom gate). 0 disables the cap.
func BuildIndex(roots []string, order int, minCnt []uint32, progress io.Writer) (*Bundle, BuildStats, error) {
	return BuildIndexBudget(roots, order, minCnt, 48_000_000, 0, progress)
}

// rssBytes reads the process resident set from /proc. Returns 0 where
// /proc is unavailable.
func rssBytes() int64 {
	data, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return 0
	}
	for _, ln := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(ln, "VmRSS:") {
			f := strings.Fields(ln)
			if len(f) >= 2 {
				if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
					return kb << 10
				}
			}
		}
	}
	return 0
}

// BuildIndexBudget is BuildIndex with explicit memory guardrails:
// tightenToks gates low-order counting once the corpus passes it, and
// memMB caps the process RSS — at 55% of the cap the build tightens
// early, at 80% it freezes gated orders outright. 0 disables each.
func BuildIndexBudget(roots []string, order int, minCnt []uint32, tightenToks, memMB int, progress io.Writer) (*Bundle, BuildStats, error) {
	v := model.NewVocab()
	mb := model.NewBuilder(v, order, minCnt)
	lb := lines.NewBuilder()
	gb := lines.NewGramBuilder()
	sb := newStructBuilder()
	langs := newLangBuilder()
	sv := model.NewVocab()
	subB := model.NewBuilderPlain(sv, 3, []uint32{0, 1, 1, 2})
	syms := symbols.NewIndex()
	var st BuildStats
	var tightened, frozen bool
	memCap := int64(memMB) << 20
	eof := v.ID(tokenize.EOF) // intern now so lexed EOFs share this id
	ch := make(chan corpus.File, 64)
	go corpus.Collect(roots, ch)
	for f := range ch {
		st.Files++
		st.Bytes += len(f.Data)
		st.Meta = append(st.Meta, FileMeta{Path: f.Path, Size: int64(len(f.Data)), ModTime: f.ModTime})
		lb.AddFile(f.Data)
		gb.AddFile(f.Data)
		for _, s := range symbols.Extract(f.Path, f.Data) {
			syms.Add(s)
		}
		toks := tokenize.Lex(f.Data)
		ids := v.Intern(toks)
		mb.Add(ids)
		sb.Add(toks, ids)
		langs.Add(f.Path, ids, eof)
		subB.Add(subtokIDs(toks, sv))
		st.Tokens += len(ids)
		var rss int64
		if memCap > 0 && st.Files%256 == 0 {
			rss = rssBytes()
		}
		if !tightened && ((tightenToks > 0 && st.Tokens > tightenToks) || (memCap > 0 && rss > memCap*55/100)) {
			// Corpus outgrew the memory budget: orders >= 2 switch to
			// bloom-gated counting from here on.
			mb.Tighten(2)
			subB.Tighten(2)
			gb.Tighten()
			langs.Tighten()
			sb.Tighten()
			lb.Tighten()
			debug.FreeOSMemory()
			tightened = true
			if progress != nil {
				fmt.Fprintf(progress, "tightened at %dM tokens rss %dMB (gated orders >= 2)\n", st.Tokens/1_000_000, rss>>20)
			}
		}
		if tightened && !frozen && ((tightenToks > 0 && st.Tokens > 3*tightenToks) || (memCap > 0 && rss > memCap*4/5)) {
			// Repeat contexts alone still outgrow the budget: gated
			// orders stop accepting new keys and keep counting what
			// they already hold.
			mb.Freeze()
			subB.Freeze()
			debug.FreeOSMemory()
			frozen = true
			if progress != nil {
				fmt.Fprintf(progress, "froze at %dM tokens rss %dMB\n", st.Tokens/1_000_000, rss>>20)
			}
		}
		if progress != nil && st.Files%1000 == 0 {
			fmt.Fprintf(progress, "indexed %d files, %d MB, %dM tokens\n", st.Files, st.Bytes>>20, st.Tokens/1_000_000)
		}
	}
	m := mb.Compact()
	li := lb.Compact()
	bun := &Bundle{
		M:      m,
		Lines:  li,
		LineBi: gb.Compact(li),
		Struct: sb.Compact(),
		Langs:  langs.Compact(),
		Sub:    subB.Compact(),
	}
	// Identifier index over the main vocab: subtoken path -> ident id,
	// with the unigram continuation count as its frequency proxy.
	var uniFreq func(uint32) int32
	if len(m.Uni) > 0 {
		uniFreq = func(id uint32) int32 {
			if int(id) < len(m.Uni) {
				return m.Uni[id]
			}
			return 0
		}
	} else {
		uniFreq = func(uint32) int32 { return 0 }
	}
	bun.Idents = model.BuildIdentIndex(v, tokenize.Subtoks, uniFreq)
	st.Vocab = v.Len()
	st.Lines = li.Len()
	st.Symbols = syms
	return bun, st, nil
}
