package engine

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"runtime"
	"runtime/debug"
	"sort"
	"sync"

	"q4tab/internal/corpus"
	"q4tab/internal/esort"
	"q4tab/internal/lines"
	"q4tab/internal/model"
	"q4tab/internal/symbols"
	"q4tab/internal/tokenize"
)

// Spill build: n-gram sightings stream to sorted run files on disk and
// merge back into the packed model, so peak memory is bounded by the
// record buffers and the output tables, not by corpus size. Auxiliary
// structures (line index, gram index, lang and struct tables) stay
// in-memory on the collector goroutine; they are small relative to the
// n-gram maps and still honor the tighten budget.
//
// Record packing: A = tag<<60 | key60, B = tok<<32 | ext. The tag keeps
// each logical stream contiguous inside one merged sort. key60 is the
// context hash truncated to 60 bits; models built this way carry
// Model.KeyMask so queries mask the same way. At corpus scale a rare
// collision merges two contexts, same as full hashes do more rarely.
//
// Raw tags are the order index (order k -> tag k-1) for orders 1..N,
// then the subtoken model's raw orders 1..3 at tags 8..10. Cont stream
// tags 0..4 hold (suf, tok, ext) records for orders 2..n, which become
// continuation counts for orders 1..n-1 after grouping.
const (
	spillKeyBits = 60
	spillKeyMask = uint64(1)<<spillKeyBits - 1
	tagSubBase   = 8
)

func spillA(tag uint64, key uint64) uint64 { return tag<<spillKeyBits | key&spillKeyMask }
func spillB(tok, ext uint32) uint64        { return uint64(tok)<<32 | uint64(ext) }

// spillBufRecs bounds each worker's record buffers. 4M records at 24
// bytes is ~96MB per stream.
const spillBufRecs = 4 << 20

// mIter merges several esort.Iters into one aggregated ordered stream.
// Identical (a,b) keys across iterators are summed.
type mIter struct {
	its  []*esort.Iter
	head []recEnt
	live int
}

type recEnt struct {
	a, b uint64
	cnt  uint64
	ok   bool
}

func newMIter(its []*esort.Iter) *mIter {
	m := &mIter{its: its, head: make([]recEnt, len(its))}
	for i, it := range its {
		a, b, c, ok := it.Next()
		m.head[i] = recEnt{a, b, c, ok}
		if ok {
			m.live++
		}
	}
	return m
}

// Next returns the next distinct (a,b) key with its summed count.
func (m *mIter) Next() (a, b, cnt uint64, ok bool) {
	if m.live == 0 {
		return 0, 0, 0, false
	}
	mi := -1
	for i := range m.head {
		h := m.head[i]
		if !h.ok {
			continue
		}
		if mi < 0 || h.a < m.head[mi].a || (h.a == m.head[mi].a && h.b < m.head[mi].b) {
			mi = i
		}
	}
	a, b = m.head[mi].a, m.head[mi].b
	for i := range m.head {
		h := m.head[i]
		if !h.ok || h.a != a || h.b != b {
			continue
		}
		cnt += h.cnt
		na, nb, nc, nok := m.its[i].Next()
		m.head[i] = recEnt{na, nb, nc, nok}
		if !nok {
			m.live--
		}
	}
	return a, b, cnt, true
}

func (m *mIter) Close() {
	for _, it := range m.its {
		it.Close()
	}
}

// contEnt is an aggregated continuation-count entry: the order-(k-1)
// pair (ctx,tok) has cnt distinct left extensions in the corpus.
type contEnt struct {
	ctx uint64
	tok uint32
	cnt uint32
}

// collectCPrime drains the cont iterator up to and including tag want
// and returns the (suf,tok) -> distinct-extension counts for that tag,
// sorted by (suf,tok) so it merge-joins against the raw order below.
// pending carries an overshot record into the next call.
func collectCPrime(it *mIter, wantTag uint64, pending *recEnt) []contEnt {
	var out []contEnt
	for {
		var a, b uint64
		var ok bool
		if pending.ok {
			a, b, ok = pending.a, pending.b, true
			pending.ok = false
		} else {
			a, b, _, ok = it.Next()
		}
		if !ok {
			return out
		}
		tag := a >> spillKeyBits
		if tag > wantTag {
			*pending = recEnt{a: a, b: b, ok: true}
			return out
		}
		if tag < wantTag {
			continue
		}
		// (suf, tok, ext) records: one rec per distinct ext.
		suf := a & spillKeyMask
		tok := uint32(b >> 32)
		if n := len(out); n > 0 && out[n-1].ctx == suf && out[n-1].tok == tok {
			out[n-1].cnt++
		} else {
			out = append(out, contEnt{ctx: suf, tok: tok, cnt: 1})
		}
	}
}

// orderWriter accumulates one order's rows in stream form: varint row
// length, varint row total, then (tok,cnt) varint pairs. Same encoding
// the store writes for packed orders.
type orderWriter struct {
	o      model.Order
	stream []byte
	vbuf   [binary.MaxVarintLen64]byte
	n1     int64
	n2     int64
	n3     int64
	n4     int64
}

func (w *orderWriter) put(v uint64) {
	n := binary.PutUvarint(w.vbuf[:], v)
	w.stream = append(w.stream, w.vbuf[:n]...)
}

type rowEnt struct {
	tok uint32
	cnt uint32
}

func (w *orderWriter) emitRow(key uint64, row []rowEnt) {
	sort.Slice(row, func(i, j int) bool {
		if row[i].cnt != row[j].cnt {
			return row[i].cnt > row[j].cnt
		}
		return row[i].tok < row[j].tok
	})
	var tot int64
	for _, e := range row {
		tot += int64(e.cnt)
		switch e.cnt {
		case 1:
			w.n1++
		case 2:
			w.n2++
		case 3:
			w.n3++
		case 4:
			w.n4++
		}
	}
	w.o.Keys = append(w.o.Keys, key)
	w.o.Off = append(w.o.Off, int64(len(w.stream)))
	w.put(uint64(len(row)))
	w.put(uint64(tot))
	for _, e := range row {
		w.put(uint64(e.tok))
		w.put(uint64(e.cnt))
	}
	w.o.NToks += int64(len(row))
}

func (w *orderWriter) finish() {
	w.o.Off = append(w.o.Off, int64(len(w.stream)))
	w.o.Stream = w.stream
	w.o.Disc = model.KNDiscounts(w.n1, w.n2, w.n3, w.n4)
}

// spillJob carries one file's lexed tokens back to the collector for
// the in-memory auxiliary indexes.
type spillJob struct {
	f    corpus.File
	toks []string
	ids  []uint32
}

// BuildIndexSpill is the disk-backed build path. Workers lex, intern,
// and emit n-gram records to per-worker esort streams; the collector
// goroutine feeds the in-memory aux indexes. workers <= 0 picks a
// default from GOMAXPROCS. spillDir holds the temporary run files; ""
// uses the OS temp dir. Set Q4TAB_NOS_SPILL to force the in-memory path.
func BuildIndexSpill(roots []string, order int, minCnt []uint32, tightenToks, memMB, workers int, spillDir string, progress io.Writer) (*Bundle, BuildStats, error) {
	if os.Getenv("Q4TAB_NOS_SPILL") != "" {
		return BuildIndexBudget(roots, order, minCnt, tightenToks, memMB, progress)
	}
	// A tighter GC target halves the dead-heap slack during the collect
	// phase; the record buffers and run files do the real bounding.
	old := debug.SetGCPercent(50)
	defer debug.SetGCPercent(old)
	return buildIndexSpill(roots, order, minCnt, tightenToks, memMB, workers, spillDir, progress)
}

func buildIndexSpill(roots []string, order int, minCnt []uint32, tightenToks, memMB, workers int, spillDir string, progress io.Writer) (*Bundle, BuildStats, error) {
	deep := 2
	n := order
	total := n + deep
	mins := model.DefaultMinCnt(n, deep, minCnt)
	// Effective sightings thresholds. The gated in-memory orders only
	// ever stored repeats, so the equivalents here keep a floor of 2.
	effMin := make([]uint32, total+1)
	for k := 1; k <= total; k++ {
		mc := mins[k]
		if k > n && mc < 2 {
			mc = 2
		}
		effMin[k] = mc
	}

	tmp, err := os.MkdirTemp(spillDir, "q4build-*")
	if err != nil {
		return nil, BuildStats{}, err
	}
	defer os.RemoveAll(tmp)

	v := model.NewVocab()
	sv := model.NewVocab()
	eof := v.ID(tokenize.EOF)

	if workers < 1 {
		workers = runtime.GOMAXPROCS(0) - 1
		if workers < 1 {
			workers = 1
		}
		if workers > 8 {
			workers = 8
		}
	}
	type wk struct {
		raw  *esort.Stream
		cont *esort.Stream
	}
	ws := make([]wk, workers)
	for i := range ws {
		tag := fmt.Sprintf("w%d", i)
		ws[i].raw, err = esort.NewStream(tmp, tag+"-raw", spillBufRecs)
		if err != nil {
			return nil, BuildStats{}, err
		}
		ws[i].cont, err = esort.NewStream(tmp, tag+"-cont", spillBufRecs)
		if err != nil {
			return nil, BuildStats{}, err
		}
	}

	files := make(chan corpus.File, workers*4)
	jobs := make(chan spillJob, workers*4)
	var wg sync.WaitGroup
	for i := range ws {
		w := &ws[i]
		wg.Add(1)
		go func() {
			defer wg.Done()
			for f := range files {
				if f.DupOf != "" {
					jobs <- spillJob{f: f}
					continue
				}
				toks := tokenize.Lex(f.Data)
				ids := v.Intern(toks)
				model.EmitNGrams(v, n, deep, ids, func(k int, full, suf uint64, x, tok uint32) {
					w.raw.Add(spillA(uint64(k-1), full), spillB(tok, 0))
					if k >= 2 && k <= n {
						w.cont.Add(spillA(uint64(k-2), suf), spillB(tok, x))
					}
				})
				sids := subtokIDs(toks, sv)
				model.EmitNGrams(sv, 3, 0, sids, func(k int, full, suf uint64, x, tok uint32) {
					w.raw.Add(spillA(tagSubBase+uint64(k-1), full), spillB(tok, 0))
				})
				jobs <- spillJob{f: f, toks: toks, ids: ids}
			}
		}()
	}
	go func() {
		// Collect closes the files channel when it finishes.
		corpus.Collect(roots, files)
	}()
	go func() {
		wg.Wait()
		close(jobs)
	}()

	lb := lines.NewBuilder()
	gb := lines.NewGramBuilder()
	sb := newStructBuilder()
	langs := newLangBuilder()
	starts := newStartBuilder()
	dirs := newDirBuilder()
	syms := symbols.NewIndex()
	var st BuildStats
	var tightened bool
	memCap := int64(memMB) << 20
	var filesN int
	for j := range jobs {
		f := j.f
		filesN++
		st.Files++
		st.Meta = append(st.Meta, FileMeta{Path: f.Path, Size: int64(f.Size), ModTime: f.ModTime})
		if f.DupOf != "" {
			continue
		}
		st.Bytes += len(f.Data)
		lb.AddFile(f.Data)
		gb.AddFile(f.Data)
		starts.Add(f.Path, f.Data)
		dirs.Add(f.Path, f.Data)
		for _, s := range symbols.Extract(f.Path, f.Data) {
			syms.Add(s)
		}
		sb.Add(j.toks, j.ids)
		langs.Add(f.Path, j.ids, eof)
		st.Tokens += len(j.ids)
		var rss int64
		if memCap > 0 && filesN%256 == 0 {
			rss = rssBytes()
		}
		if !tightened && ((tightenToks > 0 && st.Tokens > tightenToks) || (memCap > 0 && rss > memCap*55/100)) {
			// The spilled n-gram streams are already bounded; this
			// guards only the in-memory aux indexes.
			gb.Tighten()
			langs.Tighten()
			sb.Tighten()
			lb.Tighten()
			debug.FreeOSMemory()
			tightened = true
			if progress != nil {
				fmt.Fprintf(progress, "aux tightened at %dM tokens rss %dMB\n", st.Tokens/1_000_000, rss>>20)
			}
		}
		if progress != nil && filesN%1000 == 0 {
			fmt.Fprintf(progress, "indexed %d files, %d MB, %dM tokens\n", filesN, st.Bytes>>20, st.Tokens/1_000_000)
		}
	}
	debug.FreeOSMemory()

	rawIts := make([]*esort.Iter, 0, workers)
	contIts := make([]*esort.Iter, 0, workers)
	for i := range ws {
		ri, err := ws[i].raw.Finish()
		if err != nil {
			return nil, st, err
		}
		rawIts = append(rawIts, ri)
		ci, err := ws[i].cont.Finish()
		if err != nil {
			return nil, st, err
		}
		contIts = append(contIts, ci)
	}
	raw := newMIter(rawIts)
	defer raw.Close()
	cont := newMIter(contIts)
	defer cont.Close()

	m := &model.Model{Vocab: v, N: total, KN: true, KeyMask: spillKeyMask}
	m.Orders = make([]model.Order, total+1)
	sub := &model.Model{Vocab: sv, N: 3, KN: false, KeyMask: spillKeyMask}
	sub.Orders = make([]model.Order, 4)

	var pending recEnt
	var cprime []contEnt
	var ow *orderWriter
	var uni []rowEnt
	var row []rowEnt
	var curKey uint64
	var curTag int64 = -1
	var ci int // merge-join cursor into cprime

	closeSection := func(tag int64) {
		if len(row) > 0 && ow != nil {
			ow.emitRow(curKey, row)
			row = row[:0]
		}
		if ow == nil {
			return
		}
		ow.finish()
		if tag < tagSubBase {
			k := int(tag) + 1
			if k == 1 {
				buildUnigram(m, uni, ow)
			} else {
				m.Orders[k] = ow.o
			}
		} else {
			sub.Orders[int(tag)-tagSubBase+1] = ow.o
		}
		ow = nil
	}

	for {
		a, b, cnt, ok := raw.Next()
		if !ok {
			break
		}
		tag := int64(a >> spillKeyBits)
		key := a & spillKeyMask
		tok := uint32(b >> 32)
		if tag != curTag {
			closeSection(curTag)
			curTag = tag
			ow = &orderWriter{}
			uni = uni[:0]
			cprime = nil
			ci = 0
		}
		if tag == curTag && len(row) > 0 && key != curKey {
			ow.emitRow(curKey, row)
			row = row[:0]
		}
		curKey = key
		k := int(tag) + 1
		if tag >= tagSubBase {
			k = int(tag) - tagSubBase + 1
		}
		var minc uint64
		if tag < tagSubBase {
			minc = uint64(effMin[k])
		} else {
			minc = uint64(subMinCnt[k])
		}
		if cnt < minc {
			continue
		}
		if tag < tagSubBase && k <= n-1 {
			// KN order: stored count is the continuation count from
			// the order above, merge-joined on (ctx, tok).
			if cprime == nil {
				cprime = collectCPrime(cont, uint64(k+1)-2, &pending)
			}
			for ci < len(cprime) && (cprime[ci].ctx < key || (cprime[ci].ctx == key && cprime[ci].tok < tok)) {
				ci++
			}
			if ci >= len(cprime) || cprime[ci].ctx != key || cprime[ci].tok != tok {
				continue
			}
			if tag == 0 {
				uni = append(uni, rowEnt{tok, cprime[ci].cnt})
				continue
			}
			row = append(row, rowEnt{tok, cprime[ci].cnt})
		} else {
			row = append(row, rowEnt{tok, uint32(cnt)})
		}
	}
	closeSection(curTag)

	li := lb.Compact()
	bun := &Bundle{
		M:          m,
		Lines:      li,
		LineBi:     gb.Compact(li),
		Struct:     sb.Compact(),
		Langs:      langs.Compact(),
		Sub:        sub,
		FileStarts: starts.Compact(8, 3),
		DirIdents:  dirs.Compact(96, 3),
	}
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

// buildUnigram folds the joined order-1 entries into the flat unigram
// table and its discount header.
func buildUnigram(m *model.Model, uni []rowEnt, ow *orderWriter) {
	m.Uni = make([]int32, m.Vocab.Len())
	var n1, n2, n3, n4 int64
	for _, e := range uni {
		if int(e.tok) < len(m.Uni) {
			m.Uni[e.tok] = int32(e.cnt)
			m.UniTot += int64(e.cnt)
		}
		switch e.cnt {
		case 1:
			n1++
		case 2:
			n2++
		case 3:
			n3++
		case 4:
			n4++
		}
	}
	m.UniDisc = model.KNDiscounts(n1, n2, n3, n4)
	m.UniGamma = m.UniDisc[0]*float64(n1) + m.UniDisc[1]*float64(n2) + m.UniDisc[2]*float64(n3+n4)
	m.UniTop = uniTop(m.Uni, 64)
	ow.o.Disc = m.UniDisc
	m.Orders[1] = ow.o
}

// uniTop returns the nt tokens with the largest unigram counts.
func uniTop(uni []int32, nt int) []uint32 {
	type tc struct {
		t uint32
		c int32
	}
	top := make([]tc, 0, nt)
	for t, c := range uni {
		if c <= 0 {
			continue
		}
		top = append(top, tc{uint32(t), c})
		if len(top) > nt*4 {
			sort.Slice(top, func(i, j int) bool { return top[i].c > top[j].c })
			top = top[:nt]
		}
	}
	sort.Slice(top, func(i, j int) bool {
		if top[i].c != top[j].c {
			return top[i].c > top[j].c
		}
		return top[i].t < top[j].t
	})
	if len(top) > nt {
		top = top[:nt]
	}
	out := make([]uint32, len(top))
	for i, e := range top {
		out[i] = e.t
	}
	return out
}
