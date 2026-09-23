// Binary persistence for the trained model. The gob format decoded into
// fully resident Go structures (a 500MB file became ~3GB RSS). This
// format stores flat little-endian sections that Load mmaps read-only,
// so only touched pages occupy memory.
//
// Layout (all little-endian, every array section 8-byte aligned):
//
//	u32 magic 'Q4C3', u32 version=3, u32 order N (incl. deep orders),
//	  u32 vocabN, u8 flags (bit0: Kneser-Ney), pad to 8
//	u32 vocabOffs[vocabN+1], u64 vocabBlobLen, u8 vocabBlob[...], pad8
//
//	KN models only, unigram section:
//	  i64 uniTot, 3xf64 uniDisc, f64 uniGamma,
//	  u32 nUniTop, i32 uniTop[nUniTop], i32 uni[vocabN]
//
//	Order sections, per k (KN: 2..N, plain: 1..N):
//	  u64 nKeys, u64 nToks, 3xf64 disc, u64 streamLen,
//	  u64 keys[nKeys], i64 off[nKeys+1], u8 stream[streamLen], pad8
//	  where each row in the stream is
//	  uvarint nToks, uvarint rowTotal, uvarint toks[n], uvarint cnts[n].
//
//	u32 nLines, u32 linesOff[nLines+1], i32 linesCnts[nLines],
//	  u64 linesBlobLen, u8 linesBlob[...]
//
//	Auxiliary sections (v3): each is a count-prefixed block, zero count
//	means absent.
//	  line-ngram table: u64 nKeys, u64 streamLen, u64 keys, i64 off,
//	    stream (same row encoding, toks are line-index positions)
//	  struct table: same shape
//	  langs: u32 nLangs, per lang: u16 nameLen, name, then a unigram
//	    and a bigram order section
//	  subtoken model: u8 present; u32 subN, u32 subVocabN, vocab offs +
//	    blob, then subN plain order sections (k=1..subN)
//	  ident index: u32 n, u32 off[n+1], i32 ids[n], i32 freq[n],
//	    u64 blobLen, blob
//
// Versions 1 and 2 (magic Q4C2) still load: v1 flat CSR arrays, v2
// varint row streams, both without KN metadata or aux sections.
package engine

import (
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/bits"
	"os"
	"path/filepath"
	"strings"
	"unsafe"

	"q4tab/internal/lines"
	"q4tab/internal/model"
	"sort"
)

const (
	storeMagicV2 = 0x51344332 // 'Q4C2' with u32 version 1 or 2
	storeMagic   = 0x51344333 // 'Q4C3'
	// v4 adds keybits in flags bits 1-7 (the context-hash key width
	// used by spill builds). v3 files carry no keybits and still load.
	storeVersion = 4
)

// Save writes the bundle to path atomically in the packed binary format.
func Save(path string, b *Bundle) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := &countingWriter{w: f}
	fail := func(e error) error {
		f.Close()
		os.Remove(tmp)
		return e
	}
	m := b.M
	vblob, voffs := m.Vocab.Blob()
	putU32(w, storeMagic)
	putU32(w, storeVersion)
	putU32(w, uint32(m.N))
	putU32(w, uint32(len(voffs)-1))
	var flags byte
	if m.KN {
		flags |= 1
	}
	if m.KeyMask != 0 {
		kb := bits.Len64(m.KeyMask)
		if kb >= 1 && kb < 64 {
			flags |= byte(kb) << 1
		}
	}
	w.Write([]byte{flags})
	pad8(w)
	putSlice(w, voffs)
	putU64(w, uint64(len(vblob)))
	w.Write(vblob)
	pad8(w)

	if m.KN {
		putU64(w, uint64(m.UniTot))
		putF64(w, m.UniDisc[0])
		putF64(w, m.UniDisc[1])
		putF64(w, m.UniDisc[2])
		putF64(w, m.UniGamma)
		putU32(w, uint32(len(m.UniTop)))
		putSlice(w, u32toi32(m.UniTop))
		putSlice(w, m.Uni)
		pad8(w)
	}
	first := 1
	if m.KN {
		first = 2 // unigram lives in the flat array, not the order table
	}
	for k := first; k <= m.N; k++ {
		writeOrder(w, &m.Orders[k])
	}

	var loff []uint32
	var lcnts []int32
	var lblob []byte
	if b.Lines != nil {
		loff, lcnts, lblob = b.Lines.Off, b.Lines.Cnts, b.Lines.Blob
	}
	putU32(w, uint32(len(lcnts)))
	putSlice(w, loff)
	putSlice(w, lcnts)
	putU64(w, uint64(len(lblob)))
	w.Write(lblob)
	pad8(w)

	// Line n-gram table.
	if b.LineBi != nil {
		writeOrderSection(w, b.LineBi.Keys, b.LineBi.Off, b.LineBi.Stream, 0, [3]float64{})
	} else {
		writeOrderSection(w, nil, nil, nil, 0, [3]float64{})
	}
	// Structural-context table.
	writeOrder(w, b.Struct)

	// Per-language tables.
	names := make([]string, 0, len(b.Langs))
	for name := range b.Langs {
		names = append(names, name)
	}
	sort.Strings(names)
	putU32(w, uint32(len(names)))
	for _, name := range names {
		lt := b.Langs[name]
		putU16(w, uint16(len(name)))
		w.Write([]byte(name))
		writeOrder(w, &lt.Uni)
		writeOrder(w, &lt.Bi)
	}

	// Subtoken model: a plain (raw-count) model with its own vocab.
	if b.Sub != nil {
		w.Write([]byte{1})
		pad8(w)
		sb := b.Sub
		svblob, svoffs := sb.Vocab.Blob()
		putU32(w, uint32(sb.N))
		putU32(w, uint32(len(svoffs)-1))
		putSlice(w, svoffs)
		putU64(w, uint64(len(svblob)))
		w.Write(svblob)
		pad8(w)
		for k := 1; k <= sb.N; k++ {
			writeOrder(w, &sb.Orders[k])
		}
	} else {
		w.Write([]byte{0})
		pad8(w)
	}

	// Identifier index.
	if b.Idents != nil {
		ix := b.Idents
		putU32(w, uint32(len(ix.Ids)))
		putSlice(w, ix.Off)
		putSlice(w, ix.Ids)
		putSlice(w, ix.Freq)
		putU64(w, uint64(len(ix.Blob)))
		w.Write(ix.Blob)
	} else {
		putU32(w, 0)
	}

	// File-start priors: lang -> common first lines. A trailing
	// section: older readers stop at end of their known sections.
	if len(b.FileStarts) > 0 {
		langs := make([]string, 0, len(b.FileStarts))
		for l := range b.FileStarts {
			langs = append(langs, l)
		}
		sort.Strings(langs)
		putU32(w, uint32(len(langs)))
		for _, l := range langs {
			putU16(w, uint16(len(l)))
			w.Write([]byte(l))
			putU32(w, uint32(len(b.FileStarts[l])))
			for _, s := range b.FileStarts[l] {
				putU16(w, uint16(len(s)))
				w.Write([]byte(s))
			}
		}
	} else {
		putU32(w, 0)
	}

	// Directory identifier tables: dir -> top idents. Trailing
	// section; older readers stop at their last known section.
	if len(b.DirIdents) > 0 {
		dirs := make([]string, 0, len(b.DirIdents))
		for d := range b.DirIdents {
			dirs = append(dirs, d)
		}
		sort.Strings(dirs)
		putU32(w, uint32(len(dirs)))
		for _, d := range dirs {
			putU16(w, uint16(len(d)))
			w.Write([]byte(d))
			putU32(w, uint32(len(b.DirIdents[d])))
			for _, s := range b.DirIdents[d] {
				putU16(w, uint16(len(s)))
				w.Write([]byte(s))
			}
		}
	} else {
		putU32(w, 0)
	}
	if w.err != nil {
		return fail(w.err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// writeOrder emits one order section. In-memory orders (fresh builds)
// are encoded to the varint row stream first; already-packed orders
// (loaded models being re-saved) copy through.
func writeOrder(w *countingWriter, o *model.Order) {
	if o == nil {
		writeOrderSection(w, nil, nil, nil, 0, [3]float64{})
		return
	}
	if o.Stream != nil {
		writeOrderSection(w, o.Keys, o.Off, o.Stream, o.NToks, o.Disc)
		return
	}
	var stream []byte
	offs := make([]int64, 0, len(o.Keys)+1)
	offs = append(offs, 0)
	var vbuf [binary.MaxVarintLen64]byte
	var nt int64
	for i := range o.Keys {
		rtoks := o.Toks[o.Off[i]:o.Off[i+1]]
		rcnts := o.Cnts[o.Off[i]:o.Off[i+1]]
		n := binary.PutUvarint(vbuf[:], uint64(len(rtoks)))
		stream = append(stream, vbuf[:n]...)
		n = binary.PutUvarint(vbuf[:], uint64(o.Totals[i]))
		stream = append(stream, vbuf[:n]...)
		for j, t := range rtoks {
			n = binary.PutUvarint(vbuf[:], uint64(uint32(t)))
			stream = append(stream, vbuf[:n]...)
			n = binary.PutUvarint(vbuf[:], uint64(uint32(rcnts[j])))
			stream = append(stream, vbuf[:n]...)
		}
		nt += int64(len(rtoks))
		offs = append(offs, int64(len(stream)))
	}
	writeOrderSection(w, o.Keys, offs, stream, nt, o.Disc)
}

// writeOrderSection writes keys + row-offset + stream with the v3
// discount header (all zero for tables that do not use KN).
func writeOrderSection(w *countingWriter, keys []uint64, off []int64, stream []byte, ntoks int64, disc [3]float64) {
	putU64(w, uint64(len(keys)))
	putU64(w, uint64(ntoks))
	putF64(w, disc[0])
	putF64(w, disc[1])
	putF64(w, disc[2])
	putU64(w, uint64(len(stream)))
	putSlice(w, keys)
	putSlice(w, off)
	w.Write(stream)
	pad8(w)
}

// readOrderSection reads one v2/v3 order section into stream form.
// ver selects the encoding: v1 is flat CSR (handled by the caller),
// v2/v3 are varint row streams. v3 sections carry a 3xf64 discount
// header that v2 lacks.
func readOrderSection(r *reader, disc bool) (*model.Order, error) {
	nk := int(r.u64())
	nt := int(r.u64())
	var d [3]float64
	if disc {
		d[0] = r.f64()
		d[1] = r.f64()
		d[2] = r.f64()
	}
	slen := int(r.u64())
	if nk < 0 || nk > 1<<31 || nt < 0 || nt > 1<<33 || slen < 0 || slen > 1<<33 {
		return nil, fmt.Errorf("model: absurd order sizes nk=%d nt=%d slen=%d", nk, nt, slen)
	}
	o := &model.Order{Disc: d}
	o.Keys = r.u64s(nk)
	o.Off = r.i64s(nk + 1)
	o.Stream = r.bytes(slen)
	o.NToks = int64(nt)
	r.align8()
	if r.err != nil {
		return nil, fmt.Errorf("model: truncated order section")
	}
	if nk == 0 {
		if slen != 0 {
			return nil, fmt.Errorf("model: empty order with stream")
		}
		return o, nil
	}
	if len(o.Off) != nk+1 || len(o.Stream) != slen || !validOffs64(o.Off, slen) {
		return nil, fmt.Errorf("model: corrupt order section")
	}
	return o, nil
}

// Load reads the packed model file. It mmaps the file read-only so the
// tables stay on disk until touched. Falls back to a plain read when
// mmap is unavailable (non-unix or tiny files).
func Load(path string) (*Bundle, error) {
	data, mapped, err := mapFile(path)
	if err != nil {
		return nil, err
	}
	if os.Getenv("Q4TAB_DEBUG_RSS") != "" {
		fmt.Fprintf(os.Stderr, "post-map: %s\n", rssDebug())
	}
	r := &reader{b: data}
	magic := r.u32()
	if magic == storeMagicV2 {
		return loadV12(r, data, mapped)
	}
	if magic != storeMagic {
		return nil, fmt.Errorf("model: bad magic %x (old gob format? rebuild the index)", magic)
	}
	ver := r.u32()
	n := int(r.u32())
	vn := int(r.u32())
	flags := r.bytes(1)
	r.align8()
	if ver != 3 && ver != storeVersion {
		return nil, fmt.Errorf("model: unsupported version %d", ver)
	}
	if n < 1 || n > 16 || vn < 0 || vn > 1<<27 {
		return nil, fmt.Errorf("model: bad header order=%d vocab=%d", n, vn)
	}
	kn := len(flags) > 0 && flags[0]&1 != 0
	var keyMask uint64
	if ver >= 4 && len(flags) > 0 {
		if kb := int(flags[0] >> 1); kb > 0 {
			keyMask = uint64(1)<<kb - 1
		}
	}
	voffs := r.u32s(vn + 1)
	vlen := r.u64()
	vblob := r.bytes(int(vlen))
	r.align8()
	if r.err != nil || len(voffs) != vn+1 || len(vblob) != int(vlen) {
		return nil, fmt.Errorf("model: truncated vocab section")
	}
	if !validOffs32(voffs, len(vblob)) {
		return nil, fmt.Errorf("model: corrupt vocab offsets")
	}
	v := model.VocabFromBlob(vblob, voffs)
	m := &model.Model{Vocab: v, N: n, KN: kn, KeyMask: keyMask}
	m.Orders = make([]model.Order, n+1)

	if kn {
		m.UniTot = int64(r.u64())
		m.UniDisc[0] = r.f64()
		m.UniDisc[1] = r.f64()
		m.UniDisc[2] = r.f64()
		m.UniGamma = r.f64()
		ntop := int(r.u32())
		if ntop < 0 || ntop > 1<<20 {
			return nil, fmt.Errorf("model: bad unigram top %d", ntop)
		}
		top32 := r.i32s(ntop)
		m.UniTop = make([]uint32, len(top32))
		for i, t := range top32 {
			m.UniTop[i] = uint32(t)
		}
		m.Uni = r.i32s(vn)
		r.align8()
		if r.err != nil || len(m.Uni) != vn {
			return nil, fmt.Errorf("model: truncated unigram section")
		}
	}
	first := 1
	if kn {
		first = 2
	}
	for k := first; k <= n; k++ {
		o, err := readOrderSection(r, true)
		if err != nil {
			return nil, fmt.Errorf("model: order %d: %w", k, err)
		}
		m.Orders[k] = *o
	}
	// Remaining sections are optional: a truncated tail degrades to a
	// smaller bundle rather than a hard error only when the required
	// parts are already sound. Line index is required.
	nl := int(r.u32())
	if r.err != nil {
		return nil, fmt.Errorf("model: truncated line index header")
	}
	bun := &Bundle{M: m}
	if nl > 0 {
		loff := r.u32s(nl + 1)
		lcnts := r.i32s(nl)
		llen := r.u64()
		lblob := r.bytes(int(llen))
		if r.err != nil {
			return nil, fmt.Errorf("model: truncated line index")
		}
		if len(loff) != nl+1 || len(lcnts) != nl || !validOffs32(loff, len(lblob)) {
			return nil, fmt.Errorf("model: corrupt line index")
		}
		bun.Lines = &lines.Index{Blob: lblob, Off: loff, Cnts: lcnts}
	}
	r.align8()
	// Aux sections: absent (end of file) is fine, but a section that
	// starts and then truncates is corrupt, not degraded.
	hasAux := func() bool { return r.err == nil && r.off < len(r.b) }
	if hasAux() {
		o, err := readOrderSection(r, true)
		if err != nil {
			return nil, fmt.Errorf("model: line-gram section: %w", err)
		}
		if len(o.Keys) > 0 {
			bun.LineBi = &lines.GramIndex{Keys: o.Keys, Off: o.Off, Stream: o.Stream}
		}
	}
	if hasAux() {
		o, err := readOrderSection(r, true)
		if err != nil {
			return nil, fmt.Errorf("model: struct section: %w", err)
		}
		if len(o.Keys) > 0 {
			bun.Struct = o
		}
	}
	if hasAux() {
		nl := int(r.u32())
		if nl < 0 || nl > 128 || r.err != nil {
			return nil, fmt.Errorf("model: bad langs section")
		}
		bun.Langs = map[string]*model.LangTable{}
		for i := 0; i < nl; i++ {
			nlen := int(r.u16())
			name := string(r.bytes(nlen))
			uni, e1 := readOrderSection(r, true)
			bi, e2 := readOrderSection(r, true)
			if e1 != nil || e2 != nil || r.err != nil {
				return nil, fmt.Errorf("model: lang %q section: %v %v", name, e1, e2)
			}
			lt := &model.LangTable{Uni: *uni, Bi: *bi}
			bun.Langs[name] = lt
		}
	}
	if hasAux() {
		p := r.bytes(1)
		if len(p) != 1 || r.err != nil {
			return nil, fmt.Errorf("model: truncated subtoken flag")
		}
		if p[0] == 1 {
			r.align8()
			subN := int(r.u32())
			subVN := int(r.u32())
			if subN < 1 || subN > 8 || subVN <= 0 || subVN > 1<<27 {
				return nil, fmt.Errorf("model: bad subtoken header n=%d vocab=%d", subN, subVN)
			}
			svoffs := r.u32s(subVN + 1)
			svlen := r.u64()
			svblob := r.bytes(int(svlen))
			r.align8()
			if r.err != nil || len(svoffs) != subVN+1 || !validOffs32(svoffs, len(svblob)) {
				return nil, fmt.Errorf("model: corrupt subtoken vocab")
			}
			sub := &model.Model{Vocab: model.VocabFromBlob(svblob, svoffs), N: subN}
			sub.Orders = make([]model.Order, subN+1)
			for k := 1; k <= subN; k++ {
				o, err := readOrderSection(r, true)
				if err != nil {
					return nil, fmt.Errorf("model: subtoken order %d: %w", k, err)
				}
				sub.Orders[k] = *o
			}
			bun.Sub = sub
		}
	}
	if hasAux() {
		ni := int(r.u32())
		if ni < 0 || ni > 1<<24 || r.err != nil {
			return nil, fmt.Errorf("model: bad ident index header")
		}
		if ni > 0 {
			ioff := r.u32s(ni + 1)
			iids := r.i32s(ni)
			ifreq := r.i32s(ni)
			ilen := r.u64()
			iblob := r.bytes(int(ilen))
			if r.err != nil || len(ioff) != ni+1 || !validOffs32(ioff, len(iblob)) {
				return nil, fmt.Errorf("model: corrupt ident index")
			}
			bun.Idents = &model.IdentIndex{Off: ioff, Blob: iblob, Ids: iids, Freq: ifreq}
		}
	}
	if hasAux() {
		nl := int(r.u32())
		if nl < 0 || nl > 128 || r.err != nil {
			return nil, fmt.Errorf("model: bad file-starts header")
		}
		if nl > 0 {
			bun.FileStarts = map[string][]string{}
			for i := 0; i < nl; i++ {
				name := string(r.bytes(int(r.u16())))
				ns := int(r.u32())
				if ns < 0 || ns > 64 || r.err != nil {
					return nil, fmt.Errorf("model: bad file-starts section")
				}
				ls := make([]string, ns)
				for j := range ls {
					ls[j] = string(r.bytes(int(r.u16())))
				}
				bun.FileStarts[name] = ls
			}
			if r.err != nil {
				return nil, fmt.Errorf("model: truncated file-starts section")
			}
		}
	}
	if hasAux() {
		nd := int(r.u32())
		if nd < 0 || nd > 1<<20 || r.err != nil {
			return nil, fmt.Errorf("model: bad dir-ident header")
		}
		if nd > 0 {
			bun.DirIdents = map[string][]string{}
			for i := 0; i < nd; i++ {
				dir := string(r.bytes(int(r.u16())))
				ns := int(r.u32())
				if ns < 0 || ns > 1024 || r.err != nil {
					return nil, fmt.Errorf("model: bad dir-ident section")
				}
				ls := make([]string, ns)
				for j := range ls {
					ls[j] = string(r.bytes(int(r.u16())))
				}
				bun.DirIdents[dir] = ls
			}
			if r.err != nil {
				return nil, fmt.Errorf("model: truncated dir-ident section")
			}
		}
	}
	_ = mapped
	if os.Getenv("Q4TAB_DEBUG_RSS") != "" {
		fmt.Fprintf(os.Stderr, "post-load: %s\n", rssDebug())
	}
	return bun, nil
}

// loadV12 reads the legacy v1/v2 format: magic Q4C2 already consumed.
// These models carry no KN metadata or aux sections; the scorer falls
// back to Jelinek-Mercer.
func loadV12(r *reader, data []byte, mapped bool) (*Bundle, error) {
	ver := r.u32()
	n := int(r.u32())
	vn := int(r.u32())
	if ver != 1 && ver != 2 {
		return nil, fmt.Errorf("model: unsupported version %d", ver)
	}
	if n < 1 || n > 16 {
		return nil, fmt.Errorf("model: bad order %d", n)
	}
	voffs := r.u32s(vn + 1)
	vlen := r.u64()
	vblob := r.bytes(int(vlen))
	r.align8()
	if r.err != nil || len(voffs) != vn+1 || len(vblob) != int(vlen) {
		return nil, fmt.Errorf("model: truncated vocab section")
	}
	if !validOffs32(voffs, len(vblob)) {
		return nil, fmt.Errorf("model: corrupt vocab offsets")
	}
	v := model.VocabFromBlob(vblob, voffs)
	m := &model.Model{Vocab: v, N: n}
	m.Orders = make([]model.Order, n+1)
	for k := 1; k <= n; k++ {
		nk := int(r.u64())
		nt := int(r.u64())
		if nk < 0 || nk > 1<<31 || nt < 0 || nt > 1<<33 {
			return nil, fmt.Errorf("model: absurd order %d sizes", k)
		}
		o := &m.Orders[k]
		if ver == 1 {
			o.Keys = r.u64s(nk)
			o.Off = r.i64s(nk + 1)
			o.Totals = r.i64s(nk)
			o.Toks = r.i32s(nt)
			o.Cnts = r.i32s(nt)
			o.NToks = int64(nt)
			r.align8()
			if r.err != nil {
				return nil, fmt.Errorf("model: truncated order %d", k)
			}
			if len(o.Off) != nk+1 || len(o.Totals) != nk ||
				len(o.Toks) != nt || len(o.Cnts) != nt ||
				!validOffs64(o.Off, nt) {
				return nil, fmt.Errorf("model: corrupt order %d", k)
			}
			for _, t := range o.Totals[:min(64, nk)] {
				if t < 0 {
					return nil, fmt.Errorf("model: negative total in order %d", k)
				}
			}
			for _, t := range o.Totals[max(0, nk-64):] {
				if t < 0 {
					return nil, fmt.Errorf("model: negative total in order %d", k)
				}
			}
		} else {
			slen := int(r.u64())
			if slen < 0 || slen > 1<<33 {
				return nil, fmt.Errorf("model: absurd stream len in order %d", k)
			}
			o.Keys = r.u64s(nk)
			o.Off = r.i64s(nk + 1)
			o.Stream = r.bytes(slen)
			o.NToks = int64(nt)
			r.align8()
			if r.err != nil {
				return nil, fmt.Errorf("model: truncated order %d", k)
			}
			if len(o.Off) != nk+1 || len(o.Stream) != slen ||
				!validOffs64(o.Off, slen) {
				return nil, fmt.Errorf("model: corrupt order %d", k)
			}
		}
	}
	nl := int(r.u32())
	if r.err != nil {
		return nil, fmt.Errorf("model: truncated line index header")
	}
	bun := &Bundle{M: m}
	if nl > 0 {
		loff := r.u32s(nl + 1)
		lcnts := r.i32s(nl)
		llen := r.u64()
		lblob := r.bytes(int(llen))
		if r.err != nil {
			return nil, fmt.Errorf("model: truncated line index")
		}
		if len(loff) != nl+1 || len(lcnts) != nl ||
			!validOffs32(loff, len(lblob)) {
			return nil, fmt.Errorf("model: corrupt line index")
		}
		bun.Lines = &lines.Index{Blob: lblob, Off: loff, Cnts: lcnts}
	}
	_ = mapped
	return bun, nil
}

const deltaMagic = 0x51344431 // 'Q4D1'

// SaveDelta writes the incremental-index overlay: raw contents of files
// that changed since the base model was built. Kept separate from the
// packed model so a small source change never rebuilds the CSR arrays.
// LoadDelta replays them into a dynamic cache at startup.
//
// Layout: u32 magic, u32 version=1, u32 fileCount, then per file
// u32 pathLen, path, i64 mtime, u32 dataLen, data.
func SaveDelta(path string, files []DeltaFile) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := &countingWriter{w: f}
	putU32(w, deltaMagic)
	putU32(w, 1)
	putU32(w, uint32(len(files)))
	for _, df := range files {
		putU32(w, uint32(len(df.Path)))
		w.Write([]byte(df.Path))
		putU64(w, uint64(df.ModTime))
		putU32(w, uint32(len(df.Data)))
		w.Write(df.Data)
	}
	if w.err != nil {
		f.Close()
		os.Remove(tmp)
		return w.err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// LoadDelta reads the incremental overlay. A missing file is not an
// error (no incremental index has run). A corrupt one is.
func LoadDelta(path string) ([]DeltaFile, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	r := &reader{b: data}
	if r.u32() != deltaMagic || r.u32() != 1 {
		return nil, fmt.Errorf("delta: bad header")
	}
	n := int(r.u32())
	if n < 0 || n > 1<<20 {
		return nil, fmt.Errorf("delta: absurd file count %d", n)
	}
	out := make([]DeltaFile, 0, n)
	for i := 0; i < n; i++ {
		pl := int(r.u32())
		if pl <= 0 || pl > 4096 {
			return nil, fmt.Errorf("delta: bad path length")
		}
		p := string(r.bytes(pl))
		mt := int64(r.u64())
		dl := int(r.u32())
		d := r.bytes(dl)
		if r.err != nil {
			return nil, fmt.Errorf("delta: truncated at file %d", i)
		}
		// d is a sub-slice of the read buffer, which stays alive via
		// the returned entries.
		out = append(out, DeltaFile{Path: p, ModTime: mt, Data: d})
	}
	return out, nil
}

// rssDebug reports current VmRSS for the Q4TAB_DEBUG_RSS probe.
func rssDebug() string {
	d, err := os.ReadFile("/proc/self/status")
	if err != nil {
		return "?"
	}
	for _, l := range strings.Split(string(d), "\n") {
		if strings.HasPrefix(l, "VmRSS") {
			return strings.TrimSpace(l)
		}
	}
	return "?"
}

// validOffs32 checks a packed-string offset table: starts at 0, is
// monotonic across the head and tail windows, and ends exactly at the
// blob length. Interior entries are deliberately unchecked: scanning
// them would fault hundreds of MB of mmap pages at load, and Complete
// has a panic net for the residual risk.
func validOffs32(off []uint32, blobLen int) bool {
	if len(off) == 0 || off[0] != 0 {
		return false
	}
	if !monoHeadTail32(off) {
		return false
	}
	return uint64(off[len(off)-1]) == uint64(blobLen)
}

// validOffs64 is the same check for int64 row offsets ending at nt.
func validOffs64(off []int64, nt int) bool {
	if len(off) == 0 || off[0] != 0 {
		return false
	}
	if !monoHeadTail64(off) {
		return false
	}
	return off[len(off)-1] == int64(nt)
}

func monoHeadTail32(off []uint32) bool {
	head := min(64, len(off))
	for i := 1; i < head; i++ {
		if off[i] < off[i-1] {
			return false
		}
	}
	for i := max(head, len(off)-64); i < len(off); i++ {
		if off[i] < off[i-1] {
			return false
		}
	}
	return true
}

func monoHeadTail64(off []int64) bool {
	head := min(64, len(off))
	for i := 1; i < head; i++ {
		if off[i] < off[i-1] {
			return false
		}
	}
	for i := max(head, len(off)-64); i < len(off); i++ {
		if off[i] < off[i-1] {
			return false
		}
	}
	return true
}

// mapFile mmaps path read-only, falling back to ReadFile when mmap is
// unavailable. The platform-specific mmap path lives in mmap_linux.go, mmap_unix.go, and mmap_other.go.

// --- write helpers ---

type countingWriter struct {
	w   io.Writer
	off int64
	err error
}

func (c *countingWriter) Write(p []byte) (int, error) {
	if c.err != nil {
		return 0, c.err
	}
	n, err := c.w.Write(p)
	c.off += int64(n)
	if err != nil {
		c.err = err
	}
	return n, err
}

func putU32(w *countingWriter, v uint32) {
	var b [4]byte
	binary.LittleEndian.PutUint32(b[:], v)
	w.Write(b[:])
}

func putU16(w *countingWriter, v uint16) {
	var b [2]byte
	binary.LittleEndian.PutUint16(b[:], v)
	w.Write(b[:])
}

func putU64(w *countingWriter, v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	w.Write(b[:])
}

func putF64(w *countingWriter, v float64) {
	putU64(w, math.Float64bits(v))
}

func u32toi32(s []uint32) []int32 {
	out := make([]int32, len(s))
	for i, v := range s {
		out[i] = int32(v)
	}
	return out
}

func putSlice[T uint64 | int64 | int32 | uint32](w *countingWriter, s []T) {
	// Element-wise writes are slow for huge arrays. Batch via a
	// byte view when possible. All target platforms are little-endian.
	b := unsafe.Slice((*byte)(unsafe.Pointer(unsafe.SliceData(s))), len(s)*int(unsafe.Sizeof(s[0])))
	w.Write(b)
}

func pad8(w *countingWriter) {
	for w.off%8 != 0 {
		w.Write([]byte{0})
	}
}

// --- read helpers ---

type reader struct {
	b   []byte
	off int
	err error
}

func (r *reader) take(n int) []byte {
	if r.err != nil || n < 0 || r.off+n > len(r.b) {
		r.err = fmt.Errorf("short read at %d need %d", r.off, n)
		return nil
	}
	p := r.b[r.off : r.off+n]
	r.off += n
	return p
}

func (r *reader) align8() {
	r.off = (r.off + 7) &^ 7
}

func (r *reader) u32() uint32 {
	p := r.take(4)
	if p == nil {
		return 0
	}
	return binary.LittleEndian.Uint32(p)
}

func (r *reader) u16() uint16 {
	p := r.take(2)
	if p == nil {
		return 0
	}
	return binary.LittleEndian.Uint16(p)
}

func (r *reader) u64() uint64 {
	p := r.take(8)
	if p == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(p)
}

func (r *reader) f64() float64 {
	return math.Float64frombits(r.u64())
}

func (r *reader) bytes(n int) []byte { return r.take(n) }

// Typed slice readers reference the (possibly mapped) buffer directly.
// if the buffer is not aligned for the element type they fall back to
// a per-element decode.

func (r *reader) u64s(n int) []uint64 {
	p := r.take(8 * n)
	if p == nil {
		return nil
	}
	if len(p) == 0 {
		return nil
	}
	if addr := uintptr(unsafe.Pointer(&p[0])); addr%8 == 0 {
		return unsafe.Slice((*uint64)(unsafe.Pointer(&p[0])), n)
	}
	out := make([]uint64, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint64(p[i*8:])
	}
	return out
}

func (r *reader) i64s(n int) []int64 {
	p := r.take(8 * n)
	if p == nil {
		return nil
	}
	if len(p) == 0 {
		return nil
	}
	if addr := uintptr(unsafe.Pointer(&p[0])); addr%8 == 0 {
		return unsafe.Slice((*int64)(unsafe.Pointer(&p[0])), n)
	}
	if os.Getenv("Q4TAB_DEBUG_RSS") != "" {
		fmt.Fprintf(os.Stderr, "i64s fallback at off %d n=%d\n", r.off-8*n, n)
	}
	out := make([]int64, n)
	for i := range out {
		out[i] = int64(binary.LittleEndian.Uint64(p[i*8:]))
	}
	return out
}

func (r *reader) i32s(n int) []int32 {
	p := r.take(4 * n)
	if p == nil {
		return nil
	}
	if len(p) == 0 {
		return nil
	}
	if addr := uintptr(unsafe.Pointer(&p[0])); addr%4 == 0 {
		return unsafe.Slice((*int32)(unsafe.Pointer(&p[0])), n)
	}
	out := make([]int32, n)
	for i := range out {
		out[i] = int32(binary.LittleEndian.Uint32(p[i*4:]))
	}
	return out
}

func (r *reader) u32s(n int) []uint32 {
	p := r.take(4 * n)
	if p == nil {
		return nil
	}
	if len(p) == 0 {
		return nil
	}
	if addr := uintptr(unsafe.Pointer(&p[0])); addr%4 == 0 {
		return unsafe.Slice((*uint32)(unsafe.Pointer(&p[0])), n)
	}
	out := make([]uint32, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(p[i*4:])
	}
	return out
}
