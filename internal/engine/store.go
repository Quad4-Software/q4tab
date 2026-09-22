// Binary persistence for the trained model. The gob format decoded into
// fully resident Go structures (a 500MB file became ~3GB RSS). This
// format stores flat little-endian sections that Load mmaps read-only,
// so only touched pages occupy memory.
//
// Layout (all little-endian, every array section 8-byte aligned):
//
//	u32 magic 'Q4C2', u32 version, u32 order, u32 vocabN
//	u32 vocabOffs[vocabN+1], u64 vocabBlobLen, u8 vocabBlob[...]
//
//	version 1 order section, per k in 1..N: u64 nKeys, u64 nToks,
//	  u64 keys[nKeys], i64 off[nKeys+1], i64 totals[nKeys],
//	  i32 toks[nToks], i32 cnts[nToks]
//
//	version 2 order section, per k in 1..N: u64 nKeys, u64 nToks,
//	  u64 streamLen, u64 keys[nKeys], i64 off[nKeys+1],
//	  u8 stream[streamLen]
//	  where each row in the stream is
//	  uvarint nToks, uvarint rowTotal, uvarint toks[n], uvarint cnts[n].
//	  v2 folds totals/toks/cnts into varint rows, roughly halving the
//	  order section. Keys and offsets stay flat for binary search.
//
//	u32 nLines, u32 linesOff[nLines+1], i32 linesCnts[nLines],
//	  u64 linesBlobLen, u8 linesBlob[...]
package engine

import (
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
	"unsafe"

	"q4complete/internal/lines"
	"q4complete/internal/model"
)

const (
	storeMagic   = 0x51344332 // 'Q4C2'
	storeVersion = 2          // v1 flat CSR arrays, v2 varint row streams
)

// Save writes model + line index to path atomically in the packed
// binary format.
func Save(path string, m *model.Model, li *lines.Index) error {
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
	vblob, voffs := m.Vocab.Blob()
	putU32(w, storeMagic)
	putU32(w, storeVersion)
	putU32(w, uint32(m.N))
	putU32(w, uint32(len(voffs)-1))
	putSlice(w, voffs)
	putU64(w, uint64(len(vblob)))
	w.Write(vblob)
	pad8(w)
	var vbuf [binary.MaxVarintLen64]byte
	for k := 1; k <= m.N; k++ {
		o := &m.Orders[k]
		// Encode rows into a varint stream first. Row byte offsets
		// into the stream replace the v1 CSR index arrays.
		var stream []byte
		offs := make([]int64, 0, len(o.Keys)+1)
		offs = append(offs, 0)
		var nt int64
		for i := range o.Keys {
			rtoks := o.Toks[o.Off[i]:o.Off[i+1]]
			rcnts := o.Cnts[o.Off[i]:o.Off[i+1]]
			n := binary.PutUvarint(vbuf[:], uint64(len(rtoks)))
			stream = append(stream, vbuf[:n]...)
			n = binary.PutUvarint(vbuf[:], uint64(o.Totals[i]))
			stream = append(stream, vbuf[:n]...)
			// Interleave (tok, cnt) pairs so a prefix decode is valid.
			for j, t := range rtoks {
				n = binary.PutUvarint(vbuf[:], uint64(uint32(t)))
				stream = append(stream, vbuf[:n]...)
				n = binary.PutUvarint(vbuf[:], uint64(uint32(rcnts[j])))
				stream = append(stream, vbuf[:n]...)
			}
			nt += int64(len(rtoks))
			offs = append(offs, int64(len(stream)))
		}
		putU64(w, uint64(len(o.Keys)))
		putU64(w, uint64(nt))
		putU64(w, uint64(len(stream)))
		putSlice(w, o.Keys)
		putSlice(w, offs)
		w.Write(stream)
		pad8(w)
	}
	var loff []uint32
	var lcnts []int32
	var lblob []byte
	if li != nil {
		loff, lcnts, lblob = li.Off, li.Cnts, li.Blob
	}
	putU32(w, uint32(len(lcnts)))
	putSlice(w, loff)
	putSlice(w, lcnts)
	putU64(w, uint64(len(lblob)))
	w.Write(lblob)
	if w.err != nil {
		return fail(w.err)
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// Load reads the packed model file. It mmaps the file read-only so the
// CSR arrays stay on disk until touched. Falls back to a plain read when
// mmap is unavailable (non-unix or tiny files).
func Load(path string) (*model.Model, *lines.Index, error) {
	data, mapped, err := mapFile(path)
	if err != nil {
		return nil, nil, err
	}
	if os.Getenv("Q4_DEBUG_RSS") != "" {
		fmt.Fprintf(os.Stderr, "post-map: %s\n", rssDebug())
	}
	r := &reader{b: data}
	magic := r.u32()
	ver := r.u32()
	n := int(r.u32())
	vn := int(r.u32())
	if magic != storeMagic {
		return nil, nil, fmt.Errorf("model: bad magic %x (old gob format? rebuild the index)", magic)
	}
	if ver != 1 && ver != 2 {
		return nil, nil, fmt.Errorf("model: unsupported version %d", ver)
	}
	if n < 1 || n > 16 {
		return nil, nil, fmt.Errorf("model: bad order %d", n)
	}
	voffs := r.u32s(vn + 1)
	vlen := r.u64()
	vblob := r.bytes(int(vlen))
	r.align8()
	if r.err != nil || len(voffs) != vn+1 || len(vblob) != int(vlen) {
		return nil, nil, fmt.Errorf("model: truncated vocab section")
	}
	if !validOffs32(voffs, len(vblob)) {
		return nil, nil, fmt.Errorf("model: corrupt vocab offsets")
	}
	v := model.VocabFromBlob(vblob, voffs)
	if os.Getenv("Q4_DEBUG_RSS") != "" {
		fmt.Fprintf(os.Stderr, "post-vocab: %s\n", rssDebug())
	}
	m := &model.Model{Vocab: v, N: n}
	m.Orders = make([]model.Order, n+1)
	for k := 1; k <= n; k++ {
		nk := int(r.u64())
		nt := int(r.u64())
		if nk < 0 || nk > 1<<31 || nt < 0 || nt > 1<<33 {
			return nil, nil, fmt.Errorf("model: absurd order %d sizes", k)
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
				return nil, nil, fmt.Errorf("model: truncated order %d", k)
			}
			if len(o.Off) != nk+1 || len(o.Totals) != nk ||
				len(o.Toks) != nt || len(o.Cnts) != nt ||
				!validOffs64(o.Off, nt) {
				return nil, nil, fmt.Errorf("model: corrupt order %d", k)
			}
			// Check the head and tail totals only. A full scan would
			// fault the whole array into RSS for a corruption mode
			// that degrades suggestions but cannot crash.
			for _, t := range o.Totals[:min(64, nk)] {
				if t < 0 {
					return nil, nil, fmt.Errorf("model: negative total in order %d", k)
				}
			}
			for _, t := range o.Totals[max(0, nk-64):] {
				if t < 0 {
					return nil, nil, fmt.Errorf("model: negative total in order %d", k)
				}
			}
		} else {
			slen := int(r.u64())
			if slen < 0 || slen > 1<<33 {
				return nil, nil, fmt.Errorf("model: absurd stream len in order %d", k)
			}
			o.Keys = r.u64s(nk)
			o.Off = r.i64s(nk + 1)
			o.Stream = r.bytes(slen)
			o.NToks = int64(nt)
			r.align8()
			if r.err != nil {
				return nil, nil, fmt.Errorf("model: truncated order %d", k)
			}
			if len(o.Off) != nk+1 || len(o.Stream) != slen ||
				!validOffs64(o.Off, slen) {
				return nil, nil, fmt.Errorf("model: corrupt order %d", k)
			}
		}
		if os.Getenv("Q4_DEBUG_RSS") != "" {
			fmt.Fprintf(os.Stderr, "order %d (nk=%d nt=%d): %s\n", k, nk, nt, rssDebug())
		}
	}
	nl := int(r.u32())
	if r.err != nil {
		return nil, nil, fmt.Errorf("model: truncated line index header")
	}
	var li *lines.Index
	if nl > 0 {
		loff := r.u32s(nl + 1)
		lcnts := r.i32s(nl)
		llen := r.u64()
		lblob := r.bytes(int(llen))
		if r.err != nil {
			return nil, nil, fmt.Errorf("model: truncated line index")
		}
		if len(loff) != nl+1 || len(lcnts) != nl ||
			!validOffs32(loff, len(lblob)) {
			return nil, nil, fmt.Errorf("model: corrupt line index")
		}
		li = &lines.Index{Blob: lblob, Off: loff, Cnts: lcnts}
	}
	// Touch nothing else. `data` (mapped or read) stays referenced by
	// the returned structures. mapped is kept alive by the slices.
	_ = mapped
	if os.Getenv("Q4_DEBUG_RSS") != "" {
		fmt.Fprintf(os.Stderr, "post-load: %s\n", rssDebug())
	}
	return m, li, nil
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

// rssDebug reports current VmRSS for the Q4_DEBUG_RSS probe.
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
// unavailable.
func mapFile(path string) (data []byte, mapped bool, err error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, false, err
	}
	defer f.Close()
	fi, err := f.Stat()
	if err != nil {
		return nil, false, err
	}
	if fi.Size() < 4096 {
		b, err := io.ReadAll(f)
		return b, false, err
	}
	b, err := syscall.Mmap(int(f.Fd()), 0, int(fi.Size()), syscall.PROT_READ, syscall.MAP_PRIVATE)
	if err != nil {
		f.Seek(0, 0)
		b, err2 := io.ReadAll(f)
		return b, false, err2
	}
	// The access pattern is binary-search probes scattered across the
	// file. Without MADV_RANDOM the kernel readahead faults ~128KB per
	// probe and hundreds of MB go resident for no benefit.
	syscall.Madvise(b, syscall.MADV_RANDOM)
	return b, true, nil
}

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

func putU64(w *countingWriter, v uint64) {
	var b [8]byte
	binary.LittleEndian.PutUint64(b[:], v)
	w.Write(b[:])
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

func (r *reader) u64() uint64 {
	p := r.take(8)
	if p == nil {
		return 0
	}
	return binary.LittleEndian.Uint64(p)
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
	if addr := uintptr(unsafe.Pointer(&p[0])); addr%8 == 0 {
		return unsafe.Slice((*int64)(unsafe.Pointer(&p[0])), n)
	}
	if os.Getenv("Q4_DEBUG_RSS") != "" {
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
	if addr := uintptr(unsafe.Pointer(&p[0])); addr%4 == 0 {
		return unsafe.Slice((*uint32)(unsafe.Pointer(&p[0])), n)
	}
	out := make([]uint32, n)
	for i := range out {
		out[i] = binary.LittleEndian.Uint32(p[i*4:])
	}
	return out
}
