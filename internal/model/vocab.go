package model

import (
	"sync"
	"unsafe"
)

// Vocab interns token strings to dense uint32 ids. Token bytes live in
// a packed blob. Ids index into offs. When the blob is a shared read-only
// region (mmap'd model file), the first new intern copies it into owned
// memory so runtime interning keeps working.
//
// All methods are safe for concurrent use. Reads take RLock. Interning
// new tokens takes the write lock (thaw may swap the blob).
type Vocab struct {
	mu      sync.RWMutex
	strToID map[string]uint32 // runtime interns only (small)
	tab     *vtab             // static hash table over the blob
	blob    []byte
	offs    []uint32 // len = number of tokens + 1
	frozen  bool     // blob is shared read-only memory
}

// vtab is an open-addressed string->id table keyed by a 32-bit hash.
// Keys are verified against the blob on hash hit, so false positives
// only cost a compare; the heap cost is 8 bytes per entry versus a
// map[string]uint32 at roughly 10x that for large vocabularies.
type vtab struct {
	h  []uint32 // low 32 of token hash; 0 = empty slot
	id []uint32 // token id at that slot
}

func hashTok(s string) uint32 {
	var h uint32 = 2166136261
	for i := 0; i < len(s); i++ {
		h = (h ^ uint32(s[i])) * 16777619
	}
	if h == 0 {
		h = 1
	}
	return h
}

func newVtab(n int) *vtab {
	cap := uint64(1)
	for cap < uint64(n)*2 {
		cap <<= 1
	}
	return &vtab{h: make([]uint32, cap), id: make([]uint32, cap)}
}

func (t *vtab) put(h, id uint32) {
	mask := uint32(len(t.h) - 1)
	for i := h & mask; ; i = (i + 1) & mask {
		if t.h[i] == 0 {
			t.h[i] = h
			t.id[i] = id
			return
		}
	}
}

// find returns the stored id for hash h, -1 when absent.
func (t *vtab) find(h uint32) int {
	mask := uint32(len(t.h) - 1)
	for i := h & mask; ; i = (i + 1) & mask {
		if t.h[i] == 0 {
			return -1
		}
		if t.h[i] == h {
			return int(t.id[i])
		}
	}
}

// findAll walks every slot whose hash matches; cb stops the walk.
// Needed because equal hash32s of different strings chain together.
func (t *vtab) findAll(h uint32, cb func(id uint32) bool) {
	mask := uint32(len(t.h) - 1)
	for i := h & mask; t.h[i] != 0; i = (i + 1) & mask {
		if t.h[i] == h && cb(t.id[i]) {
			return
		}
	}
}

func NewVocab() *Vocab {
	v := &Vocab{strToID: make(map[string]uint32)}
	v.blob = append(v.blob, "<pad>"...)
	v.offs = []uint32{0, uint32(len(v.blob))}
	v.strToID["<pad>"] = 0
	return v
}

// VocabFromBlob builds a vocab over a shared blob. The hash table
// holds no token strings - keys live in the blob and a hit verifies
// the bytes, so static vocab memory is ~8 bytes per token.
// The blob must outlive the vocab and never be mutated while frozen.
func VocabFromBlob(blob []byte, offs []uint32) *Vocab {
	v := &Vocab{blob: blob, offs: offs, frozen: true,
		strToID: make(map[string]uint32)}
	v.tab = newVtab(len(offs))
	for i := 0; i+1 < len(offs); i++ {
		if offs[i] == offs[i+1] {
			continue
		}
		s := unsafe.String(&blob[offs[i]], int(offs[i+1]-offs[i]))
		v.tab.put(hashTok(s), uint32(i))
	}
	return v
}

// thaw copies the blob into owned memory so interning can append.
// Callers must hold the write lock.
func (v *Vocab) thaw() {
	if !v.frozen {
		return
	}
	nb := make([]byte, len(v.blob))
	copy(nb, v.blob)
	v.blob = nb
	v.frozen = false
}

// strAt reads token id from the blob without locking; callers hold a
// read or write lock.
func (v *Vocab) strAt(id uint32) string {
	if int(id)+1 >= len(v.offs) {
		return ""
	}
	return unsafe.String(&v.blob[v.offs[id]], int(v.offs[id+1]-v.offs[id]))
}

// tabLookup probes the static table and verifies the blob bytes.
// Runs under a held read lock.
func (v *Vocab) tabLookup(s string) (uint32, bool) {
	if v.tab == nil {
		return 0, false
	}
	h := hashTok(s)
	found := uint32(0)
	ok := false
	v.tab.findAll(h, func(id uint32) bool {
		if v.strAt(id) == s {
			found, ok = id, true
			return true
		}
		return false
	})
	return found, ok
}

func (v *Vocab) ID(s string) uint32 {
	v.mu.RLock()
	if id, ok := v.tabLookup(s); ok {
		v.mu.RUnlock()
		return id
	}
	if id, ok := v.strToID[s]; ok {
		v.mu.RUnlock()
		return id
	}
	v.mu.RUnlock()

	v.mu.Lock()
	defer v.mu.Unlock()
	if id, ok := v.strToID[s]; ok { // another interner beat us
		return id
	}
	if id, ok := v.tabLookup(s); ok {
		return id
	}
	v.thaw()
	id := uint32(len(v.offs) - 1)
	v.strToID[s] = id
	v.blob = append(v.blob, s...)
	v.offs = append(v.offs, uint32(len(v.blob)))
	return id
}

func (v *Vocab) Lookup(s string) (uint32, bool) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	if id, ok := v.tabLookup(s); ok {
		return id, true
	}
	id, ok := v.strToID[s]
	return id, ok
}

// Str returns the token for id without copying.
func (v *Vocab) Str(id uint32) string {
	v.mu.RLock()
	defer v.mu.RUnlock()
	if int(id+1) >= len(v.offs) {
		return ""
	}
	lo, hi := v.offs[id], v.offs[id+1]
	if lo == hi {
		return ""
	}
	return unsafe.String(&v.blob[lo], int(hi-lo))
}

func (v *Vocab) Len() int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return len(v.offs) - 1
}

// Blob returns the packed token blob and offsets (for persistence).
func (v *Vocab) Blob() ([]byte, []uint32) {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.blob, v.offs
}

func (v *Vocab) Intern(toks []string) []uint32 {
	out := make([]uint32, len(toks))
	for i, t := range toks {
		out[i] = v.ID(t)
	}
	return out
}
