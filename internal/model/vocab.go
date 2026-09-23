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
	strToID map[string]uint32
	blob    []byte
	offs    []uint32 // len = number of tokens + 1
	frozen  bool     // blob is shared read-only memory
}

func NewVocab() *Vocab {
	v := &Vocab{strToID: make(map[string]uint32)}
	v.blob = append(v.blob, "<pad>"...)
	v.offs = []uint32{0, uint32(len(v.blob))}
	v.strToID["<pad>"] = 0
	return v
}

// VocabFromBlob builds a vocab over a shared blob. strToID keys are
// zero-copy strings into the blob. The blob must outlive the vocab and
// never be mutated while frozen.
func VocabFromBlob(blob []byte, offs []uint32) *Vocab {
	v := &Vocab{blob: blob, offs: offs, frozen: true}
	v.strToID = make(map[string]uint32, len(offs)-1)
	for i := 0; i+1 < len(offs); i++ {
		if offs[i] == offs[i+1] {
			continue
		}
		s := unsafe.String(&blob[offs[i]], int(offs[i+1]-offs[i]))
		v.strToID[s] = uint32(i)
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

func (v *Vocab) ID(s string) uint32 {
	v.mu.RLock()
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
	v.thaw()
	id := uint32(len(v.offs) - 1)
	v.strToID[s] = id
	v.blob = append(v.blob, s...)
	v.offs = append(v.offs, uint32(len(v.blob)))
	return id
}

func (v *Vocab) Lookup(s string) (uint32, bool) {
	v.mu.RLock()
	id, ok := v.strToID[s]
	v.mu.RUnlock()
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
