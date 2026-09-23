package esort

import (
	"bufio"
	"container/heap"
	"encoding/binary"
	"io"
	"os"
)

// runReader decodes one run file sequentially.
type runReader struct {
	f   *os.File
	r   *bufio.Reader
	buf [recSize]byte
}

// next returns the next record in the run, or io.EOF at the end.
// ErrUnexpectedEOF marks a truncated file.
func (r *runReader) next() (Rec, uint64, error) {
	if _, err := io.ReadFull(r.r, r.buf[:]); err != nil {
		return Rec{}, 0, err
	}
	return Rec{
		binary.LittleEndian.Uint64(r.buf[0:]),
		binary.LittleEndian.Uint64(r.buf[8:]),
	}, binary.LittleEndian.Uint64(r.buf[16:]), nil
}

// heapEnt is one pending record in the merge heap: the current head of
// run number run.
type heapEnt struct {
	a, b, cnt uint64
	run       int
}

// recHeap is a min-heap of heapEnt keyed on (a, b).
type recHeap []heapEnt

func (h recHeap) Len() int { return len(h) }

func (h recHeap) Less(i, j int) bool {
	if h[i].a != h[j].a {
		return h[i].a < h[j].a
	}
	return h[i].b < h[j].b
}

func (h recHeap) Swap(i, j int) { h[i], h[j] = h[j], h[i] }

func (h *recHeap) Push(x any) { *h = append(*h, x.(heapEnt)) }

func (h *recHeap) Pop() any {
	old := *h
	n := len(old)
	e := old[n-1]
	old[n-1] = heapEnt{}
	*h = old[:n-1]
	return e
}

// Iter yields aggregated records sorted by (A, B): each call to Next
// returns the next distinct key and its total count across all runs.
type Iter struct {
	readers []*runReader
	paths   []string
	nruns   int
	h       recHeap
	err     error
	closed  bool
}

// Next returns the next distinct key, its summed count, and true, or
// ok=false when the stream is exhausted.
func (it *Iter) Next() (a, b, cnt uint64, ok bool) {
	if it.h.Len() == 0 {
		return 0, 0, 0, false
	}
	top := heap.Pop(&it.h).(heapEnt)
	a, b, cnt = top.a, top.b, top.cnt
	it.advance(top.run)
	// Drain every other run head with the same key, summing counts.
	for it.h.Len() > 0 {
		e := it.h[0]
		if e.a != a || e.b != b {
			break
		}
		heap.Pop(&it.h)
		cnt += e.cnt
		it.advance(e.run)
	}
	return a, b, cnt, true
}

// advance pushes the next record of run i onto the heap. A read error
// other than EOF ends that run and is reported by Close.
func (it *Iter) advance(i int) {
	rec, cnt, err := it.readers[i].next()
	if err == nil {
		heap.Push(&it.h, heapEnt{rec.A, rec.B, cnt, i})
		return
	}
	if err != io.EOF && it.err == nil {
		it.err = err
	}
}

// Len returns the number of runs feeding the merge.
func (it *Iter) Len() int { return it.nruns }

// Close closes and removes the stream's run files. It returns the
// first error seen while reading or cleaning up, if any.
func (it *Iter) Close() error {
	if it.closed {
		return it.err
	}
	it.closed = true
	err := it.err
	for _, r := range it.readers {
		if cerr := r.f.Close(); cerr != nil && err == nil {
			err = cerr
		}
	}
	if rerr := it.removeFiles(); rerr != nil && err == nil {
		err = rerr
	}
	it.readers = nil
	it.paths = nil
	it.h = nil
	return err
}

func (it *Iter) removeFiles() error {
	var err error
	for _, p := range it.paths {
		if rerr := os.Remove(p); rerr != nil && err == nil {
			err = rerr
		}
	}
	return err
}
