// Package esort implements a bounded-memory external merge sort for
// fixed-size key records. A Stream buffers sightings in memory; each
// time the buffer fills it is sorted, deduplicated with summed counts,
// and written to a sorted run file on disk. Finish merges the runs into
// one sorted, aggregated stream.
//
// Peak memory is one flat buffer plus one open file and read buffer per
// run, so the sorter scales to datasets far larger than RAM. Streams
// hold no shared state: distinct Streams are safe for concurrent use,
// though a single Stream is not goroutine-safe.
package esort

import (
	"bufio"
	"container/heap"
	"encoding/binary"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

const (
	// recSize is the on-disk record size: a, b, cnt as u64 each.
	recSize = 24
	// defaultBufRecs bounds the in-memory buffer when the caller does
	// not specify a size: 1M sightings, 16 MiB worst case.
	defaultBufRecs = 1 << 20
	// ioBufSize is the read and write buffer per run file.
	ioBufSize = 1 << 16
	// runExt names run files "<tag>-<seq>.run" inside dir.
	runExt = ".run"
)

// Rec is one sighting key. Ordering is lexicographic on (A, B).
type Rec struct{ A, B uint64 }

// Stream accumulates records in an in-memory buffer; when the buffer
// reaches bufRecs entries it sorts, deduplicates (summing counts), and
// writes a run file into dir.
type Stream struct {
	dir  string
	tag  string
	cap  int
	buf  []Rec
	seq  int
	runs []string
	err  error
	done bool
}

// NewStream creates a stream. dir must exist or be created by the
// caller. tag names the stream (used in run file names). bufRecs <= 0
// picks a sane default.
func NewStream(dir, tag string, bufRecs int) (*Stream, error) {
	st, err := os.Stat(dir)
	if err != nil {
		return nil, err
	}
	if !st.IsDir() {
		return nil, fmt.Errorf("esort: %s is not a directory", dir)
	}
	if bufRecs <= 0 {
		bufRecs = defaultBufRecs
	}
	return &Stream{dir: dir, tag: tag, cap: bufRecs, buf: make([]Rec, 0, bufRecs)}, nil
}

// Add appends one sighting of key (a, b). Calling Add after Finish is
// a programming error and panics.
func (s *Stream) Add(a, b uint64) {
	if s.done {
		panic("esort: Add after Finish")
	}
	if s.err != nil {
		return // a failed flush already doomed this stream
	}
	s.buf = append(s.buf, Rec{a, b})
	if len(s.buf) >= s.cap {
		s.err = s.flush()
	}
}

// Finish flushes the buffer and returns an iterator over all runs.
func (s *Stream) Finish() (*Iter, error) {
	if s.done {
		return nil, fmt.Errorf("esort: Finish called twice")
	}
	s.done = true
	if s.err == nil {
		s.err = s.flush()
	}
	s.buf = nil
	it := &Iter{paths: s.runs, nruns: len(s.runs)}
	s.runs = nil
	if s.err != nil {
		it.removeFiles()
		return nil, s.err
	}
	for i, p := range it.paths {
		f, err := os.Open(p)
		if err != nil {
			it.Close()
			return nil, err
		}
		r := &runReader{f: f, r: bufio.NewReaderSize(f, ioBufSize)}
		it.readers = append(it.readers, r)
		rec, cnt, err := r.next()
		if err == nil {
			heap.Push(&it.h, heapEnt{rec.A, rec.B, cnt, i})
		} else if err != io.EOF {
			it.err = err
			it.Close()
			return nil, err
		}
	}
	return it, nil
}

// flush sorts the buffer, aggregates identical keys, and writes one
// sorted run file. The buffer allocation is reused for the next run.
func (s *Stream) flush() error {
	if len(s.buf) == 0 {
		return nil
	}
	sort.Slice(s.buf, func(i, j int) bool {
		if s.buf[i].A != s.buf[j].A {
			return s.buf[i].A < s.buf[j].A
		}
		return s.buf[i].B < s.buf[j].B
	})
	name := fmt.Sprintf("%s-%06d%s", s.tag, s.seq, runExt)
	s.seq++
	path := filepath.Join(s.dir, name)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	w := bufio.NewWriterSize(f, ioBufSize)
	fail := func(err error) error {
		f.Close()
		os.Remove(path) // drop the partial run
		return err
	}
	var rec [recSize]byte
	buf := s.buf
	for i := 0; i < len(buf); {
		j := i + 1
		for j < len(buf) && buf[j] == buf[i] {
			j++
		}
		binary.LittleEndian.PutUint64(rec[0:], buf[i].A)
		binary.LittleEndian.PutUint64(rec[8:], buf[i].B)
		binary.LittleEndian.PutUint64(rec[16:], uint64(j-i))
		if _, err := w.Write(rec[:]); err != nil {
			return fail(err)
		}
		i = j
	}
	s.buf = buf[:0]
	if err := w.Flush(); err != nil {
		return fail(err)
	}
	if err := f.Close(); err != nil {
		return err
	}
	s.runs = append(s.runs, path)
	return nil
}

// Cleanup removes leftover run files for tag in dir, for example after
// a crash orphaned them.
func Cleanup(dir, tag string) error {
	ents, err := os.ReadDir(dir)
	if err != nil {
		return err
	}
	prefix := tag + "-"
	var rerr error
	for _, e := range ents {
		n := e.Name()
		if strings.HasPrefix(n, prefix) && strings.HasSuffix(n, runExt) {
			if err := os.Remove(filepath.Join(dir, n)); err != nil && rerr == nil {
				rerr = err
			}
		}
	}
	return rerr
}
