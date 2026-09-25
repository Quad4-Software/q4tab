package engine

import (
	"bufio"
	"fmt"
	"io"
	"os"

	"q4tab/internal/corpus"
	"q4tab/internal/tokenize"
)

// auxbuild.go: the small-table corpus pass. A full index build
// spends gigabytes of spill sorting n-gram counts; the auxiliary
// tables - type members, call members, directory idents, file-start
// priors - are tiny by comparison and power the cold-file cases
// where session facts have nothing yet. BuildAux walks the corpus
// once, lexes, and emits just those tables. The result saves as a
// sidecar next to the model and merges into the loaded bundle, so a
// server gets corpus-wide member memory without a full rebuild.

// AuxData is the corpus-wide aux table set.
type AuxData struct {
	TypeMem    map[string][]string
	CallMem    map[string][]string
	DirIdents  map[string][]string
	FileStarts map[string][]string
}

// BuildAux extracts the aux tables over roots. Progress lines match
// the index build's cadence.
func BuildAux(roots []string, progress io.Writer) (*AuxData, error) {
	fb := newFactBuilder()
	dirs := newDirBuilder()
	starts := newStartBuilder()
	var files, bytes int
	ch := make(chan corpus.File, 64)
	go corpus.Collect(roots, ch)
	for f := range ch {
		files++
		if f.DupOf != "" {
			continue
		}
		bytes += len(f.Data)
		starts.Add(f.Path, f.Data)
		dirs.Add(f.Path, f.Data)
		fb.Add(tokenize.Lex(f.Data))
		if progress != nil && files%5000 == 0 {
			fmt.Fprintf(progress, "aux: %d files, %d MB\n", files, bytes>>20)
		}
	}
	tyMem, callMem := fb.Compact()
	return &AuxData{
		TypeMem:    tyMem,
		CallMem:    callMem,
		DirIdents:  dirs.Compact(96, 3),
		FileStarts: starts.Compact(8, 3),
	}, nil
}

const auxMagic = "Q4AUX1"

// SaveAux writes the aux sidecar atomically: four string tables.
func SaveAux(path string, a *AuxData) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := &countingWriter{w: bufio.NewWriterSize(f, 1<<20)}
	w.Write([]byte(auxMagic))
	putStrTable(w, a.TypeMem)
	putStrTable(w, a.CallMem)
	putStrTable(w, a.DirIdents)
	putStrTable(w, a.FileStarts)
	if err := w.w.(*bufio.Writer).Flush(); err != nil {
		f.Close()
		os.Remove(tmp)
		return err
	}
	if err := f.Close(); err != nil {
		os.Remove(tmp)
		return err
	}
	return os.Rename(tmp, path)
}

// LoadAux reads the sidecar, returning nil on any structural error.
func LoadAux(path string) *AuxData {
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, 512<<20))
	if err != nil || len(data) < len(auxMagic) || string(data[:len(auxMagic)]) != auxMagic {
		return nil
	}
	r := &reader{b: data[len(auxMagic):]}
	a := &AuxData{
		TypeMem:    readStrTable(r, "type-member"),
		CallMem:    readStrTable(r, "call-member"),
		DirIdents:  readStrTable(r, "dir-idents"),
		FileStarts: readStrTable(r, "file-starts"),
	}
	if r.err != nil {
		return nil
	}
	return a
}

// MergeAux folds an aux sidecar into the live tables. Existing
// entries keep their values; aux entries extend coverage for files
// the base model never saw.
func (e *Engine) MergeAux(a *AuxData) {
	if a == nil {
		return
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	mergeInto := func(dst map[string][]string, src map[string][]string) map[string][]string {
		if len(src) == 0 {
			return dst
		}
		if dst == nil {
			dst = map[string][]string{}
		}
		for k, vs := range src {
			if len(dst[k]) == 0 {
				dst[k] = append([]string(nil), vs...)
			}
		}
		return dst
	}
	e.tyMem = mergeInto(e.tyMem, a.TypeMem)
	e.callMem = mergeInto(e.callMem, a.CallMem)
	if len(a.DirIdents) > 0 {
		e.dirIds = mergeInto(e.dirIds, a.DirIdents)
		e.dirSuf = buildDirSuf(e.dirIds)
	}
	e.fstarts = mergeInto(e.fstarts, a.FileStarts)
}
