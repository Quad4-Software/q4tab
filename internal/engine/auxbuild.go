package engine

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"time"

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
	Aliases    map[string]string   // type A = B across the corpus
	RetT       map[string]string   // func name -> declared result type
	SigT       map[string][]string // func name -> ordered param types
	Files      int                 // corpus files the build covered
	BuiltAt    int64               // unix seconds; freshness marker
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
		Aliases:    fb.alias,
		RetT:       fb.retT,
		SigT:       fb.sigT,
		Files:      files,
		BuiltAt:    time.Now().Unix(),
	}, nil
}

// auxMagic versions the sidecar format; the trailing byte is the
// layout version so future tables can append without breaking old
// readers (they stop at the version they know).
const auxMagic = "Q4AUX1"

// SaveAux writes the aux sidecar atomically: header (files, build
// time), then the string tables.
func SaveAux(path string, a *AuxData) error {
	tmp := path + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return err
	}
	w := &countingWriter{w: bufio.NewWriterSize(f, 1<<20)}
	w.Write([]byte(auxMagic))
	putU32(w, uint32(a.Files))
	putU64(w, uint64(a.BuiltAt))
	putStrTable(w, a.TypeMem)
	putStrTable(w, a.CallMem)
	putStrTable(w, a.DirIdents)
	putStrTable(w, a.FileStarts)
	// Aliases and retT ride the strtable encoding as singleton lists;
	// sigT is already list-valued.
	al := make(map[string][]string, len(a.Aliases))
	for k, v := range a.Aliases {
		al[k] = []string{v}
	}
	putStrTable(w, al)
	rt := make(map[string][]string, len(a.RetT))
	for k, v := range a.RetT {
		rt[k] = []string{v}
	}
	putStrTable(w, rt)
	putStrTable(w, a.SigT)
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
		Files:      int(r.u32()),
		BuiltAt:    int64(r.u64()),
		TypeMem:    readStrTable(r, "type-member"),
		CallMem:    readStrTable(r, "call-member"),
		DirIdents:  readStrTable(r, "dir-idents"),
		FileStarts: readStrTable(r, "file-starts"),
		Aliases:    map[string]string{},
	}
	for k, vs := range readStrTable(r, "alias") {
		if len(vs) > 0 {
			a.Aliases[k] = vs[0]
		}
	}
	a.RetT = map[string]string{}
	// Trailing tables are optional in both directions: absent from
	// old sidecars, and salvageable when present-but-corrupt - the
	// core member tables stand on their own.
	if r.off+4 <= len(r.b) {
		mark := r.off
		for k, vs := range readStrTable(r, "rett") {
			if len(vs) > 0 {
				a.RetT[k] = vs[0]
			}
		}
		if r.err != nil {
			r.err = nil
			r.off = mark
		}
	}
	if r.err == nil && r.off+4 <= len(r.b) {
		a.SigT = readStrTable(r, "sigt")
		if r.err != nil {
			r.err = nil
			a.SigT = nil
		}
	}
	return a
}

// auxSnap is the immutable aux layer: the base model's tables stay
// untouched so a fresh sidecar can hot-swap without a server restart.
type auxSnap struct {
	tyMem   map[string][]string
	callM   map[string][]string
	dirIds  map[string][]string
	fstarts map[string][]string
	alias   map[string]string
	retT    map[string]string
	sigT    map[string][]string
	files   int
	builtAt int64
}

// SetAux installs or replaces the aux overlay. Nil clears it.
func (e *Engine) SetAux(a *AuxData) {
	if a == nil {
		e.aux.Store(nil)
		return
	}
	e.aux.Store(&auxSnap{
		tyMem:   a.TypeMem,
		callM:   a.CallMem,
		dirIds:  a.DirIdents,
		fstarts: a.FileStarts,
		alias:   a.Aliases,
		retT:    a.RetT,
		sigT:    a.SigT,
		files:   a.Files,
		builtAt: a.BuiltAt,
	})
}

// AuxInfo reports the installed overlay for status endpoints.
func (e *Engine) AuxInfo() (types, calls, dirs, files int, builtAt int64) {
	if a := e.aux.Load(); a != nil {
		return len(a.tyMem), len(a.callM), len(a.dirIds), a.files, a.builtAt
	}
	return 0, 0, 0, 0, 0
}

// memberTabs returns the static member tables in lookup order:
// the aux overlay first (it is the freshest), then the base model.
func (e *Engine) memberTabs() (ty, call []map[string][]string, alias map[string]string) {
	if a := e.aux.Load(); a != nil {
		if len(a.tyMem) > 0 {
			ty = append(ty, a.tyMem)
		}
		if len(a.callM) > 0 {
			call = append(call, a.callM)
		}
		alias = a.alias
	}
	if len(e.tyMem) > 0 {
		ty = append(ty, e.tyMem)
	}
	if len(e.callMem) > 0 {
		call = append(call, e.callMem)
	}
	return ty, call, alias
}

// dirIdents returns dir ids from the aux overlay, falling back to
// the base model's table.
func (e *Engine) dirIdents(dir string) []string {
	if a := e.aux.Load(); a != nil {
		if ids := a.dirIds[dir]; len(ids) > 0 {
			return ids
		}
	}
	return e.dirIds[dir]
}

// fileStarts returns first-line priors for lang, aux overlay first.
func (e *Engine) fileStarts(lang string) []string {
	if a := e.aux.Load(); a != nil {
		if fs := a.fstarts[lang]; len(fs) > 0 {
			return fs
		}
	}
	return e.fstarts[lang]
}
