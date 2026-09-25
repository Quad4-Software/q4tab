// q4tab: fully local statistical code completion.
//
// Subcommands:
//
//	serve     run the LSP server on stdio, TCP (-listen), or HTTP (-http)
//	mcp       run the MCP server on stdio (for agent clients)
//	index     train the model over corpus roots
//	collect   clone a GitHub org's repos for indexing
//	complete  print completions for a file position (testing)
//	stats     print model stats
package main

import (
	"crypto/sha256"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"math/rand"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"runtime/debug"
	"sort"
	"strconv"
	"strings"
	"time"

	"q4tab/internal/corpus"
	"q4tab/internal/engine"
	"q4tab/internal/lines"
	"q4tab/internal/lsp"
	"q4tab/internal/symbols"
)

func defaultModelPath() string {
	if p := os.Getenv("Q4TAB_MODEL"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "q4tab", "model.bin")
}

// isLoopback reports whether addr binds only to localhost.
func isLoopback(addr string) bool {
	host, _, err := net.SplitHostPort(addr)
	if err != nil {
		return false
	}
	if host == "" || host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}

func defaultCorpusDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "q4tab", "corpus")
}

func defaultJournalPath() string {
	if p := os.Getenv("Q4TAB_JOURNAL"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "q4tab", "learned.jsonl")
}

func defaultConfigPath() string {
	if p := os.Getenv("Q4TAB_CONFIG"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "q4tab", "config.json")
}

type config struct {
	Roots    []string `json:"roots"`
	Order    int      `json:"order"`
	MaxLines int      `json:"maxLines"`
	MemMode  string   `json:"mem"` // auto | low | max
}

func loadConfig() config {
	cfg := config{Order: 6, MaxLines: 4}
	data, err := os.ReadFile(defaultConfigPath())
	if err == nil {
		json.Unmarshal(data, &cfg)
	}
	if cfg.Order == 0 {
		cfg.Order = 6
	}
	if cfg.MaxLines == 0 {
		cfg.MaxLines = 4
	}
	return cfg
}

// memMode resolves the memory profile: config file, then Q4TAB_MEM,
// then a flag override. auto keeps bounded caches and lets the kernel
// manage mapped pages; low bounds caches tighter, tightens GC, and
// sweeps mapped pages back to disk on a timer; max keeps everything
// resident and unbounded.
func memMode(cfg config, flag string) string {
	m := cfg.MemMode
	if v := os.Getenv("Q4TAB_MEM"); v != "" {
		m = v
	}
	if flag != "" {
		m = flag
	}
	switch m {
	case "", "auto":
		return "auto"
	case "low", "max":
		return m
	}
	return "auto"
}

func loadEngine(cfg config) *engine.Engine {
	ecfg := engine.DefaultConfig()
	if cfg.MaxLines > 0 {
		ecfg.MaxLines = cfg.MaxLines
	}
	switch memMode(cfg, "") {
	case "low":
		// 256k context rows is roughly 30-60MB per cache once warm,
		// versus unbounded growth to ~350MB.
		ecfg.CacheRows = 1 << 18
		debug.SetGCPercent(60)
	case "max":
		ecfg.CacheRows = 0
	}
	e := engine.New(ecfg)
	bun, err := engine.Load(defaultModelPath())
	if err != nil {
		fmt.Fprintf(os.Stderr, "q4tab: no model at %s (%v); running with file-local completion only\n", defaultModelPath(), err)
		return e
	}
	e.SetBundle(bun)
	// The aux sidecar overlays member/directory tables for corpus
	// files the base model predates; it is swappable at runtime.
	e.SetAux(engine.LoadAux(defaultModelPath() + ".aux"))
	if delta, err := engine.LoadDelta(defaultModelPath() + ".delta"); err == nil {
		e.InstallDelta(delta)
	}
	if syms, err := symbols.Load(defaultModelPath() + ".symbols"); err == nil {
		e.SetSymbols(syms)
	}
	if data, err := os.ReadFile(defaultModelPath() + ".weights"); err == nil {
		var w engine.Weights
		if json.Unmarshal(data, &w) == nil {
			e.SetWeights(w)
		}
	}
	e.SetJournal(defaultJournalPath())
	return e
}

// runTune searches scoring weights against a holdout corpus: for each
// knob, evaluate a few scales while holding the others at the current
// best (coordinate descent), keep the best hit@1, write the weights
// next to the model so loadEngine picks them up.
func runTune(cfg config, roots []string, nFiles, perFile int, seed int64, log func(string, ...any)) {
	files := collectEvalFiles(roots, nFiles, seed)
	if len(files) == 0 {
		log("tune: no files found")
		os.Exit(1)
	}
	e := loadEngine(cfg)
	best := engine.DefaultWeights()
	// Phase 1: journal-trained reranker. The accept/reject history
	// yields a per-source multiplier plus a rank calibrator applied at
	// merge time. With no journal, defaults stand.
	if evs, err := engine.LoadJournalEvents(defaultJournalPath(), 100000); err == nil {
		if mult := engine.TrainSourceWeights(evs, 256); mult != nil {
			best.ApplySourceWeights(mult)
			log("journal: %d events -> source multipliers %v", len(evs), mult)
		}
		if cal := engine.TrainCalibrator(evs, 256); cal != nil {
			best.Cal = cal
			log("journal: calibrator trained (bias=%.2f srcs=%d)", cal.Bias, len(cal.SrcW))
		} else {
			log("journal: %d events, too few to train (need 256)", len(evs))
		}
		// Feature-vector events (written since the reranker landed)
		// batch-train the online ranker so it starts warm.
		if rk := engine.TrainRanker(evs, 64); rk != nil {
			rk.Save(defaultJournalPath() + ".rank")
			log("journal: reranker batch-trained on feature events")
		}
	}
	// Phase 2: deleted interpolation, cross-validated on this holdout.
	// Scales are trained on the holdout token stream, then kept only if
	// they beat plain KN on the same set: backoff rescaling is a
	// second-order effect that can lose on a different distribution.
	def := engine.DefaultWeights()
	e.SetWeights(def)
	rep, _ := evalOnce(e, files, perFile, seed, false)
	defScore := rep.Hit1
	var texts []string
	for _, f := range files {
		texts = append(texts, string(f.Data))
	}
	if lamK := e.TrainLamK(texts); lamK != nil {
		w2 := def
		w2.LamK = lamK
		e.SetWeights(w2)
		rep2, _ := evalOnce(e, files, perFile, seed, false)
		if rep2.Hit1 > defScore {
			best.LamK = lamK
			log("lamK kept: %.1f%% -> %.1f%% (%v)", defScore, rep2.Hit1, lamK[1:])
		} else {
			e.SetWeights(def)
			log("lamK rejected: %.1f%% -> %.1f%%", defScore, rep2.Hit1)
		}
	}
	// Phase 3: coordinate descent over the holdout. The holdout
	// decides; the journal and interpolation only supply starting
	// points. Baseline is measured with the DEFAULT weights so a tune
	// run that finds nothing cannot regress the shipped config.
	e.SetWeights(best)
	rep, _ = evalOnce(e, files, perFile, seed, false)
	bestScore := rep.Hit1
	log("baseline hit@1=%.1f%% tuned-start=%.1f%% (files=%d tries=%d)", defScore, bestScore, rep.Files, rep.Tries)

	knobs := []struct {
		name string
		vals []float64
		get  func(*engine.Weights) float64
		set  func(*engine.Weights, float64)
	}{
		{"file", []float64{1e5, 1e6, 5e6, 1e7},
			func(w *engine.Weights) float64 { return w.File },
			func(w *engine.Weights, v float64) { w.File = v }},
		{"dyn", []float64{1e4, 1e5, 1e6},
			func(w *engine.Weights) float64 { return w.Dyn },
			func(w *engine.Weights, v float64) { w.Dyn = v }},
		{"learn", []float64{1e5, 2e5, 5e5},
			func(w *engine.Weights) float64 { return w.Learn },
			func(w *engine.Weights, v float64) { w.Learn = v }},
		{"fim", []float64{1e6, 4e6, 8e6},
			func(w *engine.Weights) float64 { return w.FIM },
			func(w *engine.Weights, v float64) { w.FIM = v }},
		{"fimIdx", []float64{1e6, 3e6, 6e6},
			func(w *engine.Weights) float64 { return w.FIMIdx },
			func(w *engine.Weights, v float64) { w.FIMIdx = v }},
		{"model", []float64{0.5, 1, 2, 4},
			func(w *engine.Weights) float64 { return w.Model },
			func(w *engine.Weights, v float64) { w.Model = v }},
		{"lineBi", []float64{1e5, 4e5, 1e6},
			func(w *engine.Weights) float64 { return w.LineBi },
			func(w *engine.Weights, v float64) { w.LineBi = v }},
		{"struct", []float64{0.5, 2, 5},
			func(w *engine.Weights) float64 { return w.Struct },
			func(w *engine.Weights, v float64) { w.Struct = v }},
		{"lang", []float64{0.5, 3, 8},
			func(w *engine.Weights) float64 { return w.Lang },
			func(w *engine.Weights, v float64) { w.Lang = v }},
		{"sub", []float64{1e5, 3e5, 1e6},
			func(w *engine.Weights) float64 { return w.Sub },
			func(w *engine.Weights, v float64) { w.Sub = v }},
	}
	for _, k := range knobs {
		cur := k.get(&best)
		for _, v := range k.vals {
			if v == cur {
				continue
			}
			cand := best
			k.set(&cand, v)
			e.SetWeights(cand)
			rep, _ := evalOnce(e, files, perFile, seed, false)
			log("  %s=%.3g -> hit@1=%.1f%%", k.name, v, rep.Hit1)
			if rep.Hit1 > bestScore {
				bestScore = rep.Hit1
				k.set(&best, v)
				cur = v
			}
		}
	}
	e.SetWeights(best)
	if bestScore <= defScore {
		log("tune: no improvement over defaults (%.1f%% <= %.1f%%); keeping existing weights", bestScore, defScore)
		return
	}
	out := defaultModelPath() + ".weights"
	data, _ := json.MarshalIndent(best, "", "  ")
	if err := os.WriteFile(out, data, 0o644); err != nil {
		log("tune: weights write: %v", err)
		return
	}
	log("best hit@1=%.1f%% -> wrote %s: %s", bestScore, out, data)
}

// version is stamped at release time via -ldflags -X main.version.
var version = "dev"

func main() {
	log := func(format string, a ...any) { fmt.Fprintf(os.Stderr, format+"\n", a...) }
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: q4tab <serve|index|collect|complete|stats|commitmsg|version> [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "version", "-version", "--version":
		fmt.Println(version)
		return
	case "serve":
		fs := flag.NewFlagSet("serve", flag.ExitOnError)
		listen := fs.String("listen", "", "serve LSP over TCP on addr (e.g. 127.0.0.1:7917)")
		httpAddr := fs.String("http", "", "serve HTTP on addr: POST /rpc, POST /mcp, GET /status, GET /healthz")
		token := fs.String("token", os.Getenv("Q4TAB_TOKEN"), "bearer token required on network endpoints (or Q4TAB_TOKEN)")
		rate := fs.Float64("rate", 0, "sustained requests/sec per client IP (0 = unlimited)")
		burst := fs.Int("burst", 0, "rate-limit burst (default 4x rate)")
		maxConc := fs.Int("maxconc", 256, "max concurrent HTTP requests")
		mem := fs.String("mem", "", "memory profile: auto | low | max (or Q4TAB_MEM / config mem)")
		var roots multiFlag
		fs.Var(&roots, "root", "corpus root allowed for MCP path reads (repeatable; unset = path reads disabled over HTTP)")
		fs.Parse(os.Args[2:])
		cfg := loadConfig()
		mode := memMode(cfg, *mem)
		e := loadEngine(cfg)
		delta := defaultModelPath() + ".delta"
		if mode == "low" {
			// Return heap and mapped pages to the OS on a timer: Go
			// holds freed heap lazily otherwise, and mapped model
			// pages stay resident until real pressure. NVMe re-fault
			// is cheap, so this is a good trade on small boxes.
			go func() {
				for range time.Tick(3 * time.Minute) {
					debug.FreeOSMemory()
					e.SweepPages()
				}
			}()
			log("mem profile: low (bounded caches, tighter GC, page sweeps)")
		} else if mode == "max" {
			log("mem profile: max (unbounded caches, no sweeps)")
		}
		if *listen == "" && *httpAddr == "" {
			conn := lsp.NewConn(os.Stdin, os.Stdout)
			srv := lsp.NewServer(e, conn)
			srv.SetDeltaPath(delta)
			if err := srv.Run(); err != nil {
				log("serve: %v", err)
				os.Exit(1)
			}
			return
		}
		opts := lsp.ServerOpts{
			Token: *token, Rate: *rate, Burst: *burst,
			MaxConc: *maxConc, Roots: roots,
		}
		go watchDelta(delta, e, log)
		go watchAux(defaultModelPath()+".aux", e, log)
		// Network modes: the model contains source code, so warn when
		// the bind address is not loopback or auth is off.
		errc := make(chan error, 2)
		if *listen != "" {
			if !isLoopback(*listen) {
				log("warning: serving source-derived completions on %s (non-loopback)", *listen)
			}
			if !isLoopback(*listen) && *token == "" {
				log("warning: no -token set; anyone who can reach %s can read the corpus", *listen)
			}
			go func() { errc <- lsp.ListenAndServe(*listen, e, delta, opts) }()
			log("lsp over tcp on %s", *listen)
		}
		if *httpAddr != "" {
			if !isLoopback(*httpAddr) {
				log("warning: serving source-derived completions on %s (non-loopback)", *httpAddr)
			}
			if !isLoopback(*httpAddr) && *token == "" {
				log("warning: no -token set; anyone who can reach %s can read the corpus", *httpAddr)
			}
			go func() { errc <- lsp.HTTPServe(*httpAddr, e, delta, opts) }()
			log("http on %s (/rpc /mcp /status /healthz)", *httpAddr)
		}
		if err := <-errc; err != nil {
			log("serve: %v", err)
			os.Exit(1)
		}

	case "watch":
		// Poll the corpus roots and fold changed files into the delta
		// overlay. Running servers pick it up via the delta watcher.
		fs := flag.NewFlagSet("watch", flag.ExitOnError)
		interval := fs.Duration("interval", 30*time.Second, "rescan interval")
		var roots multiFlag
		fs.Var(&roots, "root", "corpus root directory (repeatable)")
		fs.Parse(os.Args[2:])
		cfg := loadConfig()
		if len(roots) == 0 {
			roots = cfg.Roots
		}
		if len(roots) == 0 {
			fmt.Fprintln(os.Stderr, "watch: no roots (pass -root or set roots in config)")
			os.Exit(2)
		}
		modelPath := defaultModelPath()
		log("watch: %d roots every %s", len(roots), *interval)
		for {
			func() {
				defer func() {
					if r := recover(); r != nil {
						log("watch: incremental failed: %v", r)
					}
				}()
				runIncremental(modelPath, roots, log)
			}()
			time.Sleep(*interval)
		}

	case "mcp":
		// MCP stdio server for agent clients.
		// NDJSON framing: one JSON-RPC message per line.
		fs := flag.NewFlagSet("mcp", flag.ExitOnError)
		fs.Parse(os.Args[2:])
		cfg := loadConfig()
		e := loadEngine(cfg)
		m := lsp.NewMCPServer(e)
		if err := m.ServeMCP(os.Stdin, os.Stdout); err != nil && err != io.EOF {
			log("mcp: %v", err)
			os.Exit(1)
		}

	case "index":
		fs := flag.NewFlagSet("index", flag.ExitOnError)
		out := fs.String("o", defaultModelPath(), "output model path")
		order := fs.Int("order", 0, "n-gram order (default from config or 6)")
		budget := fs.Int("budget", 48, "memory guard: gate low-order counting after this many million tokens (0 = off)")
		memMB := fs.Int("mem", 0, "memory guard: RSS ceiling in MB (0 = 70% of physical memory, -1 = off)")
		spill := fs.Bool("spill", true, "disk-spill build: bounded memory, sorts n-gram counts on disk")
		workers := fs.Int("workers", 0, "spill build lexer/emitter workers (0 = cores - 1, capped at 8)")
		tmpdir := fs.String("tmpdir", "", "spill build run-file directory (default = system temp)")
		incr := fs.Bool("incr", false, "incremental: fold changed files into the delta overlay")
		auxOnly := fs.Bool("aux", false, "aux tables only: member/directory/file-start tables, no n-gram build")
		var roots multiFlag
		fs.Var(&roots, "root", "corpus root directory (repeatable)")
		fs.Parse(os.Args[2:])
		cfg := loadConfig()
		if *order == 0 {
			*order = cfg.Order
		}
		if *order < 1 || *order > 12 {
			log("index: order must be 1-12, got %d", *order)
			os.Exit(2)
		}
		if len(roots) == 0 {
			roots = cfg.Roots
		}
		if len(roots) == 0 {
			roots = []string{defaultCorpusDir()}
		}
		if *incr {
			if runIncremental(*out, roots, log) {
				return
			}
			log("index: falling back to a full build")
		}
		if *auxOnly {
			// The small tables need no spill: a cold file completes
			// members from TypeMem and CallMem even when a full
			// n-gram rebuild is not practical.
			ax, err := engine.BuildAux(roots, os.Stderr)
			if err != nil {
				log("index -aux: %v", err)
				os.Exit(1)
			}
			if err := engine.SaveAux(*out+".aux", ax); err != nil {
				log("index -aux: %v", err)
				os.Exit(1)
			}
			log("wrote %s.aux (%d types, %d calls, %d dirs)", *out,
				len(ax.TypeMem), len(ax.CallMem), len(ax.DirIdents))
			return
		}
		if *memMB == 0 {
			*memMB = defaultMemCapMB()
		} else if *memMB < 0 {
			*memMB = 0
		}
		log("indexing %d roots at order %d", len(roots), *order)
		var bun *engine.Bundle
		var st engine.BuildStats
		var err error
		if *spill {
			bun, st, err = engine.BuildIndexSpill(roots, *order, nil, *budget*1_000_000, *memMB, *workers, *tmpdir, os.Stderr)
		} else {
			bun, st, err = engine.BuildIndexBudget(roots, *order, nil, *budget*1_000_000, *memMB, os.Stderr)
		}
		if err != nil {
			log("index: %v", err)
			os.Exit(1)
		}
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
			log("index: %v", err)
			os.Exit(1)
		}
		if err := engine.Save(*out, bun); err != nil {
			log("index: %v", err)
			os.Exit(1)
		}
		if st.Symbols != nil {
			if err := st.Symbols.Save(*out + ".symbols"); err != nil {
				log("index: symbols write failed: %v", err)
			}
		}
		if err := writeManifest(*out+".manifest", roots, st.Meta); err != nil {
			log("index: manifest write failed: %v", err)
		}
		// A fresh base makes the old delta stale. Drop it.
		os.Remove(*out + ".delta")
		log("done: %d files, %d MB, %dM tokens, %d vocab, %d unique lines",
			st.Files, st.Bytes>>20, st.Tokens/1_000_000, st.Vocab, st.Lines)
		log("wrote %s", *out)

	case "commitmsg":
		runCommitMsg(os.Args[2:], log)

	case "collect":
		fs := flag.NewFlagSet("collect", flag.ExitOnError)
		org := fs.String("org", "Quad4-Software", "GitHub org")
		dest := fs.String("dest", defaultCorpusDir(), "clone destination")
		maxKB := fs.Int("max-size-kb", 400000, "skip repos larger than this")
		fs.Parse(os.Args[2:])
		if err := collectOrg(*org, *dest, *maxKB); err != nil {
			log("collect: %v", err)
			os.Exit(1)
		}

	case "complete":
		fs := flag.NewFlagSet("complete", flag.ExitOnError)
		file := fs.String("f", "", "file to complete in")
		line := fs.Int("line", 0, "1-based line")
		col := fs.Int("col", 0, "1-based column (byte offset in line)")
		fs.Parse(os.Args[2:])
		data, err := os.ReadFile(*file)
		if err != nil {
			log("complete: %v", err)
			os.Exit(1)
		}
		off := offsetOf(string(data), *line, *col)
		e := loadEngine(loadConfig())
		e.UpdateDoc("file://"+*file, string(data))
		items := e.Complete("file://"+*file, string(data), off)
		for _, it := range items {
			fmt.Printf("[%s] %q\n", it.Source, it.Text)
		}

	case "eval":
		fs := flag.NewFlagSet("eval", flag.ExitOnError)
		var roots multiFlag
		fs.Var(&roots, "root", "eval corpus root (repeatable)")
		nFiles := fs.Int("files", 100, "max files to sample")
		perFile := fs.Int("pos", 10, "cursor positions per file")
		seed := fs.Int64("seed", 1, "sampling seed")
		verb := fs.Bool("v", false, "print sample hits/misses")
		out := fs.String("o", "", "write JSON report to path")
		baseline := fs.String("baseline", "", "compare against a saved JSON report")
		tol := fs.Float64("tol", 0.02, "allowed hit@1 regression vs baseline (fraction)")
		fs.Parse(os.Args[2:])
		if len(roots) == 0 {
			roots = loadConfig().Roots
		}
		if len(roots) == 0 {
			log("eval: no roots (use -root or set roots in config)")
			os.Exit(2)
		}
		e := loadEngine(loadConfig())
		runEval(e, roots, *nFiles, *perFile, *seed, *verb, *out, *baseline, *tol)

	case "tune":
		// Coordinate-descent search over scoring weights against a
		// holdout corpus: the offline reranker. Evaluates each knob at
		// a few scales, keeps the best hit@1, writes model.bin.weights.
		fs := flag.NewFlagSet("tune", flag.ExitOnError)
		var roots multiFlag
		fs.Var(&roots, "root", "holdout corpus root (repeatable)")
		nFiles := fs.Int("files", 40, "files to sample")
		perFile := fs.Int("pos", 6, "cursor positions per file")
		seed := fs.Int64("seed", 7, "sampling seed")
		fs.Parse(os.Args[2:])
		if len(roots) == 0 {
			roots = loadConfig().Roots
		}
		if len(roots) == 0 {
			log("tune: no roots")
			os.Exit(2)
		}
		runTune(loadConfig(), roots, *nFiles, *perFile, *seed, log)

	case "symbol":
		fs := flag.NewFlagSet("symbol", flag.ExitOnError)
		limit := fs.Int("n", 10, "max results")
		fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fmt.Fprintln(os.Stderr, "usage: q4tab symbol [-n N] <name|prefix>")
			os.Exit(2)
		}
		e := loadEngine(loadConfig())
		for _, s := range e.LookupSymbol(fs.Arg(0), *limit) {
			fmt.Printf("%s:%d\t%s\t%s\n", s.Path, s.Line, s.Kind, s.Sig)
		}

	case "fetch":
		// Download an aux sidecar (or any model sidecar) and install
		// it atomically. A running server picks it up via watchAux.
		fs := flag.NewFlagSet("fetch", flag.ExitOnError)
		out := fs.String("o", defaultModelPath()+".aux", "destination path")
		timeout := fs.Duration("timeout", 120*time.Second, "download timeout")
		sum := fs.String("sha256", "", "expected sha256 of the payload (hex); empty skips the check")
		fs.Parse(os.Args[2:])
		if fs.NArg() < 1 {
			fmt.Fprintln(os.Stderr, "usage: q4tab fetch [-o path] [-sha256 hex] <url|file>")
			os.Exit(2)
		}
		if err := fetchAux(fs.Arg(0), *out, *sum, *timeout, log); err != nil {
			log("fetch: %v", err)
			os.Exit(1)
		}

	case "stats":
		fs := flag.NewFlagSet("stats", flag.ExitOnError)
		fs.Parse(os.Args[2:])
		bun, err := engine.Load(defaultModelPath())
		if err != nil {
			log("stats: %v", err)
			os.Exit(1)
		}
		m := bun.M
		var rows, entries int64
		for k := 1; k <= m.N; k++ {
			rows += int64(len(m.Orders[k].Keys))
			entries += m.Orders[k].NToks
		}
		nl := 0
		if bun.Lines != nil {
			nl = bun.Lines.Len()
		}
		var gramKeys, structKeys, nIdent int
		if bun.LineBi != nil {
			gramKeys = len(bun.LineBi.Keys)
		}
		if bun.Struct != nil {
			structKeys = len(bun.Struct.Keys)
		}
		if bun.Idents != nil {
			nIdent = len(bun.Idents.Ids)
		}
		fmt.Printf("vocab=%d order=%d kn=%v contexts=%d entries=%d lines=%d\n",
			m.Vocab.Len(), m.N, m.KN, rows, entries, nl)
		fmt.Printf("lineGrams=%d structCtx=%d langs=%d idents=%d subvocab=%d\n",
			gramKeys, structKeys, len(bun.Langs), nIdent, subVocabLen(bun))

	default:
		fmt.Fprintf(os.Stderr, "unknown subcommand %q\n", os.Args[1])
		os.Exit(2)
	}
}

type multiFlag []string

func (m *multiFlag) String() string { return strings.Join(*m, ",") }
func (m *multiFlag) Set(v string) error {
	*m = append(*m, v)
	return nil
}

func offsetOf(text string, line, col int) int {
	l := 1
	i := 0
	for i < len(text) && l < line {
		if text[i] == '\n' {
			l++
		}
		i++
	}
	c := 1
	for i < len(text) && text[i] != '\n' && c < col {
		i++
		c++
	}
	return i
}

// runEval measures masked-completion hit rate: for sampled positions,
// hide the rest of the line and check whether any suggestion is a
// normalized prefix of what was really there.
// evalReport is the machine-readable result of one eval run, used by
// -o (write report), -baseline (regression check), and tune.
type evalReport struct {
	Seed    int64                  `json:"seed"`
	Files   int                    `json:"files"`
	Tries   int                    `json:"tries"`
	Hit1    float64                `json:"hit1"`
	HitK    float64                `json:"hitK"`
	P50ms   float64                `json:"p50ms"`
	P95ms   float64                `json:"p95ms"`
	PerLang map[string]*langReport `json:"perLang,omitempty"`
}

type langReport struct {
	Tries int     `json:"tries"`
	Hit1  float64 `json:"hit1"`
	HitK  float64 `json:"hitK"`
}

// collectEvalFiles gathers corpus files, deterministically shuffled by
// seed and capped at nFiles.
func collectEvalFiles(roots []string, nFiles int, seed int64) []corpus.File {
	rng := rand.New(rand.NewSource(seed))
	ch := make(chan corpus.File, 64)
	go corpus.Collect(roots, ch)
	var files []corpus.File
	for f := range ch {
		files = append(files, f)
	}
	rng.Shuffle(len(files), func(i, j int) { files[i], files[j] = files[j], files[i] })
	if len(files) > nFiles {
		files = files[:nFiles]
	}
	return files
}

type evalStat struct {
	tries, hit1, hitK int
	lat               []time.Duration
}

// evalOnce runs masked-line completion over files and returns metrics.
func evalOnce(e *engine.Engine, files []corpus.File, perFile int, seed int64, verb bool) (evalReport, int) {
	rng := rand.New(rand.NewSource(seed))
	total := &evalStat{}
	perLang := map[string]*evalStat{}
	agg := func(lang string) *evalStat {
		s := perLang[lang]
		if s == nil {
			s = &evalStat{}
			perLang[lang] = s
		}
		return s
	}
	var misses int
	for _, f := range files {
		text := string(f.Data)
		uri := "file://" + f.Path
		e.UpdateDoc(uri, text)
		lang := langOf(f.Path)
		ls := agg(lang)
		var starts []int // byte offsets of line starts
		starts = append(starts, 0)
		for i, c := range f.Data {
			if c == '\n' {
				starts = append(starts, i+1)
			}
		}
		var cands [][2]int // [start,end) of lines long enough to split
		for _, s := range starts {
			end := s
			for end < len(f.Data) && f.Data[end] != '\n' {
				end++
			}
			if end-s < 8 {
				continue
			}
			cands = append(cands, [2]int{s, end})
		}
		if len(cands) == 0 {
			continue
		}
		for p := 0; p < perFile; p++ {
			l := cands[rng.Intn(len(cands))]
			lo := l[0] + 4
			if lo >= l[1]-2 {
				continue
			}
			off := lo + rng.Intn(l[1]-lo)
			truth := lines.Normalize(text[off:l[1]])
			if len(truth) < 3 {
				continue
			}
			// Mask the tail: while typing, the rest of the line does
			// not exist yet, so evaluate on the truncated document.
			masked := text[:off]
			t0 := time.Now()
			items := e.Complete(uri, masked, off)
			d := time.Since(t0)
			total.lat = append(total.lat, d)
			ls.lat = append(ls.lat, d)
			total.tries++
			ls.tries++
			hit := false
			for k, it := range items {
				// Hit when the suggestion begins with the actual
				// rest of line (it may continue past EOL).
				if strings.HasPrefix(lines.Normalize(it.Text), truth) {
					if k == 0 {
						total.hit1++
						ls.hit1++
					}
					total.hitK++
					ls.hitK++
					hit = true
					break
				}
			}
			if verb && !hit && misses < 25 {
				misses++
				line := text[l[0]:l[1]]
				fmt.Printf("%s:%d  line=%q\n  truth=%q\n", f.Path, off, line[min(off-l[0], len(line)):], truth)
				for _, it := range items {
					fmt.Printf("  [%s] %q\n", it.Source, it.Text)
				}
				if len(items) == 0 {
					fmt.Printf("  (no items)\n")
				}
			}
		}
	}
	pct := func(lat []time.Duration) (time.Duration, time.Duration) {
		if len(lat) == 0 {
			return 0, 0
		}
		sort.Slice(lat, func(i, j int) bool { return lat[i] < lat[j] })
		return lat[len(lat)/2], lat[len(lat)*95/100]
	}
	p50, p95 := pct(total.lat)
	tries := max(total.tries, 1)
	rep := evalReport{
		Seed:  seed,
		Files: len(files),
		Tries: total.tries,
		Hit1:  100 * float64(total.hit1) / float64(tries),
		HitK:  100 * float64(total.hitK) / float64(tries),
		P50ms: float64(p50.Microseconds()) / 1000,
		P95ms: float64(p95.Microseconds()) / 1000,
	}
	rep.PerLang = make(map[string]*langReport, len(perLang))
	for l, s := range perLang {
		t := max(s.tries, 1)
		rep.PerLang[l] = &langReport{
			Tries: s.tries,
			Hit1:  100 * float64(s.hit1) / float64(t),
			HitK:  100 * float64(s.hitK) / float64(t),
		}
	}
	return rep, misses
}

func runEval(e *engine.Engine, roots []string, nFiles, perFile int, seed int64, verb bool, outPath, baselinePath string, tol float64) {
	files := collectEvalFiles(roots, nFiles, seed)
	if len(files) == 0 {
		fmt.Fprintln(os.Stderr, "eval: no files found")
		os.Exit(1)
	}
	rep, _ := evalOnce(e, files, perFile, seed, verb)
	fmt.Printf("files=%d tries=%d hit@1=%.1f%% hit@k=%.1f%% p50=%.2fms p95=%.2fms\n",
		rep.Files, rep.Tries, rep.Hit1, rep.HitK, rep.P50ms, rep.P95ms)

	// Per-language breakdown, most-sampled first. Languages with a
	// handful of tries print percentages anyway. The sample count is
	// right there to keep them honest.
	var langs []string
	for l := range rep.PerLang {
		langs = append(langs, l)
	}
	sort.Slice(langs, func(i, j int) bool {
		if rep.PerLang[langs[i]].Tries != rep.PerLang[langs[j]].Tries {
			return rep.PerLang[langs[i]].Tries > rep.PerLang[langs[j]].Tries
		}
		return langs[i] < langs[j]
	})
	for _, l := range langs {
		s := rep.PerLang[l]
		fmt.Printf("  %-12s tries=%-5d hit@1=%5.1f%% hit@k=%5.1f%%\n",
			l, s.Tries, s.Hit1, s.HitK)
	}

	if outPath != "" {
		data, _ := json.MarshalIndent(rep, "", "  ")
		if err := os.WriteFile(outPath, data, 0o644); err != nil {
			fmt.Fprintf(os.Stderr, "eval: report write: %v\n", err)
		} else {
			fmt.Printf("wrote %s\n", outPath)
		}
	}
	if baselinePath != "" {
		data, err := os.ReadFile(baselinePath)
		if err != nil {
			fmt.Fprintf(os.Stderr, "eval: baseline read: %v\n", err)
			os.Exit(1)
		}
		var base evalReport
		if err := json.Unmarshal(data, &base); err != nil {
			fmt.Fprintf(os.Stderr, "eval: baseline parse: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("baseline: hit@1=%.1f%% (this run %.1f%%, tolerance %.1f)\n",
			base.Hit1, rep.Hit1, tol*100)
		if rep.Hit1 < base.Hit1-tol*100 {
			fmt.Fprintf(os.Stderr, "eval: REGRESSION: hit@1 %.1f%% below baseline %.1f%% - %.1f%%\n",
				rep.Hit1, base.Hit1, tol*100)
			os.Exit(1)
		}
		fmt.Println("no regression")
	}
}

// langOf classifies a file path into a language bucket for eval
// reporting. The shared implementation lives in engine.LangOf.
// defaultMemCapMB returns 70% of physical memory in MB, or 0 when the
// total cannot be read.
func defaultMemCapMB() int {
	data, err := os.ReadFile("/proc/meminfo")
	if err != nil {
		return 0
	}
	for _, ln := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(ln, "MemTotal:") {
			f := strings.Fields(ln)
			if len(f) >= 2 {
				if kb, err := strconv.ParseInt(f[1], 10, 64); err == nil {
					return int(kb * 7 / 10 / 1024)
				}
			}
		}
	}
	return 0
}

func langOf(path string) string { return engine.LangOf(path) }

func subVocabLen(b *engine.Bundle) int {
	if b.Sub == nil || b.Sub.Vocab == nil {
		return 0
	}
	return b.Sub.Vocab.Len()
}

// manifest records what the index knows about each corpus file so an
// incremental run can skip anything unchanged. Indexed marks files that
// passed the content filters. The rest are tracked so they do not get
// re-read on every run.
type manifest struct {
	Version int                    `json:"v"`
	Files   map[string]manifestEnt `json:"files"`
}

type manifestEnt struct {
	Size    int64 `json:"s"`
	Mtime   int64 `json:"m"`
	Indexed bool  `json:"ok"`
}

func loadManifest(path string) (*manifest, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	var m manifest
	if err := json.Unmarshal(data, &m); err != nil || m.Files == nil {
		return nil, fmt.Errorf("bad manifest")
	}
	return &m, nil
}

// writeManifest records the current corpus state: every listed file gets
// an entry, indexed only if it appears in the indexed set.
func writeManifest(path string, roots []string, indexed []engine.FileMeta) error {
	inIdx := make(map[string]bool, len(indexed))
	for _, fm := range indexed {
		inIdx[fm.Path] = true
	}
	m := manifest{Version: 1, Files: make(map[string]manifestEnt)}
	for _, f := range corpus.ListFiles(roots) {
		m.Files[f.Path] = manifestEnt{Size: f.Size, Mtime: f.ModTime, Indexed: inIdx[f.Path]}
	}
	data, err := json.Marshal(m)
	if err != nil {
		return err
	}
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, path)
}

// runIncremental diffs the corpus against the manifest and folds new or
// changed files into the delta overlay. Returns false when a full build
// is required (no manifest, no base model).
func runIncremental(modelPath string, roots []string, log func(string, ...any)) bool {
	man, err := loadManifest(modelPath + ".manifest")
	if err != nil {
		log("index: no usable manifest: %v", err)
		return false
	}
	if _, err := os.Stat(modelPath); err != nil {
		log("index: no base model: %v", err)
		return false
	}
	cur := corpus.ListFiles(roots)
	curSet := make(map[string]corpus.File, len(cur))
	for _, f := range cur {
		curSet[f.Path] = f
	}
	var changed []corpus.File
	for _, f := range cur {
		ent, ok := man.Files[f.Path]
		if !ok || ent.Size != f.Size || ent.Mtime != f.ModTime {
			changed = append(changed, f)
		}
	}
	var deleted []string
	for p := range man.Files {
		if _, ok := curSet[p]; !ok {
			deleted = append(deleted, p)
		}
	}
	if len(changed) == 0 && len(deleted) == 0 {
		log("index: up to date (%d files)", len(cur))
		return true
	}
	log("index: %d changed, %d deleted of %d", len(changed), len(deleted), len(cur))

	// Merge over any previous delta so increments accumulate.
	delta := map[string]engine.DeltaFile{}
	if prev, err := engine.LoadDelta(modelPath + ".delta"); err == nil {
		for _, d := range prev {
			delta[d.Path] = d
		}
	}
	for _, p := range deleted {
		delete(delta, p)
	}
	for _, f := range changed {
		data, err := os.ReadFile(f.Path)
		if err != nil || !corpus.Accept(f.Path, data) {
			delete(delta, f.Path)
			continue
		}
		delta[f.Path] = engine.DeltaFile{Path: f.Path, ModTime: f.ModTime, Data: data}
	}
	files := make([]engine.DeltaFile, 0, len(delta))
	var dbytes int64
	for _, d := range delta {
		files = append(files, d)
		dbytes += int64(len(d.Data))
	}
	if dbytes > 128<<20 {
		log("index: delta is %d MB; consider a full index soon", dbytes>>20)
	}
	if err := engine.SaveDelta(modelPath+".delta", files); err != nil {
		log("index: delta write: %v", err)
		os.Exit(1)
	}
	// Keep the symbol index in sync with the overlay.
	if syms, err := symbols.Load(modelPath + ".symbols"); err == nil {
		for _, p := range deleted {
			syms.RemovePath(p)
		}
		for _, f := range changed {
			syms.RemovePath(f.Path)
			if d, ok := delta[f.Path]; ok {
				for _, s := range symbols.Extract(f.Path, d.Data) {
					syms.Add(s)
				}
			}
		}
		if err := syms.Save(modelPath + ".symbols"); err != nil {
			log("index: symbols write: %v", err)
		}
	}
	// Rebuild the manifest over the current tree, carrying Indexed
	// forward for unchanged files and recomputing it for changed ones.
	inDelta := make(map[string]bool, len(delta))
	for p := range delta {
		inDelta[p] = true
	}
	nm := manifest{Version: 1, Files: make(map[string]manifestEnt, len(cur))}
	for _, f := range cur {
		ent := man.Files[f.Path]
		ent.Size = f.Size
		ent.Mtime = f.ModTime
		if _, wasChanged := man.Files[f.Path]; !wasChanged || inDelta[f.Path] {
			ent.Indexed = ent.Indexed || inDelta[f.Path]
		}
		nm.Files[f.Path] = ent
	}
	data, _ := json.Marshal(nm)
	tmp := modelPath + ".manifest.tmp"
	if err := os.WriteFile(tmp, data, 0o644); err == nil {
		os.Rename(tmp, modelPath+".manifest")
	}
	log("delta: %d files, %d MB -> %s", len(files), dbytes>>20, modelPath+".delta")
	log("note: lines from deleted files stay in the base model until a full index")
	return true
}

// watchDelta reloads the delta overlay into a running engine whenever
// the file changes, so `q4tab watch` (or a cron'd index -incr)
// updates live servers without a restart. Stat-only, every 2s.
func watchDelta(path string, e *engine.Engine, log func(string, ...any)) {
	var lastMtime, lastSize int64
	if st, err := os.Stat(path); err == nil {
		lastMtime, lastSize = st.ModTime().UnixNano(), st.Size()
	}
	for {
		time.Sleep(2 * time.Second)
		st, err := os.Stat(path)
		var mt, sz int64
		if err == nil {
			mt, sz = st.ModTime().UnixNano(), st.Size()
		}
		if mt == lastMtime && sz == lastSize {
			continue
		}
		lastMtime, lastSize = mt, sz
		var files []engine.DeltaFile
		if err == nil {
			if f, lerr := engine.LoadDelta(path); lerr == nil {
				files = f
			} else {
				log("watch: delta load: %v", lerr)
				continue
			}
		}
		// Missing file also reloads: clears a stale overlay.
		e.InstallDelta(files)
		log("watch: installed delta, %d files", len(files))
	}
}

// watchAux reloads the aux sidecar whenever it changes, the same
// contract as watchDelta: fetch or rebuild the file and the running
// server picks it up without a restart.
func watchAux(path string, e *engine.Engine, log func(string, ...any)) {
	var lastMtime, lastSize int64
	if st, err := os.Stat(path); err == nil {
		lastMtime, lastSize = st.ModTime().UnixNano(), st.Size()
	}
	for {
		time.Sleep(2 * time.Second)
		st, err := os.Stat(path)
		var mt, sz int64
		if err == nil {
			mt, sz = st.ModTime().UnixNano(), st.Size()
		}
		if mt == lastMtime && sz == lastSize {
			continue
		}
		lastMtime, lastSize = mt, sz
		if err != nil {
			e.SetAux(nil)
			log("watch: aux sidecar removed")
			continue
		}
		a := engine.LoadAux(path)
		if a == nil {
			continue // mid-write or corrupt; try again next tick
		}
		e.SetAux(a)
		log("watch: installed aux (%d types, %d calls, %d files)",
			len(a.TypeMem), len(a.CallMem), a.Files)
	}
}

func collectOrg(org, dest string, maxKB int) error {
	out, err := exec.Command("gh", "repo", "list", org,
		"--limit", "500", "--json", "name,size,isFork").Output()
	if err != nil {
		return fmt.Errorf("gh repo list: %w", err)
	}
	var repos []struct {
		Name   string `json:"name"`
		Size   int    `json:"size"`
		IsFork bool   `json:"isFork"`
	}
	if err := json.Unmarshal(out, &repos); err != nil {
		return err
	}
	if err := os.MkdirAll(dest, 0o755); err != nil {
		return err
	}
	for _, r := range repos {
		if r.Size > maxKB {
			fmt.Fprintf(os.Stderr, "skip %s (%d KB > %d KB)\n", r.Name, r.Size, maxKB)
			continue
		}
		dir := filepath.Join(dest, r.Name)
		if _, err := os.Stat(dir); err == nil {
			fmt.Fprintf(os.Stderr, "have %s\n", r.Name)
			continue
		}
		url := fmt.Sprintf("https://github.com/%s/%s", org, r.Name)
		fmt.Fprintf(os.Stderr, "clone %s\n", url)
		cmd := exec.Command("git", "clone", "--depth", "1", url, dir)
		cmd.Stdout = os.Stderr
		cmd.Stderr = os.Stderr
		cmd.Run() // continue past failures
	}
	return nil
}

// fetchAux downloads an aux sidecar from a URL (or copies a local
// path), validates the format, and installs it atomically. The size
// is capped at 512MB: aux tables are tens of MB, anything larger is
// not an aux file.
func fetchAux(src, dest, wantSHA string, timeout time.Duration, log func(string, ...any)) error {
	var data []byte
	if strings.HasPrefix(src, "http://") || strings.HasPrefix(src, "https://") {
		cl := &http.Client{Timeout: timeout}
		resp, err := cl.Get(src)
		if err != nil {
			return err
		}
		defer resp.Body.Close()
		if resp.StatusCode != 200 {
			return fmt.Errorf("%s: HTTP %d", src, resp.StatusCode)
		}
		data, err = io.ReadAll(io.LimitReader(resp.Body, 512<<20+1))
		if err != nil {
			return err
		}
		if len(data) > 512<<20 {
			return fmt.Errorf("%s: too large for an aux sidecar", src)
		}
	} else {
		var err error
		data, err = os.ReadFile(src)
		if err != nil {
			return err
		}
	}
	if wantSHA != "" {
		got := fmt.Sprintf("%x", sha256.Sum256(data))
		if !strings.EqualFold(got, wantSHA) {
			return fmt.Errorf("sha256 mismatch: got %s want %s", got, wantSHA)
		}
	}
	// Validate before touching the destination: parse a copy.
	tmp := dest + ".fetch"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	a := engine.LoadAux(tmp)
	if a == nil {
		os.Remove(tmp)
		return fmt.Errorf("%s: not a valid aux sidecar", src)
	}
	if err := os.Rename(tmp, dest); err != nil {
		os.Remove(tmp)
		return err
	}
	log("fetch: installed %s (%d types, %d calls, %d dirs, %d files)",
		dest, len(a.TypeMem), len(a.CallMem), len(a.DirIdents), a.Files)
	return nil
}
