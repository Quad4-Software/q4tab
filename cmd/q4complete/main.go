// q4complete: fully local statistical code completion.
//
// Subcommands:
//
//	serve     run the LSP server on stdio (what editors talk to)
//	index     train the model over corpus roots
//	collect   clone a GitHub org's repos for indexing
//	complete  print completions for a file position (testing)
//	stats     print model stats
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"q4complete/internal/engine"
	"q4complete/internal/lsp"
)

func defaultModelPath() string {
	if p := os.Getenv("Q4COMPLETE_MODEL"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "q4complete", "model.bin")
}

func defaultCorpusDir() string {
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".local", "share", "q4complete", "corpus")
}

func defaultConfigPath() string {
	if p := os.Getenv("Q4COMPLETE_CONFIG"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "q4complete", "config.json")
}

type config struct {
	Roots    []string `json:"roots"`
	Order    int      `json:"order"`
	MaxLines int      `json:"maxLines"`
}

func loadConfig() config {
	cfg := config{Order: 6, MaxLines: 1}
	data, err := os.ReadFile(defaultConfigPath())
	if err == nil {
		json.Unmarshal(data, &cfg)
	}
	if cfg.Order == 0 {
		cfg.Order = 6
	}
	if cfg.MaxLines == 0 {
		cfg.MaxLines = 1
	}
	return cfg
}

func loadEngine(cfg config) *engine.Engine {
	ecfg := engine.DefaultConfig()
	ecfg.MaxLines = cfg.MaxLines
	e := engine.New(ecfg)
	m, li, err := engine.Load(defaultModelPath())
	if err != nil {
		fmt.Fprintf(os.Stderr, "q4complete: no model at %s (%v); running with file-local completion only\n", defaultModelPath(), err)
		return e
	}
	e.SetModel(m, li)
	return e
}

func main() {
	log := func(format string, a ...any) { fmt.Fprintf(os.Stderr, format+"\n", a...) }
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: q4complete <serve|index|collect|complete|stats> [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "serve":
		fs := flag.NewFlagSet("serve", flag.ExitOnError)
		fs.Parse(os.Args[2:])
		cfg := loadConfig()
		e := loadEngine(cfg)
		conn := lsp.NewConn(os.Stdin, os.Stdout)
		srv := lsp.NewServer(e, conn)
		if err := srv.Run(); err != nil {
			log("serve: %v", err)
			os.Exit(1)
		}

	case "index":
		fs := flag.NewFlagSet("index", flag.ExitOnError)
		out := fs.String("o", defaultModelPath(), "output model path")
		order := fs.Int("order", 0, "n-gram order (default from config or 6)")
		var roots multiFlag
		fs.Var(&roots, "root", "corpus root directory (repeatable)")
		fs.Parse(os.Args[2:])
		cfg := loadConfig()
		if *order == 0 {
			*order = cfg.Order
		}
		if len(roots) == 0 {
			roots = cfg.Roots
		}
		if len(roots) == 0 {
			roots = []string{defaultCorpusDir()}
		}
		log("indexing %d roots at order %d", len(roots), *order)
		m, li, st, err := engine.BuildIndex(roots, *order, nil, os.Stderr)
		if err != nil {
			log("index: %v", err)
			os.Exit(1)
		}
		if err := os.MkdirAll(filepath.Dir(*out), 0o755); err != nil {
			log("index: %v", err)
			os.Exit(1)
		}
		if err := engine.Save(*out, m, li); err != nil {
			log("index: %v", err)
			os.Exit(1)
		}
		log("done: %d files, %d MB, %dM tokens, %d vocab, %d unique lines",
			st.Files, st.Bytes>>20, st.Tokens/1_000_000, st.Vocab, st.Lines)
		log("wrote %s", *out)

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

	case "stats":
		fs := flag.NewFlagSet("stats", flag.ExitOnError)
		fs.Parse(os.Args[2:])
		m, li, err := engine.Load(defaultModelPath())
		if err != nil {
			log("stats: %v", err)
			os.Exit(1)
		}
		var rows, entries int64
		for k := 1; k <= m.N; k++ {
			rows += int64(len(m.Orders[k].Keys))
			entries += int64(len(m.Orders[k].Toks))
		}
		fmt.Printf("vocab=%d order=%d contexts=%d entries=%d lines=%d\n",
			m.Vocab.Len(), m.N, rows, entries, li.Len())

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
