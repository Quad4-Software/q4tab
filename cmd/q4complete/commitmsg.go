package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"

	"q4complete/internal/commitmsg"
	"q4complete/internal/engine"
)

// runCommitMsg implements the commitmsg subcommand.
// Called from main; args are the arguments after "commitmsg".
func runCommitMsg(args []string, log func(string, ...any)) {
	fs := flag.NewFlagSet("commitmsg", flag.ExitOnError)
	repo := fs.String("repo", ".", "repository root")
	train := fs.Bool("train", false, "extract git log into a corpus dir instead of suggesting")
	outDir := fs.String("o", "", "output dir for -train (default ~/.local/share/q4complete/commits/<repo name>)")
	maxCommits := fs.Int("n", 2000, "max commits to extract with -train")
	modelPath := fs.String("model", defaultModelPath(), "model path")
	fs.Parse(args)

	if *train {
		dir := *outDir
		if dir == "" {
			abs, err := filepath.Abs(*repo)
			if err != nil {
				abs = *repo
			}
			home, _ := os.UserHomeDir()
			dir = filepath.Join(home, ".local", "share", "q4complete", "commits", filepath.Base(abs))
		}
		n, err := commitmsg.Extract(*repo, dir, *maxCommits)
		if err != nil {
			log("commitmsg: extract: %v", err)
			os.Exit(1)
		}
		log("wrote %d commit docs to %s", n, dir)
		log("index them with: q4complete index -root %s -o %s", dir, *modelPath)
		return
	}

	diff, err := gitOutput(*repo, "diff", "--cached")
	if err != nil {
		log("commitmsg: git diff --cached: %v", err)
		os.Exit(1)
	}
	names, err := gitOutput(*repo, "diff", "--cached", "--name-only")
	if err != nil {
		log("commitmsg: git diff --cached --name-only: %v", err)
		os.Exit(1)
	}
	if strings.TrimSpace(diff) == "" {
		// Nothing staged: fall back to the working-tree diff so the
		// command still answers before anything is staged.
		if d, err := gitOutput(*repo, "diff"); err == nil {
			diff = d
		}
		if f, err := gitOutput(*repo, "diff", "--name-only"); err == nil {
			names = f
		}
	}
	var files []string
	for _, f := range strings.Split(names, "\n") {
		if f = strings.TrimSpace(f); f != "" {
			files = append(files, f)
		}
	}
	if diff == "" && len(files) == 0 {
		log("commitmsg: no staged or modified files under %s", *repo)
		os.Exit(1)
	}

	bun, err := engine.Load(*modelPath)
	if err != nil {
		log("commitmsg: cannot load model %s: %v", *modelPath, err)
		os.Exit(1)
	}
	e := engine.New(engine.DefaultConfig())
	e.SetBundle(bun)
	gen := func(ctx string) []string {
		items := e.Complete("file:///commit.msg", ctx, len(ctx))
		out := make([]string, 0, len(items))
		for _, it := range items {
			out = append(out, it.Text)
		}
		return out
	}
	for _, s := range commitmsg.Suggest(gen, files, diff) {
		fmt.Println(s)
	}
}

// gitOutput runs git -C repo <args> and returns stdout.
func gitOutput(repo string, args ...string) (string, error) {
	out, err := exec.Command("git", append([]string{"-C", repo}, args...)...).Output()
	if err != nil {
		return "", err
	}
	return string(out), nil
}
