package commitmsg

import (
	"bufio"
	"bytes"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// sentinel marks commit boundaries in the git log stream so parsing
// never depends on diff content.
const sentinel = "@@@COMMIT@@@"

// Extract runs git log -p over repoRoot and writes one pseudo-document
// per commit to outDir/<sha>.commit. Commits with no diff (merges,
// empty commits) are skipped. Returns the count written.
func Extract(repoRoot, outDir string, maxCommits int) (int, error) {
	args := []string{"-C", repoRoot, "log", "--format=" + sentinel + "%H %s", "-p"}
	if maxCommits > 0 {
		args = append(args, "-n", strconv.Itoa(maxCommits))
	}
	cmd := exec.Command("git", args...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return 0, fmt.Errorf("git log: %w", err)
	}
	var errBuf bytes.Buffer
	cmd.Stderr = &errBuf
	if err := cmd.Start(); err != nil {
		return 0, fmt.Errorf("git log: %w", err)
	}
	if err := os.MkdirAll(outDir, 0o755); err != nil {
		cmd.Process.Kill()
		cmd.Wait()
		return 0, err
	}

	var doc bytes.Buffer
	var sha, subject string
	hasBody := false
	written := 0
	var p diffParser

	reset := func() {
		doc.Reset()
		sha, subject = "", ""
		hasBody = false
		p = diffParser{}
	}
	flush := func() error {
		defer reset()
		if sha == "" || !hasBody {
			return nil
		}
		doc.WriteString("msg: " + subject + "\n")
		name := filepath.Join(outDir, sha+".commit")
		if err := os.WriteFile(name, doc.Bytes(), 0o644); err != nil {
			return err
		}
		written++
		return nil
	}

	// Stream the log. ReadString has no line-length cap, so a minified
	// line cannot kill the scan.
	r := bufio.NewReaderSize(stdout, 64<<10)
	for {
		line, rerr := r.ReadString('\n')
		line = strings.TrimRight(line, "\r\n")
		if strings.HasPrefix(line, sentinel) {
			if err := flush(); err != nil {
				cmd.Process.Kill()
				cmd.Wait()
				return written, err
			}
			sha, subject, _ = strings.Cut(line[len(sentinel):], " ")
			doc.WriteString("commit " + sha + "\n")
		} else if sha != "" {
			if s := p.feed(line); s != "" {
				hasBody = true
				if doc.Len() < maxDocBytes {
					doc.WriteString(s)
					doc.WriteByte('\n')
				}
			}
		}
		if rerr != nil {
			break
		}
	}
	if err := flush(); err != nil {
		cmd.Wait()
		return written, err
	}
	if err := cmd.Wait(); err != nil {
		return written, fmt.Errorf("git log: %w: %s", err, strings.TrimSpace(errBuf.String()))
	}
	return written, nil
}
