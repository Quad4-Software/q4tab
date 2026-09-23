package engine

import (
	"fmt"
	"os"
	"strings"
	"testing"
)

func rssMB() int {
	d, _ := os.ReadFile("/proc/self/status")
	for _, l := range strings.Split(string(d), "\n") {
		if strings.HasPrefix(l, "VmRSS") {
			var n int
			fmt.Sscanf(l, "VmRSS: %d kB", &n)
			return n / 1024
		}
	}
	return -1
}

func TestRSSProbe(t *testing.T) {
	p := os.Getenv("Q4TAB_MODEL")
	if p == "" {
		t.Skip("no model")
	}
	t.Logf("before: %dMB", rssMB())
	bun, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("after Load: %dMB (vocab=%d lines=%d)", rssMB(), bun.M.Vocab.Len(), bun.Lines.Len())
	// Touch one row per order and a few line keys to see marginal cost.
	_ = bun.M.Vocab.Str(uint32(bun.M.Vocab.Len() / 2))
	_ = bun.Lines.Key(0)
	t.Logf("after touches: %dMB", rssMB())
}
