package engine

import (
	"os"
	"testing"
)

// TestV2ModelCompat loads the deployed v2 model (if present) and checks
// the JM fallback path still completes.
func TestV2ModelCompat(t *testing.T) {
	p := os.Getenv("Q4TAB_V2_MODEL")
	if p == "" {
		// The deployed model.bin is the v3 build; the v2 backup kept
		// alongside it is the compat target.
		p = "/home/user1/.local/share/q4tab/model-v2-backup.bin"
	}
	if _, err := os.Stat(p); err != nil {
		t.Skip("no deployed v2 model at", p)
	}
	bun, err := Load(p)
	if err != nil {
		t.Fatalf("v2 load: %v", err)
	}
	if bun.M.KN {
		t.Fatal("v2 model flagged KN")
	}
	if bun.Lines == nil {
		t.Fatal("v2 model lost line index")
	}
	e := New(DefaultConfig())
	e.SetBundle(bun)
	items := e.Complete("file:///x.go", "package main\n\nfunc main() {\n\tif err !", 40)
	if len(items) == 0 {
		t.Fatal("v2 model produced no completions")
	}
	t.Logf("v2 model top: %q", items[0].Text)
}
