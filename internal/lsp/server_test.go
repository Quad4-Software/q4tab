package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
	"time"

	"q4complete/internal/engine"
)

func fixtureCorpus(t *testing.T) string {
	dir := t.TempDir()
	repo := filepath.Join(dir, "demo")
	os.MkdirAll(repo, 0o755)
	os.WriteFile(filepath.Join(repo, "identity.go"), []byte(`package demo

import "fmt"

func DescribeIdentity(dest string) string {
	return fmt.Sprintf("identity hash for %s", dest)
}

func DescribeLink(dest string) string {
	return fmt.Sprintf("link id for %s", dest)
}
`), 0o644)
	os.WriteFile(filepath.Join(repo, "handler.go"), []byte(`package demo

func HandlePacket(data []byte) error {
	if len(data) == 0 {
		return fmt.Errorf("empty packet")
	}
	if len(data) > 1500 {
		return fmt.Errorf("packet too large")
	}
	return nil
}
`), 0o644)
	return dir
}

func buildTestEngine(t *testing.T) *engine.Engine {
	dir := fixtureCorpus(t)
	bun, st, err := engine.BuildIndex([]string{dir}, 6, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if st.Files != 2 {
		t.Fatalf("indexed %d files, want 2", st.Files)
	}
	e := engine.New(engine.DefaultConfig())
	e.SetBundle(bun)
	return e
}

func TestEngineComplete(t *testing.T) {
	e := buildTestEngine(t)
	text := "package x\n\nfunc f() {\n\treturn fmt.Sprintf(\"lin"
	off := len(text)
	items := e.Complete("file:///x.go", text, off)
	if len(items) == 0 {
		t.Fatal("no completions")
	}
	found := false
	for _, it := range items {
		if strings.Contains(it.Text, "link id for") || strings.Contains(it.Text, "k id for") {
			found = true
		}
		t.Logf("[%s] %q", it.Source, it.Text)
	}
	if !found {
		t.Fatalf("expected continuation of Sprintf line, got %+v", items)
	}
}

func TestEngineCacheLocality(t *testing.T) {
	e := buildTestEngine(t)
	// The open doc invents a new identifier the corpus never had.
	doc := "package x\n\nvar quadrilateralChannelCounter int\n"
	e.UpdateDoc("file:///x.go", doc)
	e.Flush()
	// Later in the same file the model should prefer it.
	text := doc + "\nfunc bump() {\n\tquadrilateralCha"
	items := e.Complete("file:///x.go", text, len(text))
	if len(items) == 0 {
		t.Fatal("no completions")
	}
	if !strings.Contains(items[0].Text, "nnelCounter") {
		t.Fatalf("expected file-local ident to win, got %q", items[0].Text)
	}
}

// rpcClient is a minimal framing client for the in-process server.
type rpcClient struct {
	w  io.Writer
	r  *bufio.Reader
	id int
}

func (c *rpcClient) call(method string, params any) (json.RawMessage, error) {
	c.id++
	body, _ := json.Marshal(struct {
		JSONRPC string `json:"jsonrpc"`
		ID      int    `json:"id"`
		Method  string `json:"method"`
		Params  any    `json:"params"`
	}{"2.0", c.id, method, params})
	fmt.Fprintf(c.w, "Content-Length: %d\r\n\r\n", len(body))
	if _, err := c.w.Write(body); err != nil {
		return nil, err
	}
	// read frames until we get a response with our id
	for {
		length := -1
		for {
			line, err := c.r.ReadString('\n')
			if err != nil {
				return nil, err
			}
			line = strings.TrimRight(line, "\r\n")
			if line == "" {
				break
			}
			if strings.HasPrefix(line, "Content-Length:") {
				length, _ = strconv.Atoi(strings.TrimSpace(line[15:]))
			}
		}
		buf := make([]byte, length)
		if _, err := io.ReadFull(c.r, buf); err != nil {
			return nil, err
		}
		var resp struct {
			ID     *int            `json:"id"`
			Result json.RawMessage `json:"result"`
			Error  json.RawMessage `json:"error"`
		}
		json.Unmarshal(buf, &resp)
		if resp.ID != nil && *resp.ID == c.id {
			if resp.Error != nil {
				return nil, fmt.Errorf("rpc error: %s", resp.Error)
			}
			return resp.Result, nil
		}
	}
}

func (c *rpcClient) notify(method string, params any) {
	body, _ := json.Marshal(struct {
		JSONRPC string `json:"jsonrpc"`
		Method  string `json:"method"`
		Params  any    `json:"params"`
	}{"2.0", method, params})
	fmt.Fprintf(c.w, "Content-Length: %d\r\n\r\n", len(body))
	c.w.Write(body)
}

func TestLSPEndToEnd(t *testing.T) {
	e := buildTestEngine(t)
	srvIn, cliIn := io.Pipe()
	cliOut, srvOut := io.Pipe()
	srv := NewServer(e, NewConn(srvIn, srvOut))
	go srv.Run()
	time.Sleep(10 * time.Millisecond)

	cli := &rpcClient{w: cliIn, r: bufio.NewReader(cliOut)}

	res, err := cli.call("initialize", map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	var caps struct {
		Capabilities struct {
			InlineCompletionProvider bool `json:"inlineCompletionProvider"`
		} `json:"capabilities"`
	}
	if err := json.Unmarshal(res, &caps); err != nil {
		t.Fatal(err)
	}
	if !caps.Capabilities.InlineCompletionProvider {
		t.Fatal("no inlineCompletionProvider capability")
	}

	doc := "package x\n\nfunc f() {\n\treturn fmt.Sprintf(\"lin"
	cli.notify("textDocument/didOpen", DidOpenTextDocumentParams{
		TextDocument: TextDocumentItem{
			URI: "file:///x.go", LanguageID: "go", Version: 1, Text: doc,
		},
	})
	time.Sleep(10 * time.Millisecond)

	res, err = cli.call("textDocument/inlineCompletion", InlineCompletionParams{
		TextDocument: TextDocumentIdentifier{URI: "file:///x.go"},
		Position:     Position{Line: 3, Character: len("\treturn fmt.Sprintf(\"lin")},
	})
	if err != nil {
		t.Fatal(err)
	}
	var list InlineCompletionList
	if err := json.Unmarshal(res, &list); err != nil {
		t.Fatal(err)
	}
	if len(list.Items) == 0 {
		t.Fatal("empty inline completion list")
	}
	found := false
	for _, it := range list.Items {
		t.Logf("item: %q", it.InsertText)
		if strings.Contains(it.InsertText, "k id for") {
			found = true
		}
	}
	if !found {
		t.Fatalf("no useful inline completion: %+v", list.Items)
	}
}

func TestIncrementalDidChange(t *testing.T) {
	e := buildTestEngine(t)
	srvIn, cliIn := io.Pipe()
	cliOut, srvOut := io.Pipe()
	srv := NewServer(e, NewConn(srvIn, srvOut))
	go srv.Run()
	time.Sleep(10 * time.Millisecond)
	cli := &rpcClient{w: cliIn, r: bufio.NewReader(cliOut)}
	if _, err := cli.call("initialize", map[string]any{}); err != nil {
		t.Fatal(err)
	}
	doc := "package x\n\nfunc f() int {\n\treturn 1\n}\n"
	cli.notify("textDocument/didOpen", DidOpenTextDocumentParams{
		TextDocument: TextDocumentItem{
			URI: "file:///x.go", LanguageID: "go", Version: 1, Text: doc,
		},
	})
	time.Sleep(10 * time.Millisecond)
	// docText synchronized via a request round-trip: the server
	// processes messages in order, so after the response the preceding
	// notify has been applied and the pipe gives a happens-before edge.
	docText := func() string {
		if _, err := cli.call("q4/status", map[string]any{}); err != nil {
			t.Fatal(err)
		}
		return srv.docs["file:///x.go"]
	}
	// Insert "2" before "1" on line 3: return 1 -> return 21.
	cli.notify("textDocument/didChange", DidChangeTextDocumentParams{
		TextDocument: VersionedTextDocumentIdentifier{URI: "file:///x.go", Version: 2},
		ContentChanges: []TextDocumentContentChangeEvent{{
			Range: &Range{Start: Position{Line: 3, Character: 8}, End: Position{Line: 3, Character: 8}},
			Text:  "2",
		}},
	})
	if got := docText(); !strings.Contains(got, "return 21") {
		t.Fatalf("incremental edit not applied: %q", got)
	}
	// A ranged delete across two lines.
	cli.notify("textDocument/didChange", DidChangeTextDocumentParams{
		TextDocument: VersionedTextDocumentIdentifier{URI: "file:///x.go", Version: 3},
		ContentChanges: []TextDocumentContentChangeEvent{{
			Range: &Range{Start: Position{Line: 3, Character: 8}, End: Position{Line: 3, Character: 9}},
			Text:  "",
		}},
	})
	if got := docText(); !strings.Contains(got, "return 1") {
		t.Fatalf("delete edit not applied: %q", got)
	}
	// Full-text change (Range nil) replaces the doc.
	cli.notify("textDocument/didChange", DidChangeTextDocumentParams{
		TextDocument:   VersionedTextDocumentIdentifier{URI: "file:///x.go", Version: 4},
		ContentChanges: []TextDocumentContentChangeEvent{{Text: "package x\n"}},
	})
	if got := docText(); got != "package x\n" {
		t.Fatalf("full replace failed: %q", got)
	}
}

func TestReindex(t *testing.T) {
	e := buildTestEngine(t)
	srvIn, cliIn := io.Pipe()
	cliOut, srvOut := io.Pipe()
	srv := NewServer(e, NewConn(srvIn, srvOut))
	go srv.Run()
	time.Sleep(10 * time.Millisecond)
	cli := &rpcClient{w: cliIn, r: bufio.NewReader(cliOut)}

	// Missing delta file is not an error: it means no changes.
	delta := filepath.Join(t.TempDir(), "model.bin.delta")
	srv.SetDeltaPath(delta)
	res, err := cli.call("q4/reindex", map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	var r struct {
		Files int `json:"files"`
	}
	if err := json.Unmarshal(res, &r); err != nil || r.Files != 0 {
		t.Fatalf("reindex empty: %v %+v", err, r)
	}

	// A real delta installs its lines into the engine.
	if err := engine.SaveDelta(delta, []engine.DeltaFile{{
		Path: "x.go", ModTime: 1,
		Data: []byte("func reindexedMarker() int {\n\treturn 42\n}\n"),
	}}); err != nil {
		t.Fatal(err)
	}
	res, err = cli.call("q4/reindex", map[string]any{})
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(res, &r); err != nil || r.Files != 1 {
		t.Fatalf("reindex 1 file: %v %+v", err, r)
	}
	if e.Stats()["delta"].(int) != 1 || e.Stats()["deltaTok"].(int) == 0 {
		t.Fatalf("delta not installed: %v", e.Stats())
	}

	// Removing the delta file then reindexing clears the overlay.
	os.Remove(delta)
	if _, err := cli.call("q4/reindex", map[string]any{}); err != nil {
		t.Fatal(err)
	}
	if e.Stats()["delta"].(int) != 0 {
		t.Fatalf("delta not cleared: %v", e.Stats())
	}
}

func TestOffsetAt(t *testing.T) {
	// UTF-16 units: BMP = 1, astral = 2.
	text := "ab\U0001F600cd\nef"
	if got := offsetAt(text, Position{Line: 0, Character: 3}); got != 6 {
		// 'a','b',emoji(2 units) -> col 3 lands right after the emoji
		t.Fatalf("astral offset = %d, want 6", got)
	}
	if got := offsetAt(text, Position{Line: 1, Character: 1}); got != 10 {
		t.Fatalf("line 1 col 1 = %d, want 10", got)
	}
	// CRLF: the \r is not part of the line's columns.
	crlf := "ab\r\ncd"
	if got := offsetAt(crlf, Position{Line: 0, Character: 5}); got != 2 {
		t.Fatalf("CRLF clamp = %d, want 2", got)
	}
	if got := offsetAt(crlf, Position{Line: 1, Character: 1}); got != 5 {
		t.Fatalf("CRLF line1 = %d, want 5", got)
	}
	// Past end of document clamps.
	if got := offsetAt(text, Position{Line: 99, Character: 0}); got != len(text) {
		t.Fatalf("past-end = %d, want %d", got, len(text))
	}
	// Truncated UTF-8 must not skip past a newline.
	bad := "a\xff\nb"
	if got := offsetAt(bad, Position{Line: 0, Character: 5}); got != 2 {
		t.Fatalf("bad utf8 clamp = %d, want 2", got)
	}
}

func TestOversizedFrameSkipped(t *testing.T) {
	e := buildTestEngine(t)
	srvIn, cliIn := io.Pipe()
	cliOut, srvOut := io.Pipe()
	srv := NewServer(e, NewConn(srvIn, srvOut))
	go srv.Run()
	time.Sleep(10 * time.Millisecond)
	cli := &rpcClient{w: cliIn, r: bufio.NewReader(cliOut)}
	// A frame over the 64 MiB cap must be discarded, not fatal.
	huge := make([]byte, 1<<26+1)
	fmt.Fprintf(cliIn, "Content-Length: %d\r\n\r\n", len(huge))
	cliIn.Write(huge)
	res, err := cli.call("initialize", map[string]any{})
	if err != nil {
		t.Fatalf("server died after oversized frame: %v", err)
	}
	if res == nil {
		t.Fatal("no initialize response")
	}
}
