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
	m, li, st, err := engine.BuildIndex([]string{dir}, 6, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if st.Files != 2 {
		t.Fatalf("indexed %d files, want 2", st.Files)
	}
	e := engine.New(engine.DefaultConfig())
	e.SetModel(m, li)
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
