package lsp

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"
)

// testLSPClient speaks Content-Length framed JSON-RPC over a conn.
type testLSPClient struct {
	conn net.Conn
	r    *bufio.Reader
	next int
}

func dialTest(t *testing.T, addr string) *testLSPClient {
	t.Helper()
	c, err := net.DialTimeout("tcp", addr, 2*time.Second)
	if err != nil {
		t.Fatal(err)
	}
	return &testLSPClient{conn: c, r: bufio.NewReader(c), next: 1}
}

func (c *testLSPClient) call(t *testing.T, method string, params any) json.RawMessage {
	t.Helper()
	p, _ := json.Marshal(params)
	id := c.next
	c.next++
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": id, "method": method, "params": json.RawMessage(p),
	})
	fmt.Fprintf(c.conn, "Content-Length: %d\r\n\r\n", len(body))
	c.conn.Write(body)
	resp, err := readMessage(c.r)
	if err != nil {
		t.Fatalf("read %s response: %v", method, err)
	}
	return resp
}

func readMessage(r *bufio.Reader) (json.RawMessage, error) {
	n := 0
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimSpace(line)
		if line == "" {
			break
		}
		if strings.HasPrefix(line, "Content-Length:") {
			fmt.Sscanf(line, "Content-Length: %d", &n)
		}
	}
	buf := make([]byte, n)
	_, err := io.ReadFull(r, buf)
	return buf, err
}

func startTCP(t *testing.T) (string, func()) {
	t.Helper()
	e := buildTestEngine(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer c.Close()
				srv := NewServer(e, NewConn(c, c))
				srv.Run()
			}()
		}
	}()
	return ln.Addr().String(), func() { ln.Close() }
}

func TestTCPLifecycle(t *testing.T) {
	addr, done := startTCP(t)
	defer done()
	c := dialTest(t, addr)
	defer c.conn.Close()

	res := c.call(t, "initialize", map[string]any{})
	var init struct {
		Capabilities struct {
			InlineCompletionProvider bool `json:"inlineCompletionProvider"`
		} `json:"capabilities"`
	}
	var env struct {
		Result json.RawMessage `json:"result"`
	}
	json.Unmarshal(res, &env)
	json.Unmarshal(env.Result, &init)
	if !init.Capabilities.InlineCompletionProvider {
		t.Fatal("no inlineCompletionProvider capability")
	}

	// didOpen notification (no id) then a completion request.
	body, _ := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "method": "textDocument/didOpen",
		"params": map[string]any{"textDocument": map[string]any{
			"uri": "file:///a.go", "languageId": "go", "version": 1,
			"text": "package x\n\nfunc main() {\n\tfmt.Spr",
		}},
	})
	fmt.Fprintf(c.conn, "Content-Length: %d\r\n\r\n", len(body))
	c.conn.Write(body)

	res = c.call(t, "textDocument/inlineCompletion", map[string]any{
		"textDocument": map[string]any{"uri": "file:///a.go"},
		"position":     map[string]any{"line": 3, "character": 7},
		"context":      map[string]any{"triggerKind": 1},
	})
	var out struct {
		Result struct {
			Items []struct {
				InsertText string `json:"insertText"`
			} `json:"items"`
		} `json:"result"`
		Error any `json:"error"`
	}
	if err := json.Unmarshal(res, &out); err != nil {
		t.Fatal(err)
	}
	if out.Error != nil {
		t.Fatalf("completion error: %v", out.Error)
	}
	if len(out.Result.Items) == 0 {
		t.Fatal("no items over tcp")
	}
	t.Logf("items: %+v", out.Result.Items)
}

func TestTCPConcurrentClients(t *testing.T) {
	addr, done := startTCP(t)
	defer done()
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			c := dialTest(t, addr)
			defer c.conn.Close()
			uri := fmt.Sprintf("file:///c%d.go", i)
			for j := 0; j < 5; j++ {
				res := c.call(t, "q4/status", map[string]any{})
				var env struct {
					Result map[string]any `json:"result"`
				}
				if err := json.Unmarshal(res, &env); err != nil || env.Result == nil {
					t.Errorf("client %d req %d: bad status", i, j)
					return
				}
			}
			_ = uri
		}(i)
	}
	wg.Wait()
}

func TestHTTPRPC(t *testing.T) {
	e := buildTestEngine(t)
	h := HTTPHandler(e, "", ServerOpts{})
	srv := httptest.NewServer(h)
	defer srv.Close()

	call := func(body string) map[string]any {
		resp, err := http.Post(srv.URL+"/rpc", "application/json", strings.NewReader(body))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		var out map[string]any
		json.NewDecoder(resp.Body).Decode(&out)
		return out
	}

	out := call(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}`)
	if _, ok := out["result"]; !ok {
		t.Fatalf("initialize failed: %v", out)
	}

	// didOpen is a notification: expect 202 and no body.
	resp, err := http.Post(srv.URL+"/rpc", "application/json", strings.NewReader(
		`{"jsonrpc":"2.0","method":"textDocument/didOpen","params":{"textDocument":{"uri":"file:///h.go","languageId":"go","version":1,"text":"package x\n\nfunc main() {\n\tfmt.Spr"}}}`))
	if err != nil {
		t.Fatal(err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusAccepted {
		t.Fatalf("notification status %d", resp.StatusCode)
	}

	out = call(`{"jsonrpc":"2.0","id":2,"method":"textDocument/inlineCompletion","params":{"textDocument":{"uri":"file:///h.go"},"position":{"line":3,"character":7},"context":{"triggerKind":1}}}`)
	res, _ := out["result"].(map[string]any)
	items, _ := res["items"].([]any)
	if len(items) == 0 {
		t.Fatalf("no items over http: %v", out)
	}

	// Malformed JSON -> parse error.
	out = call(`{bad json`)
	er, _ := out["error"].(map[string]any)
	if er == nil || er["code"].(float64) != -32700 {
		t.Fatalf("expected parse error, got %v", out)
	}

	// Unknown method -> -32601.
	out = call(`{"jsonrpc":"2.0","id":3,"method":"bogus/method","params":{}}`)
	er, _ = out["error"].(map[string]any)
	if er == nil || er["code"].(float64) != -32601 {
		t.Fatalf("expected method not found, got %v", out)
	}

	// healthz + status endpoints.
	for _, path := range []string{"/healthz", "/status"} {
		resp, err := http.Get(srv.URL + path)
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		if resp.StatusCode != 200 {
			t.Fatalf("%s: %d", path, resp.StatusCode)
		}
	}
}

func TestMCPStdio(t *testing.T) {
	e := buildTestEngine(t)
	m := NewMCPServer(e)
	inR, inW := net.Pipe()
	outR, outW := net.Pipe()
	go m.ServeMCP(inR, outW)
	defer inW.Close()
	defer outR.Close()

	send := func(msg string) map[string]any {
		fmt.Fprintln(inW, msg)
		dec := json.NewDecoder(outR)
		var out map[string]any
		if err := dec.Decode(&out); err != nil {
			t.Fatalf("decode: %v", err)
		}
		return out
	}

	out := send(`{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}`)
	res := out["result"].(map[string]any)
	if res["protocolVersion"] != "2025-06-18" {
		t.Fatalf("protocol: %v", res)
	}
	si := res["serverInfo"].(map[string]any)
	if si["name"] != "q4tab" {
		t.Fatalf("serverInfo: %v", si)
	}

	out = send(`{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}`)
	tools := out["result"].(map[string]any)["tools"].([]any)
	if len(tools) != 6 {
		t.Fatalf("tools: %d", len(tools))
	}
	names := map[string]bool{}
	for _, tl := range tools {
		names[tl.(map[string]any)["name"].(string)] = true
	}
	for _, want := range []string{"complete", "lookup_lines", "lookup_symbol", "context", "learn", "status"} {
		if !names[want] {
			t.Fatalf("missing tool %s", want)
		}
	}

	out = send(`{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"complete","arguments":{"text":"package x\n\nfunc main() {\n\tfmt.Spr"}}}`)
	res, _ = out["result"].(map[string]any)
	if res == nil {
		t.Fatalf("complete call response: %v", out)
	}
	sc := res["structuredContent"].(map[string]any)
	items, _ := sc["items"].([]any)
	if len(items) == 0 {
		t.Fatalf("complete returned no items: %v", res)
	}

	out = send(`{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"lookup_lines","arguments":{"prefix":"func "}}}`)
	sc = out["result"].(map[string]any)["structuredContent"].(map[string]any)
	if len(sc["lines"].([]any)) == 0 {
		t.Fatalf("lookup_lines empty: %v", sc)
	}

	out = send(`{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"bogus","arguments":{}}}`)
	res = out["result"].(map[string]any)
	if res["isError"] != true {
		t.Fatalf("expected tool error: %v", res)
	}

	out = send(`{"jsonrpc":"2.0","id":6,"method":"bogus/method","params":{}}`)
	er := out["error"].(map[string]any)
	if er["code"].(float64) != -32601 {
		t.Fatalf("expected -32601, got %v", out)
	}

	// Notification gets no response: send it, then ping. Ping's reply
	// must be the next frame (no ghost response consumed).
	fmt.Fprintln(inW, `{"jsonrpc":"2.0","method":"notifications/initialized"}`)
	out = send(`{"jsonrpc":"2.0","id":7,"method":"ping"}`)
	if out["id"].(float64) != 7 {
		t.Fatalf("ping id mismatch: %v", out)
	}
}

func TestMCPHTTP(t *testing.T) {
	e := buildTestEngine(t)
	m := NewMCPServer(e)
	srv := httptest.NewServer(http.HandlerFunc(m.ServeHTTP))
	defer srv.Close()

	post := func(msg string) map[string]any {
		resp, err := http.Post(srv.URL, "application/json", strings.NewReader(msg))
		if err != nil {
			t.Fatal(err)
		}
		defer resp.Body.Close()
		if ct := resp.Header.Get("Content-Type"); ct != "application/json" {
			t.Fatalf("content-type %s", ct)
		}
		var out map[string]any
		json.NewDecoder(resp.Body).Decode(&out)
		return out
	}

	out := post(`{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"status","arguments":{}}}`)
	res := out["result"].(map[string]any)
	if _, ok := res["structuredContent"]; !ok {
		t.Fatalf("status result malformed: %v", res)
	}

	out = post(`{"jsonrpc":"2.0","id":2,"method":"server/discover","params":{}}`)
	vers := out["result"].(map[string]any)["protocolVersions"].([]any)
	if len(vers) == 0 {
		t.Fatal("server/discover returned no versions")
	}
}
