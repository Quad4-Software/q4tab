package lsp

import (
	"encoding/json"
	"net"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// Token auth: every endpoint except /healthz requires the bearer token.
func TestHTTPAuth(t *testing.T) {
	e := buildTestEngine(t)
	h := HTTPHandler(e, "", ServerOpts{Token: "sekrit"})
	srv := httptest.NewServer(h)
	defer srv.Close()

	post := func(auth, body string) *http.Response {
		req, _ := http.NewRequest("POST", srv.URL+"/rpc", strings.NewReader(body))
		if auth != "" {
			req.Header.Set("Authorization", "Bearer "+auth)
		}
		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			t.Fatal(err)
		}
		return resp
	}

	resp := post("", `{"jsonrpc":"2.0","id":1,"method":"q4/status"}`)
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("no token: got %d", resp.StatusCode)
	}
	resp = post("wrong", `{"jsonrpc":"2.0","id":1,"method":"q4/status"}`)
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("bad token: got %d", resp.StatusCode)
	}
	resp = post("sekrit", `{"jsonrpc":"2.0","id":1,"method":"q4/status"}`)
	var out map[string]any
	json.NewDecoder(resp.Body).Decode(&out)
	resp.Body.Close()
	if resp.StatusCode != 200 || out["result"] == nil {
		t.Fatalf("good token: %d %v", resp.StatusCode, out)
	}
	// X-Q4-Token is accepted as an alternate header.
	req, _ := http.NewRequest("POST", srv.URL+"/rpc", strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"q4/status"}`))
	req.Header.Set("X-Q4-Token", "sekrit")
	resp, _ = http.DefaultClient.Do(req)
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatalf("X-Q4-Token: got %d", resp.StatusCode)
	}
	// healthz stays open for load balancers. Status does not.
	resp, _ = http.Get(srv.URL + "/healthz")
	resp.Body.Close()
	if resp.StatusCode != 200 {
		t.Fatal("healthz should be unauthenticated")
	}
	resp, _ = http.Get(srv.URL + "/status")
	resp.Body.Close()
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("/status without token: %d", resp.StatusCode)
	}
}

// Per-IP rate limiting: sustained traffic above the rate gets 429s.
func TestHTTPRateLimit(t *testing.T) {
	e := buildTestEngine(t)
	h := HTTPHandler(e, "", ServerOpts{Rate: 5, Burst: 3})
	srv := httptest.NewServer(h)
	defer srv.Close()

	codes := map[int]int{}
	for i := 0; i < 20; i++ {
		resp, err := http.Post(srv.URL+"/rpc", "application/json",
			strings.NewReader(`{"jsonrpc":"2.0","id":1,"method":"q4/status"}`))
		if err != nil {
			t.Fatal(err)
		}
		resp.Body.Close()
		codes[resp.StatusCode]++
	}
	if codes[429] == 0 {
		t.Fatalf("no 429s under 20 rapid requests at rate=5/burst=3: %v", codes)
	}
	if codes[200] < 3 {
		t.Fatalf("burst did not pass: %v", codes)
	}
}

// MCP path reads: denied entirely when no roots are configured, and
// confined to the roots when they are.
func TestMCPPathRestriction(t *testing.T) {
	e := buildTestEngine(t)
	dir := t.TempDir()

	callPath := func(m *MCPServer, path string) map[string]any {
		args, _ := json.Marshal(map[string]any{"path": path})
		res := m.callTool("complete", args)
		return res.(map[string]any)
	}

	// HTTP posture: allowFS off -> any path rejected.
	m := NewMCPServer(e)
	out := callPath(m, "/etc/hostname")
	if out["isError"] != true {
		t.Fatalf("path read allowed without roots: %v", out)
	}

	// Roots configured: inside ok, outside rejected, traversal rejected.
	m2 := NewMCPServer(e)
	m2.allowFS = true
	m2.roots = []string{dir}
	inner := filepath.Join(dir, "x.go")
	if err := os.WriteFile(inner, []byte("package x\n\nfunc main() {\n\tfmt.Spr"), 0o644); err != nil {
		t.Fatal(err)
	}
	out = callPath(m2, inner)
	if out["isError"] == true {
		t.Fatalf("path inside root rejected: %v", out)
	}
	out = callPath(m2, "/etc/hostname")
	if out["isError"] != true {
		t.Fatalf("path outside root allowed: %v", out)
	}
	out = callPath(m2, dir+"/../escape")
	if out["isError"] != true {
		t.Fatalf("traversal allowed: %v", out)
	}
}

// TCP auth: a server requiring a token rejects requests until q4/auth.
func TestTCPAuth(t *testing.T) {
	e := buildTestEngine(t)
	ln, err := net.Listen("tcp", "127.0.0.1:0")
	if err != nil {
		t.Fatal(err)
	}
	defer ln.Close()
	go func() {
		for {
			c, err := ln.Accept()
			if err != nil {
				return
			}
			go func() {
				defer c.Close()
				srv := NewServer(e, NewConn(c, c))
				srv.reqToken = "sekrit"
				srv.Run()
			}()
		}
	}()

	c := dialTest(t, ln.Addr().String())
	defer c.conn.Close()

	res := c.call(t, "q4/lookupSymbol", map[string]any{"name": "x"})
	var env struct {
		Error *struct {
			Code    int    `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	json.Unmarshal(res, &env)
	if env.Error == nil || env.Error.Code != -32001 {
		t.Fatalf("unauthenticated request served: %s", res)
	}

	res = c.call(t, "q4/auth", map[string]any{"token": "nope"})
	json.Unmarshal(res, &env)
	if env.Error == nil {
		t.Fatalf("bad token accepted: %s", res)
	}

	res = c.call(t, "q4/auth", map[string]any{"token": "sekrit"})
	env.Error = nil
	json.Unmarshal(res, &env)
	if env.Error != nil {
		t.Fatalf("good token rejected: %s", res)
	}

	// Same connection now serves.
	res = c.call(t, "q4/status", map[string]any{})
	var ok struct {
		Result map[string]any `json:"result"`
	}
	json.Unmarshal(res, &ok)
	if ok.Result == nil {
		t.Fatalf("authed request failed: %s", res)
	}
}
