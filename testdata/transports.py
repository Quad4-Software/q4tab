#!/usr/bin/env python3
"""Drive q4tab over TCP-LSP, HTTP-RPC, and MCP; report latencies."""
import json, socket, struct, sys, time, urllib.request

TCP = ("127.0.0.1", 7917)
HTTP = "http://127.0.0.1:7918"

DOC = ("file:///drive.go", """package main

import "fmt"

func main() {
	cfg, err := loadConfig("app.yaml")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Spr""")

def frame(msg):
    body = json.dumps(msg).encode()
    return b"Content-Length: %d\r\n\r\n" % len(body) + body

class TCPClient:
    def __init__(self):
        self.s = socket.create_connection(TCP, timeout=5)
        self.buf = b""
        self.next = 1
    def notify(self, method, params):
        self.s.sendall(frame({"jsonrpc":"2.0","method":method,"params":params}))
    def call(self, method, params):
        i = self.next; self.next += 1
        self.s.sendall(frame({"jsonrpc":"2.0","id":i,"method":method,"params":params}))
        return self._read()
    def _read(self):
        while b"\r\n\r\n" not in self.buf:
            self.buf += self.s.recv(65536)
        head, self.buf = self.buf.split(b"\r\n\r\n", 1)
        n = int(head.split(b"Content-Length: ")[1].split(b"\r\n")[0])
        while len(self.buf) < n:
            self.buf += self.s.recv(65536)
        body, self.buf = self.buf[:n], self.buf[n:]
        return json.loads(body)

def http_post(path, msg):
    req = urllib.request.Request(HTTP + path, json.dumps(msg).encode(),
                                 {"Content-Type": "application/json"})
    body = urllib.request.urlopen(req, timeout=10).read()
    return json.loads(body) if body else {}

def percentiles(ts):
    ts = sorted(ts)
    return ts[len(ts)//2], ts[int(len(ts)*.95)], ts[-1]

def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 200

    tcp = TCPClient()
    tcp.call("initialize", {})
    tcp.notify("textDocument/didOpen", {"textDocument": {
        "uri": DOC[0], "languageId": "go", "version": 1, "text": DOC[1]}})

    http_post("/rpc", {"jsonrpc":"2.0","id":1,"method":"initialize","params":{}})
    http_post("/rpc", {"jsonrpc":"2.0","method":"textDocument/didOpen","params":
        {"textDocument": {"uri": DOC[0], "languageId": "go", "version": 1, "text": DOC[1]}}})
    http_post("/mcp", {"jsonrpc":"2.0","id":1,"method":"initialize","params":
        {"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"drive","version":"0"}}})

    pos = {"line": 9, "character": 7}
    req = {"textDocument": {"uri": DOC[0]}, "position": pos,
           "context": {"triggerKind": 1}}

    lat = {"tcp": [], "http": [], "mcp": []}
    t0 = time.time()
    for i in range(rounds):
        t = time.perf_counter()
        r = tcp.call("textDocument/inlineCompletion", req)
        assert r.get("result"), r
        lat["tcp"].append(time.perf_counter() - t)

        t = time.perf_counter()
        r = http_post("/rpc", {"jsonrpc":"2.0","id":i+10,
            "method":"textDocument/inlineCompletion","params":req})
        assert "result" in r, r
        lat["http"].append(time.perf_counter() - t)

        t = time.perf_counter()
        r = http_post("/mcp", {"jsonrpc":"2.0","id":i+10,"method":"tools/call",
            "params":{"name":"complete","arguments":{"text": DOC[1],
                      "uri": "mcp://drive", "offset": len(DOC[1])}}})
        assert "result" in r, r
        lat["mcp"].append(time.perf_counter() - t)

    wall = time.time() - t0
    print(f"rounds={rounds} wall={wall:.1f}s")
    for k, ts in lat.items():
        p50, p95, mx = percentiles(ts)
        print(f"{k:5s} p50={p50*1e3:6.2f}ms p95={p95*1e3:6.2f}ms max={mx*1e3:6.2f}ms")

    # sample one item from each path for sanity
    r = tcp.call("textDocument/inlineCompletion", req)
    items = r["result"]["items"]
    print("sample tcp item:", json.dumps(items[0]["insertText"])[:80])
    r = http_post("/mcp", {"jsonrpc":"2.0","id":99,"method":"tools/call",
        "params":{"name":"lookup_lines","arguments":{"prefix":"func main()","limit":3}}})
    print("mcp lookup:", json.dumps(r["result"]["structuredContent"])[:200])

if __name__ == "__main__":
    main()
