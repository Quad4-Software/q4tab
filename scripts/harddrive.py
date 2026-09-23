#!/usr/bin/env python3
"""Hard drive: multi-file member memory, arg synthesis, concurrency."""
import json, socket, statistics, sys, threading, time

TCP = ("127.0.0.1", 7917)

def frame(msg):
    body = json.dumps(msg).encode()
    return b"Content-Length: %d\r\n\r\n" % len(body) + body

class TCPClient:
    def __init__(self):
        self.s = socket.create_connection(TCP, timeout=10)
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

BASE = "/run/media/user1/projects/shipper/q4tab/testdata/drive/queue"

def load(name):
    return open(BASE + "/" + name).read()

def probe(c, uri, text, name, expect=None):
    """Open doc truncated at cursor; complete at last line end."""
    lines = text.split("\n")
    line = len(lines) - 1
    ch = len(lines[-1])
    c.notify("textDocument/didOpen", {"textDocument": {
        "uri": uri, "languageId": "go", "version": 1, "text": text}})
    t = time.perf_counter()
    r = c.call("textDocument/inlineCompletion", {
        "textDocument": {"uri": uri},
        "position": {"line": line, "character": ch},
        "context": {"triggerKind": 1}})
    ms = (time.perf_counter() - t) * 1000
    items = (r.get("result") or {}).get("items") or []
    c.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
    top = [(it.get("insertText") or it.get("text") or "?",
            (it.get("data") or {}).get("src", "?")) for it in items[:5]]
    print(f"\n=== {name} ({ms:.0f}ms) ===")
    print(f"  cursor: {lines[-1]!r}")
    for txt, src in top:
        print(f"    [{src}] {txt!r}")
    if expect:
        flat = [t for t, _ in top]
        hit = any(any(e in t for e in [exp]) for exp in expect for t in flat)
        print(f"  EXPECT {expect}: {'HIT' if hit else 'MISS'}")
    return top, ms

def main():
    c = TCPClient()
    c.call("initialize", {})

    # open the fixture package in declaration order
    for f in ["queue.go", "worker.go", "api.go"]:
        c.notify("textDocument/didOpen", {"textDocument": {
            "uri": f"file://{BASE}/{f}", "languageId": "go",
            "version": 1, "text": load(f)}})
    time.sleep(0.6)  # let async doc builds settle

    P = "file:///probe.go"

    probe(c, P, """package queue

func (w *Worker) drain(j Job) {
	j.""", "field completion: j. (Job param)",
        expect=["ID", "Payload", "Retries", "Done"])

    probe(c, P, """package queue

func (w *Worker) tick() {
	w.q.""", "member via field: w.q. (*Queue field)",
        expect=["Push(", "Pop(", "Len(", "Close("])

    probe(c, P, """package queue

func (w *Worker) tick() {
	w.q.jobs[0].""", "index chain: w.q.jobs[0]. (Job via []Job)")

    probe(c, P, """package queue

import (
	"encoding/json"
	"net/http"
)

func (s *Server) handleGet(w http.ResponseWriter, r *http.Request) {
	var j Job
	if err := json.NewDecoder(r.Body).""", "call result: json.NewDecoder(r.Body).",
        expect=["Decode("])

    probe(c, P, """package queue

import (
	"errors"
	"net/http"
)

func (s *Server) handlePush2(w http.ResponseWriter, r *http.Request) {
	var j Job
	if err := s.q.Push(j); err != nil {
		if errors.Is(err,""", "sentinel arg: errors.Is(err,",
        expect=["ErrClosed"])

    probe(c, P, """package queue

import "net/http"

func (s *Server) handleX(w http.ResponseWriter, r *http.Request) {
	http.Error(w,""", "non-sentinel arg: http.Error(w,")

    probe(c, P, """package queue

import "net/http"

func (s *Server) routes(mux *http.ServeMux) {
	mux.HandleFunc("/jobs", s.handlePush)
	mux.HandleFunc("/len", s.handleLen)
	mux.HandleFunc(""", "string literal arg: mux.HandleFunc(")

    probe(c, P, """package queue

import "log"

func (w *Worker) once() {
	job, err := w.q.Pop()
	if err != nil {""", "multiline: if err != nil {",
        expect=["return", "log"])

    probe(c, P, """package queue

func (w *Worker) scan() {
	for _, j := range w.q.jobs {
		j.""", "range var: j. inside for-range over []Job",
        expect=["ID", "Payload"])

    probe(c, P, """package queue

func (w *Worker) finish(j Job) {
	j.Done <-""", "chan send: j.Done <-")

    probe(c, P, """package queue

func boot() {
	NewQueue().""", "ctor chain: NewQueue().")

    probe(c, "file:///probe.py", """import os
import sys

def main():
    os.""", "python: os. (no crash, corpus fallback)")

    # concurrency stress: N threads x M probes while edits stream
    print("\n=== stress: 24 threads x 40 reqs ===")
    lats, errs = [], []
    lock = threading.Lock()
    texts = [
        "package queue\n\nfunc (w *Worker) a() {\n\tw.q.",
        "package queue\n\nfunc (w *Worker) b(j Job) {\n\tj.",
        "package queue\n\nimport \"errors\"\n\nfunc f(err error) {\n\t_ = errors.Is(err,",
    ]
    def worker(tid):
        cc = TCPClient()
        cc.call("initialize", {})
        uri = f"file:///stress{tid}.go"
        for i in range(40):
            t = texts[i % len(texts)]
            cc.notify("textDocument/didChange", {"textDocument": {
                "uri": uri, "version": i + 2},
                "contentChanges": [{"text": t}]})
            t0 = time.perf_counter()
            try:
                r = cc.call("textDocument/inlineCompletion", {
                    "textDocument": {"uri": uri},
                    "position": {"line": t.count(chr(10)), "character": len(t.split(chr(10))[-1])},
                    "context": {"triggerKind": 1}})
                with lock:
                    lats.append((time.perf_counter() - t0) * 1000)
                if not (r.get("result") or {}).get("items") and r.get("error"):
                    with lock:
                        errs.append(r["error"])
            except Exception as e:
                with lock:
                    errs.append(str(e))
        cc.notify("textDocument/didClose", {"textDocument": {"uri": uri}})
    ths = [threading.Thread(target=worker, args=(i,)) for i in range(24)]
    t0 = time.time()
    for t in ths: t.start()
    for t in ths: t.join()
    lats.sort()
    n = len(lats)
    print(f"  {n} reqs in {time.time()-t0:.1f}s "
          f"p50={lats[n//2]:.1f}ms p95={lats[int(n*.95)]:.1f}ms "
          f"p99={lats[int(n*.99)]:.1f}ms max={lats[-1]:.0f}ms errors={len(errs)}")
    for e in errs[:3]:
        print(f"    err: {e}")

if __name__ == "__main__":
    main()
