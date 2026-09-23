// Concurrency driver for q4tab TCP servers.
// usage: go run ./testdata/tcpscale [clients] [reqs] [addr]
package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net"
	"os"
	"sort"
	"strconv"
	"sync"
	"sync/atomic"
	"time"
)

func frame(w io.Writer, m map[string]any) error {
	b, _ := json.Marshal(m)
	if _, err := fmt.Fprintf(w, "Content-Length: %d\r\n\r\n", len(b)); err != nil {
		return err
	}
	_, err := w.Write(b)
	return err
}

func readMsg(r *bufio.Reader) (map[string]any, error) {
	n := 0
	for {
		l, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		if l == "\r\n" {
			break
		}
		var v int
		if _, err := fmt.Sscanf(l, "Content-Length: %d", &v); err == nil {
			n = v
		}
	}
	buf := make([]byte, n)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	var m map[string]any
	return m, json.Unmarshal(buf, &m)
}

func main() {
	clients, _ := strconv.Atoi(os.Args[1])
	reqs, _ := strconv.Atoi(os.Args[2])
	addr := os.Args[3]

	var mu sync.Mutex
	var lats []float64
	var errs atomic.Int64
	var wg sync.WaitGroup
	t0 := time.Now()
	for i := 0; i < clients; i++ {
		wg.Add(1)
		go func(wid int) {
			defer wg.Done()
			c, err := net.DialTimeout("tcp", addr, 10*time.Second)
			if err != nil {
				errs.Add(1)
				return
			}
			defer c.Close()
			r := bufio.NewReaderSize(c, 1<<16)
			uri := fmt.Sprintf("file:///w%d.go", wid%64)
			frame(c, map[string]any{"jsonrpc": "2.0", "id": 0, "method": "initialize", "params": map[string]any{}})
			readMsg(r)
			frame(c, map[string]any{"jsonrpc": "2.0", "method": "textDocument/didOpen", "params": map[string]any{
				"textDocument": map[string]any{"uri": uri, "languageId": "go", "version": 1,
					"text": "package x\n\nfunc worker() {\n\tfor i := 0; i < n; i++ {\n\t\tres"}}})
			for j := 0; j < reqs; j++ {
				st := time.Now()
				frame(c, map[string]any{"jsonrpc": "2.0", "id": j + 1,
					"method": "textDocument/inlineCompletion",
					"params": map[string]any{"textDocument": map[string]any{"uri": uri},
						"position": map[string]any{"line": 4, "character": 4},
						"context":  map[string]any{"triggerKind": 1}}})
				m, err := readMsg(r)
				d := time.Since(st).Seconds() * 1e3
				if err != nil {
					errs.Add(1)
					return
				}
				if _, ok := m["result"]; !ok {
					errs.Add(1)
					continue
				}
				mu.Lock()
				lats = append(lats, d)
				mu.Unlock()
			}
		}(i)
	}
	wg.Wait()
	wall := time.Since(t0)
	sort.Float64s(lats)
	pct := func(p float64) float64 { return lats[min(len(lats)-1, int(float64(len(lats))*p))] }
	fmt.Printf("%d clients x %d reqs: wall=%.1fs n=%d errs=%d p50=%.2fms p95=%.2fms p99=%.2fms max=%.1fms rps=%.0f\n",
		clients, reqs, wall.Seconds(), len(lats), errs.Load(),
		pct(.5), pct(.95), pct(.99), lats[len(lats)-1], float64(len(lats))/wall.Seconds())
}
