package lsp

import (
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"net/http/pprof"
	"os"
	"runtime"
	"strings"
	"sync"
	"time"

	"q4tab/internal/engine"
)

// ServerOpts controls the network transports' security posture.
type ServerOpts struct {
	Token   string   // required bearer token. Empty = open (loopback only)
	Rate    float64  // sustained requests/sec per client IP. 0 = unlimited
	Burst   int      // rate-limit burst. 0 = 4x Rate
	MaxConc int      // global concurrent request cap. 0 = 256
	Roots   []string // corpus roots. MCP path reads restricted to these
}

// ListenAndServe accepts LSP connections over TCP. Each connection gets
// its own Server (isolated document store) sharing the engine, whose
// internal locking makes concurrent access safe. Content-Length framing
// is identical to the stdio transport, so any LSP client that can open
// a socket works unchanged. When opts.Token is set, the connection must
// send q4/auth before other requests are served.
func ListenAndServe(addr string, eng *engine.Engine, deltaPath string, opts ServerOpts) error {
	ln, err := net.Listen("tcp", addr)
	if err != nil {
		return err
	}
	for {
		c, err := ln.Accept()
		if err != nil {
			return err
		}
		go func() {
			defer c.Close()
			srv := NewServer(eng, NewConn(c, c))
			srv.SetDeltaPath(deltaPath)
			srv.reqToken = opts.Token
			srv.Run()
		}()
	}
}

// HTTPHandler returns an http.Handler exposing the engine:
//
//	POST /rpc      single JSON-RPC message against a shared LSP session
//	POST /mcp      Model Context Protocol (streamable-HTTP shape)
//	GET  /status   engine stats as JSON
//	GET  /healthz  liveness (unauthenticated, for load balancers)
//
// Security posture for a public deployment: set opts.Token (all
// endpoints except /healthz then require it), put TLS in front, and
// rely on the per-IP rate limit and global concurrency cap to blunt
// abuse. The model contains source code. Treat responses as sensitive.
func HTTPHandler(eng *engine.Engine, deltaPath string, opts ServerOpts) http.Handler {
	srv := NewServer(eng, nil)
	srv.SetDeltaPath(deltaPath)
	mcp := NewMCPServer(eng)
	mcp.allowFS = len(opts.Roots) > 0
	mcp.roots = opts.Roots

	maxConc := opts.MaxConc
	if maxConc <= 0 {
		maxConc = 256
	}
	sem := make(chan struct{}, maxConc)
	lim := newRateLimiter(opts.Rate, opts.Burst)

	secure := func(h http.HandlerFunc, authed bool) http.HandlerFunc {
		return func(w http.ResponseWriter, r *http.Request) {
			if authed && !checkToken(r, opts.Token) {
				http.Error(w, "unauthorized", http.StatusUnauthorized)
				return
			}
			if !lim.allow(clientIP(r)) {
				http.Error(w, "rate limited", http.StatusTooManyRequests)
				return
			}
			select {
			case sem <- struct{}{}:
				defer func() { <-sem }()
			default:
				http.Error(w, "server busy", http.StatusServiceUnavailable)
				return
			}
			h(w, r)
		}
	}

	mux := http.NewServeMux()
	mux.HandleFunc("POST /rpc", secure(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(io.LimitReader(r.Body, 1<<22))
		if err != nil {
			http.Error(w, "read body", http.StatusBadRequest)
			return
		}
		var m Message
		if err := json.Unmarshal(body, &m); err != nil {
			writeRPCError(w, nil, -32700, "parse error: "+err.Error())
			return
		}
		user := requestUser(r, opts.Token)
		if len(m.ID) == 0 {
			srv.notify(&m)
			w.WriteHeader(http.StatusAccepted)
			return
		}
		res, rerr := srv.dispatchAs(user, &m)
		writeRPC(w, m.ID, res, rerr)
	}, true))
	mux.HandleFunc("POST /mcp", secure(func(w http.ResponseWriter, r *http.Request) {
		mcp.serveHTTPAs(requestUser(r, opts.Token), w, r)
	}, true))
	mux.HandleFunc("GET /status", secure(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(eng.Stats())
	}, true))
	mux.HandleFunc("GET /healthz", secure(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte("ok"))
	}, false))
	// Optional pprof endpoints for profiling live servers. Gated on an
	// env var so public deployments never expose them by accident.
	if os.Getenv("Q4TAB_PPROF") != "" {
		runtime.SetMutexProfileFraction(5)
		runtime.SetBlockProfileRate(1000)
		mux.HandleFunc("/debug/pprof/", pprof.Index)
		mux.HandleFunc("/debug/pprof/cmdline", pprof.Cmdline)
		mux.HandleFunc("/debug/pprof/profile", pprof.Profile)
		mux.HandleFunc("/debug/pprof/symbol", pprof.Symbol)
		mux.HandleFunc("/debug/pprof/trace", pprof.Trace)
		mux.Handle("/debug/pprof/goroutine", pprof.Handler("goroutine"))
		mux.Handle("/debug/pprof/heap", pprof.Handler("heap"))
		mux.Handle("/debug/pprof/mutex", pprof.Handler("mutex"))
		mux.Handle("/debug/pprof/block", pprof.Handler("block"))
	}
	return mux
}

// checkToken verifies the bearer token with a constant-time compare.
// Empty configured token means open access (loopback deployments).
func checkToken(r *http.Request, token string) bool {
	if token == "" {
		return true
	}
	got := ""
	if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
		got = h[7:]
	} else if h := r.Header.Get("X-Q4-Token"); h != "" {
		got = h
	}
	return subtle.ConstantTimeCompare([]byte(got), []byte(token)) == 1
}

// requestUser derives the tenant id: an explicit header wins, else the
// token hash (so distinct tenants never see each other's overlays).
func requestUser(r *http.Request, token string) string {
	if u := r.Header.Get("X-Q4-User"); u != "" {
		return "u:" + u[:min(len(u), 64)]
	}
	if token != "" {
		got := ""
		if h := r.Header.Get("Authorization"); strings.HasPrefix(h, "Bearer ") {
			got = h[7:]
		} else {
			got = r.Header.Get("X-Q4-Token")
		}
		if got != "" {
			sum := sha256.Sum256([]byte(got))
			return "t:" + hex.EncodeToString(sum[:8])
		}
	}
	return ""
}

func clientIP(r *http.Request) string {
	if xff := r.Header.Get("X-Forwarded-For"); xff != "" {
		if i := strings.IndexByte(xff, ','); i >= 0 {
			return strings.TrimSpace(xff[:i])
		}
		return strings.TrimSpace(xff)
	}
	host, _, err := net.SplitHostPort(r.RemoteAddr)
	if err != nil {
		return r.RemoteAddr
	}
	return host
}

// rateLimiter is a per-key token bucket. 0 rate disables it.
type rateLimiter struct {
	mu    sync.Mutex
	rate  float64
	burst float64
	m     map[string]*bucket
}

type bucket struct {
	tokens float64
	at     time.Time
}

func newRateLimiter(rate float64, burst int) *rateLimiter {
	if rate <= 0 {
		rate = 0 // disabled
	}
	b := float64(burst)
	if b <= 0 {
		b = rate * 4
		if b < 20 {
			b = 20
		}
	}
	return &rateLimiter{rate: rate, burst: b, m: map[string]*bucket{}}
}

func (rl *rateLimiter) allow(key string) bool {
	if rl.rate <= 0 {
		return true
	}
	rl.mu.Lock()
	defer rl.mu.Unlock()
	b := rl.m[key]
	if b == nil {
		b = &bucket{tokens: rl.burst, at: time.Now()}
		rl.m[key] = b
	}
	now := time.Now()
	b.tokens += now.Sub(b.at).Seconds() * rl.rate
	if b.tokens > rl.burst {
		b.tokens = rl.burst
	}
	b.at = now
	if b.tokens < 1 {
		return false
	}
	b.tokens--
	// Lazy eviction so the map does not grow forever.
	if len(rl.m) > 10000 {
		cut := now.Add(-10 * time.Minute)
		for k, v := range rl.m {
			if v.at.Before(cut) {
				delete(rl.m, k)
			}
		}
	}
	return true
}

func writeRPC(w http.ResponseWriter, id json.RawMessage, result any, rerr *rpcError) {
	w.Header().Set("Content-Type", "application/json")
	out := struct {
		JSONRPC string          `json:"jsonrpc"`
		ID      json.RawMessage `json:"id"`
		Result  any             `json:"result,omitempty"`
		Error   any             `json:"error,omitempty"`
	}{JSONRPC: "2.0", ID: id, Result: result}
	if rerr != nil {
		out.Result = nil
		out.Error = map[string]any{"code": rerr.code, "message": rerr.msg}
	}
	json.NewEncoder(w).Encode(out)
}

func writeRPCError(w http.ResponseWriter, id json.RawMessage, code int, msg string) {
	writeRPC(w, id, nil, &rpcError{code, msg})
}

// HTTPServe runs the HTTP transport with sane timeouts for an
// editor-facing service.
func HTTPServe(addr string, eng *engine.Engine, deltaPath string, opts ServerOpts) error {
	s := &http.Server{
		Addr:              addr,
		Handler:           HTTPHandler(eng, deltaPath, opts),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		WriteTimeout:      30 * time.Second,
		IdleTimeout:       120 * time.Second,
	}
	return s.ListenAndServe()
}
