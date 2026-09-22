package holdout

import (
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"time"
)

// Fixed eval holdout. These files are never indexed into the model.
// the eval harness masks line tails and scores how often the engine
// reproduces them. Keep them idiomatic and stable: do not edit without
// regenerating testdata/eval-baseline.json.

type server struct {
	ln   net.Listener
	addr string
	mux  *http.ServeMux
}

func newServer(addr string) *server {
	s := &server{addr: addr, mux: http.NewServeMux()}
	s.mux.HandleFunc("GET /healthz", func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})
	return s
}

func (s *server) start() error {
	ln, err := net.Listen("tcp", s.addr)
	if err != nil {
		return fmt.Errorf("listen %s: %w", s.addr, err)
	}
	s.ln = ln
	srv := &http.Server{
		Handler:           s.mux,
		ReadHeaderTimeout: 10 * time.Second,
	}
	return srv.Serve(ln)
}

func decode(r *http.Request, v any) error {
	defer r.Body.Close()
	dec := json.NewDecoder(r.Body)
	if err := dec.Decode(v); err != nil {
		return fmt.Errorf("decode body: %w", err)
	}
	return nil
}
