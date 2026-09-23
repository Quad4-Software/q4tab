package queue

import (
	"encoding/json"
	"errors"
	"net/http"
)

type Server struct {
	q *Queue
}

func NewServer(q *Queue) *Server { return &Server{q: q} }

func (s *Server) handlePush(w http.ResponseWriter, r *http.Request) {
	var j Job
	if err := json.NewDecoder(r.Body).Decode(&j); err != nil {
		http.Error(w, "bad job", http.StatusBadRequest)
		return
	}
	if err := s.q.Push(j); err != nil {
		if errors.Is(err, ErrClosed) {
			http.Error(w, "closed", http.StatusGone)
			return
		}
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(http.StatusCreated)
	json.NewEncoder(w).Encode(map[string]string{"id": j.ID})
}

func (s *Server) handleLen(w http.ResponseWriter, r *http.Request) {
	json.NewEncoder(w).Encode(map[string]int{"len": s.q.Len()})
}
