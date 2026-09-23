package queue

import (
	"errors"
	"log"
)

type Worker struct {
	q    *Queue
	name string
	done chan struct{}
}

func NewWorker(name string, q *Queue) *Worker {
	return &Worker{name: name, q: q, done: make(chan struct{})}
}

func (w *Worker) Run() {
	for {
		job, err := w.q.Pop()
		if err != nil {
			if errors.Is(err, ErrClosed) {
				return
			}
			log.Printf("worker %s: pop: %v", w.name, err)
			continue
		}
		w.process(job)
	}
}

func (w *Worker) Stop() { close(w.done) }

func (w *Worker) process(j Job) {
	j.Retries++
	j.Done <- nil
}
