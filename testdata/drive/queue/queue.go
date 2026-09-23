package queue

import (
	"errors"
	"sync"
)

var ErrClosed = errors.New("queue closed")

type Job struct {
	ID      string
	Payload []byte
	Retries int
	Done    chan error
}

type Queue struct {
	mu     sync.Mutex
	jobs   []Job
	closed bool
	notify chan struct{}
}

func NewQueue() *Queue {
	return &Queue{notify: make(chan struct{}, 1)}
}

func (q *Queue) Push(j Job) error {
	q.mu.Lock()
	defer q.mu.Unlock()
	if q.closed {
		return ErrClosed
	}
	q.jobs = append(q.jobs, j)
	select {
	case q.notify <- struct{}{}:
	default:
	}
	return nil
}

func (q *Queue) Pop() (Job, error) {
	for {
		q.mu.Lock()
		if len(q.jobs) > 0 {
			j := q.jobs[0]
			q.jobs = q.jobs[1:]
			q.mu.Unlock()
			return j, nil
		}
		if q.closed {
			q.mu.Unlock()
			return Job{}, ErrClosed
		}
		q.mu.Unlock()
		<-q.notify
	}
}

func (q *Queue) Len() int {
	q.mu.Lock()
	defer q.mu.Unlock()
	return len(q.jobs)
}

func (q *Queue) Close() {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.closed = true
	close(q.notify)
}
