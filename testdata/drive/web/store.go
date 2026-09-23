package web

import (
	"errors"
	"sync"
)

var ErrNotFound = errors.New("not found")

type Item struct {
	ID   string `json:"id"`
	Text string `json:"text"`
	Done bool   `json:"done"`
}

type Store struct {
	mu    sync.RWMutex
	items map[string]Item
}

func NewStore() *Store {
	return &Store{items: make(map[string]Item)}
}

func (s *Store) Get(id string) (Item, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	it, ok := s.items[id]
	if !ok {
		return Item{}, ErrNotFound
	}
	return it, nil
}

func (s *Store) Put(it Item) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.items[it.ID] = it
}

func (s *Store) List() []Item {
	s.mu.RLock()
	defer s.mu.RUnlock()
	out := make([]Item, 0, len(s.items))
	for _, it := range s.items {
		out = append(out, it)
	}
	return out
}
