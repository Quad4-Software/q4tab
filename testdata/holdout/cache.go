package holdout

import (
	"sort"
	"sync"
)

// Fixed eval holdout file. See server.go for the rules.

type lruCache struct {
	mu      sync.Mutex
	max     int
	entries map[string]*lruEntry
	order   []string
}

type lruEntry struct {
	key   string
	value int64
}

func newLRU(max int) *lruCache {
	return &lruCache{max: max, entries: make(map[string]*lruEntry)}
}

func (c *lruCache) get(key string) (int64, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[key]
	if !ok {
		return 0, false
	}
	return e.value, true
}

func (c *lruCache) set(key string, value int64) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if _, ok := c.entries[key]; !ok {
		c.entries[key] = &lruEntry{key: key, value: value}
		c.order = append(c.order, key)
	} else {
		c.entries[key].value = value
	}
	for len(c.order) > c.max {
		old := c.order[0]
		c.order = c.order[1:]
		delete(c.entries, old)
	}
}

func (c *lruCache) keys() []string {
	c.mu.Lock()
	defer c.mu.Unlock()
	out := append([]string(nil), c.order...)
	sort.Strings(out)
	return out
}
