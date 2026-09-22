package engine

import (
	"fmt"
	"sync"
	"testing"
)

// Hammers Complete/UpdateDoc/Learn/Reject/Stats from many goroutines
// sharing a handful of documents. Run with -race: any unsynchronized
// access in the engine, caches, or vocab fails the run.
func TestConcurrentMixed(t *testing.T) {
	e := buildEngine(t, DefaultConfig())
	var wg sync.WaitGroup
	for i := 0; i < 32; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			uri := fmt.Sprintf("file:///c%d.go", i%6)
			text := "package x\n\nfunc worker() {\n\tfor i := 0; i < n; i++ {\n\t\tres"
			for j := 0; j < 40; j++ {
				e.UpdateDoc(uri, fmt.Sprintf("%s%d", text, j))
				items := e.Complete(uri, text, len(text))
				_ = items
				if j%5 == 0 {
					e.Learn(uri, "\t\tres = append(res, i)", 4)
				}
				if j%7 == 0 {
					e.Reject(uri, 4)
				}
				if j%9 == 0 {
					e.Stats()
					e.LookupLines("func ", 4)
				}
				if j%11 == 0 {
					e.CloseDoc(fmt.Sprintf("file:///ghost%d.go", i))
				}
			}
		}(i)
	}
	wg.Wait()
}

// Parallel completion throughput: the read path must scale with
// goroutines, not serialize behind the engine mutex.
func BenchmarkCompleteParallel(b *testing.B) {
	e := buildEngine(&testing.T{}, DefaultConfig())
	texts := []string{
		"package x\n\nfunc a() {\n\treturn fmt.Sprintf(\"lin",
		"package x\n\nfunc b() {\n\tfor i := 0; i < n; i++ {\n\t\tres",
		"package x\n\nfunc c() {\n\tcfg, err := loadConfig(\"x\")\n\tif err",
	}
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			txt := texts[i%len(texts)]
			e.Complete("file:///bench.go", txt, len(txt))
			i++
		}
	})
}
