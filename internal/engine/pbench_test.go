package engine

import "testing"

func BenchmarkCompleteParallelReal(b *testing.B) {
	e := benchEngine(b)
	text := "package main\n\nfunc check() error {\n\terr := doThing()\n\tif err !="
	e.UpdateDoc("file:///t.go", text)
	e.Flush()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			e.Complete("file:///t.go", text, len(text))
		}
	})
}
