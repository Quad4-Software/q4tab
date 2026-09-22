package engine

import "testing"

func BenchmarkLineIndex(b *testing.B) {
	e := benchEngine(b)
	prefix := "	if err !="
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.li.Complete(prefix, 8, e.cfg.ScanCap)
	}
}

func BenchmarkGenerate(b *testing.B) {
	e := benchEngine(b)
	text := "package main\n\nfunc check() error {\n\terr := doThing()\n\tif err !="
	uri := "file:///t.go"
	e.UpdateDoc(uri, text)
	e.Flush()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.generate("", uri, text, len(text), "\tif err !=")
	}
}
