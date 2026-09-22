package engine

import "testing"

func benchEngine(b *testing.B) *Engine {
	m, li, err := Load("../../bin/model.bin")
	if err != nil {
		b.Skip(err)
	}
	e := New(DefaultConfig())
	e.SetModel(m, li)
	return e
}

func BenchmarkComplete(b *testing.B) {
	e := benchEngine(b)
	text := "package main\n\nfunc check() error {\n\terr := doThing()\n\tif err !="
	uri := "file:///t.go"
	e.UpdateDoc(uri, text)
	e.Flush()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Complete(uri, text, len(text))
	}
}

func BenchmarkCompleteMidFile(b *testing.B) {
	e := benchEngine(b)
	text := "package main\n\nfunc check() error {\n\tif err := doThing(); e"
	uri := "file:///t.go"
	e.UpdateDoc(uri, text)
	e.Flush()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Complete(uri, text, len(text))
	}
}
