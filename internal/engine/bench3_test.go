package engine

import (
	"fmt"
	"testing"
)

// Scenario ladder, easy to complex, plus per-method benchmarks.
// All run against the real model when bin/model.bin is present.

// Easy: the most common Go completion prefix in any corpus.
func BenchmarkCompleteEasy(b *testing.B) {
	e := benchEngine(b)
	text := "package main\n\nfunc f() error {\n\tif err !="
	uri := "file:///t.go"
	e.UpdateDoc(uri, text)
	e.Flush()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Complete(uri, text, len(text))
	}
}

// Medium: mid-file, mid-expression, suffix present after the cursor.
func BenchmarkCompleteFIM(b *testing.B) {
	e := benchEngine(b)
	text := "package main\n\nfunc f() {\n\tres := compute(a, )\n\tuse(res)\n}\n"
	uri := "file:///t.go"
	e.UpdateDoc(uri, text)
	e.Flush()
	off := len("\tres := compute(a, ") + len("package main\n\nfunc f() {\n")
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Complete(uri, text, off)
	}
}

// Hard: after several open blocks, multi-line generation territory.
func BenchmarkCompleteNested(b *testing.B) {
	e := benchEngine(b)
	text := `package main

func process(items []Item) error {
	for _, it := range items {
		if it.Ready() {
			for j := 0; j < it.Attempts(); j++ {
				if err := it.Run(j);`
	uri := "file:///t.go"
	e.UpdateDoc(uri, text)
	e.Flush()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Complete(uri, text, len(text))
	}
}

// Hardest: cold context in a file the model has never seen, deep in.
func BenchmarkCompleteCold(b *testing.B) {
	e := benchEngine(b)
	text := `package novel

type zorkmidAssembler struct {
	quuxers []int
}

func (z *zorkmidAssembler) splice(idx int) error {
	for _, q := range z.quuxers {
		if q >`
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.Complete("file:///novel.go", text, len(text))
	}
}

// UpdateDoc churn: full edit-to-indexed cost (enqueue + drain) under
// continuous editing.
func BenchmarkUpdateDoc(b *testing.B) {
	e := benchEngine(b)
	text := "package main\n\nfunc f() {\n\tvar x = 1\n\t_ = x\n}\n"
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.UpdateDoc(fmt.Sprintf("file:///d%d.go", i%16), text)
		e.Flush()
	}
}

func BenchmarkLookupLines(b *testing.B) {
	e := benchEngine(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.LookupLines("\tif err", 8)
	}
}

func BenchmarkLookupSymbol(b *testing.B) {
	e := benchEngine(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.LookupSymbol("main", 8)
	}
}

// Learn path cost: journal + cache add under the write lock.
func BenchmarkLearn(b *testing.B) {
	e := benchEngine(b)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		e.LearnFor(fmt.Sprintf("u%d", i%4), "file:///t.go", "\treturn nil", 3)
	}
}

// Tenant-complete throughput: overlays must not serialize the world.
func BenchmarkCompleteTenants(b *testing.B) {
	e := benchEngine(b)
	text := "package main\n\nfunc f() error {\n\tif err !="
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			e.CompleteFor(fmt.Sprintf("t%d", i%64), "file:///t.go", text, len(text))
			i++
		}
	})
}
