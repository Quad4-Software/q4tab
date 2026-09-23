package lines

import "testing"

func TestGramIndex(t *testing.T) {
	lb := NewBuilder()
	gb := NewGramBuilder()
	// "if err != nil {" -> "return err" pairs repeat. A pairing seen
	// only once must not survive compaction (min count 2).
	src := []byte(`if err != nil {
	return err
}
x := 1
if err != nil {
	return err
}
y := 2
if cond {
	solo line here
}
`)
	lb.AddFile(src)
	lb.AddFile(src)
	gb.AddFile(src)
	gb.AddFile(src) // same file twice: pairs repeat
	// A context whose transition appears once is gated out.
	once := []byte("uniqctx1 := call()\n\tuniqresult1 += 1\n")
	lb.AddFile(once)
	gb.AddFile(once)
	li := lb.Compact()
	g := gb.Compact(li)
	if len(g.Keys) == 0 {
		t.Fatal("no gram keys built")
	}
	toks, cnts, tot := g.Next("if err != nil {")
	if len(toks) == 0 {
		t.Fatal("no next line for 'if err != nil {'")
	}
	if got := li.Key(int(toks[0])); got != "return err" {
		t.Fatalf("top next line = %q want %q", got, "return err")
	}
	if tot <= 0 || cnts[0] < 2 {
		t.Fatalf("bad counts cnt=%d tot=%d", cnts[0], tot)
	}
	// Unknown context returns nothing.
	if toks, _, _ = g.Next("never seen this line"); len(toks) != 0 {
		t.Fatal("unknown context returned rows")
	}
	// Trigram key over two previous lines.
	toks, _, _ = g.Next("x := 1", "if err != nil {")
	if len(toks) == 0 {
		t.Fatal("trigram context missed")
	}
	// Singleton transition gated out.
	if toks, _, _ = g.Next("uniqctx1 := call()"); len(toks) != 0 {
		t.Fatal("singleton gram survived min-count pruning")
	}
}

func TestGramNilSafe(t *testing.T) {
	var g *GramIndex
	if toks, cnts, tot := g.Next("a"); toks != nil || cnts != nil || tot != 0 {
		t.Fatal("nil gram index returned data")
	}
}

// TestGramTighten: after Tighten, contexts with no repeated transition
// drop out and new contexts stop accumulating.
func TestGramTighten(t *testing.T) {
	lb := NewBuilder()
	gb := NewGramBuilder()
	pair := []byte("aaaa aaaa\nbbbb bbbb\n")
	solo := []byte("cccc cccc\ndddd dddd\n")
	frozen := []byte("eeee eeee\nffff ffff\n")
	for i := 0; i < 2; i++ {
		gb.AddFile(pair)
	}
	gb.AddFile(solo)
	lb.AddFile(pair)
	lb.AddFile(solo)
	lb.AddFile(frozen)
	gb.Tighten()
	// Post-freeze: known pair keeps counting, new contexts ignored.
	gb.AddFile(frozen)
	gb.AddFile(frozen)
	li := lb.Compact()
	g := gb.Compact(li)
	toks, _, _ := g.Next("aaaa aaaa")
	if len(toks) == 0 || li.Key(int(toks[0])) != "bbbb bbbb" {
		t.Fatalf("pre-tighten repeat lost: %v", toks)
	}
	if toks, _, _ = g.Next("cccc cccc"); len(toks) != 0 {
		t.Fatalf("singleton context kept: %v", toks)
	}
	if toks, _, _ = g.Next("eeee eeee"); len(toks) != 0 {
		t.Fatalf("post-freeze context kept: %v", toks)
	}
}
