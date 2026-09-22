package tokenize

import (
	"slices"
	"testing"
)

func TestLexBasic(t *testing.T) {
	src := "func main() {\n\tfmt.Println(\"hi\")\n}\n"
	got := Lex([]byte(src))
	want := []string{
		"func", " ", "main", "(", ")", " ", "{", NL,
		"\t", "fmt", ".", "Println", "(", "\"hi\"", ")", NL,
		"}", NL, EOF,
	}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q\nwant %q", got, want)
	}
}

func TestLexSpacesNormalized(t *testing.T) {
	got := Lex([]byte("a   =   1\n"))
	want := []string{"a", " ", "=", " ", "1", NL, EOF}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestLexStringEscape(t *testing.T) {
	got := Lex([]byte(`x = "a\"b"` + "\n"))
	want := []string{"x", " ", "=", " ", `"a\"b"`, NL, EOF}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestLexMultiOps(t *testing.T) {
	got := Lex([]byte("if a == b {\n\tch <- v := f()\n}\n"))
	for _, tok := range []string{"==", "<-", ":="} {
		if !slices.Contains(got, tok) {
			t.Fatalf("missing %q in %q", tok, got)
		}
	}
}

func TestLexLineStripsIndent(t *testing.T) {
	got := LexLine([]byte("\t\treturn result"))
	want := []string{"return", " ", "result"}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestBlankLineNoIndent(t *testing.T) {
	got := Lex([]byte("a\n   \nb\n"))
	want := []string{"a", NL, NL, "b", NL, EOF}
	if !slices.Equal(got, want) {
		t.Fatalf("got %q want %q", got, want)
	}
}

func TestDetokRoundTrip(t *testing.T) {
	src := "func f() int {\n\treturn 42\n}\n"
	toks := Lex([]byte(src))
	if Detok(toks[:len(toks)-1]) != src {
		t.Fatalf("round trip failed: %q", Detok(toks))
	}
}

// TestLexTruncatedEscape covers the panic where a backslash at EOF
// overshot the string scan.
func TestLexTruncatedEscape(t *testing.T) {
	for _, src := range []string{`"a\`, `x = "ab\`, "'\\", "`\\", `"`} {
		got := Lex([]byte(src))
		if len(got) == 0 || got[len(got)-1] != EOF {
			t.Fatalf("bad tokens for %q: %q", src, got)
		}
	}
}

// FuzzLex must never panic and always end with EOF.
func FuzzLex(f *testing.F) {
	f.Add(`x = "a\"b"` + "\nif err != nil {\n\treturn err\n}\n")
	f.Add("a\t\tb /* c */ d // e\n")
	f.Add("`raw\\`str`\n")
	f.Fuzz(func(t *testing.T, src string) {
		got := Lex([]byte(src))
		if len(got) == 0 || got[len(got)-1] != EOF {
			t.Fatalf("no EOF for %q", src)
		}
	})
}
