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
