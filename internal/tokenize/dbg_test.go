package tokenize

import (
	"fmt"
	"testing"
)

func TestDbgToks(t *testing.T) {
	for _, t := range Lex([]byte("import \"encoding/json\"\nimport \"net/http\"\nfrom foo import bar\nuse std::collections::HashMap\n")) {
		fmt.Printf("%q\n", t)
	}
}
