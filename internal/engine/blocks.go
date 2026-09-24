package engine

import (
	"strings"

	"q4tab/internal/lines"
)

// blocks.go: snippet-level retrieval. The line index answers "what
// comes after this line"; the block map answers "what comes after
// that line AND the one after it" - attested multi-line chunks.
// Verbatim block recall is strongest inside the open document (table
// tests, handler pairs, per-field boilerplate) so the map lives on
// session docs, and the current file gets a direct text scan.

// blocksFor maps each normalized line of a document to the next up to
// two nonblank normalized lines that follow it. Keys are the same
// normalized lines the dynamic index stores, so a dyn continuation
// looks up its followers by the reconstructed full line.
func blocksFor(text string) map[string][]string {
	raw := strings.Split(text, "\n")
	var nl []string
	for _, r := range raw {
		nl = append(nl, lines.Normalize(r))
	}
	out := map[string][]string{}
	for i, l := range nl {
		if len(l) < 4 {
			continue
		}
		var fol []string
		for j := i + 1; j < len(nl) && len(fol) < 2; j++ {
			if nl[j] != "" {
				fol = append(fol, nl[j])
			}
		}
		if len(fol) > 0 {
			if _, dup := out[l]; !dup {
				out[l] = fol
			}
		}
	}
	return out
}

// fileBlockLines scans the current document text for lines matching
// the prefix and returns multi-line continuations: the rest of the
// matched line plus the next up to two nonblank lines verbatim. This
// is the same-file sibling of the session block map.
func fileBlockLines(text, prefix, exclude string, curLine, limit int) []string {
	norm := lines.Normalize(prefix)
	if len(norm) < 4 {
		return nil
	}
	raw := strings.Split(text, "\n")
	nl := make([]string, len(raw))
	for i, r := range raw {
		nl[i] = lines.Normalize(r)
	}
	best := map[string]int{}
	var order []string
	for i, l := range nl {
		if l == exclude || len(l) <= len(norm)+1 || !strings.HasPrefix(l, norm) {
			continue
		}
		d := i - curLine
		if d < 0 {
			d = -d
		}
		rest := l[len(norm):]
		var fol []string
		for j := i + 1; j < len(nl) && len(fol) < 2; j++ {
			if nl[j] != "" {
				fol = append(fol, nl[j])
			}
		}
		if len(fol) == 0 {
			continue
		}
		full := rest + "\n" + strings.Join(fol, "\n")
		if bd, ok := best[full]; !ok {
			best[full] = d
			order = append(order, full)
		} else if d < bd {
			best[full] = d
		}
	}
	// Sort by distance like fileLines.
	for i := 1; i < len(order); i++ {
		for j := i; j > 0 && best[order[j]] < best[order[j-1]]; j-- {
			order[j], order[j-1] = order[j-1], order[j]
		}
	}
	if len(order) > limit {
		order = order[:limit]
	}
	return order
}
