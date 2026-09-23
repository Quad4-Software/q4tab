package engine

import (
	"strings"

	"q4complete/internal/lines"
	"q4complete/internal/model"
	"q4complete/internal/tokenize"
)

// sub.go: identifier-subtoken completion and next-line retrieval.

// subCans returns injected first-token candidates for a partially typed
// identifier, plus a synthesized identifier string for names not in the
// vocabulary at all.
//
// The mechanism: partial "parseH" splits into complete parts [parse]
// and a partial last part "H". The subtoken model predicts what comes
// after the typed parts (marked continuation subtokens), and the ident
// index maps subtoken paths back to real vocabulary identifiers. Known
// idents become first-token candidates so the main model scores their
// continuations; a predicted part sequence that maps to nothing is
// joined into a brand-new identifier suggestion.
func (e *Engine) subCands(ctx []uint32, partial string) (extra []model.Cand, synth string) {
	if e.sub == nil || e.idents == nil || len(partial) < 2 || !isIdentByte(partial[0]) {
		return nil, ""
	}
	parts := tokenize.Subtoks(partial)
	if len(parts) == 0 {
		return nil, ""
	}
	complete := parts[:len(parts)-1]
	lastPart := parts[len(parts)-1]

	// Translate the tail of the token context into subvocab ids.
	var sc []uint32
	lo := 0
	if len(ctx) > 6 {
		lo = len(ctx) - 6
	}
	for _, id := range ctx[lo:] {
		s := e.m.Vocab.Str(id)
		if tokenize.IsIdentTok(s) {
			for i, p := range tokenize.Subtoks(s) {
				key := p
				if i > 0 {
					key = "\x01" + p
				}
				if sid, ok := e.sub.Vocab.Lookup(key); ok {
					sc = append(sc, sid)
				}
			}
		} else if sid, ok := e.sub.Vocab.Lookup(s); ok {
			sc = append(sc, sid)
		}
	}
	// The complete typed parts extend the context.
	for i, p := range complete {
		key := p
		if i > 0 {
			key = "\x01" + p
		}
		if sid, ok := e.sub.Vocab.Lookup(key); ok {
			sc = append(sc, sid)
		}
	}
	sq := e.sub.NewQuery()

	// Predicted continuation subtokens that extend the partial part.
	// e.g. typed "parse|H" -> predicted "\x01HTTP" completes to paths
	// like "parse|http|response" in the ident index.
	var next []model.Cand
	if e.sub.N > 0 {
		next = e.sub.TopUnion(sc, e.cfg.Lam, 8, sq)
	}
	seenPath := map[string]bool{}
	addIdents := func(prefix string, weight float64) {
		ids, _ := e.idents.Prefix(prefix, 6)
		for _, id := range ids {
			extra = append(extra, model.Cand{Tok: uint32(id), P: weight})
		}
	}
	for _, c := range next {
		s := e.sub.Vocab.Str(c.Tok)
		if !strings.HasPrefix(s, "\x01") || len(s) < 2 {
			continue
		}
		part := s[1:]
		if !strings.HasPrefix(part, lastPart) || part == lastPart {
			continue
		}
		path := model.IdentKey(append(append([]string{}, complete...), part))
		if seenPath[path] {
			continue
		}
		seenPath[path] = true
		addIdents(path, 0.05+0.3*c.P)
	}
	// Direct index hits on the typed prefix, regardless of what the
	// subtoken model predicts: "parseH" still finds parseHeaderSize.
	direct := model.IdentKey(complete)
	if len(complete) > 0 {
		direct += "|" + lowerASCII(lastPart)
	} else {
		direct = lowerASCII(lastPart)
	}
	addIdents(direct, 0.05)

	// Synthesis: greedily extend the typed parts with the subtoken
	// model's top continuation until it stops producing continuations,
	// then join. This proposes identifiers that never appeared in the
	// corpus whole, as long as each part transition did.
	if len(next) > 0 {
		cur := append([]string{}, complete...)
		cur = append(cur, lastPart)
		sctx := append([]uint32{}, sc...)
		for len(cur) < 6 {
			tops := e.sub.TopUnion(sctx, e.cfg.Lam, 1, sq)
			if len(tops) == 0 {
				break
			}
			s := e.sub.Vocab.Str(tops[0].Tok)
			if !strings.HasPrefix(s, "\x01") || len(s) < 2 {
				break
			}
			cur = append(cur, s[1:])
			sctx = append(sctx, tops[0].Tok)
		}
		joined := strings.Join(cur, "")
		if len(joined) > len(partial) && strings.HasPrefix(joined, partial) {
			// Only synthesize when the joined name is not already a
			// vocab ident (those surface through `extra` anyway).
			if _, known := e.m.Vocab.Lookup(joined); !known {
				synth = joined[len(partial):]
			}
		}
	}
	return extra, synth
}

// lineBiItems proposes next lines from the line n-gram table. Two
// shapes: at the start of an empty line the suggestion is the predicted
// line itself; at the end of a full line it is a newline plus the
// predicted line reindented to the current depth.
func (e *Engine) lineBiItems(text string, offset int, linePrefix string) []Item {
	if e.lineBi == nil || e.li == nil {
		return nil
	}
	prefix := text[:offset]
	trimmed := strings.TrimLeft(linePrefix, " \t")
	bol := trimmed == ""
	var restBlank bool
	if !bol {
		rest := text[offset:]
		if i := strings.IndexByte(rest, '\n'); i >= 0 {
			rest = rest[:i]
		}
		restBlank = strings.TrimSpace(rest) == ""
	}
	if !bol && !restBlank {
		return nil // mid-line: the token model owns this
	}
	// Previous normalized lines above the cursor's line.
	var prev1, prev2 string
	start := 0
	curStart := strings.LastIndexByte(prefix, '\n') + 1
	for i := 0; i <= len(prefix); i++ {
		if i == len(prefix) || prefix[i] == '\n' {
			end := i
			if end > curStart {
				end = curStart
			}
			if end > start {
				if n := lines.Normalize(prefix[start:end]); n != "" {
					prev2, prev1 = prev1, n
				}
			}
			start = i + 1
			if start > curStart {
				break
			}
		}
	}
	if !bol && trimmed != "" {
		// At EOL the current line is the freshest context.
		prev2, prev1 = prev1, lines.Normalize(linePrefix)
	}
	if prev1 == "" {
		return nil
	}
	// Try the 2-line context first, then the 1-line fallback.
	var toks, cnts []int32
	var tot int64
	if prev2 != "" {
		toks, cnts, tot = e.lineBi.Next(prev2, prev1)
	}
	if len(toks) == 0 {
		toks, cnts, tot = e.lineBi.Next(prev1)
	}
	if len(toks) == 0 || tot == 0 {
		return nil
	}
	var out []Item
	indent := leadingWS(linePrefix)
	if !bol {
		indent = leadingWS(linePrefix)
	}
	for i, li := range toks {
		if i >= e.cfg.MaxItems*2 {
			break
		}
		key := e.li.Key(int(li))
		if key == "" {
			continue
		}
		var text string
		if bol {
			text = key
		} else {
			text = "\n" + indent + key
		}
		out = append(out, Item{Text: text, Source: "linebi",
			Score: e.w.LineBi * (1 + 0.2*float64(cnts[i]))})
	}
	return out
}

func lowerASCII(s string) string {
	b := []byte(s)
	for i, c := range b {
		if c >= 'A' && c <= 'Z' {
			b[i] = c + ('a' - 'A')
		}
	}
	return string(b)
}
