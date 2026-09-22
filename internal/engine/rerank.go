package engine

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"sort"
)

// rerank.go: offline reranking from the accept/reject journal.
//
// Every shown suggestion that later correlates with a Learn or Reject
// is journaled as a labeled event carrying the features available at
// display time: source, score, and rank. TrainSourceWeights fits a
// small logistic model over those events and returns a per-source
// score multiplier. Nothing runs in the completion hot path: the
// result is baked into Weights once, offline, by `q4complete tune`.

// JournalEvent is one labeled accept ("a") or reject ("r") record.
type JournalEvent struct {
	Kind  string  `json:"e"` // "a" or "r"
	Src   string  `json:"s"` // base source: model, file, corpus, dyn, learn
	Score float64 `json:"v"` // item score at display time
	Rank  int     `json:"r"` // position in the shown list
}

// LoadJournalEvents reads labeled events from the journal, keeping at
// most max of the most recent (the file also contains {"t":...} learn
// records, which are skipped).
func LoadJournalEvents(path string, max int) ([]JournalEvent, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	var evs []JournalEvent
	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 1<<16), 1<<20)
	for sc.Scan() {
		line := sc.Bytes()
		if len(line) == 0 || line[0] != '{' {
			continue
		}
		var ev JournalEvent
		if json.Unmarshal(line, &ev) != nil || (ev.Kind != "a" && ev.Kind != "r") {
			continue
		}
		evs = append(evs, ev)
		if len(evs) > max {
			evs = evs[len(evs)-max:]
		}
	}
	return evs, sc.Err()
}

// TrainSourceWeights fits P(accept) = sigmoid(b_src + w1*log(1+score)
// + w2*rank) by gradient descent and returns per-source multipliers
// exp(b_src), normalized so the largest is 1.0. Sources with too few
// events fall back to multiplier 1.0. Returns nil when the journal has
// too little signal (< minEvents total).
func TrainSourceWeights(evs []JournalEvent, minEvents int) map[string]float64 {
	if len(evs) < minEvents {
		return nil
	}
	srcs := map[string]int{}
	for _, ev := range evs {
		srcs[ev.Src]++
	}
	names := make([]string, 0, len(srcs))
	for s := range srcs {
		names = append(names, s)
	}
	sort.Strings(names)
	sidx := make(map[string]int, len(names))
	for i, s := range names {
		sidx[s] = i
	}
	// Features: intercept per source + shared score slope + rank slope.
	nw := len(names) + 2
	w := make([]float64, nw)
	const lr, epochs = 0.05, 400
	for ep := 0; ep < epochs; ep++ {
		grad := make([]float64, nw)
		for _, ev := range evs {
			x := make([]float64, nw)
			x[sidx[ev.Src]] = 1
			x[len(names)] = math.Log1p(ev.Score) / 10 // scale: scores vary wildly by source
			x[len(names)+1] = float64(ev.Rank)
			z := 0.0
			for i := range x {
				z += w[i] * x[i]
			}
			p := 1 / (1 + math.Exp(-z))
			y := 0.0
			if ev.Kind == "a" {
				y = 1
			}
			for i := range x {
				grad[i] += (p - y) * x[i]
			}
		}
		n := float64(len(evs))
		for i := range w {
			w[i] -= lr * grad[i] / n
		}
	}
	out := make(map[string]float64, len(names))
	maxB := math.Inf(-1)
	for i, s := range names {
		if srcs[s] < 32 { // too few events to trust
			continue
		}
		if w[i] > maxB {
			maxB = w[i]
		}
	}
	for i, s := range names {
		if srcs[s] < 32 {
			out[s] = 1.0
			continue
		}
		out[s] = math.Exp(w[i] - maxB)
	}
	return out
}

// ApplySourceWeights scales the source knobs of w by the learned
// multipliers, clamped so a noisy journal cannot collapse a source.
func (w *Weights) ApplySourceWeights(mult map[string]float64) {
	clamp := func(v float64) float64 {
		if v < 0.25 {
			return 0.25
		}
		if v > 4 {
			return 4
		}
		return v
	}
	if m, ok := mult["file"]; ok {
		w.File *= clamp(m)
	}
	if m, ok := mult["dyn"]; ok {
		w.Dyn *= clamp(m)
	}
	if m, ok := mult["learn"]; ok {
		w.Learn *= clamp(m)
	}
	if m, ok := mult["model"]; ok {
		w.Model *= clamp(m)
	}
}
