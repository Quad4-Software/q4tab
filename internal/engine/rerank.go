package engine

import (
	"bufio"
	"encoding/json"
	"math"
	"os"
	"sort"
	"strings"
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
	Kind  string  `json:"e"`           // "a" or "r"
	Src   string  `json:"s"`           // base source: model, file, corpus, dyn, learn
	Score float64 `json:"v"`           // item score at display time
	Rank  int     `json:"r"`           // position in the shown list
	Len   int     `json:"l,omitempty"` // suggestion length in bytes
	Multi bool    `json:"m,omitempty"` // suggestion spans multiple lines
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

// Calibrator is a small logistic model over display-time features that
// rescales merged candidates. Unlike ApplySourceWeights (which shifts a
// whole source's score unit), the calibrator sees each item's blended
// picture: source, score magnitude, shown rank, length, and whether it
// spans lines. Trained offline by `q4complete tune` from the journal,
// applied at merge time in CompleteFor.
type Calibrator struct {
	Bias   float64            `json:"b"`
	SrcW   map[string]float64 `json:"s"` // per-source weight
	ScoreW float64            `json:"v"` // log-score slope
	RankW  float64            `json:"r"` // rank slope
	LenW   float64            `json:"l"` // log-length slope
	MultiW float64            `json:"m"` // multi-line flag weight
}

// TrainCalibrator fits P(accept) = sigmoid(bias + srcW[src] +
// scoreW*log1p(score)/10 + rankW*rank + lenW*log1p(len)/5 +
// multiW*multi) by gradient descent with L2 shrinkage. Returns nil
// under minEvents so a thin journal leaves ranking untouched.
func TrainCalibrator(evs []JournalEvent, minEvents int) *Calibrator {
	if len(evs) < minEvents {
		return nil
	}
	srcs := map[string]int{}
	for _, ev := range evs {
		srcs[ev.Src]++
	}
	names := make([]string, 0, len(srcs))
	for s, n := range srcs {
		if n >= 32 {
			names = append(names, s)
		}
	}
	if len(names) == 0 {
		return nil
	}
	sort.Strings(names)
	sidx := make(map[string]int, len(names))
	for i, s := range names {
		sidx[s] = i
	}
	// Weight vector: bias | per-source | score | rank | len | multi
	nw := 1 + len(names) + 4
	w := make([]float64, nw)
	const lr, epochs, l2 = 0.05, 500, 1e-4
	feat := func(ev JournalEvent) []float64 {
		x := make([]float64, nw)
		x[0] = 1
		if i, ok := sidx[ev.Src]; ok {
			x[1+i] = 1
		}
		x[1+len(names)] = math.Log1p(ev.Score) / 10
		x[2+len(names)] = float64(ev.Rank)
		x[3+len(names)] = math.Log1p(float64(ev.Len)) / 5
		if ev.Multi {
			x[4+len(names)] = 1
		}
		return x
	}
	n := float64(len(evs))
	for ep := 0; ep < epochs; ep++ {
		grad := make([]float64, nw)
		for _, ev := range evs {
			x := feat(ev)
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
		for i := range w {
			// Shrink per-source weights toward 0 harder: sparse
			// sources should not drift the shared features.
			reg := l2
			if i >= 1 && i <= len(names) {
				reg = l2 * 4
			}
			w[i] -= lr * (grad[i]/n + reg*w[i])
		}
	}
	c := &Calibrator{
		Bias:   w[0],
		SrcW:   map[string]float64{},
		ScoreW: w[1+len(names)],
		RankW:  w[2+len(names)],
		LenW:   w[3+len(names)],
		MultiW: w[4+len(names)],
	}
	for i, s := range names {
		c.SrcW[s] = w[1+i]
	}
	return c
}

// Boost maps an item's features to a score multiplier, clamped to
// [0.4, 2.5] so a skewed journal can nudge but never dominate. The
// multiplier is relative to the calibrator's intercept-only prediction,
// so a featureless item stays neutral.
func (c *Calibrator) Boost(it *Item, rank int) float64 {
	if c == nil {
		return 1
	}
	z := c.Bias + c.SrcW[baseSource(it.Source)] +
		c.ScoreW*math.Log1p(it.Score)/10 +
		c.RankW*float64(rank) +
		c.LenW*math.Log1p(float64(len(it.Text)))/5
	if strings.IndexByte(it.Text, '\n') >= 0 {
		z += c.MultiW
	}
	p := 1 / (1 + math.Exp(-z))
	base := 1 / (1 + math.Exp(-c.Bias))
	if base <= 0 {
		return 1
	}
	b := p / base
	if b < 0.4 {
		return 0.4
	}
	if b > 2.5 {
		return 2.5
	}
	return b
}
