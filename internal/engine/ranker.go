package engine

import (
	"encoding/json"
	"math"
	"os"
	"strings"
	"sync"
)

// ranker.go: online learned reranker. The journal already knows what
// was shown and what was accepted; this layer turns that into a
// per-candidate probability instead of a fixed per-source multiplier.
// A logistic model over request-time features is trained by SGD on
// every accept (positive) and shown-but-skipped sibling (negative).
// Weights persist next to the journal so the model survives restarts.
//
// The output is a bounded score multiplier: a cold model (logit ~0)
// is a no-op, and even a confident one can only reorder, never veto.

// featNames is the fixed feature order; the persisted weight vector
// is indexed by it. Append only.
var featNames = []string{
	"bias",
	"logScore",
	"srcFile", "srcDyn", "srcAdapt", "srcCorpus", "srcModel",
	"srcMem", "srcLineBi", "srcPrior", "srcIter", "srcUnit", "srcEdit",
	"multiLine", "scopeHits", "memHit", "thinCtx",
	"candLen", "atDot", "argPos", "learnHit", "modelProb",
	"srcIdent", "impHit", "indentFit",
}

var featDim = len(featNames)

type Ranker struct {
	mu    sync.Mutex
	w     []float64
	n     int // updates applied this session
	dirty int // updates since last save
}

func NewRanker() *Ranker {
	return &Ranker{w: make([]float64, featDim)}
}

// featIdx maps a base source tag to its feature index, or -1 for
// sources without a dedicated slot.
func featSrcIdx(src string) int {
	switch src {
	case "file":
		return 2
	case "dyn":
		return 3
	case "adapt":
		return 4
	case "corpus":
		return 5
	case "model":
		return 6
	case "mem":
		return 7
	case "lineBi":
		return 8
	case "prior":
		return 9
	case "iter":
		return 10
	case "unit":
		return 11
	case "edit":
		return 12
	case "ident":
		return 22
	}
	return -1
}

// itemFeat computes the fixed-order feature vector for a candidate in
// its request context.
func itemFeat(it *Item, scopeHits int, memHit, thinCtx, atDot, argPos, learnHit bool) []float32 {
	return itemFeatExt(it, scopeHits, memHit, thinCtx, atDot, argPos, learnHit, false, false)
}

// itemFeatExt is itemFeat with the newer trailing features; the
// fixed-order vector stays append-only for ranker compatibility.
func itemFeatExt(it *Item, scopeHits int, memHit, thinCtx, atDot, argPos, learnHit, impHit, indentFit bool) []float32 {
	f := make([]float32, featDim)
	f[0] = 1
	f[1] = float32(math.Log1p(math.Max(it.Score, 0)))
	// Compound sources ("file+adapt") light every component slot.
	for _, s := range strings.Split(it.Source, "+") {
		if i := featSrcIdx(s); i >= 0 {
			f[i] = 1
		}
	}
	if strings.IndexByte(it.Text, '\n') >= 0 {
		f[13] = 1
	}
	f[14] = float32(math.Min(float64(scopeHits), 6) / 6)
	if memHit {
		f[15] = 1
	}
	if thinCtx {
		f[16] = 1
	}
	f[17] = float32(math.Log1p(float64(len(it.Text))) / 8)
	if atDot {
		f[18] = 1
	}
	if argPos {
		f[19] = 1
	}
	if learnHit {
		f[20] = 1
	}
	// Model attestation: the mean per-token n-gram probability of the
	// first line. Retrieval hits score high here too when they fit the
	// context; adapt noise and stale variants score low.
	f[21] = float32(math.Min(math.Max(it.modelP, 0), 1))
	if impHit {
		f[23] = 1
	}
	if indentFit {
		f[24] = 1
	}
	return f
}

func sigmoid(x float64) float64 {
	if x >= 0 {
		return 1 / (1 + math.Exp(-x))
	}
	e := math.Exp(x)
	return e / (1 + e)
}

// mult returns the score multiplier for a feature vector: sigmoid
// remapped to [0.45, 2.2] so a trained model can reorder strongly but
// never zero a candidate out.
func (r *Ranker) mult(f []float32) float64 {
	r.mu.Lock()
	defer r.mu.Unlock()
	var z float64
	for i, v := range f {
		z += r.w[i] * float64(v)
	}
	p := sigmoid(z)
	return 0.45 + 1.75*p
}

// update applies one labeled example by SGD with light L2 pull.
func (r *Ranker) update(f []float32, accept bool) {
	if len(f) == 0 {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	var z float64
	for i, v := range f {
		z += r.w[i] * float64(v)
	}
	p := sigmoid(z)
	y := 0.0
	if accept {
		y = 1
	}
	const lr = 0.06
	g := lr * (y - p)
	for i, v := range f {
		if v != 0 {
			r.w[i] += g*float64(v) - 1e-5*r.w[i]
		}
	}
	r.n++
	r.dirty++
}

// saveIfDirty persists weights when enough updates accumulated. Path
// is alongside the journal; a missing file is a cold start.
func (r *Ranker) saveIfDirty(path string) {
	r.mu.Lock()
	if r.dirty < 64 || path == "" {
		r.mu.Unlock()
		return
	}
	w := append([]float64(nil), r.w...)
	r.dirty = 0
	r.mu.Unlock()
	data, _ := json.Marshal(w)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err == nil {
		os.Rename(tmp, path)
	}
}

// TrainRanker batch-fits the online model over journaled feature
// vectors: several SGD epochs over the recorded shown/accept pairs.
// Returns nil when too few events carry features.
func TrainRanker(evs []JournalEvent, minEvents int) *Ranker {
	n := 0
	for _, e := range evs {
		if len(e.Feat) > 0 {
			n++
		}
	}
	if n < minEvents {
		return nil
	}
	r := NewRanker()
	for epoch := 0; epoch < 6; epoch++ {
		for _, e := range evs {
			if len(e.Feat) == 0 {
				continue
			}
			r.update(e.Feat, e.Kind == "a")
		}
	}
	return r
}

// Save writes the weight vector, atomically.
func (r *Ranker) Save(path string) {
	r.mu.Lock()
	w := append([]float64(nil), r.w...)
	r.mu.Unlock()
	data, _ := json.Marshal(w)
	tmp := path + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err == nil {
		os.Rename(tmp, path)
	}
}

// LoadRanker restores persisted weights, tolerating a missing or
// stale-length file.
func LoadRanker(path string) *Ranker {
	r := NewRanker()
	data, err := os.ReadFile(path)
	if err != nil {
		return r
	}
	var w []float64
	if json.Unmarshal(data, &w) != nil {
		return r
	}
	for i := range w {
		if i < featDim {
			r.w[i] = w[i]
		}
	}
	return r
}
