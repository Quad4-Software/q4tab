package engine

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// auxFixture is a small corpus with enough repetition for every
// auxiliary layer to have signal: repeated line transitions for the
// line-gram, camelCase identifiers for the subtoken model, and enough
// tokens to keep the language table over its floor.
var auxFixture = map[string]string{
	"a.go": `package demo

func fetchRemoteConfig(ctx Context) (*RemoteConfig, error) {
	remoteConfigValue, err := fetchRemoteConfigRaw(ctx)
	if err != nil {
		return nil, err
	}
	return parseRemoteConfig(remoteConfigValue)
}

func loadRemoteConfig(ctx Context) (*RemoteConfig, error) {
	remoteConfigValue, err := fetchRemoteConfigRaw(ctx)
	if err != nil {
		return nil, err
	}
	return parseRemoteConfig(remoteConfigValue)
}

func saveRemoteConfig(ctx Context) error {
	remoteConfigValue, err := fetchRemoteConfigRaw(ctx)
	if err != nil {
		return err
	}
	return writeRemoteConfig(remoteConfigValue)
}
`,
}

func buildAuxEngine(t *testing.T) *Engine {
	t.Helper()
	dir := t.TempDir()
	for name, src := range auxFixture {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(src), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	bun, st, err := BuildIndex([]string{dir}, 6, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	if st.Tokens == 0 {
		t.Fatal("empty corpus stats")
	}
	e := New(DefaultConfig())
	e.SetBundle(bun)
	return e
}

func TestBundleHasAllLayers(t *testing.T) {
	e := buildAuxEngine(t)
	if e.m == nil || !e.m.KN {
		t.Fatal("KN model missing")
	}
	if e.li == nil {
		t.Fatal("line index missing")
	}
	if e.lineBi == nil {
		t.Fatal("line-gram table missing")
	}
	if e.struc == nil {
		t.Fatal("struct table missing")
	}
	if e.sub == nil || e.idents == nil {
		t.Fatal("subtoken model or ident index missing")
	}
	if e.m.N != 8 {
		t.Fatalf("N=%d want 8 (order 6 + 2 deep)", e.m.N)
	}
}

func TestLineBiSurfaces(t *testing.T) {
	e := buildAuxEngine(t)
	// After "remoteConfigValue, err := fetchRemoteConfigRaw(ctx)" the
	// corpus always continues "if err != nil {". Typing that same line
	// in a fresh buffer should surface the retrieval hit.
	text := "package demo\n\nfunc f(ctx Context) error {\n\tremoteConfigValue, err := fetchRemoteConfigRaw(ctx)\n"
	items := e.Complete("file:///f.go", text, len(text))
	found := false
	for _, it := range items {
		if it.Source == "linebi" && strings.Contains(it.Text, "if err != nil") {
			found = true
		}
	}
	if !found {
		texts := make([]string, 0, len(items))
		for _, it := range items {
			texts = append(texts, it.Source+":"+strings.ReplaceAll(it.Text, "\n", "\\n"))
		}
		t.Fatalf("no linebi item for repeated transition, got %v", texts)
	}
}

func TestSubtokenInjection(t *testing.T) {
	e := buildAuxEngine(t)
	// Partial "fetchRem" inside a call context: the subtoken model +
	// ident index should inject candidates that complete the name.
	extra, _ := e.subCands(nil, "fetchRem")
	if len(extra) == 0 {
		t.Fatal("no subtoken candidates for fetchRem")
	}
	gotName := false
	for _, c := range extra {
		if s := e.m.Vocab.Str(c.Tok); strings.HasPrefix(s, "fetchRem") {
			gotName = true
		}
	}
	if !gotName {
		t.Fatal("injected candidates do not include a fetchRem* ident")
	}
}

func TestLangOfMapping(t *testing.T) {
	cases := map[string]string{
		"a.go":                 "go",
		"file:///x/y.py":       "python",
		"file:///x/y.TS":       "typescript",
		"x.mjs":                "javascript",
		"Dockerfile":           "dockerfile",
		"x/Dockerfile?query=1": "dockerfile",
		"Makefile":             "makefile",
		"a.unknownext":         "other",
		"":                     "other",
	}
	for in, want := range cases {
		if got := LangOf(in); got != want {
			t.Errorf("LangOf(%q) = %q want %q", in, got, want)
		}
	}
}

func TestAuxRoundTrip(t *testing.T) {
	e := buildAuxEngine(t)
	p := filepath.Join(t.TempDir(), "m.q4")
	if err := Save(p, e.bundle()); err != nil {
		t.Fatal(err)
	}
	bun2, err := Load(p)
	if err != nil {
		t.Fatal(err)
	}
	if bun2.LineBi == nil || bun2.Struct == nil || bun2.Sub == nil || bun2.Idents == nil {
		t.Fatal("aux sections dropped on round trip")
	}
	if !bun2.M.KN || bun2.M.UniTot == 0 || len(bun2.M.UniTop) == 0 {
		t.Fatal("KN metadata dropped on round trip")
	}
	e2 := New(DefaultConfig())
	e2.SetBundle(bun2)
	// Same prompt must produce the same top items on the loaded bundle.
	text := "package demo\n\nfunc f(ctx Context) error {\n\tremoteConfigValue, err := fetchRemoteConfigRaw(ctx)\n"
	i1 := texts(e.Complete("file:///f.go", text, len(text)))
	i2 := texts(e2.Complete("file:///f.go", text, len(text)))
	if strings.Join(i1, "|") != strings.Join(i2, "|") {
		t.Fatalf("post-load items differ:\nbefore %v\nafter  %v", i1, i2)
	}
	// Line-gram retrieval works on the loaded bundle.
	_, cnts, tot := bun2.LineBi.Next("remoteConfigValue, err := fetchRemoteConfigRaw(ctx)")
	if len(cnts) == 0 || tot == 0 {
		t.Fatal("loaded line-gram lost its rows")
	}
}

func TestTrainLamK(t *testing.T) {
	e := buildAuxEngine(t)
	lamK := e.TrainLamK([]string{auxFixture["a.go"]})
	if len(lamK) == 0 {
		t.Fatal("TrainLamK returned nothing on a KN model")
	}
	// TrainLamK only computes; SetWeights installs.
	w := DefaultWeights()
	w.LamK = lamK
	e.SetWeights(w)
	if e.m.LamK == nil {
		t.Fatal("LamK not installed via SetWeights")
	}
	e.SetWeights(DefaultWeights())
	if e.m.LamK != nil {
		t.Fatal("LamK not cleared by weights without it")
	}
}

func TestCalibratorBoost(t *testing.T) {
	// Synthetic journal: "corpus" accepts score high, "model" rejects
	// score low. A trained calibrator must boost corpus over model.
	var evs []JournalEvent
	for i := 0; i < 200; i++ {
		evs = append(evs,
			JournalEvent{Kind: "a", Src: "corpus", Score: 100, Rank: 0},
			JournalEvent{Kind: "r", Src: "model", Score: 1, Rank: 0},
		)
	}
	cal := TrainCalibrator(evs, 50)
	if cal == nil {
		t.Fatal("calibrator did not train")
	}
	corpusIt := Item{Text: "x", Source: "corpus", Score: 100}
	modelIt := Item{Text: "y", Source: "model", Score: 1}
	bc := cal.Boost(&corpusIt, 0)
	bm := cal.Boost(&modelIt, 0)
	if bc <= bm {
		t.Fatalf("calibrator ranks model (%g) >= corpus (%g)", bm, bc)
	}
	// Untrained: Boost must be a no-op (nil receiver safe).
	var nilCal *Calibrator
	if b := nilCal.Boost(&corpusIt, 0); b != 1 {
		t.Fatalf("nil calibrator boost = %g want 1", b)
	}
}
