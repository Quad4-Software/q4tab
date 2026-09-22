package engine

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

// A journal where "model" accepts dominate must yield a higher model
// multiplier than file. A journal with too few events returns nil.
func TestTrainSourceWeights(t *testing.T) {
	var evs []JournalEvent
	for i := 0; i < 500; i++ {
		// model accepts on high scores, rejects on low
		evs = append(evs, JournalEvent{Kind: "a", Src: "model", Score: 1.0 + float64(i%3), Rank: 0})
		evs = append(evs, JournalEvent{Kind: "r", Src: "model", Score: 0.01, Rank: 2})
		// file mostly rejects
		evs = append(evs, JournalEvent{Kind: "r", Src: "file", Score: 1e6, Rank: 1})
		evs = append(evs, JournalEvent{Kind: "r", Src: "file", Score: 1e6, Rank: 2})
	}
	mult := TrainSourceWeights(evs, 256)
	if mult == nil {
		t.Fatal("nil multipliers on 2000 events")
	}
	if mult["model"] <= mult["file"] {
		t.Fatalf("model mult %g should exceed file %g", mult["model"], mult["file"])
	}
	w := DefaultWeights()
	w.ApplySourceWeights(mult)
	// Multipliers normalize the best source to 1.0: model stays put
	// while the mostly-rejected file source drops.
	if w.File >= DefaultWeights().File {
		t.Fatalf("file weight not reduced: %+v", w)
	}
	if got := TrainSourceWeights(evs[:10], 256); got != nil {
		t.Fatalf("tiny journal should not train, got %v", got)
	}
}

// Journal events round-trip through LoadJournalEvents and the learn
// records ({"t":...}) are skipped.
func TestLoadJournalEvents(t *testing.T) {
	p := filepath.Join(t.TempDir(), "j.jsonl")
	f, _ := os.Create(p)
	f.WriteString(`{"t":"accepted text"}` + "\n")
	for _, ev := range []JournalEvent{
		{Kind: "a", Src: "model", Score: 2.0, Rank: 0},
		{Kind: "r", Src: "file", Score: 1e6, Rank: 1},
	} {
		b, _ := json.Marshal(ev)
		f.Write(b)
		f.WriteString("\n")
	}
	f.WriteString("garbage line\n")
	f.Close()
	evs, err := LoadJournalEvents(p, 1000)
	if err != nil {
		t.Fatal(err)
	}
	if len(evs) != 2 || evs[0].Src != "model" || evs[1].Kind != "r" {
		t.Fatalf("events: %+v", evs)
	}
}
