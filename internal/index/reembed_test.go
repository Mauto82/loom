package index

import (
	"context"
	"fmt"
	"testing"
)

// stalePaths returns the files whose vector is missing or not from model.
func stalePaths(t *testing.T, ix *Indexer, model string) map[string]bool {
	t.Helper()
	stale, err := ix.Store.StaleVectorPaths(context.Background(), model)
	if err != nil {
		t.Fatalf("stale paths: %v", err)
	}
	out := map[string]bool{}
	for _, p := range stale {
		out[p] = true
	}
	return out
}

// A model switch must not need --force: the next normal scan re-embeds the
// unchanged files from the stored content, without calling the LLM again.
func TestScanReembedsAfterModelSwitchWithoutLLM(t *testing.T) {
	notesDir, store, lc := setupNotes(t)
	writeNote(t, notesDir, "a.md", "# A\nhello world")
	writeNote(t, notesDir, "b.md", "# B\nciao mondo")

	ix := New([]string{notesDir}, store, lc)
	ix.Embedder = &stubEmbedder{name: "old-model"}
	if _, err := ix.Scan(context.Background()); err != nil {
		t.Fatalf("first scan: %v", err)
	}
	llmCalls := lc.callCount()

	newEmb := &stubEmbedder{name: "new-model"}
	ix.Embedder = newEmb
	res, err := ix.Scan(context.Background())
	if err != nil {
		t.Fatalf("second scan: %v", err)
	}
	if res.Skipped != 2 || res.Reembedded != 2 {
		t.Errorf("skipped=%d reembedded=%d, want 2 and 2", res.Skipped, res.Reembedded)
	}
	if got := lc.callCount(); got != llmCalls {
		t.Errorf("LLM called %d more times; re-embedding must not re-summarise", got-llmCalls)
	}
	if newEmb.calls != 2 {
		t.Errorf("new embedder called for %d files, want 2", newEmb.calls)
	}
	if stale := stalePaths(t, ix, "new-model"); len(stale) != 0 {
		t.Errorf("still stale under new-model: %v", stale)
	}

	// Nothing left to do: a third scan re-embeds nothing.
	res, err = ix.Scan(context.Background())
	if err != nil {
		t.Fatalf("third scan: %v", err)
	}
	if res.Reembedded != 0 || newEmb.calls != 2 {
		t.Errorf("third scan reembedded=%d calls=%d, want 0 and 2", res.Reembedded, newEmb.calls)
	}
}

// Files indexed while the embedder was down get their vector on a later scan.
func TestScanFillsMissingVectorsOnceEmbedderIsBack(t *testing.T) {
	notesDir, store, lc := setupNotes(t)
	writeNote(t, notesDir, "a.md", "# A\nhello world")

	ix := New([]string{notesDir}, store, lc)
	emb := &stubEmbedder{fail: true}
	ix.Embedder = emb
	res, err := ix.Scan(context.Background())
	if err != nil {
		t.Fatalf("scan with embedder down: %v", err)
	}
	if res.Added != 1 || res.Failed != 0 || res.Reembedded != 0 {
		t.Errorf("added=%d failed=%d reembedded=%d, want 1/0/0 (embedding is best-effort)",
			res.Added, res.Failed, res.Reembedded)
	}
	if ok, _ := store.HasVectors(context.Background()); ok {
		t.Fatal("no vector expected while the embedder is down")
	}

	emb.fail = false
	res, err = ix.Scan(context.Background())
	if err != nil {
		t.Fatalf("scan with embedder back: %v", err)
	}
	if res.Reembedded != 1 {
		t.Errorf("reembedded=%d, want 1", res.Reembedded)
	}
	if ok, _ := store.HasVectors(context.Background()); !ok {
		t.Error("vector expected once the embedder is back")
	}
}

// More stale files than one batch: all of them are re-embedded.
func TestScanReembedsInBatches(t *testing.T) {
	notesDir, store, lc := setupNotes(t)
	n := embedBatch*2 + 3
	for i := 0; i < n; i++ {
		writeNote(t, notesDir, fmt.Sprintf("n%02d.md", i), fmt.Sprintf("# N%d\ntext %d", i, i))
	}
	ix := New([]string{notesDir}, store, lc) // first scan without embeddings
	if _, err := ix.Scan(context.Background()); err != nil {
		t.Fatalf("first scan: %v", err)
	}
	ix.Embedder = &stubEmbedder{}
	res, err := ix.Scan(context.Background())
	if err != nil {
		t.Fatalf("second scan: %v", err)
	}
	if res.Reembedded != n {
		t.Errorf("reembedded=%d, want %d", res.Reembedded, n)
	}
}

func TestResultPayloadReportsReembedded(t *testing.T) {
	if p := (&Result{Reembedded: 7}).Payload(); p["reembedded"] != 7 {
		t.Errorf("reembedded = %v, want 7", p["reembedded"])
	}
}
