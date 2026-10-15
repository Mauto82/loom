package main

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Mauto82/loom/internal/config"
	"github.com/Mauto82/loom/internal/storage"
)

func TestValidateCorpus(t *testing.T) {
	ok := []string{"speedy", "tenant_1", "a-b-c", "ABC123"}
	for _, n := range ok {
		if err := validateCorpus(n); err != nil {
			t.Errorf("expected %q valid, got %v", n, err)
		}
	}
	bad := []string{"", "..", ".", "a/b", "../etc", "a.b", "foo/", "with space", "a:b"}
	for _, n := range bad {
		if err := validateCorpus(n); err == nil {
			t.Errorf("expected %q rejected", n)
		}
	}
}

func TestCorpusPathsStayUnderRoot(t *testing.T) {
	db, notes := corpusPaths("/srv/knowledge", "speedy")
	if db != "/srv/knowledge/speedy/index.db" {
		t.Errorf("db path: %s", db)
	}
	if notes != "/srv/knowledge/speedy/notes" {
		t.Errorf("notes path: %s", notes)
	}
}

// newTestServer builds a server with a real single store and a temp corpus root,
// no LLM/embedder (search is LLM-free).
func newTestServer(t *testing.T, multiCorpus bool) *server {
	t.Helper()
	single, err := storage.Open(filepath.Join(t.TempDir(), "single.db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { single.Close() })
	s := &server{
		single: single,
		cfg:    &config.Config{NotesDirs: []string{t.TempDir()}},
		stores: map[string]*storage.Store{},
	}
	if multiCorpus {
		s.root = t.TempDir()
	}
	t.Cleanup(s.closeAll)
	return s
}

func doSearch(t *testing.T, s *server, body string) (int, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(http.MethodPost, "/search", bytes.NewBufferString(body))
	rec := httptest.NewRecorder()
	s.search(rec, req)
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec.Code, out
}

func seedCorpus(t *testing.T, s *server, corpus, relPath, content string) {
	t.Helper()
	store, _, err := s.storeFor(corpus)
	if err != nil {
		t.Fatalf("storeFor %q: %v", corpus, err)
	}
	if err := store.Upsert(context.Background(), &storage.File{
		RelPath: relPath, Hash: relPath, MTime: 1, Kind: "md",
		Title: relPath, Content: content, Summary: content, Keywords: []string{relPath},
	}); err != nil {
		t.Fatal(err)
	}
}

func TestMultiCorpusIsolation(t *testing.T) {
	s := newTestServer(t, true)
	seedCorpus(t, s, "alpha", "a.md", "carbonara guanciale")
	seedCorpus(t, s, "beta", "b.md", "carbonara guanciale")

	// alpha sees its own doc...
	code, out := doSearch(t, s, `{"query":"carbonara","corpus":"alpha"}`)
	if code != http.StatusOK {
		t.Fatalf("alpha search code %d", code)
	}
	hits, _ := out["hits"].([]any)
	if len(hits) != 1 {
		t.Fatalf("alpha should have exactly its own 1 hit, got %d", len(hits))
	}
	if rp := hits[0].(map[string]any)["rel_path"]; rp != "a.md" {
		t.Errorf("alpha returned the wrong corpus' file: %v", rp)
	}

	// ...and never beta's, even with an identical query.
	_, out = doSearch(t, s, `{"query":"carbonara","corpus":"beta"}`)
	hits, _ = out["hits"].([]any)
	if len(hits) != 1 || hits[0].(map[string]any)["rel_path"] != "b.md" {
		t.Errorf("beta isolation broken: %v", hits)
	}
}

func TestSearchRejectsBadCorpus(t *testing.T) {
	s := newTestServer(t, true)
	code, out := doSearch(t, s, `{"query":"x","corpus":"../etc"}`)
	if code != http.StatusBadRequest {
		t.Errorf("traversal corpus should be 400, got %d (%v)", code, out)
	}
}

func TestCorpusRejectedWhenMultiCorpusOff(t *testing.T) {
	s := newTestServer(t, false) // root == ""
	code, _ := doSearch(t, s, `{"query":"x","corpus":"alpha"}`)
	if code != http.StatusBadRequest {
		t.Errorf("corpus on single-mode server should be 400, got %d", code)
	}
	// ...but the default (no corpus) still works.
	code, _ = doSearch(t, s, `{"query":"x"}`)
	if code != http.StatusOK {
		t.Errorf("default corpus search should be 200, got %d", code)
	}
}

func TestSearchExcerptChars(t *testing.T) {
	s := newTestServer(t, false)
	long := "Guida crociera\n" + strings.Repeat("informazioni generali sul viaggio. ", 1000) +
		"\nImbarco a Civitavecchia dalle 13:00 alle 16:00.\n" +
		strings.Repeat("informazioni generali sul viaggio. ", 1000)
	seedCorpus(t, s, "", "crociera.md", long)

	// Default: full content, no content_len — unchanged response shape.
	_, out := doSearch(t, s, `{"query":"imbarco Civitavecchia"}`)
	hit := out["hits"].([]any)[0].(map[string]any)
	if hit["content"] != long {
		t.Error("without excerpt_chars content must be the full file")
	}
	if _, ok := hit["content_len"]; ok {
		t.Error("content_len must only appear when excerpt_chars is set")
	}

	// With excerpt_chars: the matching passage, within budget, plus total length.
	_, out = doSearch(t, s, `{"query":"imbarco Civitavecchia","excerpt_chars":2000}`)
	hit = out["hits"].([]any)[0].(map[string]any)
	content := hit["content"].(string)
	if !strings.Contains(content, "Civitavecchia dalle 13:00") {
		t.Error("excerpt should carry the passage matching the query")
	}
	if !strings.HasPrefix(content, "Guida crociera") {
		t.Error("excerpt should keep the file's opening")
	}
	if len(content) > 2000+len("\n[...truncated]") {
		t.Errorf("excerpt over budget: %d bytes", len(content))
	}
	if n, _ := hit["content_len"].(float64); int(n) != len(long) {
		t.Errorf("content_len = %v, want %d", hit["content_len"], len(long))
	}
}

func TestSearchRejectsNegativeExcerpt(t *testing.T) {
	s := newTestServer(t, false)
	code, _ := doSearch(t, s, `{"query":"x","excerpt_chars":-1}`)
	if code != http.StatusBadRequest {
		t.Errorf("negative excerpt_chars should be 400, got %d", code)
	}
}
