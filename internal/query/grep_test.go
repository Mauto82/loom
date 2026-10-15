package query

import (
	"context"
	"strings"
	"testing"

	"github.com/Mauto82/loom/internal/storage"
)

func seedGrep(t *testing.T) *storage.Store {
	t.Helper()
	s := newStore(t)
	seed(t, s, []storage.File{
		{RelPath: "a.md", Hash: "1", MTime: 1, Kind: "md", Title: "A",
			Content: "prima riga\nFattura 123 pagata\nterza riga\nfattura 456 aperta"},
		{RelPath: "b.md", Hash: "2", MTime: 1, Kind: "md", Title: "B",
			Content: "niente qui\nFATTURA 789"},
	})
	return s
}

func TestGrepAcrossFiles(t *testing.T) {
	s := seedGrep(t)
	res, err := Grep(context.Background(), s, GrepOptions{Pattern: `fattura\s+\d+`, IgnoreCase: true})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Matches) != 3 || res.Truncated {
		t.Fatalf("expected 3 matches, got %+v", res)
	}
	m := res.Matches[0]
	if m.RelPath != "a.md" || m.Line != 2 || m.Text != "Fattura 123 pagata" {
		t.Errorf("unexpected first match %+v", m)
	}
	content := "prima riga\nFattura 123 pagata\nterza riga\nfattura 456 aperta"
	if !strings.HasPrefix(content[m.Offset:], "Fattura 123") {
		t.Errorf("offset %d does not point at the matching line", m.Offset)
	}
}

func TestGrepCaseSensitive(t *testing.T) {
	s := seedGrep(t)
	res, err := Grep(context.Background(), s, GrepOptions{Pattern: "Fattura"})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Matches) != 1 {
		t.Errorf("case-sensitive grep should find 1 match, got %d", len(res.Matches))
	}
}

func TestGrepSingleFileAndContext(t *testing.T) {
	s := seedGrep(t)
	res, err := Grep(context.Background(), s, GrepOptions{Pattern: "terza", RelPath: "a.md", Context: 1})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Matches) != 1 {
		t.Fatalf("expected 1 match, got %d", len(res.Matches))
	}
	m := res.Matches[0]
	if len(m.Before) != 1 || m.Before[0] != "Fattura 123 pagata" ||
		len(m.After) != 1 || m.After[0] != "fattura 456 aperta" {
		t.Errorf("unexpected context %+v", m)
	}
}

func TestGrepTruncates(t *testing.T) {
	s := seedGrep(t)
	res, err := Grep(context.Background(), s, GrepOptions{Pattern: "a", MaxMatches: 2})
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Matches) != 2 || !res.Truncated {
		t.Errorf("expected 2 matches and truncated=true, got %d/%v", len(res.Matches), res.Truncated)
	}
}

func TestGrepErrors(t *testing.T) {
	s := seedGrep(t)
	for _, opts := range []GrepOptions{
		{Pattern: ""},
		{Pattern: "("},
		{Pattern: "x", RelPath: "missing.md"},
	} {
		if _, err := Grep(context.Background(), s, opts); err == nil {
			t.Errorf("expected error for %+v", opts)
		}
	}
}

func TestGrepClipsLongLines(t *testing.T) {
	s := newStore(t)
	long := strings.Repeat("x", 5000) + "AGO" + strings.Repeat("y", 5000)
	seed(t, s, []storage.File{{RelPath: "p.pdf", Hash: "1", MTime: 1, Kind: "pdf", Content: long}})
	res, err := Grep(context.Background(), s, GrepOptions{Pattern: "AGO"})
	if err != nil {
		t.Fatal(err)
	}
	text := res.Matches[0].Text
	if !strings.Contains(text, "AGO") || len(text) > grepLineChars+len("……") {
		t.Errorf("long line not clipped around match: %d bytes", len(text))
	}
}
