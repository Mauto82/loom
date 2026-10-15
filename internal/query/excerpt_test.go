package query

import (
	"context"
	"strings"
	"testing"
	"unicode/utf8"

	"github.com/Mauto82/loom/internal/storage"
)

func filler(n int) string {
	return strings.Repeat("lorem ipsum dolor sit amet. ", n/28+1)[:n]
}

func TestExcerptShortContentUnchanged(t *testing.T) {
	if got := excerpt("breve nota", "nota", 100); got != "breve nota" {
		t.Errorf("short content must pass through, got %q", got)
	}
}

func TestExcerptFindsAnswerDeepInFile(t *testing.T) {
	content := "Titolo del documento\n" + filler(20000) +
		"\nLa configurazione di pgbouncer usa pool_mode transaction.\n" + filler(20000)
	got := excerpt(content, "come ho configurato pgbouncer?", PerFileCharBudget)

	if !strings.Contains(got, "pool_mode transaction") {
		t.Error("excerpt should include the window mentioning the query term")
	}
	if !strings.HasPrefix(got, "Titolo del documento") {
		t.Error("excerpt should keep the file's opening")
	}
	if !strings.Contains(got, "[...]") || !strings.HasSuffix(got, "[...truncated]") {
		t.Errorf("excerpt should mark skipped and truncated parts")
	}
	if len(got) > PerFileCharBudget+len(truncatedMarker) {
		t.Errorf("excerpt over budget: %d bytes", len(got))
	}
}

func TestExcerptPrefersRareTerm(t *testing.T) {
	// "database" appears everywhere; "pgbouncer" only once, far from the start.
	block := strings.Repeat("database tuning notes. ", 400)
	content := block + block + "pgbouncer qui. " + block + block
	got := excerpt(content, "database pgbouncer", 4000)
	if !strings.Contains(got, "pgbouncer qui") {
		t.Error("rare term should decide the window")
	}
}

func TestExcerptFallsBackToHead(t *testing.T) {
	content := "inizio " + filler(20000)
	for _, q := range []string{"zzzz assente", "", "di il la"} {
		got := excerpt(content, q, 1000)
		if !strings.HasPrefix(got, "inizio ") || strings.Contains(got, "[...]\n") {
			t.Errorf("q=%q: expected plain head truncation, got %q", q, got[:40])
		}
		if !strings.HasSuffix(got, truncatedMarker) {
			t.Errorf("q=%q: missing truncation marker", q)
		}
	}
}

func TestExcerptMatchNearStartUsesHead(t *testing.T) {
	content := "pgbouncer in apertura. " + filler(20000)
	got := excerpt(content, "pgbouncer", 4000)
	if strings.Contains(got, skippedMarker) {
		t.Error("a match already visible in the head should not add a second window")
	}
}

func TestExcerptKeepsValidUTF8(t *testing.T) {
	content := strings.Repeat("perché è così ", 2000) + "àncora unica " + strings.Repeat("perché è così ", 2000)
	got := excerpt(content, "àncora", 3001)
	if !utf8.ValidString(got) {
		t.Error("excerpt split a multi-byte rune")
	}
	if !strings.Contains(got, "àncora unica") {
		t.Error("expected window around accented term")
	}
}

func TestQueryTerms(t *testing.T) {
	got := queryTerms("Cosa ho scritto sulle ricette di pasta? Ricette!")
	want := []string{"ricett", "pasta"}
	if strings.Join(got, ",") != strings.Join(want, ",") {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestRange(t *testing.T) {
	s := "abcàdef"
	cases := []struct {
		off, n     int
		want       string
		start, end int
	}{
		{0, 0, s, 0, len(s)},
		{2, 3, "cà", 2, 5}, // "c" + 2-byte "à"
		{4, 2, "à", 3, 5},  // offset inside "à" snaps back
		{100, 5, "", len(s), len(s)},
		{-3, 2, "ab", 0, 2},
	}
	for _, c := range cases {
		got, st, en := Range(s, c.off, c.n)
		if got != c.want || st != c.start || en != c.end {
			t.Errorf("Range(%d,%d) = %q [%d,%d), want %q [%d,%d)", c.off, c.n, got, st, en, c.want, c.start, c.end)
		}
	}
}

func TestAnswerPromptUsesRelevantWindow(t *testing.T) {
	s := newStore(t)
	content := filler(30000) + "\nIl codice del cancello è 4471.\n" + filler(30000)
	seed(t, s, []storage.File{
		{RelPath: "casa.md", Hash: "1", MTime: 1, Kind: "md", Title: "Casa",
			Content: content, Summary: "note sulla casa", Keywords: []string{"casa"}},
	})
	lc := &stubLLM{}
	if _, err := Answer(context.Background(), s, lc, nil, "codice cancello", 5); err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(lc.lastPrompt, "codice del cancello è 4471") {
		t.Error("prompt should carry the passage matching the question, not just the file head")
	}
}
