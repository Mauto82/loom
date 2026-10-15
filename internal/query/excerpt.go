package query

import (
	"math"
	"strings"
	"unicode"
	"unicode/utf8"
)

const (
	// excerptHeadChars is how much of the file's opening we always keep when a
	// file is too long to send whole: titles, abstracts and intros often answer
	// "what is this document about" questions the relevant window would miss.
	excerptHeadChars = 1200

	// excerptStemLen caps query terms to a prefix so Italian/English inflections
	// still match ("ricetta" → "ricett" hits "ricette"). Crude, but the window
	// only needs to land near the answer, not rank documents.
	excerptStemLen = 6

	// excerptMaxCount stops one term repeated 200 times in a window from
	// drowning out a window that mentions every term once.
	excerptMaxCount = 3

	truncatedMarker = "\n[...truncated]"
	skippedMarker   = "\n\n[...]\n\n"
)

// excerpt returns at most ~budget bytes of content to show the LLM for a
// question. Short content is returned as-is. Long content keeps the opening
// excerptHeadChars plus the window that best matches the question's terms, so
// an answer buried on page 20 of a PDF is no longer cut off. When no window
// matches any term (or the best one overlaps the opening) it falls back to the
// plain head truncation Loom always used.
func excerpt(content, question string, budget int) string {
	if len(content) <= budget {
		return content
	}
	head := content[:snapBack(content, budget)] + truncatedMarker

	terms := queryTerms(question)
	if len(terms) == 0 {
		return head
	}
	headLen := snapBack(content, excerptHeadChars)
	winLen := budget - headLen - len(skippedMarker)
	if winLen <= 0 {
		return head
	}

	start, ok := bestWindow(content, terms, winLen)
	if !ok || start <= headLen+len(skippedMarker) {
		// Best window fits inside what head truncation shows anyway.
		return head
	}

	end := start + winLen
	if end > len(content) {
		end = len(content)
	}
	start, end = snapForward(content, start), snapBack(content, end)
	out := content[:headLen] + skippedMarker + content[start:end]
	if end < len(content) {
		out += truncatedMarker
	}
	return out
}

// bestWindow slides a winLen-byte window over content in quarter-window steps
// and returns the start of the one scoring highest. A term's weight is its
// inverse window frequency, so rare terms ("pgbouncer") decide over ones that
// appear everywhere in the document ("database"). ok is false when no window
// contains any term.
func bestWindow(content string, terms []string, winLen int) (start int, ok bool) {
	step := winLen / 4
	if step < 1 {
		step = 1
	}
	var starts []int
	var counts [][]int
	df := make([]int, len(terms))
	for s := 0; ; s += step {
		e := s + winLen
		if e > len(content) {
			e = len(content)
		}
		lower := strings.ToLower(content[s:e])
		c := make([]int, len(terms))
		for i, t := range terms {
			c[i] = strings.Count(lower, t)
			if c[i] > 0 {
				df[i]++
			}
		}
		starts = append(starts, s)
		counts = append(counts, c)
		if e == len(content) {
			break
		}
	}

	n := float64(len(starts))
	best := 0.0
	for w, c := range counts {
		score := 0.0
		for i, k := range c {
			if k == 0 {
				continue
			}
			if k > excerptMaxCount {
				k = excerptMaxCount
			}
			idf := math.Log(1 + n/float64(df[i]))
			score += idf * (1 + math.Log(float64(k)))
		}
		if score > best {
			best, start, ok = score, starts[w], true
		}
	}
	return start, ok
}

// queryTerms lowercases the question, keeps words of 3+ runes that aren't
// stopwords, and truncates each to excerptStemLen runes.
func queryTerms(question string) []string {
	seen := map[string]bool{}
	var out []string
	for _, w := range strings.FieldsFunc(strings.ToLower(question), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	}) {
		if utf8.RuneCountInString(w) < 3 || stopwords[w] {
			continue
		}
		if r := []rune(w); len(r) > excerptStemLen {
			w = string(r[:excerptStemLen])
		}
		if !seen[w] {
			seen[w] = true
			out = append(out, w)
		}
	}
	return out
}

// stopwords are frequent Italian and English words of 3+ letters that would
// otherwise pull the window toward any paragraph at all.
var stopwords = map[string]bool{
	// Italian
	"che": true, "chi": true, "cosa": true, "come": true, "dove": true, "quando": true,
	"perché": true, "perche": true, "quale": true, "quali": true, "quanto": true, "quanti": true,
	"per": true, "con": true, "tra": true, "fra": true, "una": true, "uno": true,
	"del": true, "della": true, "dello": true, "dei": true, "degli": true, "delle": true,
	"nel": true, "nella": true, "nei": true, "nelle": true, "sul": true, "sulla": true,
	"dal": true, "dalla": true, "dai": true, "dagli": true, "dalle": true, "sui": true,
	"sugli": true, "sulle": true, "negli": true, "alla": true, "allo": true, "agli": true, "alle": true,
	"non": true, "sono": true, "era": true, "erano": true, "essere": true, "stato": true,
	"hai": true, "abbiamo": true, "hanno": true, "questo": true, "questa": true,
	"quello": true, "quella": true, "anche": true, "più": true, "molto": true,
	"mio": true, "mia": true, "miei": true, "mie": true, "tuo": true, "tua": true,
	"suo": true, "sua": true, "loro": true, "scritto": true, "note": true, "nota": true,
	// English
	"the": true, "and": true, "for": true, "with": true, "what": true, "which": true,
	"who": true, "how": true, "why": true, "when": true, "where": true, "this": true,
	"that": true, "these": true, "those": true, "are": true, "was": true, "were": true,
	"have": true, "has": true, "had": true, "did": true, "does": true, "about": true,
	"from": true, "into": true, "not": true, "you": true, "your": true, "my": true,
	"wrote": true, "notes": true, "any": true, "all": true,
}

// snapBack moves i back to the nearest rune boundary so a slice never splits a
// UTF-8 sequence.
func snapBack(s string, i int) int {
	if i >= len(s) {
		return len(s)
	}
	for i > 0 && !utf8.RuneStart(s[i]) {
		i--
	}
	return i
}

// snapForward moves i to a rune boundary, then to just after the next
// whitespace if one is close, so a window doesn't open mid-word.
func snapForward(s string, i int) int {
	for i < len(s) && !utf8.RuneStart(s[i]) {
		i++
	}
	if j := strings.IndexFunc(s[i:min(len(s), i+80)], unicode.IsSpace); j >= 0 {
		i += j + 1
	}
	return i
}

// Range slices content by byte offset and length, snapped to rune boundaries,
// and returns the slice with its actual [start, end). length <= 0 means "to the
// end". An offset past the end yields an empty slice, not an error, so callers
// paging through a file can stop when they see it.
func Range(content string, offset, length int) (string, int, int) {
	if offset < 0 {
		offset = 0
	}
	start := snapBack(content, offset)
	end := len(content)
	if length > 0 && start+length < end {
		end = snapBack(content, start+length)
	}
	return content[start:end], start, end
}
