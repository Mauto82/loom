package query

import (
	"context"
	"fmt"
	"regexp"
	"strings"

	"github.com/Mauto82/loom/internal/storage"
)

const (
	// DefaultGrepMaxMatches / MaxGrepMaxMatches bound how many matching lines
	// Grep returns, so a pattern like "e" can't dump the whole corpus.
	DefaultGrepMaxMatches = 50
	MaxGrepMaxMatches     = 500

	// MaxGrepContext caps the context lines returned around each match.
	MaxGrepContext = 10

	// grepLineChars caps each returned line. Extracted PDFs often come out as a
	// few enormous lines; past this we keep the part around the match.
	grepLineChars = 400
)

// GrepOptions configures a Grep call.
type GrepOptions struct {
	Pattern    string // RE2 regular expression
	RelPath    string // restrict to one file; empty = every indexed file
	IgnoreCase bool
	Context    int // lines of context before and after each match
	MaxMatches int
}

// GrepMatch is one matching line.
type GrepMatch struct {
	RelPath string   `json:"rel_path"`
	Line    int      `json:"line"`   // 1-based line number in the extracted content
	Offset  int      `json:"offset"` // byte offset of the line, usable with loom_get_file
	Text    string   `json:"text"`
	Before  []string `json:"before,omitempty"`
	After   []string `json:"after,omitempty"`
}

// GrepResult is what Grep returns.
type GrepResult struct {
	Matches   []GrepMatch `json:"matches"`
	Truncated bool        `json:"truncated"` // true when MaxMatches cut the results short
}

// Grep runs a regular expression over the extracted content of indexed files,
// line by line. It complements Retrieve: ranked search finds files that are
// about a topic, Grep finds every exact mention (names, codes, dates) and where
// it sits, so an agent can read just that part of a long file. No LLM call.
// Go's RE2 engine runs in linear time, so a caller-supplied pattern can't
// trigger catastrophic backtracking.
func Grep(ctx context.Context, store *storage.Store, opts GrepOptions) (*GrepResult, error) {
	pattern := opts.Pattern
	if pattern == "" {
		return nil, fmt.Errorf("empty pattern")
	}
	if opts.IgnoreCase {
		pattern = "(?i)" + pattern
	}
	re, err := regexp.Compile(pattern)
	if err != nil {
		return nil, fmt.Errorf("invalid pattern: %w", err)
	}
	maxMatches := opts.MaxMatches
	if maxMatches <= 0 {
		maxMatches = DefaultGrepMaxMatches
	}
	maxMatches = min(maxMatches, MaxGrepMaxMatches)
	ctxLines := min(max(opts.Context, 0), MaxGrepContext)

	var files []*storage.File
	if opts.RelPath != "" {
		f, err := store.GetByRelPath(ctx, opts.RelPath)
		if err != nil {
			return nil, err
		}
		files = []*storage.File{f}
	} else if files, err = store.List(ctx); err != nil {
		return nil, err
	}

	res := &GrepResult{Matches: []GrepMatch{}}
	for _, f := range files {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		lines := strings.Split(f.Content, "\n")
		offset := 0
		for i, line := range lines {
			lineOffset := offset
			offset += len(line) + 1
			loc := re.FindStringIndex(line)
			if loc == nil {
				continue
			}
			if len(res.Matches) == maxMatches {
				res.Truncated = true
				return res, nil
			}
			m := GrepMatch{
				RelPath: f.RelPath,
				Line:    i + 1,
				Offset:  lineOffset,
				Text:    clipAround(line, loc[0], loc[1]),
			}
			for j := max(0, i-ctxLines); j < i; j++ {
				m.Before = append(m.Before, clipAround(lines[j], 0, 0))
			}
			for j := i + 1; j <= min(len(lines)-1, i+ctxLines); j++ {
				m.After = append(m.After, clipAround(lines[j], 0, 0))
			}
			res.Matches = append(res.Matches, m)
		}
	}
	return res, nil
}

// clipAround shortens line to about grepLineChars, centred on [from, to).
func clipAround(line string, from, to int) string {
	if len(line) <= grepLineChars {
		return line
	}
	pad := max((grepLineChars-(to-from))/2, 0)
	start := snapBack(line, max(from-pad, 0))
	end := snapBack(line, min(to+pad, len(line)))
	if end-start < grepLineChars && end < len(line) {
		end = snapBack(line, min(start+grepLineChars, len(line)))
	}
	out := line[start:end]
	if start > 0 {
		out = "…" + out
	}
	if end < len(line) {
		out += "…"
	}
	return out
}
