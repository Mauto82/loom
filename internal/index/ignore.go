package index

import (
	"bufio"
	"errors"
	"fmt"
	"os"
	"path"
	"path/filepath"
	"strings"
)

// IgnoreFile is the name of the per-folder ignore file. It lives at the root
// of a notes dir and applies only to that dir. Being a dotfile, the walk never
// indexes it.
const IgnoreFile = ".loomignore"

// ignoreRule is one compiled gitignore-style pattern.
//
// Supported syntax (a subset of .gitignore):
//   - blank lines and lines starting with "#" are skipped
//   - a trailing "/" matches directories only ("build/")
//   - a pattern with no "/" (other than a trailing one) matches the name at any
//     depth ("*.bak", "drafts/")
//   - a pattern with a "/" is anchored to the notes dir root ("archive/v1/",
//     "/old.md"); a leading "/" is dropped
//   - "*", "?" and "[...]" as in path.Match; "**" as a whole segment matches
//     zero or more directories ("**/tmp/", "papers/**/*.draft.md")
//
// Negation ("!pattern") is not supported.
type ignoreRule struct {
	segments []string // pattern split on "/"; a single segment when unanchored
	anchored bool
	dirOnly  bool
}

// ignoreMatcher holds the rules that apply to one notes dir.
type ignoreMatcher struct {
	rules []ignoreRule
}

// compileIgnore turns raw pattern lines into rules. source names where the
// lines came from (file path or "config") for error messages.
func compileIgnore(lines []string, source string) ([]ignoreRule, error) {
	var rules []ignoreRule
	for i, raw := range lines {
		line := strings.TrimSpace(raw)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "!") {
			return nil, fmt.Errorf("%s:%d: negation (%q) is not supported", source, i+1, line)
		}
		r := ignoreRule{}
		if strings.HasSuffix(line, "/") {
			r.dirOnly = true
			line = strings.TrimRight(line, "/")
		}
		if strings.HasPrefix(line, "/") {
			r.anchored = true
			line = strings.TrimLeft(line, "/")
		}
		if line == "" {
			continue // "/" alone would ignore the whole folder: treat as no-op
		}
		if strings.Contains(line, "/") {
			r.anchored = true
		}
		r.segments = strings.Split(line, "/")
		for _, seg := range r.segments {
			if seg == "**" {
				continue
			}
			if _, err := path.Match(seg, ""); err != nil {
				return nil, fmt.Errorf("%s:%d: bad pattern %q: %w", source, i+1, raw, err)
			}
		}
		rules = append(rules, r)
	}
	return rules, nil
}

// loadIgnore builds the matcher for root from its .loomignore (if any) plus
// the extra patterns from config. A missing .loomignore is not an error.
func loadIgnore(root string, extra []string) (*ignoreMatcher, error) {
	m := &ignoreMatcher{}

	fromCfg, err := compileIgnore(extra, "config ignore")
	if err != nil {
		return nil, err
	}
	m.rules = append(m.rules, fromCfg...)

	p := filepath.Join(root, IgnoreFile)
	f, err := os.Open(p)
	if errors.Is(err, os.ErrNotExist) {
		return m, nil
	}
	if err != nil {
		return nil, fmt.Errorf("read %s: %w", p, err)
	}
	defer f.Close()
	var lines []string
	sc := bufio.NewScanner(f)
	for sc.Scan() {
		lines = append(lines, sc.Text())
	}
	if err := sc.Err(); err != nil {
		return nil, fmt.Errorf("read %s: %w", p, err)
	}
	fromFile, err := compileIgnore(lines, p)
	if err != nil {
		return nil, err
	}
	m.rules = append(m.rules, fromFile...)
	return m, nil
}

// Match reports whether relPath (slash-separated, relative to the notes dir
// root) is ignored. The walk calls it for every entry and skips ignored
// directories whole, so a rule never needs to match a path's descendants.
func (m *ignoreMatcher) Match(relPath string, isDir bool) bool {
	if m == nil || len(m.rules) == 0 {
		return false
	}
	parts := strings.Split(relPath, "/")
	for _, r := range m.rules {
		if r.dirOnly && !isDir {
			continue
		}
		if !r.anchored {
			if ok, _ := path.Match(r.segments[0], parts[len(parts)-1]); ok {
				return true
			}
			continue
		}
		if matchSegments(r.segments, parts) {
			return true
		}
	}
	return false
}

// matchSegments matches pattern segments against path segments, with "**"
// standing for zero or more path segments.
func matchSegments(pat, parts []string) bool {
	for len(pat) > 0 {
		if pat[0] == "**" {
			rest := pat[1:]
			if len(rest) == 0 {
				return true
			}
			for i := 0; i <= len(parts); i++ {
				if matchSegments(rest, parts[i:]) {
					return true
				}
			}
			return false
		}
		if len(parts) == 0 {
			return false
		}
		if ok, _ := path.Match(pat[0], parts[0]); !ok {
			return false
		}
		pat, parts = pat[1:], parts[1:]
	}
	return len(parts) == 0
}
