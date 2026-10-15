package index

import "testing"

func TestIgnoreMatch(t *testing.T) {
	cases := []struct {
		pattern string
		path    string
		isDir   bool
		want    bool
	}{
		{"build/", "build", true, true},
		{"build/", "a/b/build", true, true},
		{"build/", "build", false, false}, // dir-only rule, file named build
		{"/build/", "a/build", true, false},
		{"*.bak", "x.bak", false, true},
		{"*.bak", "deep/dir/x.bak", false, true},
		{"*.bak", "x.md", false, false},
		{"archive/v1/", "archive/v1", true, true},
		{"archive/v1/", "other/archive/v1", true, false},
		{"/notes.md", "notes.md", false, true},
		{"/notes.md", "sub/notes.md", false, false},
		{"**/tmp/", "tmp", true, true},
		{"**/tmp/", "a/b/tmp", true, true},
		{"papers/**/*.draft.md", "papers/x.draft.md", false, true},
		{"papers/**/*.draft.md", "papers/a/b/x.draft.md", false, true},
		{"papers/**/*.draft.md", "other/x.draft.md", false, false},
		{"docs/*.pdf", "docs/a.pdf", false, true},
		{"docs/*.pdf", "docs/sub/a.pdf", false, false},
	}
	for _, c := range cases {
		rules, err := compileIgnore([]string{c.pattern}, "test")
		if err != nil {
			t.Fatalf("%q: %v", c.pattern, err)
		}
		m := &ignoreMatcher{rules: rules}
		if got := m.Match(c.path, c.isDir); got != c.want {
			t.Errorf("pattern %q path %q dir=%v: got %v, want %v", c.pattern, c.path, c.isDir, got, c.want)
		}
	}
}

func TestCompileIgnoreSkipsCommentsAndBlanks(t *testing.T) {
	rules, err := compileIgnore([]string{"", "  ", "# comment", "build/"}, "test")
	if err != nil {
		t.Fatal(err)
	}
	if len(rules) != 1 {
		t.Errorf("expected 1 rule, got %d", len(rules))
	}
}

func TestCompileIgnoreRejectsNegation(t *testing.T) {
	if _, err := compileIgnore([]string{"!keep.md"}, "test"); err == nil {
		t.Error("expected error for unsupported negation")
	}
}

func TestNilMatcherMatchesNothing(t *testing.T) {
	var m *ignoreMatcher
	if m.Match("anything", true) {
		t.Error("nil matcher should match nothing")
	}
}
