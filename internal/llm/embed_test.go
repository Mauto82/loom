package llm

import "testing"

func TestGemmaTaskPrompts(t *testing.T) {
	for _, model := range []string{"embeddinggemma:300m", "embeddinggemma-2:270m", "google/EmbeddingGemma-300m"} {
		e := NewOllamaEmbedder("", model, 0)
		if got, want := QueryText(e, " carbonara "), "task: search result | query: carbonara"; got != want {
			t.Errorf("%s query = %q, want %q", model, got, want)
		}
		if got, want := DocumentText(e, "Ricette", "guanciale e pecorino"), "title: Ricette | text: guanciale e pecorino"; got != want {
			t.Errorf("%s doc = %q, want %q", model, got, want)
		}
		if got, want := DocumentText(e, "", "body"), "title: none | text: body"; got != want {
			t.Errorf("%s untitled doc = %q, want %q", model, got, want)
		}
		if got, want := e.Name(), "ollama-embed:"+model+promptVersion; got != want {
			t.Errorf("name = %q, want %q", got, want)
		}
	}
}

func TestNonGemmaModelsUsePlainText(t *testing.T) {
	embs := []Embedder{
		NewOllamaEmbedder("", "mxbai-embed-large", 0),
		NewOpenAIEmbedder("", "text-embedding-3-small", "k", 0),
	}
	for _, e := range embs {
		if got := QueryText(e, "carbonara"); got != "carbonara" {
			t.Errorf("%s query = %q, want plain", e.Name(), got)
		}
		if got, want := DocumentText(e, "Ricette", "guanciale"), "Ricette\n\nguanciale"; got != want {
			t.Errorf("%s doc = %q, want %q", e.Name(), got, want)
		}
	}
	if got, want := embs[0].Name(), "ollama-embed:mxbai-embed-large"; got != want {
		t.Errorf("name changed for non-gemma model: %q", got)
	}
}

func TestGemmaPromptsOnOpenAICompatibleEndpoint(t *testing.T) {
	e := NewOpenAIEmbedder("http://localhost:8080", "embeddinggemma-2", "k", 0)
	if got := QueryText(e, "q"); got != "task: search result | query: q" {
		t.Errorf("query = %q", got)
	}
}
