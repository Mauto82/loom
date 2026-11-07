package llm

// embed.go — embedding clients for optional hybrid (BM25 + vector) search.
//
// Embeddings are separate from the chat Client: a small embedding model
// (e.g. Ollama "embeddinggemma-2:270m") turns text into a dense vector. They
// share the same HTTP plumbing (postJSON / newHTTPClients) as the chat clients.

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sort"
	"strings"
	"time"
)

// Embedder turns text into dense vectors. One call may embed a batch of texts;
// the returned slice is aligned 1:1 with the input order.
type Embedder interface {
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	Name() string
}

// --- Task prompts -------------------------------------------------------------
//
// Some embedding models are trained with task-specific prompts and retrieve
// noticeably better when queries and documents carry them. EmbeddingGemma
// (v1 and v2) expects "task: search result | query: …" for queries and
// "title: … | text: …" for documents. Embedders whose model needs prompts
// implement taskFormatter; for every other model QueryText/DocumentText return
// the plain text, so behaviour is unchanged.

// promptVersion tags vectors built with task prompts. It is appended to the
// embedder Name (stored per vector), so vectors built without prompts — or with
// a future prompt format — are ignored at query time until re-embedded.
const promptVersion = "+tp1"

type taskFormatter interface {
	formatQuery(q string) string
	formatDocument(title, content string) string
}

// QueryText returns the text to embed for a search query.
func QueryText(e Embedder, q string) string {
	if f, ok := e.(taskFormatter); ok {
		if out := f.formatQuery(q); out != "" {
			return out
		}
	}
	return q
}

// DocumentText returns the text to embed for an indexed file.
func DocumentText(e Embedder, title, content string) string {
	if f, ok := e.(taskFormatter); ok {
		if out := f.formatDocument(title, content); out != "" {
			return out
		}
	}
	return strings.TrimSpace(title + "\n\n" + content)
}

// usesGemmaPrompts reports whether model is an EmbeddingGemma variant
// (e.g. "embeddinggemma:300m", "embeddinggemma-2:270m", "google/embeddinggemma-300m").
func usesGemmaPrompts(model string) bool {
	return strings.Contains(strings.ToLower(model), "embeddinggemma")
}

// gemmaPrompts implements taskFormatter for EmbeddingGemma models; it returns
// "" (→ plain text fallback) for any other model.
type gemmaPrompts struct{ model string }

func (g gemmaPrompts) formatQuery(q string) string {
	if !usesGemmaPrompts(g.model) {
		return ""
	}
	return "task: search result | query: " + strings.TrimSpace(q)
}

func (g gemmaPrompts) formatDocument(title, content string) string {
	if !usesGemmaPrompts(g.model) {
		return ""
	}
	title = strings.TrimSpace(title)
	if title == "" {
		title = "none"
	}
	return "title: " + title + " | text: " + strings.TrimSpace(content)
}

func (g gemmaPrompts) nameSuffix() string {
	if usesGemmaPrompts(g.model) {
		return promptVersion
	}
	return ""
}

// BuildEmbedder constructs an Embedder from plain config values. It mirrors the
// per-command buildLLM helpers so the cmd packages don't depend on config here.
func BuildEmbedder(provider, model, endpoint, apiKey string) (Embedder, error) {
	switch provider {
	case "ollama", "":
		if model == "" {
			return nil, fmt.Errorf("ollama embeddings require a model (e.g. embeddinggemma-2:270m)")
		}
		return NewOllamaEmbedder(endpoint, model, 0), nil
	case "openai":
		if apiKey == "" {
			return nil, fmt.Errorf("openai embeddings require api_key_env to point at a non-empty env var")
		}
		if model == "" {
			model = "text-embedding-3-small"
		}
		return NewOpenAIEmbedder(endpoint, model, apiKey, 0), nil
	default:
		return nil, fmt.Errorf("unknown embeddings provider %q (supported: ollama, openai)", provider)
	}
}

// --- Ollama -------------------------------------------------------------------

type OllamaEmbedder struct {
	gemmaPrompts
	endpoint string
	model    string
	hc       *http.Client
}

func NewOllamaEmbedder(endpoint, model string, timeout time.Duration) *OllamaEmbedder {
	if endpoint == "" {
		endpoint = "http://localhost:11434"
	}
	hc, _ := newHTTPClients(timeout)
	return &OllamaEmbedder{gemmaPrompts: gemmaPrompts{model}, endpoint: strings.TrimRight(endpoint, "/"), model: model, hc: hc}
}

func (e *OllamaEmbedder) Name() string { return "ollama-embed:" + e.model + e.nameSuffix() }

type ollamaEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type ollamaEmbedResponse struct {
	Embeddings [][]float32 `json:"embeddings"`
}

func (e *OllamaEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(ollamaEmbedRequest{Model: e.model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("marshal ollama embed request: %w", err)
	}
	resp, err := postJSON(ctx, e.hc, e.endpoint+"/api/embed", nil, body)
	if err != nil {
		return nil, fmt.Errorf("ollama embed: %w", err)
	}
	defer resp.Body.Close()

	var out ollamaEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode ollama embed response: %w", err)
	}
	if len(out.Embeddings) != len(texts) {
		return nil, fmt.Errorf("ollama embed: got %d vectors for %d inputs", len(out.Embeddings), len(texts))
	}
	return out.Embeddings, nil
}

// --- OpenAI (and OpenAI-compatible endpoints) ---------------------------------

type OpenAIEmbedder struct {
	gemmaPrompts
	endpoint string
	model    string
	apiKey   string
	hc       *http.Client
}

func NewOpenAIEmbedder(endpoint, model, apiKey string, timeout time.Duration) *OpenAIEmbedder {
	if endpoint == "" {
		endpoint = "https://api.openai.com"
	}
	hc, _ := newHTTPClients(timeout)
	return &OpenAIEmbedder{gemmaPrompts: gemmaPrompts{model}, endpoint: strings.TrimRight(endpoint, "/"), model: model, apiKey: apiKey, hc: hc}
}

func (e *OpenAIEmbedder) Name() string { return "openai-embed:" + e.model + e.nameSuffix() }

type openAIEmbedRequest struct {
	Model string   `json:"model"`
	Input []string `json:"input"`
}

type openAIEmbedResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

func (e *OpenAIEmbedder) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}
	body, err := json.Marshal(openAIEmbedRequest{Model: e.model, Input: texts})
	if err != nil {
		return nil, fmt.Errorf("marshal openai embed request: %w", err)
	}
	resp, err := postJSON(ctx, e.hc, e.endpoint+"/v1/embeddings",
		map[string]string{"Authorization": "Bearer " + e.apiKey}, body)
	if err != nil {
		return nil, fmt.Errorf("openai embed: %w", err)
	}
	defer resp.Body.Close()

	var out openAIEmbedResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return nil, fmt.Errorf("decode openai embed response: %w", err)
	}
	if len(out.Data) != len(texts) {
		return nil, fmt.Errorf("openai embed: got %d vectors for %d inputs", len(out.Data), len(texts))
	}
	// The API may return data out of order — sort by Index before extracting.
	sort.SliceStable(out.Data, func(i, j int) bool { return out.Data[i].Index < out.Data[j].Index })
	vecs := make([][]float32, len(out.Data))
	for i := range out.Data {
		vecs[i] = out.Data[i].Embedding
	}
	return vecs, nil
}
