package query

import (
	"context"
	"testing"

	"github.com/erasulov/rag-pipeline/internal/domain"
	"github.com/erasulov/rag-pipeline/internal/vectorstore"
)

type mockEmbedder struct {
	dim       int
	embedding []float32
}

func (m *mockEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range texts {
		result[i] = make([]float32, m.dim)
		copy(result[i], m.embedding)
	}
	return result, nil
}

func (m *mockEmbedder) Dimension() int { return m.dim }

type mockLLM struct {
	response string
}

func (m *mockLLM) Generate(_ context.Context, prompt string) (string, error) {
	return m.response, nil
}

// mockRetriever wraps a store + embedder as a simple retriever for tests.
type mockRetriever struct {
	store    *vectorstore.MemoryStore
	embedder *mockEmbedder
}

func (r *mockRetriever) Retrieve(ctx context.Context, query string, k int) ([]domain.SearchResult, error) {
	embs, err := r.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	return r.store.Search(ctx, embs[0], k)
}

func setupQueryTest(t *testing.T) *Service {
	t.Helper()

	store := vectorstore.NewMemoryStore()
	ctx := context.Background()

	chunks := []domain.Chunk{
		{ID: "doc1-0", DocumentID: "doc1", Content: "Paris is the capital of France."},
		{ID: "doc1-1", DocumentID: "doc1", Content: "France is a country in Europe."},
		{ID: "doc2-0", DocumentID: "doc2", Content: "Go is a programming language."},
	}
	embeddings := [][]float32{
		{1, 0, 0},
		{0.8, 0.2, 0},
		{0, 0, 1},
	}
	store.Store(ctx, chunks, embeddings)

	embedder := &mockEmbedder{dim: 3, embedding: []float32{0.9, 0.1, 0}}
	retriever := &mockRetriever{store: store, embedder: embedder}
	llm := &mockLLM{response: "The capital of France is Paris."}

	return New(retriever, llm, nil, nil, 5)
}

func TestService_Query(t *testing.T) {
	svc := setupQueryTest(t)

	resp, err := svc.Query(context.Background(), QueryRequest{
		Question: "What is the capital of France?",
		K:        2,
	})
	if err != nil {
		t.Fatalf("query: %v", err)
	}

	if resp.Answer != "The capital of France is Paris." {
		t.Fatalf("unexpected answer: %s", resp.Answer)
	}
	if len(resp.Sources) != 2 {
		t.Fatalf("expected 2 sources, got %d", len(resp.Sources))
	}
	if resp.Sources[0].DocumentID != "doc1" {
		t.Fatalf("expected doc1 as top source, got %s", resp.Sources[0].DocumentID)
	}
}

func TestService_Query_EmptyQuestion(t *testing.T) {
	svc := setupQueryTest(t)
	_, err := svc.Query(context.Background(), QueryRequest{})
	if err == nil {
		t.Fatal("expected error for empty question")
	}
}

func TestService_Retrieve(t *testing.T) {
	svc := setupQueryTest(t)

	results, err := svc.Retrieve(context.Background(), "Tell me about France", 2)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Chunk.ID != "doc1-0" {
		t.Fatalf("expected doc1-0, got %s", results[0].Chunk.ID)
	}
}

func TestService_Query_NoLLM(t *testing.T) {
	store := vectorstore.NewMemoryStore()
	store.Store(context.Background(),
		[]domain.Chunk{{ID: "a", DocumentID: "d", Content: "test"}},
		[][]float32{{1, 0}},
	)

	embedder := &mockEmbedder{dim: 2, embedding: []float32{1, 0}}
	retriever := &mockRetriever{store: store, embedder: embedder}
	svc := New(retriever, nil, nil, nil, 5)

	resp, err := svc.Query(context.Background(), QueryRequest{Question: "test?"})
	if err != nil {
		t.Fatalf("query: %v", err)
	}
	if resp.Answer == "" {
		t.Fatal("expected fallback answer when no LLM configured")
	}
}

func TestBuildPrompt(t *testing.T) {
	results := []domain.SearchResult{
		{Chunk: domain.Chunk{Content: "Paris is the capital."}, Score: 0.9},
	}
	prompt := buildPrompt("What is the capital?", results)

	if len(prompt) == 0 {
		t.Fatal("prompt should not be empty")
	}
	if !containsStr(prompt, "Paris is the capital.") {
		t.Fatal("prompt should contain context")
	}
	if !containsStr(prompt, "What is the capital?") {
		t.Fatal("prompt should contain question")
	}
}

func containsStr(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
