package search

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

func setupHybridTest(t *testing.T) (*HybridRetriever, *vectorstore.MemoryStore, *BM25Index) {
	t.Helper()

	store := vectorstore.NewMemoryStore()
	bm25 := NewBM25Index()
	ctx := context.Background()

	chunks := []domain.Chunk{
		{ID: "doc1-0", DocumentID: "doc1", Content: "Go is a programming language created at Google"},
		{ID: "doc1-1", DocumentID: "doc1", Content: "Goroutines enable lightweight concurrency in Go"},
		{ID: "doc2-0", DocumentID: "doc2", Content: "Kubernetes orchestrates Docker containers at scale"},
	}
	embeddings := [][]float32{
		{1, 0, 0},     // Go topic
		{0.8, 0.2, 0}, // Go concurrency topic
		{0, 0, 1},     // K8s topic
	}

	store.Store(ctx, chunks, embeddings)

	// Feed BM25 index.
	for _, c := range chunks {
		bm25.Add(c.ID, c.DocumentID, c.Content)
	}

	embedder := &mockEmbedder{dim: 3, embedding: []float32{0.9, 0.1, 0}} // query about Go
	retriever := NewHybridRetriever(store, embedder, bm25)
	return retriever, store, bm25
}

func TestHybridRetriever_CombinesResults(t *testing.T) {
	retriever, _, _ := setupHybridTest(t)

	results, err := retriever.Retrieve(context.Background(), "Go programming language", 2)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(results) < 1 {
		t.Fatal("expected at least 1 result")
	}
	// Top result should be about Go (matches both vector and BM25).
	if results[0].Chunk.ID != "doc1-0" {
		t.Fatalf("expected doc1-0 as top result, got %s", results[0].Chunk.ID)
	}
}

func TestHybridRetriever_BM25BoostsKeywordMatch(t *testing.T) {
	retriever, _, _ := setupHybridTest(t)

	// Query "Kubernetes" — BM25 should strongly boost the K8s chunk
	// even if vector similarity is low.
	retriever.embedder = &mockEmbedder{dim: 3, embedding: []float32{0.3, 0.3, 0.4}}
	results, err := retriever.Retrieve(context.Background(), "Kubernetes containers", 3)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(results) == 0 {
		t.Fatal("expected results")
	}

	// K8s chunk should appear in results due to BM25 keyword match.
	found := false
	for _, r := range results {
		if r.Chunk.ID == "doc2-0" {
			found = true
			break
		}
	}
	if !found {
		t.Fatal("expected K8s chunk (doc2-0) in results due to keyword match")
	}
}

func TestHybridRetriever_EmptyBM25FallsBackToVector(t *testing.T) {
	store := vectorstore.NewMemoryStore()
	bm25 := NewBM25Index() // empty index
	ctx := context.Background()

	store.Store(ctx,
		[]domain.Chunk{{ID: "c1", Content: "test content"}},
		[][]float32{{1, 0}},
	)

	embedder := &mockEmbedder{dim: 2, embedding: []float32{1, 0}}
	retriever := NewHybridRetriever(store, embedder, bm25)

	results, err := retriever.Retrieve(ctx, "test", 1)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result from vector fallback, got %d", len(results))
	}
}
