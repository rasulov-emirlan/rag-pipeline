package vectorstore

import (
	"context"
	"testing"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

func TestMemoryStore_StoreAndSearch(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	chunks := []domain.Chunk{
		{ID: "doc1-0", DocumentID: "doc1", Content: "Go is a programming language"},
		{ID: "doc1-1", DocumentID: "doc1", Content: "Python is also a language"},
		{ID: "doc2-0", DocumentID: "doc2", Content: "Kubernetes orchestrates containers"},
	}
	// Simple embeddings: first chunk is [1,0,0], second [0,1,0], third [0,0,1]
	embeddings := [][]float32{
		{1, 0, 0},
		{0, 1, 0},
		{0, 0, 1},
	}

	if err := store.Store(ctx, chunks, embeddings); err != nil {
		t.Fatalf("store: %v", err)
	}

	// Query closest to first chunk.
	results, err := store.Search(ctx, []float32{0.9, 0.1, 0}, 2)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(results))
	}
	if results[0].Chunk.ID != "doc1-0" {
		t.Fatalf("expected doc1-0 as top result, got %s", results[0].Chunk.ID)
	}
	if results[0].Score <= results[1].Score {
		t.Fatal("results should be ordered by descending score")
	}
}

func TestMemoryStore_SearchEmpty(t *testing.T) {
	store := NewMemoryStore()
	results, err := store.Search(context.Background(), []float32{1, 0, 0}, 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}

func TestMemoryStore_Delete(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	chunks := []domain.Chunk{
		{ID: "doc1-0", DocumentID: "doc1", Content: "chunk a"},
		{ID: "doc1-1", DocumentID: "doc1", Content: "chunk b"},
		{ID: "doc2-0", DocumentID: "doc2", Content: "chunk c"},
	}
	embeddings := [][]float32{{1, 0}, {0, 1}, {1, 1}}
	store.Store(ctx, chunks, embeddings)

	if err := store.Delete(ctx, "doc1"); err != nil {
		t.Fatalf("delete: %v", err)
	}

	results, _ := store.Search(ctx, []float32{1, 1}, 10)
	if len(results) != 1 {
		t.Fatalf("expected 1 result after delete, got %d", len(results))
	}
	if results[0].Chunk.DocumentID != "doc2" {
		t.Fatalf("expected doc2, got %s", results[0].Chunk.DocumentID)
	}
}

func TestMemoryStore_SearchKLargerThanStore(t *testing.T) {
	store := NewMemoryStore()
	ctx := context.Background()

	store.Store(ctx, []domain.Chunk{{ID: "a", Content: "hello"}}, [][]float32{{1, 0}})

	results, err := store.Search(ctx, []float32{1, 0}, 100)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result (capped), got %d", len(results))
	}
}

func TestCosineSimilarity(t *testing.T) {
	// Identical vectors → 1.0
	sim := cosineSimilarity([]float32{1, 0, 0}, []float32{1, 0, 0})
	if sim < 0.999 {
		t.Fatalf("identical vectors should have similarity ~1.0, got %f", sim)
	}

	// Orthogonal vectors → 0.0
	sim = cosineSimilarity([]float32{1, 0, 0}, []float32{0, 1, 0})
	if sim > 0.001 {
		t.Fatalf("orthogonal vectors should have similarity ~0.0, got %f", sim)
	}

	// Zero vector → 0.0
	sim = cosineSimilarity([]float32{0, 0, 0}, []float32{1, 0, 0})
	if sim != 0 {
		t.Fatalf("zero vector should have similarity 0, got %f", sim)
	}
}
