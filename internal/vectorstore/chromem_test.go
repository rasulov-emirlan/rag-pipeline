package vectorstore

import (
	"context"
	"testing"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

func TestChromemStore_StoreAndSearch(t *testing.T) {
	store, err := NewChromemStore("") // in-memory
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	ctx := context.Background()

	chunks := []domain.Chunk{
		{ID: "doc1-0", DocumentID: "doc1", Content: "Go is a programming language"},
		{ID: "doc1-1", DocumentID: "doc1", Content: "Python is also a language"},
		{ID: "doc2-0", DocumentID: "doc2", Content: "Kubernetes orchestrates containers"},
	}
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
}

func TestChromemStore_Delete(t *testing.T) {
	store, err := NewChromemStore("")
	if err != nil {
		t.Fatalf("create: %v", err)
	}
	ctx := context.Background()

	store.Store(ctx,
		[]domain.Chunk{
			{ID: "doc1-0", DocumentID: "doc1", Content: "chunk a"},
			{ID: "doc2-0", DocumentID: "doc2", Content: "chunk b"},
		},
		[][]float32{{1, 0}, {0, 1}},
	)

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

func TestChromemStore_SearchEmpty(t *testing.T) {
	store, err := NewChromemStore("")
	if err != nil {
		t.Fatalf("create: %v", err)
	}

	results, err := store.Search(context.Background(), []float32{1, 0, 0}, 5)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}
