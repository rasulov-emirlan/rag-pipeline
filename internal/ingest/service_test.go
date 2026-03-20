package ingest

import (
	"context"
	"testing"

	"github.com/erasulov/rag-pipeline/internal/chunker"
	"github.com/erasulov/rag-pipeline/internal/domain"
	"github.com/erasulov/rag-pipeline/internal/vectorstore"
)

type mockEmbedder struct {
	dim int
}

func (m *mockEmbedder) Embed(_ context.Context, texts []string) ([][]float32, error) {
	result := make([][]float32, len(texts))
	for i := range texts {
		// Simple deterministic embedding: hash-like based on text length.
		vec := make([]float32, m.dim)
		for j := range vec {
			vec[j] = float32(len(texts[i])+i+j) / 100.0
		}
		result[i] = vec
	}
	return result, nil
}

func (m *mockEmbedder) Dimension() int { return m.dim }

func TestService_IngestDocument(t *testing.T) {
	store := vectorstore.NewMemoryStore()
	ch := chunker.NewRecursive(50, 0)
	embedder := &mockEmbedder{dim: 3}
	svc := New(ch, embedder, store)

	doc := domain.Document{
		ID:      "test-doc",
		Content: "This is a test document with enough content to be split into multiple chunks by the chunker.",
		Metadata: map[string]any{
			"source": "test",
		},
	}

	result, err := svc.IngestDocument(context.Background(), doc)
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}

	if result.DocumentID != "test-doc" {
		t.Fatalf("expected id test-doc, got %s", result.DocumentID)
	}
	if result.ChunkCount < 2 {
		t.Fatalf("expected multiple chunks, got %d", result.ChunkCount)
	}

	// Verify chunks are searchable.
	results, err := store.Search(context.Background(), []float32{0.5, 0.5, 0.5}, 10)
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(results) != result.ChunkCount {
		t.Fatalf("expected %d results, got %d", result.ChunkCount, len(results))
	}
}

func TestService_IngestDocument_AutoGeneratesID(t *testing.T) {
	store := vectorstore.NewMemoryStore()
	ch := chunker.NewRecursive(1000, 0)
	svc := New(ch, nil, store)

	result, err := svc.IngestDocument(context.Background(), domain.Document{
		Content: "Some content",
	})
	if err != nil {
		t.Fatalf("ingest: %v", err)
	}
	if result.DocumentID == "" {
		t.Fatal("expected auto-generated ID")
	}
}

func TestService_DeleteDocument(t *testing.T) {
	store := vectorstore.NewMemoryStore()
	ch := chunker.NewRecursive(1000, 0)
	embedder := &mockEmbedder{dim: 3}
	svc := New(ch, embedder, store)

	svc.IngestDocument(context.Background(), domain.Document{
		ID: "doc1", Content: "Document one content",
	})
	svc.IngestDocument(context.Background(), domain.Document{
		ID: "doc2", Content: "Document two content",
	})

	if err := svc.DeleteDocument(context.Background(), "doc1"); err != nil {
		t.Fatalf("delete: %v", err)
	}

	docs := svc.ListDocuments()
	if len(docs) != 1 {
		t.Fatalf("expected 1 document, got %d", len(docs))
	}
	if docs[0].ID != "doc2" {
		t.Fatalf("expected doc2, got %s", docs[0].ID)
	}
}

func TestService_ListDocuments(t *testing.T) {
	store := vectorstore.NewMemoryStore()
	ch := chunker.NewRecursive(1000, 0)
	svc := New(ch, nil, store)

	svc.IngestDocument(context.Background(), domain.Document{ID: "a", Content: "aaa"})
	svc.IngestDocument(context.Background(), domain.Document{ID: "b", Content: "bbb"})

	docs := svc.ListDocuments()
	if len(docs) != 2 {
		t.Fatalf("expected 2 documents, got %d", len(docs))
	}
}
