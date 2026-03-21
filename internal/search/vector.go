package search

import (
	"context"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

// VectorRetriever wraps a VectorStore + Embedder as a simple Retriever
// (vector-only search, no hybrid). Used as fallback when BM25 is not available.
type VectorRetriever struct {
	store    domain.VectorStore
	embedder domain.Embedder
}

func NewVectorRetriever(store domain.VectorStore, embedder domain.Embedder) *VectorRetriever {
	return &VectorRetriever{store: store, embedder: embedder}
}

func (v *VectorRetriever) Retrieve(ctx context.Context, query string, k int) ([]domain.SearchResult, error) {
	embeddings, err := v.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	return v.store.Search(ctx, embeddings[0], k)
}
