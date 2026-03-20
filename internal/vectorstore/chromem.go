package vectorstore

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"runtime"

	chromem "github.com/philippgille/chromem-go"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

const chromemCollectionName = "documents"

// ChromemStore implements domain.VectorStore using chromem-go,
// an embeddable vector database with zero external dependencies.
type ChromemStore struct {
	db         *chromem.DB
	collection *chromem.Collection
}

// NewChromemStore creates a persistent chromem-go store at the given directory.
// If dir is empty, uses in-memory storage.
func NewChromemStore(dir string) (*ChromemStore, error) {
	var db *chromem.DB
	var err error

	if dir != "" {
		db, err = chromem.NewPersistentDB(dir, false)
	} else {
		db = chromem.NewDB()
	}
	if err != nil {
		return nil, fmt.Errorf("create chromem db: %w", err)
	}

	// Use a dummy embedding function since we provide pre-computed embeddings.
	col, err := db.GetOrCreateCollection(chromemCollectionName, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("create collection: %w", err)
	}

	if dir != "" {
		slog.Info("chromem store opened", "dir", dir)
	} else {
		slog.Info("chromem store opened (in-memory)")
	}

	return &ChromemStore{db: db, collection: col}, nil
}

func (s *ChromemStore) Store(ctx context.Context, chunks []domain.Chunk, embeddings [][]float32) error {
	docs := make([]chromem.Document, len(chunks))
	for i, chunk := range chunks {
		metaJSON, _ := json.Marshal(chunk.Metadata)
		docs[i] = chromem.Document{
			ID:        chunk.ID,
			Content:   chunk.Content,
			Embedding: embeddings[i],
			Metadata: map[string]string{
				"document_id":   chunk.DocumentID,
				"metadata_json": string(metaJSON),
			},
		}
	}

	return s.collection.AddDocuments(ctx, docs, runtime.NumCPU())
}

func (s *ChromemStore) Search(ctx context.Context, embedding []float32, k int) ([]domain.SearchResult, error) {
	count := s.collection.Count()
	if count == 0 {
		return nil, nil
	}
	if k > count {
		k = count
	}

	results, err := s.collection.QueryEmbedding(ctx, embedding, k, nil, nil)
	if err != nil {
		return nil, fmt.Errorf("chromem query: %w", err)
	}

	out := make([]domain.SearchResult, len(results))
	for i, r := range results {
		var metadata map[string]any
		if raw, ok := r.Metadata["metadata_json"]; ok {
			json.Unmarshal([]byte(raw), &metadata)
		}

		out[i] = domain.SearchResult{
			Chunk: domain.Chunk{
				ID:         r.ID,
				DocumentID: r.Metadata["document_id"],
				Content:    r.Content,
				Metadata:   metadata,
			},
			Score: float64(r.Similarity),
		}
	}

	return out, nil
}

func (s *ChromemStore) Delete(ctx context.Context, documentID string) error {
	// Use chromem-go's where filter to match by document_id metadata.
	return s.collection.Delete(ctx,
		map[string]string{"document_id": documentID}, // where
		nil, // whereDocument
	)
}
