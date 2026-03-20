package ingest

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

// DocumentInfo stores metadata about an ingested document.
type DocumentInfo struct {
	ID         string         `json:"id"`
	ChunkCount int            `json:"chunk_count"`
	Metadata   map[string]any `json:"metadata,omitempty"`
	IngestedAt time.Time      `json:"ingested_at"`
}

// IngestResult is returned after successful ingestion.
type IngestResult struct {
	DocumentID string        `json:"document_id"`
	ChunkCount int           `json:"chunk_count"`
	Duration   time.Duration `json:"duration_ms"`
}

// Service orchestrates the document ingestion pipeline:
// chunk → embed → store.
type Service struct {
	chunker  domain.Chunker
	embedder domain.Embedder
	store    domain.VectorStore

	mu   sync.RWMutex
	docs map[string]DocumentInfo
}

func New(chunker domain.Chunker, embedder domain.Embedder, store domain.VectorStore) *Service {
	return &Service{
		chunker:  chunker,
		embedder: embedder,
		store:    store,
		docs:     make(map[string]DocumentInfo),
	}
}

// IngestDocument chunks, embeds, and stores a document.
func (s *Service) IngestDocument(ctx context.Context, doc domain.Document) (*IngestResult, error) {
	start := time.Now()

	if doc.ID == "" {
		doc.ID = generateID()
	}

	// 1. Chunk the document.
	texts, err := s.chunker.Split(doc.Content)
	if err != nil {
		return nil, fmt.Errorf("chunking: %w", err)
	}

	if len(texts) == 0 {
		return nil, fmt.Errorf("document produced no chunks")
	}

	// 2. Build chunk objects.
	chunks := make([]domain.Chunk, len(texts))
	for i, text := range texts {
		chunks[i] = domain.Chunk{
			ID:         fmt.Sprintf("%s-%d", doc.ID, i),
			DocumentID: doc.ID,
			Content:    text,
			Metadata:   doc.Metadata,
		}
	}

	// 3. Embed all chunks.
	embeddings, err := s.embed(ctx, texts)
	if err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}

	// 4. Store in vector store.
	if err := s.store.Store(ctx, chunks, embeddings); err != nil {
		return nil, fmt.Errorf("storing: %w", err)
	}

	// 5. Track document metadata.
	s.mu.Lock()
	s.docs[doc.ID] = DocumentInfo{
		ID:         doc.ID,
		ChunkCount: len(chunks),
		Metadata:   doc.Metadata,
		IngestedAt: time.Now(),
	}
	s.mu.Unlock()

	duration := time.Since(start)
	slog.Info("document ingested",
		"id", doc.ID,
		"chunks", len(chunks),
		"duration_ms", duration.Milliseconds(),
	)

	return &IngestResult{
		DocumentID: doc.ID,
		ChunkCount: len(chunks),
		Duration:   duration,
	}, nil
}

// DeleteDocument removes a document and all its chunks.
func (s *Service) DeleteDocument(ctx context.Context, documentID string) error {
	if err := s.store.Delete(ctx, documentID); err != nil {
		return fmt.Errorf("deleting chunks: %w", err)
	}

	s.mu.Lock()
	delete(s.docs, documentID)
	s.mu.Unlock()

	slog.Info("document deleted", "id", documentID)
	return nil
}

// ListDocuments returns metadata for all ingested documents.
func (s *Service) ListDocuments() []DocumentInfo {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]DocumentInfo, 0, len(s.docs))
	for _, info := range s.docs {
		result = append(result, info)
	}
	return result
}

// embed generates embeddings, returning zero vectors if no embedder is configured.
func (s *Service) embed(ctx context.Context, texts []string) ([][]float32, error) {
	if s.embedder == nil {
		// No embedder configured — return zero vectors (dev/test mode).
		dim := 3
		embeddings := make([][]float32, len(texts))
		for i := range embeddings {
			embeddings[i] = make([]float32, dim)
		}
		return embeddings, nil
	}
	return s.embedder.Embed(ctx, texts)
}

func generateID() string {
	var buf [8]byte
	rand.Read(buf[:])
	return hex.EncodeToString(buf[:])
}
