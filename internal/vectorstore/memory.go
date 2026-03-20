package vectorstore

import (
	"context"
	"math"
	"sort"
	"sync"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

type storedChunk struct {
	chunk     domain.Chunk
	embedding []float32
}

// MemoryStore is an in-memory vector store using brute-force cosine similarity.
// Suitable for tests, development, and small datasets.
type MemoryStore struct {
	mu     sync.RWMutex
	chunks []storedChunk
}

func NewMemoryStore() *MemoryStore {
	return &MemoryStore{}
}

func (s *MemoryStore) Store(_ context.Context, chunks []domain.Chunk, embeddings [][]float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for i, c := range chunks {
		s.chunks = append(s.chunks, storedChunk{
			chunk:     c,
			embedding: embeddings[i],
		})
	}
	return nil
}

func (s *MemoryStore) Search(_ context.Context, embedding []float32, k int) ([]domain.SearchResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if len(s.chunks) == 0 {
		return nil, nil
	}

	type scored struct {
		chunk domain.Chunk
		score float64
	}

	results := make([]scored, 0, len(s.chunks))
	for _, sc := range s.chunks {
		sim := cosineSimilarity(embedding, sc.embedding)
		results = append(results, scored{chunk: sc.chunk, score: sim})
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if k > len(results) {
		k = len(results)
	}

	out := make([]domain.SearchResult, k)
	for i := 0; i < k; i++ {
		out[i] = domain.SearchResult{
			Chunk: results[i].chunk,
			Score: results[i].score,
		}
	}
	return out, nil
}

func (s *MemoryStore) Delete(_ context.Context, documentID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	filtered := s.chunks[:0]
	for _, sc := range s.chunks {
		if sc.chunk.DocumentID != documentID {
			filtered = append(filtered, sc)
		}
	}
	s.chunks = filtered
	return nil
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) != len(b) {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom == 0 {
		return 0
	}
	return dot / denom
}
