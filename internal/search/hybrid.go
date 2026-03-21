package search

import (
	"context"
	"sort"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

const (
	defaultVectorK = 20   // retrieve top-N from each source
	defaultAlpha   = 0.7  // vector weight in fusion (0-1)
	rrfK           = 60   // RRF constant
)

// HybridRetriever combines vector similarity search with BM25 keyword search
// using Reciprocal Rank Fusion (RRF).
type HybridRetriever struct {
	store    domain.VectorStore
	embedder domain.Embedder
	bm25     *BM25Index
}

func NewHybridRetriever(store domain.VectorStore, embedder domain.Embedder, bm25 *BM25Index) *HybridRetriever {
	return &HybridRetriever{
		store:    store,
		embedder: embedder,
		bm25:     bm25,
	}
}

func (h *HybridRetriever) Retrieve(ctx context.Context, query string, k int) ([]domain.SearchResult, error) {
	fetchK := defaultVectorK
	if fetchK < k*3 {
		fetchK = k * 3
	}

	// 1. Vector search.
	embeddings, err := h.embedder.Embed(ctx, []string{query})
	if err != nil {
		return nil, err
	}
	vectorResults, err := h.store.Search(ctx, embeddings[0], fetchK)
	if err != nil {
		return nil, err
	}

	// 2. BM25 search.
	bm25Results := h.bm25.Search(query, fetchK)

	// 3. If BM25 has no results, fall back to vector-only.
	if len(bm25Results) == 0 {
		if k > len(vectorResults) {
			k = len(vectorResults)
		}
		return vectorResults[:k], nil
	}

	// 4. Reciprocal Rank Fusion.
	// Build chunk lookup from vector results (we need full Chunk data).
	chunkMap := make(map[string]domain.SearchResult)
	vectorRank := make(map[string]int)
	for i, r := range vectorResults {
		chunkMap[r.Chunk.ID] = r
		vectorRank[r.Chunk.ID] = i + 1
	}

	bm25Rank := make(map[string]int)
	for i, r := range bm25Results {
		bm25Rank[r.ID] = i + 1
	}

	// Collect all unique chunk IDs.
	allIDs := make(map[string]bool)
	for _, r := range vectorResults {
		allIDs[r.Chunk.ID] = true
	}
	for _, r := range bm25Results {
		allIDs[r.ID] = true
	}

	// Compute RRF scores.
	type fused struct {
		id    string
		score float64
	}
	var fusedResults []fused

	for id := range allIDs {
		var score float64
		if rank, ok := vectorRank[id]; ok {
			score += defaultAlpha / float64(rank+rrfK)
		}
		if rank, ok := bm25Rank[id]; ok {
			score += (1 - defaultAlpha) / float64(rank+rrfK)
		}
		fusedResults = append(fusedResults, fused{id: id, score: score})
	}

	sort.Slice(fusedResults, func(i, j int) bool {
		return fusedResults[i].score > fusedResults[j].score
	})

	if k > len(fusedResults) {
		k = len(fusedResults)
	}

	// Build final results. Use chunk data from vector results where available.
	results := make([]domain.SearchResult, 0, k)
	for i := 0; i < k; i++ {
		f := fusedResults[i]
		if sr, ok := chunkMap[f.id]; ok {
			sr.Score = f.score
			results = append(results, sr)
		}
		// Chunks only in BM25 results are skipped since we don't have full Chunk data.
		// In practice, most relevant chunks appear in both.
	}

	return results, nil
}
