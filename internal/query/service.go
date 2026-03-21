package query

import (
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

// QueryRequest holds parameters for a RAG query.
type QueryRequest struct {
	Question string `json:"question"`
	K        int    `json:"k,omitempty"`
}

// QueryResponse is returned after a RAG query.
type QueryResponse struct {
	Answer   string   `json:"answer"`
	Sources  []Source `json:"sources"`
	Duration int64    `json:"duration_ms"`
}

// Source represents a retrieved chunk used to generate the answer.
type Source struct {
	DocumentID string         `json:"document_id"`
	Content    string         `json:"content"`
	Score      float64        `json:"score"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// Reranker re-scores search results for better precision.
type Reranker interface {
	Rerank(ctx context.Context, query string, results []domain.SearchResult, topN int) ([]domain.SearchResult, error)
}

// Cache provides optional response caching using raw JSON.
type Cache interface {
	Get(ctx context.Context, key string) (json.RawMessage, bool)
	Set(ctx context.Context, key string, data json.RawMessage)
}

// Service orchestrates the RAG query pipeline:
// cache check → retrieve → rerank → build prompt → LLM → answer.
type Service struct {
	retriever domain.Retriever
	llm       domain.LLM
	reranker  Reranker // optional, nil = skip
	cache     Cache    // optional, nil = skip
	defaultK  int
}

func New(retriever domain.Retriever, llm domain.LLM, reranker Reranker, cache Cache, defaultK int) *Service {
	if defaultK <= 0 {
		defaultK = 5
	}
	return &Service{
		retriever: retriever,
		llm:       llm,
		reranker:  reranker,
		cache:     cache,
		defaultK:  defaultK,
	}
}

// Query performs retrieval-augmented generation.
func (s *Service) Query(ctx context.Context, req QueryRequest) (*QueryResponse, error) {
	start := time.Now()

	if req.Question == "" {
		return nil, fmt.Errorf("question is required")
	}

	k := req.K
	if k <= 0 {
		k = s.defaultK
	}

	// 1. Cache check.
	cacheKey := normalizeCacheKey(req.Question, k)
	if s.cache != nil {
		if data, ok := s.cache.Get(ctx, cacheKey); ok {
			var cached QueryResponse
			if err := json.Unmarshal(data, &cached); err == nil {
				slog.Info("cache hit", "question_len", len(req.Question))
				return &cached, nil
			}
		}
	}

	// 2. Retrieve relevant chunks.
	// Fetch more candidates if reranking is enabled.
	retrieveK := k
	if s.reranker != nil {
		retrieveK = k * 3
		if retrieveK < 15 {
			retrieveK = 15
		}
	}

	results, err := s.retriever.Retrieve(ctx, req.Question, retrieveK)
	if err != nil {
		return nil, fmt.Errorf("retrieval: %w", err)
	}

	// 3. Rerank if configured.
	if s.reranker != nil && len(results) > 0 {
		results, err = s.reranker.Rerank(ctx, req.Question, results, k)
		if err != nil {
			slog.Warn("reranking failed, using original order", "error", err)
			// Fall back to original results, trimmed to k.
			if k < len(results) {
				results = results[:k]
			}
		}
	} else if k < len(results) {
		results = results[:k]
	}

	// 4. Build RAG prompt.
	prompt := buildPrompt(req.Question, results)

	// 5. Generate answer.
	answer, err := s.generate(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("generation: %w", err)
	}

	// 6. Build response.
	sources := make([]Source, len(results))
	for i, r := range results {
		sources[i] = Source{
			DocumentID: r.Chunk.DocumentID,
			Content:    r.Chunk.Content,
			Score:      r.Score,
			Metadata:   r.Chunk.Metadata,
		}
	}

	duration := time.Since(start)
	resp := &QueryResponse{
		Answer:   answer,
		Sources:  sources,
		Duration: duration.Milliseconds(),
	}

	// 7. Cache store.
	if s.cache != nil {
		if data, err := json.Marshal(resp); err == nil {
			s.cache.Set(ctx, cacheKey, data)
		}
	}

	slog.Info("query completed",
		"question_len", len(req.Question),
		"chunks_retrieved", len(results),
		"duration_ms", duration.Milliseconds(),
	)

	return resp, nil
}

// Retrieve returns relevant chunks without generating an answer.
func (s *Service) Retrieve(ctx context.Context, question string, k int) ([]domain.SearchResult, error) {
	return s.retriever.Retrieve(ctx, question, k)
}

// generate calls the LLM or returns a fallback if none is configured.
func (s *Service) generate(ctx context.Context, prompt string) (string, error) {
	if s.llm == nil {
		return "[no LLM configured — retrieval only mode]", nil
	}
	return s.llm.Generate(ctx, prompt)
}

func buildPrompt(question string, results []domain.SearchResult) string {
	var b strings.Builder
	b.WriteString("Answer the question based on the following context. ")
	b.WriteString("If the context doesn't contain the answer, say \"I don't have enough information to answer that.\"\n\n")
	b.WriteString("Context:\n")

	for _, r := range results {
		b.WriteString("---\n")
		b.WriteString(r.Chunk.Content)
		b.WriteString("\n")
	}

	b.WriteString("---\n\n")
	b.WriteString("Question: ")
	b.WriteString(question)
	b.WriteString("\n\nAnswer:")
	return b.String()
}

func normalizeCacheKey(question string, k int) string {
	return fmt.Sprintf("%s:%d", strings.ToLower(strings.TrimSpace(question)), k)
}
