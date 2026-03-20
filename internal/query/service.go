package query

import (
	"context"
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

// Service orchestrates the RAG query pipeline:
// embed query → search → build prompt → LLM → answer.
type Service struct {
	embedder domain.Embedder
	store    domain.VectorStore
	llm      domain.LLM
	defaultK int
}

func New(embedder domain.Embedder, store domain.VectorStore, llm domain.LLM, defaultK int) *Service {
	if defaultK <= 0 {
		defaultK = 5
	}
	return &Service{
		embedder: embedder,
		store:    store,
		llm:      llm,
		defaultK: defaultK,
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

	// 1. Retrieve relevant chunks.
	results, err := s.Retrieve(ctx, req.Question, k)
	if err != nil {
		return nil, fmt.Errorf("retrieval: %w", err)
	}

	// 2. Build RAG prompt.
	prompt := buildPrompt(req.Question, results)

	// 3. Generate answer.
	answer, err := s.generate(ctx, prompt)
	if err != nil {
		return nil, fmt.Errorf("generation: %w", err)
	}

	// 4. Build response.
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
	slog.Info("query completed",
		"question_len", len(req.Question),
		"chunks_retrieved", len(results),
		"duration_ms", duration.Milliseconds(),
	)

	return &QueryResponse{
		Answer:   answer,
		Sources:  sources,
		Duration: duration.Milliseconds(),
	}, nil
}

// Retrieve returns relevant chunks without generating an answer.
// Useful for debugging retrieval quality.
func (s *Service) Retrieve(ctx context.Context, question string, k int) ([]domain.SearchResult, error) {
	if s.embedder == nil {
		return nil, fmt.Errorf("no embedder configured")
	}

	embeddings, err := s.embedder.Embed(ctx, []string{question})
	if err != nil {
		return nil, fmt.Errorf("embedding question: %w", err)
	}

	results, err := s.store.Search(ctx, embeddings[0], k)
	if err != nil {
		return nil, fmt.Errorf("searching: %w", err)
	}

	return results, nil
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
