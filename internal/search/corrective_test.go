package search

import (
	"context"
	"testing"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

type mockLLM struct {
	response string
}

func (m *mockLLM) Generate(_ context.Context, _ string) (string, error) {
	return m.response, nil
}

type mockRetriever struct {
	results []domain.SearchResult
	calls   int
}

func (m *mockRetriever) Retrieve(_ context.Context, _ string, _ int) ([]domain.SearchResult, error) {
	m.calls++
	return m.results, nil
}

func TestCorrectiveRetriever_GoodResults(t *testing.T) {
	base := &mockRetriever{results: []domain.SearchResult{
		{Chunk: domain.Chunk{ID: "c1", Content: "Go is great"}, Score: 0.9},
	}}
	llm := &mockLLM{response: "GOOD"}
	cr := NewCorrectiveRetriever(base, llm)

	results, err := cr.Retrieve(context.Background(), "Tell me about Go", 5)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result, got %d", len(results))
	}
	if base.calls != 1 {
		t.Fatalf("expected 1 base call (no re-retrieve), got %d", base.calls)
	}
}

func TestCorrectiveRetriever_BadResults_Rephrases(t *testing.T) {
	base := &mockRetriever{results: []domain.SearchResult{
		{Chunk: domain.Chunk{ID: "c1", Content: "irrelevant stuff"}, Score: 0.3},
	}}
	llm := &mockLLM{response: "BAD"}
	cr := NewCorrectiveRetriever(base, llm)

	_, err := cr.Retrieve(context.Background(), "original query", 5)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	// Should have called base twice: initial + re-retrieve.
	if base.calls != 2 {
		t.Fatalf("expected 2 base calls (initial + re-retrieve), got %d", base.calls)
	}
}

func TestCorrectiveRetriever_EmptyResults(t *testing.T) {
	base := &mockRetriever{results: nil}
	llm := &mockLLM{response: "GOOD"}
	cr := NewCorrectiveRetriever(base, llm)

	results, err := cr.Retrieve(context.Background(), "test", 5)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}
