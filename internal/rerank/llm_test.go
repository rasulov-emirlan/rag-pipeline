package rerank

import (
	"context"
	"fmt"
	"testing"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

type mockLLM struct {
	scores map[string]string // chunk content → score response
}

func (m *mockLLM) Generate(_ context.Context, prompt string) (string, error) {
	for content, score := range m.scores {
		if len(prompt) > 0 && containsStr(prompt, content) {
			return score, nil
		}
	}
	return "5", nil
}

func containsStr(s, sub string) bool {
	for i := 0; i <= len(s)-len(sub); i++ {
		if s[i:i+len(sub)] == sub {
			return true
		}
	}
	return false
}

func TestLLMReranker_Rerank(t *testing.T) {
	llm := &mockLLM{scores: map[string]string{
		"Go is great":     "9",
		"Python is cool":  "3",
		"Rust is fast":    "7",
	}}
	reranker := NewLLMReranker(llm)

	results := []domain.SearchResult{
		{Chunk: domain.Chunk{ID: "c1", Content: "Python is cool"}, Score: 0.9},
		{Chunk: domain.Chunk{ID: "c2", Content: "Rust is fast"}, Score: 0.8},
		{Chunk: domain.Chunk{ID: "c3", Content: "Go is great"}, Score: 0.7},
	}

	reranked, err := reranker.Rerank(context.Background(), "Tell me about Go", results, 2)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}

	if len(reranked) != 2 {
		t.Fatalf("expected 2 results, got %d", len(reranked))
	}
	// "Go is great" should be first (score 9), then "Rust is fast" (score 7).
	if reranked[0].Chunk.ID != "c3" {
		t.Fatalf("expected c3 (Go) first, got %s", reranked[0].Chunk.ID)
	}
	if reranked[1].Chunk.ID != "c2" {
		t.Fatalf("expected c2 (Rust) second, got %s", reranked[1].Chunk.ID)
	}
}

func TestLLMReranker_Empty(t *testing.T) {
	reranker := NewLLMReranker(&mockLLM{})
	results, err := reranker.Rerank(context.Background(), "test", nil, 5)
	if err != nil {
		t.Fatalf("rerank: %v", err)
	}
	if len(results) != 0 {
		t.Fatalf("expected 0 results, got %d", len(results))
	}
}

func TestParseScore(t *testing.T) {
	tests := []struct {
		input    string
		expected float64
	}{
		{"8", 8},
		{"  9  ", 9},
		{"7/10", 5}, // "7/10" is one token, ParseFloat fails → default 5
		{"10 out of 10", 10},
		{"abc", 5}, // default
		{"0", 1},   // clamped
		{"15", 10}, // clamped
	}

	for _, tt := range tests {
		got := parseScore(tt.input)
		if got != tt.expected {
			t.Errorf("parseScore(%q) = %v, want %v", tt.input, got, tt.expected)
		}
	}
}

func TestTruncate(t *testing.T) {
	if truncate("short", 100) != "short" {
		t.Fatal("short string should not be truncated")
	}
	result := truncate("a long string that exceeds the limit", 10)
	if len(result) > 13 { // 10 + "..."
		t.Fatalf("expected truncated string, got %q (len %d)", result, len(result))
	}
	_ = fmt.Sprintf("") // avoid unused import
}
