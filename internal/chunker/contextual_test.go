package chunker

import (
	"context"
	"strings"
	"testing"
)

type mockLLM struct {
	response string
}

func (m *mockLLM) Generate(_ context.Context, _ string) (string, error) {
	return m.response, nil
}

func TestContextual_EnrichesChunks(t *testing.T) {
	base := NewRecursive(50, 0)
	llm := &mockLLM{response: "This chunk discusses Go programming basics."}
	ctx := NewContextual(base, llm)

	text := "Go is a programming language. It was created at Google. Goroutines enable concurrency."
	chunks, err := ctx.Split(text)
	if err != nil {
		t.Fatalf("split: %v", err)
	}

	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}

	// Each chunk should be prefixed with context.
	for i, chunk := range chunks {
		if !strings.Contains(chunk, "This chunk discusses Go programming basics.") {
			t.Errorf("chunk %d missing context prefix: %q", i, chunk[:min(len(chunk), 80)])
		}
	}
}

func TestContextual_SingleChunkNoContext(t *testing.T) {
	base := NewRecursive(1000, 0)
	llm := &mockLLM{response: "should not be called"}
	ctx := NewContextual(base, llm)

	chunks, err := ctx.Split("Short text")
	if err != nil {
		t.Fatalf("split: %v", err)
	}

	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	// Single chunk should NOT have context prepended.
	if strings.Contains(chunks[0], "should not be called") {
		t.Fatal("single chunk should not have context")
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
