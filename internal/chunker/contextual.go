package chunker

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

const contextWindowChars = 2000 // first N chars of document used for context

// Contextual wraps a base chunker and enriches each chunk with document-level
// context using an LLM (Anthropic's "Contextual Retrieval" approach).
// This prepends a 1-2 sentence description to each chunk before embedding,
// reducing retrieval failures by up to 49%.
type Contextual struct {
	base domain.Chunker
	llm  domain.LLM
}

func NewContextual(base domain.Chunker, llm domain.LLM) *Contextual {
	return &Contextual{base: base, llm: llm}
}

func (c *Contextual) Split(text string) ([]string, error) {
	// 1. Split with base chunker.
	chunks, err := c.base.Split(text)
	if err != nil {
		return nil, err
	}

	if len(chunks) <= 1 {
		// Single chunk — no context needed.
		return chunks, nil
	}

	// 2. Extract document summary window.
	docWindow := text
	if len(docWindow) > contextWindowChars {
		docWindow = docWindow[:contextWindowChars]
	}

	// 3. Enrich each chunk with context.
	enriched := make([]string, len(chunks))
	for i, chunk := range chunks {
		ctx := context.Background()
		contextDesc, err := c.generateContext(ctx, docWindow, chunk)
		if err != nil {
			slog.Warn("contextual chunking failed for chunk, using raw",
				"chunk_index", i, "error", err)
			enriched[i] = chunk
			continue
		}
		enriched[i] = contextDesc + "\n\n" + chunk
	}

	return enriched, nil
}

func (c *Contextual) generateContext(ctx context.Context, docWindow, chunk string) (string, error) {
	prompt := fmt.Sprintf(
		"Here is a document:\n%s\n\n"+
			"Here is a chunk from that document:\n%s\n\n"+
			"Write a short 1-2 sentence context that explains where this chunk fits "+
			"in the document and what topic it covers. Be specific and concise. "+
			"Only output the context, nothing else.",
		docWindow, truncateChunk(chunk, 500),
	)

	resp, err := c.llm.Generate(ctx, prompt)
	if err != nil {
		return "", fmt.Errorf("generate context: %w", err)
	}

	return strings.TrimSpace(resp), nil
}

func truncateChunk(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
