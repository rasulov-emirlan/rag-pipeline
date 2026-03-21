package search

import (
	"context"
	"fmt"
	"log/slog"
	"strings"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

// CorrectiveRetriever wraps a base Retriever with self-correction.
// It evaluates retrieval quality using an LLM and re-retrieves with
// a rephrased query if results are poor (CRAG pattern).
type CorrectiveRetriever struct {
	base    domain.Retriever
	llm     domain.LLM
}

func NewCorrectiveRetriever(base domain.Retriever, llm domain.LLM) *CorrectiveRetriever {
	return &CorrectiveRetriever{
		base: base,
		llm:  llm,
	}
}

func (c *CorrectiveRetriever) Retrieve(ctx context.Context, query string, k int) ([]domain.SearchResult, error) {
	// 1. Initial retrieval.
	results, err := c.base.Retrieve(ctx, query, k)
	if err != nil {
		return nil, err
	}

	if len(results) == 0 {
		return results, nil
	}

	// 2. Evaluate retrieval quality.
	assessment := c.evaluate(ctx, query, results)

	switch assessment {
	case "GOOD":
		slog.Debug("corrective RAG: results are good", "query_len", len(query))
		return results, nil

	case "PARTIAL":
		slog.Info("corrective RAG: filtering weak results", "query_len", len(query))
		return c.filterRelevant(ctx, query, results), nil

	default: // "BAD"
		slog.Info("corrective RAG: rephrasing and re-retrieving", "query_len", len(query))
		rephrased, err := c.rephrase(ctx, query)
		if err != nil {
			slog.Warn("corrective RAG: rephrase failed, returning original", "error", err)
			return results, nil
		}
		return c.base.Retrieve(ctx, rephrased, k)
	}
}

// evaluate asks the LLM to assess retrieval quality.
func (c *CorrectiveRetriever) evaluate(ctx context.Context, query string, results []domain.SearchResult) string {
	var snippets strings.Builder
	for i, r := range results {
		if i >= 5 {
			break
		}
		content := r.Chunk.Content
		if len(content) > 200 {
			content = content[:200]
		}
		fmt.Fprintf(&snippets, "- %s\n", content)
	}

	prompt := fmt.Sprintf(
		"You are evaluating search results for a question.\n\n"+
			"Question: %s\n\n"+
			"Search results:\n%s\n"+
			"Are these results relevant to answering the question?\n"+
			"Answer with exactly one word: GOOD, PARTIAL, or BAD.",
		query, snippets.String(),
	)

	resp, err := c.llm.Generate(ctx, prompt)
	if err != nil {
		slog.Warn("corrective RAG: evaluation failed", "error", err)
		return "GOOD" // fail-open
	}

	upper := strings.ToUpper(strings.TrimSpace(resp))
	if strings.Contains(upper, "BAD") {
		return "BAD"
	}
	if strings.Contains(upper, "PARTIAL") {
		return "PARTIAL"
	}
	return "GOOD"
}

// filterRelevant asks the LLM to score each result and keeps only relevant ones.
func (c *CorrectiveRetriever) filterRelevant(ctx context.Context, query string, results []domain.SearchResult) []domain.SearchResult {
	var filtered []domain.SearchResult
	for _, r := range results {
		prompt := fmt.Sprintf(
			"Is this passage relevant to the question? Answer YES or NO only.\n\n"+
				"Question: %s\n"+
				"Passage: %s",
			query, truncateStr(r.Chunk.Content, 300),
		)

		resp, err := c.llm.Generate(ctx, prompt)
		if err != nil {
			filtered = append(filtered, r) // keep on error
			continue
		}

		if strings.Contains(strings.ToUpper(resp), "YES") {
			filtered = append(filtered, r)
		}
	}

	if len(filtered) == 0 {
		return results // don't return empty
	}
	return filtered
}

// rephrase asks the LLM to rewrite the query for better retrieval.
func (c *CorrectiveRetriever) rephrase(ctx context.Context, query string) (string, error) {
	prompt := fmt.Sprintf(
		"Rephrase this search query to get better results. "+
			"Keep the same meaning but use different words. "+
			"Only output the rephrased query, nothing else.\n\n"+
			"Original query: %s\n\nRephrased query:",
		query,
	)

	resp, err := c.llm.Generate(ctx, prompt)
	if err != nil {
		return "", err
	}

	rephrased := strings.TrimSpace(resp)
	if rephrased == "" {
		return query, nil
	}
	return rephrased, nil
}

func truncateStr(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
