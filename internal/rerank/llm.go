package rerank

import (
	"context"
	"fmt"
	"log/slog"
	"sort"
	"strconv"
	"strings"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

// LLMReranker uses an LLM to score (query, chunk) pairs for more precise ranking.
type LLMReranker struct {
	llm domain.LLM
}

func NewLLMReranker(llm domain.LLM) *LLMReranker {
	return &LLMReranker{llm: llm}
}

func (r *LLMReranker) Rerank(ctx context.Context, query string, results []domain.SearchResult, topN int) ([]domain.SearchResult, error) {
	if len(results) == 0 {
		return results, nil
	}

	type scored struct {
		result domain.SearchResult
		score  float64
	}
	scoredResults := make([]scored, 0, len(results))

	for _, result := range results {
		prompt := fmt.Sprintf(
			"Rate how relevant this passage is to the question on a scale of 1 to 10.\n"+
				"Only respond with a single number, nothing else.\n\n"+
				"Question: %s\n\n"+
				"Passage: %s\n\n"+
				"Score:",
			query, truncate(result.Chunk.Content, 500),
		)

		resp, err := r.llm.Generate(ctx, prompt)
		if err != nil {
			slog.Warn("rerank score failed", "chunk_id", result.Chunk.ID, "error", err)
			scoredResults = append(scoredResults, scored{result: result, score: 0})
			continue
		}

		score := parseScore(resp)
		scoredResults = append(scoredResults, scored{result: result, score: score})
	}

	sort.Slice(scoredResults, func(i, j int) bool {
		return scoredResults[i].score > scoredResults[j].score
	})

	if topN > len(scoredResults) {
		topN = len(scoredResults)
	}

	out := make([]domain.SearchResult, topN)
	for i := 0; i < topN; i++ {
		out[i] = scoredResults[i].result
		out[i].Score = scoredResults[i].score / 10.0 // normalize to 0-1
	}
	return out, nil
}

// parseScore extracts a numeric score from the LLM response.
func parseScore(resp string) float64 {
	cleaned := strings.TrimSpace(resp)
	// Try parsing first word/number.
	parts := strings.Fields(cleaned)
	if len(parts) > 0 {
		if f, err := strconv.ParseFloat(parts[0], 64); err == nil {
			if f < 1 {
				f = 1
			}
			if f > 10 {
				f = 10
			}
			return f
		}
	}
	return 5 // default middle score if parsing fails
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}
