package eval

import (
	"context"
	"fmt"
	"log/slog"
	"strconv"
	"strings"
	"time"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

// TestCase defines a single evaluation case.
type TestCase struct {
	Question       string   `json:"question"`
	ExpectedAnswer string   `json:"expected_answer,omitempty"`
	ExpectedDocIDs []string `json:"expected_doc_ids,omitempty"`
}

// Result holds evaluation results for a single test case.
type Result struct {
	Question        string   `json:"question"`
	Answer          string   `json:"answer"`
	RetrievedDocIDs []string `json:"retrieved_doc_ids"`
	RetrievalRecall float64  `json:"retrieval_recall"`
	AnswerRelevance float64  `json:"answer_relevance"`
	Faithfulness    float64  `json:"faithfulness"`
	DurationMs      int64    `json:"duration_ms"`
}

// Report aggregates results across all test cases.
type Report struct {
	Results         []Result `json:"results"`
	AvgRecall       float64  `json:"avg_retrieval_recall"`
	AvgRelevance    float64  `json:"avg_answer_relevance"`
	AvgFaithfulness float64  `json:"avg_faithfulness"`
	TotalDurationMs int64    `json:"total_duration_ms"`
	TestCount       int      `json:"test_count"`
}

// QueryFunc abstracts the query service to avoid circular imports.
type QueryFunc func(ctx context.Context, question string, k int) (answer string, docIDs []string, sources []string, err error)

// Evaluator runs evaluation test cases against the RAG pipeline.
type Evaluator struct {
	queryFn QueryFunc
	llm     domain.LLM
}

func NewEvaluator(queryFn QueryFunc, llm domain.LLM) *Evaluator {
	return &Evaluator{queryFn: queryFn, llm: llm}
}

// Run executes all test cases and returns an aggregated report.
func (e *Evaluator) Run(ctx context.Context, cases []TestCase) (*Report, error) {
	if len(cases) == 0 {
		return nil, fmt.Errorf("no test cases provided")
	}

	report := &Report{
		Results:   make([]Result, 0, len(cases)),
		TestCount: len(cases),
	}

	totalStart := time.Now()

	for i, tc := range cases {
		start := time.Now()

		answer, docIDs, sources, err := e.queryFn(ctx, tc.Question, 5)
		if err != nil {
			slog.Warn("eval query failed", "question", tc.Question, "error", err)
			continue
		}

		result := Result{
			Question:        tc.Question,
			Answer:          answer,
			RetrievedDocIDs: docIDs,
			DurationMs:      time.Since(start).Milliseconds(),
		}

		// Compute retrieval recall.
		if len(tc.ExpectedDocIDs) > 0 {
			result.RetrievalRecall = recallAtK(tc.ExpectedDocIDs, docIDs)
		}

		// LLM-as-judge metrics.
		if e.llm != nil {
			result.AnswerRelevance = e.scoreRelevance(ctx, tc.Question, answer)
			result.Faithfulness = e.scoreFaithfulness(ctx, answer, sources)
		}

		report.Results = append(report.Results, result)
		slog.Info("eval case completed",
			"index", i+1,
			"recall", result.RetrievalRecall,
			"relevance", result.AnswerRelevance,
			"faithfulness", result.Faithfulness,
		)
	}

	report.TotalDurationMs = time.Since(totalStart).Milliseconds()

	// Compute averages.
	if len(report.Results) > 0 {
		var sumRecall, sumRelevance, sumFaithfulness float64
		for _, r := range report.Results {
			sumRecall += r.RetrievalRecall
			sumRelevance += r.AnswerRelevance
			sumFaithfulness += r.Faithfulness
		}
		n := float64(len(report.Results))
		report.AvgRecall = sumRecall / n
		report.AvgRelevance = sumRelevance / n
		report.AvgFaithfulness = sumFaithfulness / n
	}

	return report, nil
}

// recallAtK computes the fraction of expected doc IDs that were retrieved.
func recallAtK(expected, retrieved []string) float64 {
	if len(expected) == 0 {
		return 1.0
	}

	retrievedSet := make(map[string]bool)
	for _, id := range retrieved {
		retrievedSet[id] = true
	}

	hits := 0
	for _, id := range expected {
		if retrievedSet[id] {
			hits++
		}
	}

	return float64(hits) / float64(len(expected))
}

// scoreRelevance asks the LLM: does this answer address the question?
func (e *Evaluator) scoreRelevance(ctx context.Context, question, answer string) float64 {
	prompt := fmt.Sprintf(
		"Rate how well this answer addresses the question on a scale of 1 to 5.\n"+
			"Only respond with a single number.\n\n"+
			"Question: %s\n\nAnswer: %s\n\nScore:",
		question, answer,
	)
	return e.llmScore(ctx, prompt, 5.0)
}

// scoreFaithfulness asks the LLM: is the answer grounded in the sources?
func (e *Evaluator) scoreFaithfulness(ctx context.Context, answer string, sources []string) float64 {
	if len(sources) == 0 {
		return 0
	}

	context := strings.Join(sources, "\n---\n")
	prompt := fmt.Sprintf(
		"Rate how faithfully this answer uses only information from the provided sources "+
			"on a scale of 1 to 5. A score of 5 means the answer only contains information "+
			"from the sources. A score of 1 means the answer fabricates information.\n"+
			"Only respond with a single number.\n\n"+
			"Sources:\n%s\n\nAnswer: %s\n\nScore:",
		context, answer,
	)
	return e.llmScore(ctx, prompt, 5.0)
}

func (e *Evaluator) llmScore(ctx context.Context, prompt string, maxScore float64) float64 {
	resp, err := e.llm.Generate(ctx, prompt)
	if err != nil {
		return 0
	}

	cleaned := strings.TrimSpace(resp)
	parts := strings.Fields(cleaned)
	if len(parts) > 0 {
		if f, err := strconv.ParseFloat(parts[0], 64); err == nil {
			if f < 1 {
				f = 1
			}
			if f > maxScore {
				f = maxScore
			}
			return f / maxScore // normalize to 0-1
		}
	}
	return 0.5
}
