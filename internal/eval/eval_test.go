package eval

import (
	"context"
	"testing"
)

type mockLLM struct {
	response string
}

func (m *mockLLM) Generate(_ context.Context, _ string) (string, error) {
	return m.response, nil
}

func mockQueryFn(answer string, docIDs []string) QueryFunc {
	return func(_ context.Context, _ string, _ int) (string, []string, []string, error) {
		return answer, docIDs, []string{"source text 1", "source text 2"}, nil
	}
}

func TestEvaluator_Run(t *testing.T) {
	llm := &mockLLM{response: "4"}
	queryFn := mockQueryFn("Paris is the capital.", []string{"doc1", "doc2"})
	evaluator := NewEvaluator(queryFn, llm)

	cases := []TestCase{
		{
			Question:       "What is the capital of France?",
			ExpectedAnswer: "Paris",
			ExpectedDocIDs: []string{"doc1"},
		},
		{
			Question:       "What language is Go?",
			ExpectedDocIDs: []string{"doc3"},
		},
	}

	report, err := evaluator.Run(context.Background(), cases)
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	if report.TestCount != 2 {
		t.Fatalf("expected 2 test cases, got %d", report.TestCount)
	}
	if len(report.Results) != 2 {
		t.Fatalf("expected 2 results, got %d", len(report.Results))
	}

	// First case: doc1 was expected and retrieved → recall = 1.0.
	if report.Results[0].RetrievalRecall != 1.0 {
		t.Fatalf("expected recall 1.0, got %f", report.Results[0].RetrievalRecall)
	}

	// Second case: doc3 was expected but not retrieved → recall = 0.0.
	if report.Results[1].RetrievalRecall != 0.0 {
		t.Fatalf("expected recall 0.0, got %f", report.Results[1].RetrievalRecall)
	}

	// LLM scores should be normalized (4/5 = 0.8).
	if report.Results[0].AnswerRelevance != 0.8 {
		t.Fatalf("expected relevance 0.8, got %f", report.Results[0].AnswerRelevance)
	}
}

func TestEvaluator_EmptyCases(t *testing.T) {
	evaluator := NewEvaluator(mockQueryFn("", nil), nil)
	_, err := evaluator.Run(context.Background(), nil)
	if err == nil {
		t.Fatal("expected error for empty cases")
	}
}

func TestRecallAtK(t *testing.T) {
	tests := []struct {
		expected  []string
		retrieved []string
		want      float64
	}{
		{[]string{"a", "b"}, []string{"a", "b", "c"}, 1.0},
		{[]string{"a", "b"}, []string{"a", "c"}, 0.5},
		{[]string{"a", "b"}, []string{"c", "d"}, 0.0},
		{[]string{}, []string{"a"}, 1.0},
	}

	for i, tt := range tests {
		got := recallAtK(tt.expected, tt.retrieved)
		if got != tt.want {
			t.Errorf("case %d: recallAtK(%v, %v) = %f, want %f", i, tt.expected, tt.retrieved, got, tt.want)
		}
	}
}

func TestEvaluator_NoLLM(t *testing.T) {
	queryFn := mockQueryFn("answer", []string{"doc1"})
	evaluator := NewEvaluator(queryFn, nil) // no LLM

	report, err := evaluator.Run(context.Background(), []TestCase{
		{Question: "test?", ExpectedDocIDs: []string{"doc1"}},
	})
	if err != nil {
		t.Fatalf("run: %v", err)
	}

	// Recall should still be computed.
	if report.Results[0].RetrievalRecall != 1.0 {
		t.Fatalf("expected recall 1.0, got %f", report.Results[0].RetrievalRecall)
	}
	// LLM metrics should be 0 (no LLM).
	if report.Results[0].AnswerRelevance != 0 {
		t.Fatalf("expected 0 relevance without LLM, got %f", report.Results[0].AnswerRelevance)
	}
}
