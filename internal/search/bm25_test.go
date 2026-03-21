package search

import "testing"

func TestBM25Index_SearchBasic(t *testing.T) {
	idx := NewBM25Index()
	idx.Add("c1", "d1", "Go is a programming language designed at Google")
	idx.Add("c2", "d1", "Python is popular for machine learning")
	idx.Add("c3", "d2", "Kubernetes orchestrates container deployments")

	results := idx.Search("programming language Go", 2)
	if len(results) == 0 {
		t.Fatal("expected results")
	}
	if results[0].ID != "c1" {
		t.Fatalf("expected c1 as top result, got %s", results[0].ID)
	}
}

func TestBM25Index_SearchNoMatch(t *testing.T) {
	idx := NewBM25Index()
	idx.Add("c1", "d1", "Go programming language")

	results := idx.Search("quantum physics", 5)
	if len(results) != 0 {
		t.Fatalf("expected 0 results for unrelated query, got %d", len(results))
	}
}

func TestBM25Index_SearchEmpty(t *testing.T) {
	idx := NewBM25Index()
	results := idx.Search("anything", 5)
	if len(results) != 0 {
		t.Fatalf("expected 0 results on empty index, got %d", len(results))
	}
}

func TestBM25Index_Remove(t *testing.T) {
	idx := NewBM25Index()
	idx.Add("c1", "d1", "chunk one about Go")
	idx.Add("c2", "d1", "chunk two about Go")
	idx.Add("c3", "d2", "chunk about Python")

	idx.Remove("d1")

	if idx.Count() != 1 {
		t.Fatalf("expected 1 doc after remove, got %d", idx.Count())
	}

	results := idx.Search("Go", 5)
	if len(results) != 0 {
		t.Fatalf("expected 0 results for removed doc, got %d", len(results))
	}

	results = idx.Search("Python", 5)
	if len(results) != 1 {
		t.Fatalf("expected 1 result for remaining doc, got %d", len(results))
	}
}

func TestBM25Index_KLargerThanResults(t *testing.T) {
	idx := NewBM25Index()
	idx.Add("c1", "d1", "Go language")

	results := idx.Search("Go", 100)
	if len(results) != 1 {
		t.Fatalf("expected 1 result (capped), got %d", len(results))
	}
}

func TestTokenize(t *testing.T) {
	tokens := tokenize("Hello, World! This is a Test-123.")
	expected := []string{"hello", "world", "this", "is", "a", "test", "123"}
	if len(tokens) != len(expected) {
		t.Fatalf("expected %d tokens, got %d: %v", len(expected), len(tokens), tokens)
	}
	for i, tok := range tokens {
		if tok != expected[i] {
			t.Errorf("token %d: expected %q, got %q", i, expected[i], tok)
		}
	}
}
