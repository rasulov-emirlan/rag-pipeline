// Package domain defines the core types and interfaces for the RAG pipeline.
// This package has zero external dependencies. All other packages depend on it.
package domain

import "context"

// Document represents a source document before chunking.
type Document struct {
	ID       string         `json:"id"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

// Chunk represents a piece of a document after splitting.
type Chunk struct {
	ID         string         `json:"id"`
	DocumentID string         `json:"document_id"`
	Content    string         `json:"content"`
	Metadata   map[string]any `json:"metadata,omitempty"`
}

// SearchResult is a chunk with a relevance score.
type SearchResult struct {
	Chunk Chunk   `json:"chunk"`
	Score float64 `json:"score"` // 0.0 to 1.0, higher is more relevant
}

// Embedder generates vector embeddings from text.
type Embedder interface {
	// Embed returns embeddings for the given texts. len(result) == len(texts).
	Embed(ctx context.Context, texts []string) ([][]float32, error)
	// Dimension returns the embedding vector dimension (e.g., 768 for nomic-embed-text).
	Dimension() int
}

// VectorStore persists and queries document chunks with their embeddings.
type VectorStore interface {
	// Store saves chunks with their pre-computed embeddings.
	// len(chunks) must equal len(embeddings).
	Store(ctx context.Context, chunks []Chunk, embeddings [][]float32) error
	// Search finds the k most similar chunks to the query embedding.
	Search(ctx context.Context, embedding []float32, k int) ([]SearchResult, error)
	// Delete removes all chunks belonging to a document.
	Delete(ctx context.Context, documentID string) error
}

// Retriever finds relevant chunks for a query. Unlike VectorStore.Search,
// it may combine multiple search strategies (vector + keyword), reranking,
// and self-correction.
type Retriever interface {
	Retrieve(ctx context.Context, query string, k int) ([]SearchResult, error)
}

// Chunker splits text into smaller pieces for embedding.
type Chunker interface {
	Split(text string) ([]string, error)
}

// HealthChecker is implemented by components that can report their health.
type HealthChecker interface {
	HealthCheck(ctx context.Context) error
}

// Loader reads a source and returns its text content.
type Loader interface {
	Load(ctx context.Context) (string, error)
}

// LLM generates text completions.
type LLM interface {
	Generate(ctx context.Context, prompt string) (string, error)
}
