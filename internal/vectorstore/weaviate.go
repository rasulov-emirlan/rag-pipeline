package vectorstore

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"log/slog"
	"net/http"
	"time"

	"github.com/erasulov/rag-pipeline/internal/domain"
)

const weaviateClassName = "Document"

// WeaviateStore implements domain.VectorStore using Weaviate's REST API.
// No SDK — raw HTTP, consistent with project philosophy.
type WeaviateStore struct {
	baseURL string
	client  *http.Client
}

func NewWeaviateStore(host, scheme string) (*WeaviateStore, error) {
	s := &WeaviateStore{
		baseURL: fmt.Sprintf("%s://%s", scheme, host),
		client:  &http.Client{Timeout: 30 * time.Second},
	}

	if err := s.ensureClass(context.Background()); err != nil {
		return nil, fmt.Errorf("ensure weaviate class: %w", err)
	}

	slog.Info("weaviate store connected", "url", s.baseURL)
	return s, nil
}

// ensureClass creates the Document class if it doesn't exist.
func (s *WeaviateStore) ensureClass(ctx context.Context) error {
	// Check if class exists.
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet,
		s.baseURL+"/v1/schema/"+weaviateClassName, nil)
	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("check class: %w", err)
	}
	resp.Body.Close()

	if resp.StatusCode == http.StatusOK {
		return nil // Already exists.
	}

	// Create the class.
	classSchema := map[string]any{
		"class":      weaviateClassName,
		"vectorizer": "none", // We provide our own embeddings.
		"properties": []map[string]any{
			{"name": "content", "dataType": []string{"text"}},
			{"name": "document_id", "dataType": []string{"text"}},
			{"name": "chunk_id", "dataType": []string{"text"}},
			{"name": "metadata_json", "dataType": []string{"text"}},
		},
	}

	body, _ := json.Marshal(classSchema)
	req, _ = http.NewRequestWithContext(ctx, http.MethodPost,
		s.baseURL+"/v1/schema", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")

	resp, err = s.client.Do(req)
	if err != nil {
		return fmt.Errorf("create class: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("create class: status %d", resp.StatusCode)
	}

	slog.Info("weaviate class created", "class", weaviateClassName)
	return nil
}

func (s *WeaviateStore) Store(ctx context.Context, chunks []domain.Chunk, embeddings [][]float32) error {
	// Use batch API for efficiency.
	objects := make([]map[string]any, len(chunks))
	for i, chunk := range chunks {
		metaJSON, _ := json.Marshal(chunk.Metadata)
		objects[i] = map[string]any{
			"class": weaviateClassName,
			"properties": map[string]any{
				"content":       chunk.Content,
				"document_id":   chunk.DocumentID,
				"chunk_id":      chunk.ID,
				"metadata_json": string(metaJSON),
			},
			"vector": embeddings[i],
		}
	}

	body, _ := json.Marshal(map[string]any{"objects": objects})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		s.baseURL+"/v1/batch/objects", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("batch store: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("batch store: status %d", resp.StatusCode)
	}

	return nil
}

func (s *WeaviateStore) Search(ctx context.Context, embedding []float32, k int) ([]domain.SearchResult, error) {
	// GraphQL nearVector query.
	gql := fmt.Sprintf(`{
		Get {
			%s(
				nearVector: {vector: %s}
				limit: %d
			) {
				content
				document_id
				chunk_id
				metadata_json
				_additional {
					distance
				}
			}
		}
	}`, weaviateClassName, vectorToJSON(embedding), k)

	body, _ := json.Marshal(map[string]string{"query": gql})
	req, err := http.NewRequestWithContext(ctx, http.MethodPost,
		s.baseURL+"/v1/graphql", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return nil, fmt.Errorf("search: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("search: status %d", resp.StatusCode)
	}

	var gqlResp graphqlResponse
	if err := json.NewDecoder(resp.Body).Decode(&gqlResp); err != nil {
		return nil, fmt.Errorf("decode: %w", err)
	}

	docs := gqlResp.Data.Get[weaviateClassName]
	results := make([]domain.SearchResult, 0, len(docs))
	for _, doc := range docs {
		var metadata map[string]any
		if doc.MetadataJSON != "" {
			json.Unmarshal([]byte(doc.MetadataJSON), &metadata)
		}

		// Weaviate returns distance (lower = better). Convert to similarity score.
		score := 1.0 - doc.Additional.Distance

		results = append(results, domain.SearchResult{
			Chunk: domain.Chunk{
				ID:         doc.ChunkID,
				DocumentID: doc.DocumentID,
				Content:    doc.Content,
				Metadata:   metadata,
			},
			Score: score,
		})
	}

	return results, nil
}

func (s *WeaviateStore) Delete(ctx context.Context, documentID string) error {
	// Batch delete using where filter.
	body, _ := json.Marshal(map[string]any{
		"match": map[string]any{
			"class": weaviateClassName,
			"where": map[string]any{
				"path":        []string{"document_id"},
				"operator":    "Equal",
				"valueText":   documentID,
			},
		},
	})

	req, err := http.NewRequestWithContext(ctx, http.MethodDelete,
		s.baseURL+"/v1/batch/objects", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("batch delete: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("batch delete: status %d", resp.StatusCode)
	}

	return nil
}

// HealthCheck verifies Weaviate is reachable.
func (s *WeaviateStore) HealthCheck(ctx context.Context) error {
	req, _ := http.NewRequestWithContext(ctx, http.MethodGet,
		s.baseURL+"/v1/.well-known/ready", nil)
	resp, err := s.client.Do(req)
	if err != nil {
		return fmt.Errorf("weaviate unreachable: %w", err)
	}
	resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("weaviate unhealthy: status %d", resp.StatusCode)
	}
	return nil
}

// GraphQL response types.
type graphqlResponse struct {
	Data struct {
		Get map[string][]weaviateDoc `json:"Get"`
	} `json:"data"`
}

type weaviateDoc struct {
	Content      string `json:"content"`
	DocumentID   string `json:"document_id"`
	ChunkID      string `json:"chunk_id"`
	MetadataJSON string `json:"metadata_json"`
	Additional   struct {
		Distance float64 `json:"distance"`
	} `json:"_additional"`
}

func vectorToJSON(v []float32) string {
	b, _ := json.Marshal(v)
	return string(b)
}
