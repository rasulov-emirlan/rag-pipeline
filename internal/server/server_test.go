package server

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/erasulov/rag-pipeline/internal/chunker"
	"github.com/erasulov/rag-pipeline/internal/ingest"
	"github.com/erasulov/rag-pipeline/internal/query"
	"github.com/erasulov/rag-pipeline/internal/vectorstore"
)

func setupTestServer(t *testing.T) *Server {
	t.Helper()
	store := vectorstore.NewMemoryStore()
	ch := chunker.NewRecursive(1000, 0)
	ingestSvc := ingest.New(ch, nil, store)
	querySvc := query.New(nil, store, nil, 5)
	return New(ingestSvc, querySvc)
}

func TestServer_Health(t *testing.T) {
	srv := setupTestServer(t)
	req := httptest.NewRequest(http.MethodGet, "/health", nil)
	w := httptest.NewRecorder()
	srv.Router().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
}

func TestServer_IngestDocument(t *testing.T) {
	srv := setupTestServer(t)

	body := `{"id":"test","content":"This is a test document with some content."}`
	req := httptest.NewRequest(http.MethodPost, "/v1/documents", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Router().ServeHTTP(w, req)

	if w.Code != http.StatusCreated {
		t.Fatalf("expected 201, got %d: %s", w.Code, w.Body.String())
	}

	var result map[string]any
	json.NewDecoder(w.Body).Decode(&result)
	if result["document_id"] != "test" {
		t.Fatalf("expected document_id 'test', got %v", result["document_id"])
	}
}

func TestServer_IngestDocument_EmptyContent(t *testing.T) {
	srv := setupTestServer(t)

	body := `{"content":""}`
	req := httptest.NewRequest(http.MethodPost, "/v1/documents", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Router().ServeHTTP(w, req)

	if w.Code != http.StatusBadRequest {
		t.Fatalf("expected 400, got %d", w.Code)
	}
}

func TestServer_ListDocuments(t *testing.T) {
	srv := setupTestServer(t)

	// Ingest a document first.
	body := `{"id":"doc1","content":"Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/documents", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Router().ServeHTTP(w, req)

	// List documents.
	req = httptest.NewRequest(http.MethodGet, "/v1/documents", nil)
	w = httptest.NewRecorder()
	srv.Router().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}

	var result map[string]any
	json.NewDecoder(w.Body).Decode(&result)
	count := result["count"].(float64)
	if count != 1 {
		t.Fatalf("expected 1 document, got %v", count)
	}
}

func TestServer_DeleteDocument(t *testing.T) {
	srv := setupTestServer(t)

	// Ingest.
	body := `{"id":"doc1","content":"Hello world"}`
	req := httptest.NewRequest(http.MethodPost, "/v1/documents", bytes.NewBufferString(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	srv.Router().ServeHTTP(w, req)

	// Delete.
	req = httptest.NewRequest(http.MethodDelete, "/v1/documents/doc1", nil)
	w = httptest.NewRecorder()
	srv.Router().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d: %s", w.Code, w.Body.String())
	}

	// Verify deleted.
	req = httptest.NewRequest(http.MethodGet, "/v1/documents", nil)
	w = httptest.NewRecorder()
	srv.Router().ServeHTTP(w, req)

	var result map[string]any
	json.NewDecoder(w.Body).Decode(&result)
	if result["count"].(float64) != 0 {
		t.Fatalf("expected 0 documents after delete")
	}
}
