package server

import (
	"encoding/json"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"github.com/erasulov/rag-pipeline/internal/domain"
	"github.com/erasulov/rag-pipeline/internal/ingest"
	"github.com/erasulov/rag-pipeline/internal/query"
)

// Server is the HTTP transport layer. It decodes requests, calls services,
// and encodes responses. No business logic lives here.
type Server struct {
	ingest *ingest.Service
	query  *query.Service
}

func New(ingest *ingest.Service, query *query.Service) *Server {
	return &Server{
		ingest: ingest,
		query:  query,
	}
}

func (s *Server) Router() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", s.handleHealth)
	mux.HandleFunc("POST /v1/documents", s.handleIngestDocument)
	mux.HandleFunc("GET /v1/documents", s.handleListDocuments)
	mux.HandleFunc("DELETE /v1/documents/{id}", s.handleDeleteDocument)
	mux.HandleFunc("POST /v1/query", s.handleQuery)
	return withLogging(mux)
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	writeJSON(w, http.StatusOK, map[string]string{"status": "ok"})
}

// IngestRequest is the JSON body for document ingestion.
type IngestRequest struct {
	ID       string         `json:"id,omitempty"`
	Content  string         `json:"content"`
	Metadata map[string]any `json:"metadata,omitempty"`
}

func (s *Server) handleIngestDocument(w http.ResponseWriter, r *http.Request) {
	ct := r.Header.Get("Content-Type")

	// Multipart file upload.
	if strings.HasPrefix(ct, "multipart/form-data") {
		s.handleFileUpload(w, r)
		return
	}

	// JSON body.
	var req IngestRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}

	if req.Content == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "content is required"})
		return
	}

	result, err := s.ingest.IngestDocument(r.Context(), domain.Document{
		ID:       req.ID,
		Content:  req.Content,
		Metadata: req.Metadata,
	})
	if err != nil {
		slog.Error("ingest failed", "error", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusCreated, result)
}

func (s *Server) handleFileUpload(w http.ResponseWriter, r *http.Request) {
	// Max 10MB.
	r.ParseMultipartForm(10 << 20)

	file, header, err := r.FormFile("file")
	if err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "file is required"})
		return
	}
	defer file.Close()

	data, err := io.ReadAll(file)
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": "failed to read file"})
		return
	}

	content := string(data)
	if content == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "file is empty"})
		return
	}

	metadata := map[string]any{
		"source":   "upload",
		"filename": header.Filename,
	}

	// Parse optional metadata from form field.
	if metaStr := r.FormValue("metadata"); metaStr != "" {
		var extra map[string]any
		if err := json.Unmarshal([]byte(metaStr), &extra); err == nil {
			for k, v := range extra {
				metadata[k] = v
			}
		}
	}

	id := r.FormValue("id")
	if id == "" {
		id = header.Filename
	}

	result, err := s.ingest.IngestDocument(r.Context(), domain.Document{
		ID:       id,
		Content:  content,
		Metadata: metadata,
	})
	if err != nil {
		slog.Error("file ingest failed", "error", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusCreated, result)
}

func (s *Server) handleListDocuments(w http.ResponseWriter, r *http.Request) {
	docs := s.ingest.ListDocuments()
	writeJSON(w, http.StatusOK, map[string]any{
		"documents": docs,
		"count":     len(docs),
	})
}

func (s *Server) handleDeleteDocument(w http.ResponseWriter, r *http.Request) {
	id := r.PathValue("id")
	if id == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "document id is required"})
		return
	}

	if err := s.ingest.DeleteDocument(r.Context(), id); err != nil {
		slog.Error("delete failed", "error", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, map[string]string{"status": "deleted", "id": id})
}

func (s *Server) handleQuery(w http.ResponseWriter, r *http.Request) {
	var req query.QueryRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "invalid request body"})
		return
	}

	if req.Question == "" {
		writeJSON(w, http.StatusBadRequest, map[string]string{"error": "question is required"})
		return
	}

	resp, err := s.query.Query(r.Context(), req)
	if err != nil {
		slog.Error("query failed", "error", err)
		writeJSON(w, http.StatusInternalServerError, map[string]string{"error": err.Error()})
		return
	}

	writeJSON(w, http.StatusOK, resp)
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	json.NewEncoder(w).Encode(v)
}

// withLogging wraps a handler with request logging.
func withLogging(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		slog.Info("request",
			"method", r.Method,
			"path", r.URL.Path,
			"duration_ms", time.Since(start).Milliseconds(),
		)
	})
}
