package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/erasulov/rag-pipeline/internal/chunker"
	"github.com/erasulov/rag-pipeline/internal/config"
	"github.com/erasulov/rag-pipeline/internal/domain"
	"github.com/erasulov/rag-pipeline/internal/embedding"
	"github.com/erasulov/rag-pipeline/internal/ingest"
	"github.com/erasulov/rag-pipeline/internal/llm"
	"github.com/erasulov/rag-pipeline/internal/query"
	"github.com/erasulov/rag-pipeline/internal/server"
	"github.com/erasulov/rag-pipeline/internal/telemetry"
	"github.com/erasulov/rag-pipeline/internal/vectorstore"
)

func main() {
	cfg := config.Load()

	logger := slog.New(slog.NewJSONHandler(os.Stdout, &slog.HandlerOptions{
		Level: cfg.LogLevel,
	}))
	slog.SetDefault(logger)

	ctx := context.Background()

	// Initialize telemetry.
	metrics, shutdownMetrics, err := telemetry.New(ctx, cfg.OTelEndpoint)
	if err != nil {
		slog.Error("failed to initialize metrics", "error", err)
		os.Exit(1)
	}
	defer shutdownMetrics()
	_ = metrics // Will be wired into services when we add instrumentation.

	shutdownTracing, err := telemetry.InitTracing(ctx, cfg.OTelEndpoint)
	if err != nil {
		slog.Error("failed to initialize tracing", "error", err)
		os.Exit(1)
	}
	defer shutdownTracing()

	// Embedder.
	embedder := embedding.NewOllamaEmbedder(cfg.OllamaURL, cfg.EmbeddingModel, cfg.EmbeddingDim)

	// Vector store.
	store := buildVectorStore(cfg)

	// LLM.
	ollamaLLM := llm.NewOllamaLLM(cfg.OllamaURL, cfg.ChatModel)

	// Chunker.
	ch := chunker.NewRecursive(cfg.ChunkSize, cfg.ChunkOverlap)

	// Services.
	ingestSvc := ingest.New(ch, embedder, store)
	querySvc := query.New(embedder, store, ollamaLLM, cfg.DefaultK)

	// HTTP server.
	srv := server.New(ingestSvc, querySvc)
	httpSrv := &http.Server{
		Addr:         ":" + cfg.Port,
		Handler:      srv.Router(),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 120 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	go func() {
		slog.Info("starting RAG pipeline",
			"port", cfg.Port,
			"vector_store", cfg.VectorStoreType,
			"embedding_model", cfg.EmbeddingModel,
			"chat_model", cfg.ChatModel,
		)
		if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			slog.Error("server error", "error", err)
			os.Exit(1)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit

	slog.Info("shutting down server")
	shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	if err := httpSrv.Shutdown(shutdownCtx); err != nil {
		slog.Error("server forced shutdown", "error", err)
	}
}

func buildVectorStore(cfg *config.Config) domain.VectorStore {
	switch cfg.VectorStoreType {
	case "weaviate":
		store, err := vectorstore.NewWeaviateStore(cfg.WeaviateHost, cfg.WeaviateScheme)
		if err != nil {
			slog.Error("weaviate init failed, falling back to memory", "error", err)
			return vectorstore.NewMemoryStore()
		}
		return store
	case "chromem":
		store, err := vectorstore.NewChromemStore(cfg.ChromemDir)
		if err != nil {
			slog.Error("chromem init failed, falling back to memory", "error", err)
			return vectorstore.NewMemoryStore()
		}
		return store
	default:
		slog.Info("vector store: in-memory")
		return vectorstore.NewMemoryStore()
	}
}
