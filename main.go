package main

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/erasulov/rag-pipeline/internal/cache"
	"github.com/erasulov/rag-pipeline/internal/chunker"
	"github.com/erasulov/rag-pipeline/internal/config"
	"github.com/erasulov/rag-pipeline/internal/domain"
	"github.com/erasulov/rag-pipeline/internal/embedding"
	"github.com/erasulov/rag-pipeline/internal/eval"
	"github.com/erasulov/rag-pipeline/internal/ingest"
	"github.com/erasulov/rag-pipeline/internal/llm"
	"github.com/erasulov/rag-pipeline/internal/query"
	"github.com/erasulov/rag-pipeline/internal/rerank"
	"github.com/erasulov/rag-pipeline/internal/search"
	"github.com/erasulov/rag-pipeline/internal/server"
	"github.com/erasulov/rag-pipeline/internal/telemetry"
	"github.com/erasulov/rag-pipeline/internal/vectorstore"
	"github.com/redis/go-redis/v9"
)

// Stateful components (per-instance, lost on restart):
//   - BM25 keyword index: rebuilt as documents are ingested, empty on start
//   - Document registry: in-memory map for listing, empty on start
//
// Persistent state (survives restarts):
//   - Vector store embeddings (Weaviate/chromem with persistence)
//   - Redis cache (if configured)
//
// For multi-replica deployments: use Weaviate (shared state), accept that
// BM25 index and document registry rebuild per pod as documents are ingested.
// The document registry is informational (listing endpoint), not critical path.
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
	_ = metrics

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

	// Chunker (optionally contextual).
	var ch domain.Chunker
	ch = chunker.NewRecursive(cfg.ChunkSize, cfg.ChunkOverlap)
	if cfg.ContextualChunking {
		ch = chunker.NewContextual(ch, ollamaLLM)
		slog.Info("contextual chunking enabled")
	}

	// BM25 keyword index (shared between ingest and search).
	bm25Idx := search.NewBM25Index()

	// Ingest service.
	ingestSvc := ingest.New(ch, embedder, store, bm25Idx)

	// Build retriever chain.
	var retriever domain.Retriever
	retriever = search.NewHybridRetriever(store, embedder, bm25Idx)
	if cfg.CorrectiveRAG {
		retriever = search.NewCorrectiveRetriever(retriever, ollamaLLM)
		slog.Info("corrective RAG enabled")
	}

	// Optional reranker.
	var reranker query.Reranker
	if cfg.Reranking {
		reranker = rerank.NewLLMReranker(ollamaLLM)
		slog.Info("LLM reranking enabled")
	}

	// Optional cache.
	var queryCache query.Cache
	if cfg.RedisURL != "" {
		rdb := buildRedisClient(cfg.RedisURL)
		if rdb != nil {
			queryCache = cache.New(rdb, cfg.CacheTTL)
			defer rdb.Close()
		}
	}

	// Query service.
	querySvc := query.New(retriever, ollamaLLM, reranker, queryCache, cfg.DefaultK)

	// Evaluator.
	queryFn := func(ctx context.Context, question string, k int) (string, []string, []string, error) {
		resp, err := querySvc.Query(ctx, query.QueryRequest{Question: question, K: k})
		if err != nil {
			return "", nil, nil, err
		}
		var docIDs, sources []string
		for _, s := range resp.Sources {
			docIDs = append(docIDs, s.DocumentID)
			sources = append(sources, s.Content)
		}
		return resp.Answer, docIDs, sources, nil
	}
	evaluator := eval.NewEvaluator(queryFn, ollamaLLM)

	// HTTP server with health checks.
	serverOpts := []server.Option{
		server.WithEvaluator(evaluator),
		server.WithHealthCheck("ollama", embedder),
	}
	if hc, ok := store.(domain.HealthChecker); ok {
		serverOpts = append(serverOpts, server.WithHealthCheck("vector_store", hc))
	}
	srv := server.New(ingestSvc, querySvc, serverOpts...)
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
			"contextual_chunking", cfg.ContextualChunking,
			"corrective_rag", cfg.CorrectiveRAG,
			"reranking", cfg.Reranking,
			"cache", cfg.RedisURL != "",
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

func buildRedisClient(redisURL string) *redis.Client {
	opts, err := redis.ParseURL(redisURL)
	if err != nil {
		slog.Error("redis url parse failed", "error", err)
		return nil
	}
	client := redis.NewClient(opts)
	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Second)
	defer cancel()
	if err := client.Ping(ctx).Err(); err != nil {
		slog.Error("redis ping failed", "error", err)
		client.Close()
		return nil
	}
	slog.Info("redis connected", "url", redisURL)
	return client
}
