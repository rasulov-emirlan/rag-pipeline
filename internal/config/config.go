package config

import (
	"log/slog"
	"os"
	"strconv"
	"strings"
	"time"
)

type Config struct {
	// Server
	Port     string
	LogLevel slog.Level

	// Ollama
	OllamaURL      string
	EmbeddingModel string
	EmbeddingDim   int
	ChatModel      string

	// Vector store
	VectorStoreType string // "memory", "weaviate", "chromem"
	WeaviateHost    string
	WeaviateScheme  string
	ChromemDir      string

	// Chunking
	ChunkSize    int
	ChunkOverlap int

	// Query defaults
	DefaultK int

	// Advanced features
	ContextualChunking bool
	CorrectiveRAG      bool
	Reranking          bool
	RedisURL           string
	CacheTTL           time.Duration

	// Telemetry
	OTelEndpoint string
}

func Load() *Config {
	return &Config{
		Port:            getEnv("PORT", "8080"),
		LogLevel:        parseLogLevel(getEnv("LOG_LEVEL", "info")),
		OllamaURL:       getEnv("OLLAMA_URL", "http://localhost:11434"),
		EmbeddingModel:  getEnv("EMBEDDING_MODEL", "nomic-embed-text"),
		EmbeddingDim:    getEnvInt("EMBEDDING_DIM", 768),
		ChatModel:       getEnv("CHAT_MODEL", "llama3.2"),
		VectorStoreType: getEnv("VECTOR_STORE", "memory"),
		WeaviateHost:    getEnv("WEAVIATE_HOST", "localhost:8081"),
		WeaviateScheme:  getEnv("WEAVIATE_SCHEME", "http"),
		ChromemDir:      getEnv("CHROMEM_DIR", "./data/chromem"),
		ChunkSize:       getEnvInt("CHUNK_SIZE", 1000),
		ChunkOverlap:    getEnvInt("CHUNK_OVERLAP", 200),
		DefaultK:           getEnvInt("DEFAULT_K", 5),
		ContextualChunking: getEnvBool("CONTEXTUAL_CHUNKING", false),
		CorrectiveRAG:      getEnvBool("CORRECTIVE_RAG", false),
		Reranking:          getEnvBool("RERANKING", false),
		RedisURL:           getEnv("REDIS_URL", ""),
		CacheTTL:           getEnvDuration("CACHE_TTL", 10*time.Minute),
		OTelEndpoint:       getEnv("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
	}
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func getEnvInt(key string, fallback int) int {
	if v := os.Getenv(key); v != "" {
		if i, err := strconv.Atoi(v); err == nil {
			return i
		}
	}
	return fallback
}

func getEnvBool(key string, fallback bool) bool {
	if v := os.Getenv(key); v != "" {
		return strings.ToLower(v) == "true" || v == "1"
	}
	return fallback
}

func getEnvDuration(key string, fallback time.Duration) time.Duration {
	if v := os.Getenv(key); v != "" {
		if d, err := time.ParseDuration(v); err == nil {
			return d
		}
	}
	return fallback
}

func parseLogLevel(s string) slog.Level {
	switch strings.ToLower(s) {
	case "debug":
		return slog.LevelDebug
	case "warn":
		return slog.LevelWarn
	case "error":
		return slog.LevelError
	default:
		return slog.LevelInfo
	}
}
