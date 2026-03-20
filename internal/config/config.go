package config

import (
	"log/slog"
	"os"
	"strconv"
	"strings"
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
		DefaultK:        getEnvInt("DEFAULT_K", 5),
		OTelEndpoint:    getEnv("OTEL_EXPORTER_OTLP_ENDPOINT", ""),
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
