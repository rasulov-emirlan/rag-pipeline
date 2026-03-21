package cache

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/redis/go-redis/v9"
)

// SemanticCache provides Redis-backed query response caching.
// It stores and retrieves JSON-serialized responses keyed by normalized question.
// Nil-safe — all methods no-op when cache is nil (disabled).
type SemanticCache struct {
	client *redis.Client
	ttl    time.Duration
}

// New creates a semantic cache. Returns nil if client is nil (disabled).
func New(client *redis.Client, ttl time.Duration) *SemanticCache {
	if client == nil {
		slog.Info("query cache disabled (no redis client)")
		return nil
	}
	slog.Info("query cache enabled", "ttl", ttl)
	return &SemanticCache{client: client, ttl: ttl}
}

// Get retrieves a cached response. Returns nil, false on miss or if disabled.
func (c *SemanticCache) Get(ctx context.Context, key string) (json.RawMessage, bool) {
	if c == nil {
		return nil, false
	}

	data, err := c.client.Get(ctx, key).Bytes()
	if err != nil {
		return nil, false
	}

	return data, true
}

// Set stores a response in cache. No-op if disabled.
func (c *SemanticCache) Set(ctx context.Context, key string, data json.RawMessage) {
	if c == nil {
		return
	}

	if err := c.client.Set(ctx, key, []byte(data), c.ttl).Err(); err != nil {
		slog.Error("cache set error", "error", err)
	}
}

// Key generates a cache key from a normalized question and k value.
func Key(question string, k int) string {
	normalized := strings.ToLower(strings.TrimSpace(question))
	payload := fmt.Sprintf("%s:%d", normalized, k)
	hash := sha256.Sum256([]byte(payload))
	return fmt.Sprintf("ragcache:%x", hash)
}
