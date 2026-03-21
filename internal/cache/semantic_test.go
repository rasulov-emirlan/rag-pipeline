package cache

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/alicebob/miniredis/v2"
	"github.com/redis/go-redis/v9"
)

func setupCache(t *testing.T) (*SemanticCache, *miniredis.Miniredis) {
	t.Helper()
	mr := miniredis.RunT(t)
	client := redis.NewClient(&redis.Options{Addr: mr.Addr()})
	t.Cleanup(func() { client.Close() })
	return New(client, 5*time.Minute), mr
}

func TestSemanticCache_SetAndGet(t *testing.T) {
	cache, _ := setupCache(t)
	ctx := context.Background()

	resp := map[string]any{
		"answer":      "Paris is the capital of France.",
		"duration_ms": 100,
	}
	data, _ := json.Marshal(resp)

	key := Key("What is the capital of France?", 5)
	cache.Set(ctx, key, data)

	got, ok := cache.Get(ctx, key)
	if !ok {
		t.Fatal("expected cache hit")
	}

	var result map[string]any
	json.Unmarshal(got, &result)
	if result["answer"] != "Paris is the capital of France." {
		t.Fatalf("expected correct answer, got %v", result["answer"])
	}
}

func TestSemanticCache_Miss(t *testing.T) {
	cache, _ := setupCache(t)
	_, ok := cache.Get(context.Background(), "nonexistent")
	if ok {
		t.Fatal("expected cache miss")
	}
}

func TestSemanticCache_Nil(t *testing.T) {
	var cache *SemanticCache
	cache.Set(context.Background(), "key", json.RawMessage(`{}`))
	_, ok := cache.Get(context.Background(), "key")
	if ok {
		t.Fatal("nil cache should always miss")
	}
}

func TestKey_Normalized(t *testing.T) {
	k1 := Key("What is Go?", 5)
	k2 := Key("  what is go?  ", 5)
	if k1 != k2 {
		t.Fatalf("keys should match after normalization: %s vs %s", k1, k2)
	}

	k3 := Key("What is Go?", 3)
	if k1 == k3 {
		t.Fatal("different k should produce different keys")
	}
}

func TestSemanticCache_TTLExpiry(t *testing.T) {
	cache, mr := setupCache(t)
	ctx := context.Background()

	key := Key("test question", 5)
	cache.Set(ctx, key, json.RawMessage(`{"answer":"cached"}`))

	mr.FastForward(10 * time.Minute)

	_, ok := cache.Get(ctx, key)
	if ok {
		t.Fatal("expected cache miss after TTL expiry")
	}
}
