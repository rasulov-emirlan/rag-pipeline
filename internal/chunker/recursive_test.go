package chunker

import (
	"strings"
	"testing"
)

func TestRecursive_ShortText(t *testing.T) {
	c := NewRecursive(1000, 200)
	chunks, err := c.Split("Hello world")
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if len(chunks) != 1 {
		t.Fatalf("expected 1 chunk, got %d", len(chunks))
	}
	if chunks[0] != "Hello world" {
		t.Fatalf("expected 'Hello world', got %q", chunks[0])
	}
}

func TestRecursive_EmptyText(t *testing.T) {
	c := NewRecursive(1000, 200)
	chunks, err := c.Split("")
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if len(chunks) != 0 {
		t.Fatalf("expected 0 chunks, got %d", len(chunks))
	}
}

func TestRecursive_SplitsLongText(t *testing.T) {
	c := NewRecursive(100, 20)

	// Create text with paragraphs.
	text := strings.Repeat("This is a sentence. ", 50)
	chunks, err := c.Split(text)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d", len(chunks))
	}

	// No chunk should greatly exceed chunk size (some tolerance for word boundaries).
	for i, chunk := range chunks {
		if len(chunk) > 150 { // 100 + some tolerance
			t.Errorf("chunk %d too long: %d chars", i, len(chunk))
		}
	}
}

func TestRecursive_PrefersParagraphBoundaries(t *testing.T) {
	c := NewRecursive(40, 0)

	text := "First paragraph here with text.\n\nSecond paragraph here with text.\n\nThird paragraph here with text."
	chunks, err := c.Split(text)
	if err != nil {
		t.Fatalf("split: %v", err)
	}
	if len(chunks) < 2 {
		t.Fatalf("expected multiple chunks, got %d: %v", len(chunks), chunks)
	}

	// Should split on paragraph boundaries — no chunk contains "\n\n".
	for _, chunk := range chunks {
		if strings.Contains(chunk, "\n\n") {
			t.Errorf("chunk should not contain paragraph separator: %q", chunk)
		}
	}
}

func TestRecursive_AllContentPreserved(t *testing.T) {
	c := NewRecursive(50, 0) // No overlap for this test.

	text := "Alpha bravo charlie. Delta echo foxtrot. Golf hotel india. Juliet kilo lima."
	chunks, err := c.Split(text)
	if err != nil {
		t.Fatalf("split: %v", err)
	}

	// Every word from the original should appear in at least one chunk.
	words := strings.Fields(text)
	joined := strings.Join(chunks, " ")
	for _, word := range words {
		if !strings.Contains(joined, word) {
			t.Errorf("word %q missing from chunks", word)
		}
	}
}
