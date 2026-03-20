package loader

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestFileLoader_TextFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "test.txt")
	os.WriteFile(path, []byte("Hello, world!"), 0644)

	l := FromFile(path)
	content, err := l.Load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if content != "Hello, world!" {
		t.Fatalf("expected 'Hello, world!', got %q", content)
	}
}

func TestFileLoader_MarkdownFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "readme.md")
	os.WriteFile(path, []byte("# Title\n\nSome content."), 0644)

	l := FromFile(path)
	content, err := l.Load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if !strings.Contains(content, "# Title") {
		t.Fatalf("expected markdown content, got %q", content)
	}
}

func TestFileLoader_NotFound(t *testing.T) {
	l := FromFile("/nonexistent/path.txt")
	_, err := l.Load(context.Background())
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestReaderLoader(t *testing.T) {
	r := strings.NewReader("streamed content")
	l := FromReader(r)
	content, err := l.Load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if content != "streamed content" {
		t.Fatalf("expected 'streamed content', got %q", content)
	}
}

func TestTextLoader(t *testing.T) {
	l := FromText("inline content")
	content, err := l.Load(context.Background())
	if err != nil {
		t.Fatalf("load: %v", err)
	}
	if content != "inline content" {
		t.Fatalf("expected 'inline content', got %q", content)
	}
}
