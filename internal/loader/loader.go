package loader

import (
	"context"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// FileLoader reads a file and returns its text content.
type FileLoader struct {
	path string
}

// FromFile creates a loader for the given file path.
// Supports .txt, .md, and plain text files.
func FromFile(path string) *FileLoader {
	return &FileLoader{path: path}
}

func (l *FileLoader) Load(_ context.Context) (string, error) {
	ext := strings.ToLower(filepath.Ext(l.path))

	switch ext {
	case ".txt", ".md", ".markdown", ".text", ".log", ".csv", ".json", ".yaml", ".yml":
		return readTextFile(l.path)
	default:
		// Try reading as text for unknown extensions.
		return readTextFile(l.path)
	}
}

func readTextFile(path string) (string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read file: %w", err)
	}
	return string(data), nil
}

// ReaderLoader reads all content from an io.Reader.
type ReaderLoader struct {
	reader io.Reader
}

// FromReader creates a loader from an io.Reader.
func FromReader(r io.Reader) *ReaderLoader {
	return &ReaderLoader{reader: r}
}

func (l *ReaderLoader) Load(_ context.Context) (string, error) {
	data, err := io.ReadAll(l.reader)
	if err != nil {
		return "", fmt.Errorf("read: %w", err)
	}
	return string(data), nil
}

// TextLoader wraps a string as a loader.
type TextLoader struct {
	text string
}

// FromText creates a loader from a string.
func FromText(text string) *TextLoader {
	return &TextLoader{text: text}
}

func (l *TextLoader) Load(_ context.Context) (string, error) {
	return l.text, nil
}
