package chunker

import "strings"

// Recursive splits text by trying separators in order, recursively splitting
// chunks that exceed the target size. This mirrors LangChain's
// RecursiveCharacterTextSplitter algorithm.
type Recursive struct {
	chunkSize    int
	chunkOverlap int
	separators   []string
}

// NewRecursive creates a recursive text chunker.
// chunkSize is the target maximum chunk length in characters.
// chunkOverlap is the number of characters shared between adjacent chunks.
func NewRecursive(chunkSize, chunkOverlap int) *Recursive {
	return &Recursive{
		chunkSize:    chunkSize,
		chunkOverlap: chunkOverlap,
		separators:   []string{"\n\n", "\n", ". ", " ", ""},
	}
}

func (r *Recursive) Split(text string) ([]string, error) {
	if len(text) <= r.chunkSize {
		if text == "" {
			return nil, nil
		}
		return []string{text}, nil
	}

	// Find the best separator that splits this text.
	sep := r.findSeparator(text)
	parts := strings.Split(text, sep)

	var chunks []string
	var current strings.Builder

	for _, part := range parts {
		piece := part
		if sep != "" {
			piece = part + sep
		}

		// If adding this piece exceeds chunk size and we have content, flush.
		if current.Len()+len(piece) > r.chunkSize && current.Len() > 0 {
			chunk := strings.TrimSpace(current.String())
			if chunk != "" {
				chunks = append(chunks, chunk)
			}

			// Start new chunk with overlap from the end of the current one.
			overlap := r.getOverlap(current.String())
			current.Reset()
			current.WriteString(overlap)
		}

		current.WriteString(piece)
	}

	// Flush remaining.
	if chunk := strings.TrimSpace(current.String()); chunk != "" {
		chunks = append(chunks, chunk)
	}

	return chunks, nil
}

// findSeparator returns the first separator found in the text.
func (r *Recursive) findSeparator(text string) string {
	for _, sep := range r.separators {
		if sep == "" || strings.Contains(text, sep) {
			return sep
		}
	}
	return ""
}

// getOverlap returns the last chunkOverlap characters of s.
func (r *Recursive) getOverlap(s string) string {
	if r.chunkOverlap <= 0 || len(s) <= r.chunkOverlap {
		return ""
	}
	return s[len(s)-r.chunkOverlap:]
}
