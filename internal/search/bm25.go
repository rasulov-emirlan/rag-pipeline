package search

import (
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

const (
	bm25K1 = 1.2
	bm25B  = 0.75
)

// ScoredID is a chunk ID with a relevance score.
type ScoredID struct {
	ID    string
	Score float64
}

type tokenizedDoc struct {
	chunkID    string
	documentID string
	tokens     []string
}

// BM25Index is an in-memory inverted index for keyword search.
type BM25Index struct {
	mu       sync.RWMutex
	docs     map[string]tokenizedDoc // chunkID → doc
	df       map[string]int          // term → document frequency
	totalDL  int                     // sum of all document lengths
	docCount int
}

func NewBM25Index() *BM25Index {
	return &BM25Index{
		docs: make(map[string]tokenizedDoc),
		df:   make(map[string]int),
	}
}

// Add indexes a chunk for keyword search.
func (idx *BM25Index) Add(chunkID, documentID, text string) {
	tokens := tokenize(text)

	idx.mu.Lock()
	defer idx.mu.Unlock()

	// If already indexed, remove first.
	if old, ok := idx.docs[chunkID]; ok {
		idx.removeDocLocked(old)
	}

	idx.docs[chunkID] = tokenizedDoc{
		chunkID:    chunkID,
		documentID: documentID,
		tokens:     tokens,
	}

	// Update document frequency (count each unique term once per doc).
	seen := make(map[string]bool)
	for _, t := range tokens {
		if !seen[t] {
			idx.df[t]++
			seen[t] = true
		}
	}

	idx.totalDL += len(tokens)
	idx.docCount++
}

// Remove removes all chunks belonging to a document.
func (idx *BM25Index) Remove(documentID string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	for _, doc := range idx.docs {
		if doc.documentID == documentID {
			idx.removeDocLocked(doc)
			delete(idx.docs, doc.chunkID)
		}
	}
}

func (idx *BM25Index) removeDocLocked(doc tokenizedDoc) {
	seen := make(map[string]bool)
	for _, t := range doc.tokens {
		if !seen[t] {
			idx.df[t]--
			if idx.df[t] <= 0 {
				delete(idx.df, t)
			}
			seen[t] = true
		}
	}
	idx.totalDL -= len(doc.tokens)
	idx.docCount--
}

// Search returns the top-k chunks matching the query, scored by BM25.
func (idx *BM25Index) Search(query string, k int) []ScoredID {
	queryTokens := tokenize(query)
	if len(queryTokens) == 0 {
		return nil
	}

	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if idx.docCount == 0 {
		return nil
	}

	avgDL := float64(idx.totalDL) / float64(idx.docCount)

	type scored struct {
		id    string
		score float64
	}
	var results []scored

	for chunkID, doc := range idx.docs {
		score := idx.scoreBM25(queryTokens, doc.tokens, avgDL)
		if score > 0 {
			results = append(results, scored{id: chunkID, score: score})
		}
	}

	sort.Slice(results, func(i, j int) bool {
		return results[i].score > results[j].score
	})

	if k > len(results) {
		k = len(results)
	}

	out := make([]ScoredID, k)
	for i := 0; i < k; i++ {
		out[i] = ScoredID{ID: results[i].id, Score: results[i].score}
	}
	return out
}

// Count returns the number of indexed chunks.
func (idx *BM25Index) Count() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.docCount
}

func (idx *BM25Index) scoreBM25(queryTokens, docTokens []string, avgDL float64) float64 {
	dl := float64(len(docTokens))

	// Build term frequency map for this document.
	tf := make(map[string]int)
	for _, t := range docTokens {
		tf[t]++
	}

	var score float64
	for _, qt := range queryTokens {
		termFreq := float64(tf[qt])
		if termFreq == 0 {
			continue
		}

		docFreq := float64(idx.df[qt])
		n := float64(idx.docCount)

		// IDF: log((N - df + 0.5) / (df + 0.5) + 1)
		idf := math.Log((n-docFreq+0.5)/(docFreq+0.5) + 1.0)

		// TF component: (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl/avgDL))
		tfNorm := (termFreq * (bm25K1 + 1)) / (termFreq + bm25K1*(1-bm25B+bm25B*dl/avgDL))

		score += idf * tfNorm
	}

	return score
}

// tokenize lowercases and splits text on non-alphanumeric characters.
func tokenize(text string) []string {
	lower := strings.ToLower(text)
	words := strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	return words
}
