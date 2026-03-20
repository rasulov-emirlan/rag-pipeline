package telemetry

import (
	"context"
	"log/slog"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetricgrpc"
	"go.opentelemetry.io/otel/metric"
	"go.opentelemetry.io/otel/metric/noop"
	sdkmetric "go.opentelemetry.io/otel/sdk/metric"
)

// Metrics holds all OTel instruments for the RAG pipeline.
type Metrics struct {
	IngestTotal       metric.Int64Counter
	IngestChunks      metric.Int64Counter
	IngestDuration    metric.Float64Histogram
	QueryTotal        metric.Int64Counter
	QueryDuration     metric.Float64Histogram
	ChunksRetrieved   metric.Int64Counter
	EmbeddingDuration metric.Float64Histogram
	LLMDuration       metric.Float64Histogram
}

// WithAttr is a convenience wrapper for metric.WithAttributes.
func WithAttr(attrs ...attribute.KeyValue) metric.MeasurementOption {
	return metric.WithAttributes(attrs...)
}

// New initializes OTel metrics. If endpoint is empty, returns no-op metrics.
func New(ctx context.Context, endpoint string) (*Metrics, func(), error) {
	if endpoint == "" {
		slog.Info("otel metrics disabled (no endpoint configured)")
		return newNoopMetrics(), func() {}, nil
	}

	exporter, err := otlpmetricgrpc.New(ctx,
		otlpmetricgrpc.WithEndpoint(endpoint),
		otlpmetricgrpc.WithInsecure(),
	)
	if err != nil {
		return nil, nil, err
	}

	provider := sdkmetric.NewMeterProvider(
		sdkmetric.WithReader(sdkmetric.NewPeriodicReader(exporter, sdkmetric.WithInterval(10*time.Second))),
	)
	otel.SetMeterProvider(provider)

	meter := provider.Meter("rag-pipeline")
	m, err := newMetrics(meter)
	if err != nil {
		return nil, nil, err
	}

	shutdown := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := provider.Shutdown(ctx); err != nil {
			slog.Error("otel shutdown error", "error", err)
		}
	}

	slog.Info("otel metrics enabled", "endpoint", endpoint)
	return m, shutdown, nil
}

func newMetrics(meter metric.Meter) (*Metrics, error) {
	ingestTotal, err := meter.Int64Counter("rag_pipeline_ingest_documents_total",
		metric.WithDescription("Total documents ingested"))
	if err != nil {
		return nil, err
	}
	ingestChunks, err := meter.Int64Counter("rag_pipeline_ingest_chunks_total",
		metric.WithDescription("Total chunks created"))
	if err != nil {
		return nil, err
	}
	ingestDuration, err := meter.Float64Histogram("rag_pipeline_ingest_duration_seconds",
		metric.WithDescription("Document ingestion duration"))
	if err != nil {
		return nil, err
	}
	queryTotal, err := meter.Int64Counter("rag_pipeline_query_total",
		metric.WithDescription("Total RAG queries"))
	if err != nil {
		return nil, err
	}
	queryDuration, err := meter.Float64Histogram("rag_pipeline_query_duration_seconds",
		metric.WithDescription("RAG query duration"))
	if err != nil {
		return nil, err
	}
	chunksRetrieved, err := meter.Int64Counter("rag_pipeline_chunks_retrieved_total",
		metric.WithDescription("Total chunks retrieved"))
	if err != nil {
		return nil, err
	}
	embeddingDuration, err := meter.Float64Histogram("rag_pipeline_embedding_duration_seconds",
		metric.WithDescription("Embedding generation duration"))
	if err != nil {
		return nil, err
	}
	llmDuration, err := meter.Float64Histogram("rag_pipeline_llm_duration_seconds",
		metric.WithDescription("LLM generation duration"))
	if err != nil {
		return nil, err
	}

	return &Metrics{
		IngestTotal:       ingestTotal,
		IngestChunks:      ingestChunks,
		IngestDuration:    ingestDuration,
		QueryTotal:        queryTotal,
		QueryDuration:     queryDuration,
		ChunksRetrieved:   chunksRetrieved,
		EmbeddingDuration: embeddingDuration,
		LLMDuration:       llmDuration,
	}, nil
}

func newNoopMetrics() *Metrics {
	meter := noop.Meter{}
	ingestTotal, _ := meter.Int64Counter("rag_pipeline_ingest_documents_total")
	ingestChunks, _ := meter.Int64Counter("rag_pipeline_ingest_chunks_total")
	ingestDuration, _ := meter.Float64Histogram("rag_pipeline_ingest_duration_seconds")
	queryTotal, _ := meter.Int64Counter("rag_pipeline_query_total")
	queryDuration, _ := meter.Float64Histogram("rag_pipeline_query_duration_seconds")
	chunksRetrieved, _ := meter.Int64Counter("rag_pipeline_chunks_retrieved_total")
	embeddingDuration, _ := meter.Float64Histogram("rag_pipeline_embedding_duration_seconds")
	llmDuration, _ := meter.Float64Histogram("rag_pipeline_llm_duration_seconds")

	return &Metrics{
		IngestTotal:       ingestTotal,
		IngestChunks:      ingestChunks,
		IngestDuration:    ingestDuration,
		QueryTotal:        queryTotal,
		QueryDuration:     queryDuration,
		ChunksRetrieved:   chunksRetrieved,
		EmbeddingDuration: embeddingDuration,
		LLMDuration:       llmDuration,
	}
}
