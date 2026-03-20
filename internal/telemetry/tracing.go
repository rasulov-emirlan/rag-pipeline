package telemetry

import (
	"context"
	"log/slog"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlptrace/otlptracegrpc"
	"go.opentelemetry.io/otel/propagation"
	sdktrace "go.opentelemetry.io/otel/sdk/trace"
	"go.opentelemetry.io/otel/trace"
)

// Tracer is the package-level tracer used across the pipeline.
var Tracer trace.Tracer

func init() {
	Tracer = trace.NewNoopTracerProvider().Tracer("rag-pipeline")
}

// InitTracing sets up the OpenTelemetry trace pipeline with OTLP gRPC export.
// If endpoint is empty, tracing is disabled (noop).
func InitTracing(ctx context.Context, endpoint string) (func(), error) {
	if endpoint == "" {
		slog.Info("otel tracing disabled (no endpoint configured)")
		return func() {}, nil
	}

	exporter, err := otlptracegrpc.New(ctx,
		otlptracegrpc.WithEndpoint(endpoint),
		otlptracegrpc.WithInsecure(),
	)
	if err != nil {
		return nil, err
	}

	provider := sdktrace.NewTracerProvider(
		sdktrace.WithBatcher(exporter,
			sdktrace.WithBatchTimeout(5*time.Second),
		),
		sdktrace.WithSampler(sdktrace.AlwaysSample()),
	)

	otel.SetTracerProvider(provider)
	otel.SetTextMapPropagator(propagation.NewCompositeTextMapPropagator(
		propagation.TraceContext{},
		propagation.Baggage{},
	))

	Tracer = provider.Tracer("rag-pipeline")

	shutdown := func() {
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		if err := provider.Shutdown(ctx); err != nil {
			slog.Error("otel trace shutdown error", "error", err)
		}
	}

	slog.Info("otel tracing enabled", "endpoint", endpoint)
	return shutdown, nil
}
