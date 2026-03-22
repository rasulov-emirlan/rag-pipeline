#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${RAG_URL:-http://localhost:8080}"

echo "=== Seeding RAG Pipeline with demo documents ==="
echo ""

# Document 1: Go programming
curl -sf -X POST "$BASE_URL/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "go-overview",
    "content": "Go is an open-source programming language created at Google in 2009 by Robert Griesemer, Rob Pike, and Ken Thompson. It is statically typed and compiled, with syntax loosely derived from C. Go features garbage collection, structural typing, and CSP-style concurrency with goroutines and channels. Goroutines are lightweight threads managed by the Go runtime, making it easy to write concurrent programs. The language was designed for building scalable, high-performance server-side applications. Go is widely used for cloud infrastructure (Docker, Kubernetes), web services, CLI tools, and DevOps tooling. The standard library provides comprehensive HTTP, JSON, crypto, and testing packages.",
    "metadata": {"topic": "programming", "source": "seed"}
  }' | jq .
echo ""

# Document 2: Kubernetes
curl -sf -X POST "$BASE_URL/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "kubernetes-overview",
    "content": "Kubernetes is an open-source container orchestration platform originally developed by Google and now maintained by the Cloud Native Computing Foundation (CNCF). It automates deployment, scaling, and management of containerized applications. Key concepts include Pods (smallest deployable units), Deployments (declarative updates), Services (stable networking), ConfigMaps and Secrets (configuration), and Namespaces (resource isolation). Kubernetes uses a control plane architecture with an API server, scheduler, controller manager, and etcd for state storage. Worker nodes run the kubelet agent and container runtime. Horizontal Pod Autoscaler (HPA) automatically scales workloads based on CPU, memory, or custom metrics.",
    "metadata": {"topic": "infrastructure", "source": "seed"}
  }' | jq .
echo ""

# Document 3: RAG systems
curl -sf -X POST "$BASE_URL/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "rag-overview",
    "content": "Retrieval-Augmented Generation (RAG) is a technique that enhances Large Language Model responses by retrieving relevant documents from a knowledge base before generating an answer. The pipeline consists of document ingestion (chunking, embedding, storing in a vector database), and query processing (embedding the question, similarity search, prompt construction, and LLM generation). Key challenges include chunking strategy (size, overlap, semantic boundaries), embedding model selection, retrieval quality (hybrid search combining dense vectors and sparse BM25), reranking for precision, and evaluation metrics (retrieval recall, answer faithfulness, relevance). Advanced techniques include Corrective RAG (self-correcting retrieval), Contextual Chunking (enriching chunks with document context), and Agentic RAG (agent-driven multi-step retrieval).",
    "metadata": {"topic": "ai-infrastructure", "source": "seed"}
  }' | jq .
echo ""

# Document 4: OpenTelemetry
curl -sf -X POST "$BASE_URL/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "otel-overview",
    "content": "OpenTelemetry (OTel) is a vendor-neutral observability framework for generating, collecting, and exporting telemetry data including traces, metrics, and logs. It provides SDKs for many languages including Go, Java, Python, and JavaScript. Traces follow requests across service boundaries using context propagation. Metrics track system behavior (counters, histograms, gauges). The OTel Collector receives, processes, and exports telemetry to backends like Prometheus (metrics), Jaeger or Tempo (traces), and Loki (logs). Instrumentation can be automatic or manual. In Go, the go.opentelemetry.io/otel package provides the core API, with exporters for OTLP/gRPC, Prometheus, and more.",
    "metadata": {"topic": "observability", "source": "seed"}
  }' | jq .
echo ""

echo "=== Seed complete. Ingested 4 documents. ==="
echo "Try: curl -X POST $BASE_URL/v1/query -H 'Content-Type: application/json' -d '{\"question\": \"What is Go?\"}'"
