FROM golang:1.25-alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o rag-pipeline .

FROM alpine:3.21
RUN apk add --no-cache ca-certificates curl
RUN adduser -D -u 1000 app
USER app
COPY --from=builder /app/rag-pipeline /usr/local/bin/rag-pipeline
EXPOSE 8080
HEALTHCHECK --interval=10s --timeout=3s --retries=3 \
  CMD curl -f http://localhost:8080/healthz || exit 1
ENTRYPOINT ["rag-pipeline"]
