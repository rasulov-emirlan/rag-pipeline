FROM golang:alpine AS builder
WORKDIR /app
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -o rag-pipeline .

FROM alpine:3.21
RUN apk add --no-cache ca-certificates
COPY --from=builder /app/rag-pipeline /usr/local/bin/rag-pipeline
EXPOSE 8080
ENTRYPOINT ["rag-pipeline"]
