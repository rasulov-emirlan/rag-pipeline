.PHONY: build run test lint up down logs smoke seed

build:
	go build -o bin/rag-pipeline .

run:
	go run .

test:
	go test ./...

lint:
	golangci-lint run ./...

up:
	cp -n .env.example .env 2>/dev/null || true
	docker compose up --build -d
	@echo ""
	@echo "Waiting for all services (this may take a few minutes on first run — pulling models)..."
	@timeout=300; elapsed=0; \
	while [ $$elapsed -lt $$timeout ]; do \
		if curl -sf http://localhost:8080/healthz > /dev/null 2>&1; then \
			echo "All services ready!"; \
			echo "  App:        http://localhost:8080/health"; \
			echo "  Grafana:    http://localhost:3000"; \
			echo "  Prometheus: http://localhost:9090"; \
			echo "  Weaviate:   http://localhost:8081"; \
			echo ""; \
			echo "Run 'make seed' to load demo data, then 'make smoke' to verify."; \
			exit 0; \
		fi; \
		sleep 5; \
		elapsed=$$((elapsed + 5)); \
		printf "."; \
	done; \
	echo ""; \
	echo "Timeout waiting for services. Check 'docker compose logs' for errors."; \
	exit 1

down:
	docker compose down

logs:
	docker compose logs -f rag-pipeline

smoke:
	@bash scripts/smoke.sh

seed:
	@bash scripts/seed.sh
