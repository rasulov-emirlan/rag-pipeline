#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${RAG_URL:-http://localhost:8080}"
PASSED=0
FAILED=0

pass() { echo "  PASS: $1"; PASSED=$((PASSED + 1)); }
fail() { echo "  FAIL: $1 — $2"; FAILED=$((FAILED + 1)); }

echo "=== RAG Pipeline Smoke Tests ==="
echo "Target: $BASE_URL"
echo ""

# 1. Liveness probe
echo "[1/8] Liveness probe..."
if curl -sf "$BASE_URL/healthz" | grep -q '"ok"'; then
  pass "GET /healthz returns ok"
else
  fail "GET /healthz" "did not return ok"
fi

# 2. Readiness probe
echo "[2/8] Readiness probe..."
STATUS=$(curl -sf -o /dev/null -w "%{http_code}" "$BASE_URL/readyz" 2>/dev/null || echo "000")
if [ "$STATUS" = "200" ]; then
  pass "GET /readyz returns 200 (all deps healthy)"
else
  fail "GET /readyz" "status $STATUS (dependencies may be unhealthy)"
fi

# 3. Rich health check
echo "[3/8] Rich health check..."
if curl -sf "$BASE_URL/health" | grep -q '"status"'; then
  pass "GET /health returns status with dependency checks"
else
  fail "GET /health" "missing status field"
fi

# 4. Ingest a document
echo "[4/8] Ingest document..."
INGEST=$(curl -sf -X POST "$BASE_URL/v1/documents" \
  -H "Content-Type: application/json" \
  -d '{"id":"smoke-test","content":"The Eiffel Tower is a wrought-iron lattice tower in Paris, France. It was constructed from 1887 to 1889 as the centerpiece of the 1889 World Fair. The tower is 330 metres tall and was the tallest man-made structure in the world until 1930.","metadata":{"source":"smoke-test"}}' 2>/dev/null)

if echo "$INGEST" | grep -q '"smoke-test"'; then
  CHUNKS=$(echo "$INGEST" | grep -o '"chunk_count":[0-9]*' | grep -o '[0-9]*')
  pass "POST /v1/documents ingested ($CHUNKS chunks)"
else
  fail "POST /v1/documents" "ingest failed: $INGEST"
fi

# 5. List documents
echo "[5/8] List documents..."
LIST=$(curl -sf "$BASE_URL/v1/documents" 2>/dev/null)
if echo "$LIST" | grep -q '"smoke-test"'; then
  pass "GET /v1/documents lists the ingested document"
else
  fail "GET /v1/documents" "document not listed: $LIST"
fi

# 6. Query
echo "[6/8] Query document..."
QUERY=$(curl -sf -X POST "$BASE_URL/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question":"How tall is the Eiffel Tower?","k":3}' 2>/dev/null)

if echo "$QUERY" | grep -q '"answer"'; then
  pass "POST /v1/query returned an answer"
  # Check if sources reference our document
  if echo "$QUERY" | grep -q '"smoke-test"'; then
    pass "Query sources reference the correct document"
    PASSED=$((PASSED - 1)) # don't double count, just log
  fi
else
  fail "POST /v1/query" "no answer returned: $QUERY"
fi

# 7. Delete document
echo "[7/8] Delete document..."
DEL=$(curl -sf -X DELETE "$BASE_URL/v1/documents/smoke-test" 2>/dev/null)
if echo "$DEL" | grep -q '"deleted"'; then
  pass "DELETE /v1/documents/smoke-test succeeded"
else
  fail "DELETE /v1/documents/smoke-test" "delete failed: $DEL"
fi

# 8. Verify deletion
echo "[8/8] Verify deletion..."
LIST2=$(curl -sf "$BASE_URL/v1/documents" 2>/dev/null)
COUNT=$(echo "$LIST2" | grep -o '"count":[0-9]*' | grep -o '[0-9]*')
if [ "${COUNT:-1}" = "0" ]; then
  pass "Document confirmed deleted (count=0)"
else
  fail "Verify deletion" "count=$COUNT, expected 0"
fi

echo ""
echo "=== Results: $PASSED passed, $FAILED failed ==="

if [ "$FAILED" -gt 0 ]; then
  exit 1
fi
