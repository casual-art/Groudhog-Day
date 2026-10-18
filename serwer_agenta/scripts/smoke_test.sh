#!/usr/bin/env bash
# Smoke test serwera (bez GPU, agent mock). Uzycie: bash scripts/smoke_test.sh [PORT]
# Serwer startuje z ustawionym API_TOKEN -> middleware auth (Bearer) jest
# wymuszany na wszystkich endpointach poza /health.
set -euo pipefail
PORT="${1:-8091}"
TOKEN="smoke-test-token"
cd "$(dirname "$0")/.."

API_TOKEN="$TOKEN" uvicorn app.main:app --port "$PORT" > /tmp/swistak_smoke.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
sleep 4

AUTH="Authorization: Bearer ${TOKEN}"

echo "== /health (publiczne, bez tokenu)"
curl -sf "localhost:${PORT}/health"; echo

echo "== auth: brak tokenu -> 401"
CODE=$(curl -s -o /dev/null -w '%{http_code}' "localhost:${PORT}/experiments")
[ "$CODE" = "401" ] || { echo "FAIL: oczekiwano 401, jest $CODE"; exit 1; }
echo "401 OK"

echo "== /experiments (z tokenem)"
curl -sf -H "$AUTH" "localhost:${PORT}/experiments" | head -c 400; echo

echo "== POST /runs scenariusz-stub -> 422 (odrzut juz przy POST)"
CODE=$(curl -s -o /dev/null -w '%{http_code}' -X POST "localhost:${PORT}/runs" \
  -H "$AUTH" -H 'Content-Type: application/json' \
  -d '{"experiment":"s2-llm-tooluse","background":false}')
[ "$CODE" = "422" ] || { echo "FAIL: oczekiwano 422, jest $CODE"; exit 1; }
echo "422 OK"

echo "== POST /runs (mock agent, sync)"
RESP=$(curl -sf -X POST "localhost:${PORT}/runs" -H "$AUTH" \
  -H 'Content-Type: application/json' \
  -d '{"experiment":"s1-mock-baseline","background":false,"overrides":{"seed":5}}')
echo "$RESP"
RUN_ID=$(echo "$RESP" | python -c "import sys,json;print(json.load(sys.stdin)['run_id'])")
echo "== GET /runs/$RUN_ID"; curl -sf -H "$AUTH" "localhost:${PORT}/runs/${RUN_ID}"; echo
echo "== log JSONL (pola DG 8.3a):"; head -3 "runs/${RUN_ID}/episodes.jsonl"
grep -q '"episode_id"' "runs/${RUN_ID}/episodes.jsonl" \
  || { echo "FAIL: brak pola episode_id w logu"; exit 1; }
echo "SMOKE TEST OK"
