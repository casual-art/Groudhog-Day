#!/usr/bin/env bash
# ============================================================================
# deploy.sh — jednokrotne postawienie serwera "Pętla Świstaka"
#
# Co robi (kroki):
#   1. Wirtualne środowisko Pythona + zależności API.
#   2. Pobranie modelu z Hugging Face (domyślnie Qwen3-8B-AWQ, ~6 GB — bezpieczne
#      na darmowe T4 16 GB; nadpisz: MODEL_ID=Qwen/Qwen3-14B-AWQ ./deploy.sh).
#   3. Start Model Servera (vLLM, OpenAI-compatible) — pomijany w trybie --no-gpu
#      lub gdy LLM_BASE_URL wskazuje zewnętrzny endpoint (Groq/Cerebras).
#   4. Start API (FastAPI :8080) z tokenem API_TOKEN.
#   5. Opcjonalnie tunel cloudflared (--tunnel) -> publiczny URL dla klienta.
#
# Użycie:
#   API_TOKEN=trudne-haslo ./deploy.sh                 # pełny tryb z GPU
#   API_TOKEN=trudne-haslo ./deploy.sh --tunnel        # + publiczny URL
#   ./deploy.sh --no-gpu                               # test bez GPU (stub LLM)
#   LLM_BASE_URL=https://api.groq.com/openai/v1 \
#   LLM_API_KEY=gsk_... LLM_MODEL=llama-3.3-70b-versatile \
#   API_TOKEN=trudne-haslo ./deploy.sh --external-llm  # bez własnego GPU
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

MODEL_ID="${MODEL_ID:-Qwen/Qwen3-8B-AWQ}"
VLLM_PORT="${VLLM_PORT:-8000}"
API_PORT="${API_PORT:-8080}"
GPU_UTIL="${GPU_UTIL:-0.85}"
DO_TUNNEL=0; NO_GPU=0; EXTERNAL_LLM=0
for arg in "$@"; do case "$arg" in
  --tunnel) DO_TUNNEL=1;; --no-gpu) NO_GPU=1;; --external-llm) EXTERNAL_LLM=1;;
  *) echo "Nieznany argument: $arg"; exit 2;; esac; done

# Token: wygeneruj, jeśli nie podano (komunikacja po tunelu musi być chroniona).
API_TOKEN="${API_TOKEN:-$(head -c 24 /dev/urandom | base64 | tr -dc 'a-zA-Z0-9')}"

echo "==> [1/5] Środowisko Python"
python3 -m venv .venv 2>/dev/null || true
source .venv/bin/activate
pip -q install --upgrade pip
pip -q install -r requirements.txt

if [ "$EXTERNAL_LLM" = "1" ]; then
  : "${LLM_BASE_URL:?podaj LLM_BASE_URL dla --external-llm}"
  echo "==> [2/5] LLM zewnętrzny: $LLM_BASE_URL (model: ${LLM_MODEL:-local-model})"
elif [ "$NO_GPU" = "1" ]; then
  echo "==> [2/5] Tryb --no-gpu: LLM = stub lokalny (tylko test API)"
else
  echo "==> [2/5] Pobieranie modelu: $MODEL_ID"
  pip -q install "huggingface_hub[cli]"
  HF_DL="hf download $MODEL_ID"; [ -n "${HF_TOKEN:-}" ] && HF_DL="$HF_DL --token $HF_TOKEN"
  $HF_DL   # idempotentne: cache ~/.cache/huggingface
  echo "==> [3/5] Start vLLM (port $VLLM_PORT, gpu_util=$GPU_UTIL)"
  pip -q install vllm
  nohup vllm serve "$MODEL_ID" --port "$VLLM_PORT" \
        --gpu-memory-utilization "$GPU_UTIL" --max-model-len 8192 \
        > vllm.log 2>&1 &
  echo $! > vllm.pid
  echo "    czekam na gotowość vLLM (tail -f vllm.log pokazuje postęp)..."
  for i in $(seq 1 120); do
    curl -sf "http://127.0.0.1:$VLLM_PORT/health" >/dev/null && break
    sleep 5; [ "$i" = "120" ] && { echo "vLLM nie wstał w 10 min — patrz vllm.log"; exit 1; }
  done
  export LLM_BASE_URL="http://127.0.0.1:$VLLM_PORT/v1" LLM_MODEL="$MODEL_ID"
fi

echo "==> [4/5] Start API (port $API_PORT)"
export API_TOKEN RUNS_DIR="$PWD/runs"
nohup .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port "$API_PORT" \
      > api.log 2>&1 &
echo $! > api.pid
for i in $(seq 1 30); do curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null && break; sleep 2; done
curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null || { echo "API nie wstało — patrz api.log"; exit 1; }

PUBLIC_URL="http://127.0.0.1:$API_PORT"
if [ "$DO_TUNNEL" = "1" ]; then
  echo "==> [5/5] Tunel cloudflared (quick tunnel, bez konta)"
  if ! command -v cloudflared >/dev/null; then
    curl -sL -o cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
    chmod +x cloudflared; mv cloudflared ~/.local/bin/cloudflared 2>/dev/null || sudo mv cloudflared /usr/local/bin/
  fi
  nohup cloudflared tunnel --url "http://127.0.0.1:$API_PORT" --no-autoupdate \
        > tunnel.log 2>&1 &
  echo $! > tunnel.pid
  for i in $(seq 1 30); do
    PUBLIC_URL=$(grep -o 'https://[a-z0-9-]*\.trycloudflare\.com' tunnel.log | head -1) && break
    sleep 2
  done
  [ -z "$PUBLIC_URL" ] && { echo "Brak URL tunelu — patrz tunnel.log"; exit 1; }
fi

echo
echo "=================== SERWER GOTOWY ==================="
echo " URL:        $PUBLIC_URL"
echo " API_TOKEN:  $API_TOKEN"
echo " Test:       curl -H \"Authorization: Bearer $API_TOKEN\" $PUBLIC_URL/experiments"
echo " Czat:       ./termux_client.sh $PUBLIC_URL $API_TOKEN"
echo " Stop:       kill \$(cat api.pid) 2>/dev/null; [ -f vllm.pid ] && kill \$(cat vllm.pid); [ -f tunnel.pid ] && kill \$(cat tunnel.pid)"
echo " Logi:       api.log, vllm.log, tunnel.log"
echo "====================================================="
