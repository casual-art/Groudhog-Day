#!/usr/bin/env bash
# ============================================================================
# termux_client.sh — minimalny klient terminalowy agenta (Android/Termux, Linux)
#
# Instalacja na Androidzie (raz):
#   pkg install bash curl jq
#   chmod +x termux_client.sh
#
# Użycie:
#   ./termux_client.sh https://xxx.trycloudflare.com TWOJ_TOKEN
#
# Komendy w trakcie rozmowy:
#   /reset              — czyści pamięć sesji czatu (DELETE /chat/{sid})
#   /run <eksperyment>  — start runu z rejestru (POST /runs), np. /run s1-mock-baseline
#   /status <run_id>    — status runu (GET /runs/{id})
#   /exp                — lista eksperymentów
#   /quit               — wyjście
# ============================================================================
set -euo pipefail

BASE="${1:?podaj URL serwera, np. https://xxx.trycloudflare.com}"
TOKEN="${2:?podaj API_TOKEN}"
command -v jq >/dev/null || { echo "Brak jq: pkg install jq"; exit 1; }

SID=""   # session_id bieżącej rozmowy (nadawane przez serwer przy 1. wiadomości)

auth() { printf 'Authorization: Bearer %s' "$TOKEN"; }

echo "Łączę z $BASE ... "
curl -sf -H "$(auth)" "$BASE/experiments" >/dev/null \
  || { echo "Błąd połączenia/autoryzacji (sprawdź URL i token)"; exit 1; }
echo "Połączono. Pisz wiadomości; /quit kończy."

while true; do
  printf '\nTy> '; read -r LINE || break
  case "$LINE" in
    "") continue;;
    /quit) break;;
    /exp)
      curl -sf -H "$(auth)" "$BASE/experiments" | jq -r '.experiments[] | "\(.name) [\(.group)] — \(.description)"' ; continue;;
    /reset)
      [ -z "$SID" ] && { echo "(brak aktywnej sesji)"; continue; }
      curl -sf -X DELETE -H "$(auth)" "$BASE/chat/$SID" >/dev/null && SID=""
      echo "(pamięć sesji wyczyszczona)"; continue;;
    /run\ *)
      EXP="${LINE#/run }"
      RID=$(curl -sf -X POST -H "$(auth)" -H 'Content-Type: application/json' \
            -d "{\"experiment\": \"$EXP\", \"background\": true}" "$BASE/runs" | jq -r .run_id) \
        && echo "run_id: $RID  (status: /status $RID)" || echo "błąd startu runu"
      continue;;
    /status\ *)
      RID="${LINE#/status }"
      curl -sf -H "$(auth)" "$BASE/runs/$RID" | jq '{status, result, error}'; continue;;
    /*) echo "nieznana komenda ($LINE)"; continue;;
  esac

  # Zwykła wiadomość -> /chat (z session_id, jeśli rozmowa już trwa)
  BODY=$(jq -nc --arg m "$LINE" --arg s "$SID" \
         'if $s == "" then {message:$m} else {message:$m, session_id:$s} end')
  RESP=$(curl -sf -X POST -H "$(auth)" -H 'Content-Type: application/json' \
         -d "$BODY" "$BASE/chat") || { echo "(błąd HTTP — LLM niedostępny?)"; continue; }
  SID=$(printf '%s' "$RESP" | jq -r .session_id)
  printf 'Agent> %s\n' "$(printf '%s' "$RESP" | jq -r .reply)"
done
echo "Do widzenia."
