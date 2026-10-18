#!/usr/bin/env bash
# Tunel cloudflared dla Kaggle/Colab: wystawia lokalny serwer API na
# publiczny URL trycloudflare (bez konta). Uzycie:
#   bash scripts/tunnel.sh [PORT=8080]
# Wymaga pobranego binarium cloudflared (patrz scripts/kaggle_notebook.md).
set -euo pipefail
PORT="${1:-8080}"

if ! command -v cloudflared >/dev/null 2>&1; then
  echo "Pobieranie cloudflared..."
  ARCH="$(uname -m)"; [ "$ARCH" = "x86_64" ] && ARCH="amd64"
  curl -sSL -o cloudflared \
    "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${ARCH}"
  chmod +x cloudflared
  CLOUDFLARED="./cloudflared"
else
  CLOUDFLARED="cloudflared"
fi

echo "Start tunelu na porcie ${PORT} (URL pojawi sie ponizej)..."
exec "$CLOUDFLARED" tunnel --url "http://localhost:${PORT}" --no-autoupdate
