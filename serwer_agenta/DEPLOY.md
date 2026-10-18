# Szybki start — serwer + agent + komunikacja (15 minut)

## A. Na serwerze (Kaggle / Colab / VPS z GPU)

```bash
# 1. Wgraj katalog serwer_agenta/ na serwer (git/scp/wget).
cd serwer_agenta

# 2. Jeden skrypt stawia wszystko (venv, pobranie modelu, vLLM, API, tunel):
API_TOKEN='wybierz-trudne-haslo' ./deploy.sh --tunnel
#    Bez GPU (sam test API):        ./deploy.sh --no-gpu
#    Z zewnętrznym LLM (bez GPU):   LLM_BASE_URL=https://api.groq.com/openai/v1 \
#                                   LLM_API_KEY=gsk_... LLM_MODEL=llama-3.3-70b-versatile \
#                                   API_TOKEN=haslo ./deploy.sh --external-llm --tunnel
```

Na koniec skrypt wypisze **URL** (np. `https://xxx.trycloudflare.com`) i **API_TOKEN** — zapisz oba.

Domyślny model: `Qwen/Qwen3-8B-AWQ` (~6 GB, bezpieczny na T4 16 GB).
Większy: `MODEL_ID=Qwen/Qwen3-14B-AWQ API_TOKEN=... ./deploy.sh --tunnel`.
Na Kaggle: wklej zawartość kroków do komórek wg `scripts/kaggle_notebook.md`
(pamiętaj o limicie sesji 9 h — `runs/` to jedyne nieodtwarzalne dane, kopiuj je na Dataset).

## B. Na Androidzie (Termux)

```bash
pkg install bash curl jq
# wgraj termux_client.sh (np. curl -O z repo / termux-setup-storage + cp)
chmod +x termux_client.sh
./termux_client.sh https://xxx.trycloudflare.com 'wybierz-trudne-haslo'
```

Komendy w rozmowie: `/exp` (lista eksperymentów), `/run <nazwa>` (start runu),
`/status <run_id>`, `/reset` (czyszczenie pamięci sesji), `/quit`.

## C. Test bez Androida (z dowolnego Linuksa)

Identyczna komenda jak w B — klient to zwykły bash+curl.

## Zatrzymanie serwera

```bash
kill $(cat api.pid); [ -f vllm.pid ] && kill $(cat vllm.pid); [ -f tunnel.pid ] && kill $(cat tunnel.pid)
```
