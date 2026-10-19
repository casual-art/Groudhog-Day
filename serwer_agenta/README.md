[![DOI](https://zenodo.org/badge/1424913646.svg)](https://doi.org/10.5281/zenodo.23483377)

# Serwer agenta — „Pętla Świstaka" (scaffold wielokrotnego użytku)

Infrastruktura serwerowa dla projektu „Pętla Świstaka" (DG v1.2): harness
środowiska z checkpointami i mutacjami + runtime agenta z pamięcią Φ,
wystawione jako usługa FastAPI. Warstwa modelu (vLLM / Groq / Cerebras) jest
wymienna przez `base_url` — nowy eksperyment to nowy wpis w
`experiments/registry.yaml`, bez ruszania kodu usług.

Szczegóły wdrożenia per platforma (Kaggle / Colab / Lightning AI / GCP A100):
`../wdrozenie_serwera_agenta.md`.

## Quickstart lokalny (CPU, bez GPU)

```bash
pip install -r requirements.txt
uvicorn app.main:app --port 8080

# w drugim terminalu:
curl -s localhost:8080/health
curl -s localhost:8080/experiments
curl -s -X POST localhost:8080/runs \
  -H 'Content-Type: application/json' \
  -d '{"experiment": "s1-mock-baseline", "background": false}'
# logi epizodu: runs/<run_id>/episodes.jsonl (schemat DG 8.3a)
```

Run `s1-mock-baseline` używa agenta mock (BFS + pamięć znanych pułapek) —
nie wymaga GPU ani sieci i służy walidacji harnessu (faza F0, DG 12).

## Quickstart na Kaggle (darmowe GPU 16 GB)

Patrz `scripts/kaggle_notebook.md` — gotowe komórki: instalacja, vLLM w tle,
serwer API, tunel cloudflared (`scripts/tunnel.sh`), start runu, zapis
artefaktów przed końcem sesji (limit 9 h/sesja, 30 h/tydz.).

## Podpięcie własnego modelu LLM

```bash
export LLM_BASE_URL=http://localhost:8000      # lokalny vLLM
# albo: export LLM_BASE_URL=https://api.groq.com/openai
export LLM_API_KEY=...
curl -X POST localhost:8080/runs -d '{"experiment": "s1-llm-full"}' \
  -H 'Content-Type: application/json'
```

## Dodanie nowego scenariusza (3 kroki)

1. Utwórz `app/harness/envs/<twoj>.py`: podklasy `Env` i `Verifier`
   (kontrakt w `app/harness/base.py`, wzorzec: `envs/gridworld.py`).
2. Zarejestruj fabrykę w `_make_env()` w `app/main.py` i operacje mutacji
   w `Mutator.OPS` (`app/harness/mutator.py`).
3. Dodaj `configs/<SCENARIUSZ>.yaml` i wpis w `experiments/registry.yaml`.

## Struktura

```
app/main.py        serwer FastAPI: /health /experiments /runs /env/*
app/registry.py    loader rejestru eksperymentow (YAML)
app/harness/       srodowisko: base, checkpoints, mutator (R0-R4), logi JSONL
app/harness/envs/  scenariusze (S1 gridworld)
app/agent/         runtime: kontroler petli, pamiec Phi, klienci LLM
experiments/       rejestr eksperymentow (YAML)
configs/           konfiguracje scenariuszy (progi [START])
scripts/           notebook Kaggle + tunel cloudflared
runs/              artefakty runow (episodes.jsonl)
```
