# Notebook Kaggle — szkielet kroków (wklej kolejne bloki do komórek)

## Komórka 1: przygotowanie repo i zależności
```bash
!git clone <URL_REPO> /kaggle/working/serwer_agenta || true
%cd /kaggle/working/serwer_agenta
!pip install -q -r requirements.txt
```

## Komórka 2 (GPU, opcjonalnie): serwer modelu vLLM w tle
```bash
# Kaggle: 1x T4 16 GB (P100 też 16 GB). Model z tabeli VRAM z dok. wdrożeniowego.
!pip install -q vllm
!nohup vllm serve Qwen/Qwen3-14B-AWQ \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 --port 8000 \
    > vllm.log 2>&1 &
# poczekaj na "Application startup complete" w vllm.log (kilka minut)
```

## Komórka 3: serwer API (harness + runtime) w tle
```bash
import os
os.environ["LLM_BASE_URL"] = "http://localhost:8000"   # albo https://api.groq.com/openai
os.environ["LLM_API_KEY"]  = "none"                    # klucz, jeśli endpoint zewnętrzny
!nohup uvicorn app.main:app --host 0.0.0.0 --port 8080 > api.log 2>&1 &
!sleep 5 && curl -s http://localhost:8080/health
```

## Komórka 4 (opcjonalnie): tunel publiczny
```bash
!nohup bash scripts/tunnel.sh 8080 > tunnel.log 2>&1 &
!sleep 8 && grep -o "https://[^ ]*trycloudflare.com" tunnel.log | head -1
```

## Komórka 5: start runu i polling statusu
```python
import httpx, time
r = httpx.post("http://localhost:8080/runs",
               json={"experiment": "s1-mock-baseline", "background": True})
run_id = r.json()["run_id"]
while True:
    st = httpx.get(f"http://localhost:8080/runs/{run_id}").json()
    if st["status"] in ("finished", "failed"):
        print(st); break
    time.sleep(2)
```

## Komórka 6 (PRZED KOŃCEM SESJI): auto-checkpointing artefaktów
Sesja Kaggle to max 9 h; katalog /kaggle/working ginie po sesji.
Zapisz `runs/` jako Kaggle Dataset (trwały) — uruchamiaj co ~30 min i na końcu:
```python
import shutil, json
shutil.make_archive("/kaggle/working/runs_backup", "zip", "runs")
# Następnie: zakładka Output -> "New Dataset" z /kaggle/working,
# albo kaggle datasets create -p /kaggle/working/runs_backup
# Wariant chmurowy: rclone copy runs/ remote:swistak/runs (config w Secrets)
```
Alternatywa bez ręcznego klikania: `!rclone copy /kaggle/working/runs gdrive:swistak --min-age 10s`
z konfiguracją rclone trzymaną w Kaggle Secrets.
