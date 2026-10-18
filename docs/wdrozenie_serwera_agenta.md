# Wdrożenie serwera agenta — „Pętla Świstaka"

**Status:** dokument wdrożeniowy, wersja 1.1 (changelog: sekcja 9)
**Powiązane dokumenty:** DG v1.2 (`projekt_petla_swistaka.md` — szczególnie sekcje 4, 6, 8.3, 9), dokument towarzyszący (`badanie_elastycznosci_strategii.md` — grupy A7–A14)
**Kod:** scaffold w `serwer_agenta/` (działający, smoke test przechodzi bez GPU)
**Cel:** infrastruktura serwerowa WIELOKROTNEGO UŻYTKU — postawiona raz, kolejne eksperymenty podpinane jak wtyczki (nowy wpis YAML + ewentualnie nowy scenariusz, bez ruszania usług). Docelowe zasoby: darmowe GPU 16 GB (Kaggle/Colab/Lightning AI), opcja skalowania na A100 40 GB (kredyty GCP).

---

## 1. Architektura: 3 warstwy jako osobne usługi z czystymi API

Fundament projektu to rozdzielenie trzech odpowiedzialności, które w DG są rozdzielone koncepcyjnie (środowisko / kontroler / agent), na trzy **procesy komunikujące się wyłącznie przez jawne API**. Dzięki temu każdą warstwę można wymienić (inny model, inny agent RL, inny scenariusz) bez dotykania pozostałych.

```
                        klient / notebook / CI
                                │  HTTP (token API)
                                v
┌────────────────────────────────────────────────────────────────┐
│  USŁUGA: API + HARNESS  (FastAPI, CPU, wystarczy 2 GB RAM)      │
│  app/main.py, app/harness/                                      │
│                                                                 │
│  Endpoints operacyjne:  /health  /metrics                       │
│  Rejestr:               /experiments  (experiments/registry.yaml)│
│  Runy:                  POST /runs, GET /runs/{id}              │
│  Kontrakt env (DG 8.3b):                                        │
│    /env/make_checkpoint  /env/restore  /env/step                │
│    /env/verify  /env/state  /env/mutate                         │
│                                                                 │
│  Odpowiada za: środowisko (Env), checkpointy C0..Ck (I1/I2),    │
│  mutator Ψ (reguły R0-R4, gwarancja G1), weryfikator V,         │
│  logi JSONL (DG 8.3a) → runs/<run_id>/episodes.jsonl            │
└───────────────▲────────────────────────────────┬───────────────┘
                │ wywołania wewnętrzne           │ aktualizacja Φ
                │ (ten sam proces,               │ tylko przez
                │  kontrakt LoopEnv/Verifier)    │ kontroler (I3)
┌───────────────┴────────────────────────────────v───────────────┐
│  AGENT RUNTIME  (app/agent/, w procesie API; wydzielalny)       │
│                                                                 │
│  - polityka: MockGridAgent (testy) | LLMGridAgent | RL (TODO)   │
│  - pamięć Φ_k = (k, τ, H_k, Q_k) — POZA snapshotem env (I1)     │
│  - kontroler pętli (DG 8.1): restore → loop → settle →          │
│    detekcja lock-in → decyzja Ψ → log                           │
│  - kontrakt akcji: {"move": ...} (grid) / {name, args} (tool)   │
└───────────────▲────────────────────────────────────────────────┘
                │ OpenAI-compatible API: POST /v1/chat/completions
                │ (base_url konfigurowalny — to jest "wtyczka" modelu)
┌───────────────┴────────────────────────────────────────────────┐
│  MODEL SERVER  (wymienna warstwa modelu)                        │
│                                                                 │
│  Opcja A: vLLM lokalnie (docker-compose profil "gpu")           │
│  Opcja B: zewnętrzny endpoint (Groq/Cerebras) przez LLM_BASE_URL│
│  Opcja C: MockLLMClient — stub do smoke testu, zero GPU         │
│                                                                 │
│  Serwer NIE wie o pętlach, checkpointach ani mutacjach.         │
│  Powtarzalność (DG 4.1): temperature=0, przypięta wersja        │
│  modelu w konfiguracji epizodu, seed inference jeśli wspierany. │
└─────────────────────────────────────────────────────────────────┘
```

**Zasada wtyczkowości:** nowy eksperyment = nowy wpis w `experiments/registry.yaml` (+ ewentualnie nowy plik `configs/<scenariusz>.yaml` lub nowa klasa `Env`). Żaden z trzech bloków nie wymaga zmiany kodu. Wymiana modelu to zmiana pola `model.base_url`/`model.model` w YAML — nie deploymentu.

**Dlaczego env API jest też wystawione przez HTTP** (`/env/*`): pozwala (a) debugować scenariusz ręcznie z notebooka, (b) podpiąć zewnętrzny runtime agenta (np. RL w PyTorch w innym procesie) bez portowania go do naszego kodu, (c) testować kontrakt DG 8.3b niezależnie od agenta. W trybie produkcyjnym kontroler pętli wywołuje te same klasy bezpośrednio (bez narzutu HTTP) — API HTTP i kontroler dzielą implementację.

---

## 2. Rejestr eksperymentów

### 2.1. Konwencja YAML (`experiments/registry.yaml`)

Każdy eksperyment to jeden wpis:

| Pole | Znaczenie | Uwagi |
|---|---|---|
| `description` | opis dla ludzi | cel / hipoteza |
| `scenario` | klucz scenariusza | konfiguracja w `configs/<scenario>.yaml` |
| `group` | grupa kontrolna A0–A6 (DG 10.2) lub A7–A14 (dok. towarzyszący 6.1) | decyduje o wariantach Φ/Ψ |
| `agent_type` | `mock` / `llm` (docelowo `rl`) | wybiera politykę w runtime |
| `model` | `{base_url, model, api_key_env}` | puste `base_url` = stub lokalny |
| `seed` | ziarno instancji zadania | determinizm (DG 4.1); sweep seedów = wiele runów tego samego eksperymentu |
| `k_max`, `tau_max`, `t_max` | budżety `[START]` (DG 7.4, 5.1) | nadpisują wartości z `configs/` |
| `overrides` (w POST /runs) | dowolne nadpisanie progu `[START]` | np. sweep kar w E1 |

### 2.2. run_id i katalog artefaktów

- `run_id = "<nazwa_eksperymentu>-<6 znaków hex>"`, np. `s1-llm-full-a3f1c2`.
- Artefakty: `runs/<run_id>/episodes.jsonl` (schemat DG 8.3a; pierwszy rekord = pełna, spłaszczona konfiguracja epizodu z `config_hash`). Pole identyfikatora w rekordach JSONL nosi nazwę **`episode_id`** (zgodnie z DG 8.3a) i jest tożsame z `run_id` warstwy API. Docelowo obok: `traces.jsonl` (pełne trajektorie, DG 3.3) i `llm_pairs.jsonl` (pary prompt/odpowiedź z hashami, DG 4.1d).
- Katalog `runs/` jest jedyną rzeczą wymagającą trwałości — wszystko inne da się odtworzyć z repo.

### 2.3. Dwie przykładowe konfiguracje (dostarczone w scaffoldzie)

**S1 — gridworld, agent mock/RL (walidacja harnessu, faza F0):**

```yaml
s1-mock-baseline:
  description: "S1 gridworld z ukrytą pułapką, agent mock BFS z pamięcią Φ"
  scenario: S1
  group: A3
  agent_type: mock
  seed: 42
  k_max: 8
  tau_max: 3000
  t_max: 200
```

**S2 — LLM tool-use (szkielet; scenariusz do implementacji wg checklisty z sekcji 7):**

```yaml
s2-llm-tooluse:
  description: "S2 tool-use LLM — sekwencja API z ukrytym błędnym krokiem"
  scenario: S2
  group: A3
  agent_type: llm
  model:
    base_url: ""              # wypełnić: vLLM lokalny albo Groq/Cerebras
    model: "qwen3-32b-awq"    # przykład — zweryfikuj wariant (sekcja 3)
  seed: 7
  k_max: 15
  tau_max: 400                # budżet wywołań narzędzi (DG 7.4)
  t_max: 30                   # T_max = 30 wywołań narzędzi (DG 5.1)
```

**Sweepy (E1, E4 z dok. towarzyszącego):** nie trzeba wpisów per komórka — jeden eksperyment + `overrides` w POST /runs, np. `{"experiment": "s1-llm-full", "overrides": {"seed": 101, "p_fatal_base": 2.0}}`. Zalecenie: przy sweepach trzymaj macierz komórek w osobnym pliku CSV/YAML poza rejestrem i generuj wywołania skryptem.

---

## 3. Dobór modeli do VRAM

**Zastrzeżenie:** rynek kwantyzacji i wariantów modeli zmienia się szybko. Poniższa tabela to punkt startowy (stan wiedzy na moment pisania); **przed każdą kampanią runów zweryfikuj**: (a) dostępność oficjalnego wariantu AWQ/GPTQ danego modelu, (b) wsparcie w bieżącej wersji vLLM (`vllm serve --model ...` loguje rzeczywiste zużycie VRAM), (c) czy kontekst 8k mieści się w limicie po załadowaniu wag. Szacunek VRAM: wagi 4-bit ≈ 0.55–0.6 GB na 1B parametrów + KV-cache (rosnący z `max_model_len` i równoległością) + ~1–2 GB narzutu CUDA.

| VRAM | Rekomendowane modele (przykłady) | Konfiguracja vLLM | Zastosowanie w projekcie |
|---|---|---|---|
| **16 GB** (Kaggle T4/P100, Colab T4, Lightning L4?) | Qwen3-14B AWQ (~9–10 GB wag); Mistral-Small 24B AWQ (~13 GB — ciasno, `max_model_len` 4–8k); Qwen3-8B bf16 (~16 GB — na styk, lepiej AWQ) | `--max-model-len 8192 --gpu-memory-utilization 0.9`, 1 równoległe żądanie | agent LLM (S1/S2); blok Φ rośnie z k — pilnuj długości kontekstu |
| **24 GB** (Colab A100? nie — L4 24 GB w Lightning/GCP) | Qwen3-32B AWQ (~18–19 GB wag) | `--max-model-len 8192`, ewentualnie `--enforce-eager` na wąskich kartach | agent LLM lepszej klasy; podsumowywacz na tym samym serwerze (drugi endpoint lub sekwencyjnie) |
| **40 GB** (A100, GCP kredyty) | Qwen3-32B bf16 (~65 GB — NIE mieści się; bf16 wymaga ~64 GB+, więc: 32B w bf16 tylko na 2×40 GB lub 80 GB) — realnie: **Qwen3-32B AWQ z dużym kontekstem 32k** albo **Llama-3.3-70B / Qwen3-72B AWQ (~38–40 GB, na styk)** | 32B AWQ: pełny komfort; 70B AWQ: `--max-model-len 8192`, `--gpu-memory-utilization 0.95` | ewaluacja końcowa F5 na najmocniejszym modelu; tryb replay kontrolny |

Korekta do powyższej tabeli (ważna): **32B w bf16 zajmuje ~64 GB i NIE mieści się na A100 40 GB** — na 40 GB sensowne są: 32B AWQ z dużym marginesem na KV-cache (rekomendowane) lub 70B AWQ na styk. Wpis „32B bf16" w zadaniu był zbyt optymistyczny.

**Alternatywa zero-VRAM: darmowe endpointy inferencyjne.** Groq i Cerebras oferują darmowe tiery API (OpenAI-compatible, z limitami rate) z szybkimi modelami klasy Llama/Qwen. Dla tego projektu to często **lepszy wybór niż własne 16 GB**: (a) całe GPU zostaje na ewentualny RL w S1, (b) brak problemów z kwantyzacją, (c) przez kontrakt OpenAI-compatible przełączenie to jedna zmienna `LLM_BASE_URL`. Ograniczenia: rate limits darmowego tieru (planuj backoff, DG 8.3c), brak gwarancji przypięcia wersji modelu na zawsze (przypinaj `model-YYYYMMDD` w konfiguracji epizodu i loguj w pierwszym rekordzie JSONL), brak seed inference na części endpointów (akceptowalne — DG 4.1 przewiduje rozluźnienie + tryb replay). **Uwaga DG 4.1(b):** zewnętrzny dostawca może podmienić wersję modelu w trakcie kampanii — przed każdą serią runów sprawdzaj nagłówek/metadane wersji i porównuj z `config_hash`.

---

## 4. Runbooki per platforma

### 4.1. Kaggle (darmowe: 30 h GPU/tydz., sesje do 9 h, T4/P100 16 GB)

**Kroki:**
1. Notebook z GPU (Settings → Accelerator → T4). Sklonuj repo do `/kaggle/working/serwer_agenta`.
2. Wykonaj komórki z `scripts/kaggle_notebook.md`: instalacja → vLLM w tle (komórka 2, ~5–10 min ładowania modelu) → API w tle (komórka 3) → opcjonalnie tunel (komórka 4).
3. Runy przez POST /runs (komórka 5) albo z zewnątrz przez tunel.
4. **Auto-checkpointing (komórka 6):** co ~30 min i zawsze przed końcem sesji — `runs/` do Kaggle Dataset (trwały) lub `rclone copy` na zewnętrzny storage. Katalog `/kaggle/working` **ginie po sesji**.

**Ograniczenia:** 30 h GPU/tydz. (pilnuj budżetu tygodniowego — duże kampanie rozbijaj na partie seedów), sesja max 9 h (planuj checkpoint co ≤ 8 h), brak gwarancji ciągłości procesów w tle po bezczynności interaktywnej, wyjście sieciowe wymaga włączenia „Internet" w notebooku (weryfikacja telefoniczna konta). Tunel `trycloudflare` jest efemeryczny (URL zmienia się przy restarcie).

### 4.2. Colab (darmowe: T4 16 GB, sesje ~12 h, agresywne odłączanie przy bezczynności)

Analogicznie do Kaggle, z różnicami:
- Limit sesji ~12 h, ale **bezczynność (~90 min) ubija runtime** — trzymaj „heartbeat" (komórka z pętlą `time.sleep` + okresowy `curl /health`) albo uruchamiaj z colab pro / przez `colabctl`.
- Trwały dysk: zamontuj Google Drive (`from google.colab import drive; drive.mount('/content/drive')`) i skonfiguruj `rclone` albo prosty `rsync runs/ /content/drive/MyDrive/swistak/runs/` co N minut (np. pętla w tle `while true; do rsync ...; sleep 900; done`).
- Tunel cloudflared identyczny (`scripts/tunnel.sh`); alternatywnie `pyngrok` (wymaga konta ngrok).
- Brak gwarantowanej alokacji GPU w godzinach szczytu — runbook musi zakładać odmowę GPU (fallback: runtime CPU + Groq jako model server).

### 4.3. Lightning AI (darmowy tier: studio z persistent storage, publiczny port, SSH)

Najbliższe „prawdziwemu serwerowi" z darmowych opcji:
1. Utwórz Studio z GPU (L4 24 GB w płatnych/darmowych kredytach — sprawdź aktualną pulę). Kod w `/teamspace/studios/<studio>/` jest **trwały** — `runs/` nie ginie między sesjami.
2. Uruchom usługi jak lokalnie: `uvicorn app.main:app --port 8080` (+ vLLM na 8000). Wystaw port 8080 przez „Public ports" studia — dostajesz stabilny URL bez tunelu.
3. SSH: `lightning connect` lub web terminal; procesy w tle przez `tmux`/`systemd --user`/`nohup`.
4. Auto-sleep: Studio usypia po bezczynności — ustaw webhook/cron wybudzający albo akceptuj przerwy (stan na dysku przetrwa; wznowienie = restart usług, run_id kontynuują się z logów).

### 4.4. GCP A100 40 GB (kredyty; pełna kontrola)

1. VM: typ maszyny `a2-highgpu-1g` ma GPU A100 **wbudowane w definicję** — flaga `--accelerator` jest zbędna (i odrzucana przez gcloud dla tej rodziny). Obraz z rodziny `deeplearning-platform-release` z metadanymi instalującymi sterownik NVIDIA (bez ręcznej konfiguracji `nvidia-container-toolkit`):

```bash
gcloud compute instances create swistak-a100 \
  --zone=us-central1-a \
  --machine-type=a2-highgpu-1g \
  --maintenance-policy=TERMINATE \
  --image-project=deeplearning-platform-release \
  --image-family=common-cu124-debian-12 \
  --metadata=install-nvidia-driver=True \
  --boot-disk-size=200GB
```

Dla kalibracji rozważ instancję spot/preemptible (`--provisioning-model=SPOT`, wtedy `--maintenance-policy=TERMINATE` jest wymuszane) — znacznie taniej, z obsługą preemption przez zapis `runs/` na trwały dysk co N minut.
2. Deploy: `docker compose --profile gpu up -d` (usługi `api` + `model-server` z `docker-compose.yml` z scaffolda; model ustaw zmienną `VLLM_MODEL`).
3. Artefakty: dysk PD zamontowany pod `runs/` + okresowy `gsutil rsync runs/ gs://<bucket>/runs/`. Przy preemptible: skrypt `preemption-handler` (metadata `preempted`) robiący natychmiastowy rsync.
4. **Koszty:** A100 40 GB on-demand ~2.9–4.1 USD/h (zależnie od regionu) — rezerwuj na fazy F3/F5; kalibrację F2 rób na Kaggle/Lightning. Ustaw **budget alert** i skrypt auto-stop VM po zakończeniu kampanii (`gcloud compute instances stop` z Cloud Scheduler).

**Wspólne dla wszystkich platform — auto-checkpointing artefaktów:** logger JSONL flushuje każdy rekord (`episode_log.py`), więc utrata sesji kosztuje maksymalnie bieżącą pętlę. Dodatkowo: kopia `runs/` na trwały storage co 15–30 min + na żądanie przed końcem sesji. Przy wznowieniu na innej maszynie runy są kompletne od ostatniego zflushowanego zdarzenia.

---

## 5. Operacyjne

**Auth.** Token API w nagłówku `Authorization: Bearer <API_TOKEN>`, **wymuszany middleware w `app/main.py`** gdy zmienna `API_TOKEN` jest ustawiona (pusta = tryb lokalny bez auth, wyłącznie localhost/notebook). `/health` pozostaje publiczne (proby liveness platformy); pozostałe endpointy bez poprawnego tokenu zwracają 401. Na tunelach publicznych (cloudflared) token **obowiązkowy** — URL trycloudflare jest nieodgadywalny, ale to nie jest kontrola dostępu. Klucze do Groq/Cerebras/GCS trzymaj w sekretach platformy (Kaggle Secrets, Colab userdata, GCP Secret Manager), nigdy w repo ani w YAML rejestru (w YAML tylko nazwa zmiennej: `api_key_env` — klient LLM odczytuje klucz z tej zmiennej w `build_client`).

**Rate limiting.** Dwa kierunki: (a) przychodzący — prosty limiter per token (np. slowapi, 60 req/min `[START]`), by przypadkowa pętla klienta nie zabiła sesji; (b) wychodzący — klient LLM respektuje limity dostawcy: retry z backoffem wykładniczym 1s→2s→4s→…, max 5 prób, jitter (dokładnie procedura SYS z DG 8.3c; odpowiedzi częściowe odrzucane w całości). Przy darmowych tierach Groq/Cerebras limit RPM jest głównym bottleneckiem kampanii — planuj sekwencyjne runy, nie równoległe.

**Persystencja logów poza sesją.** Priorytet: `runs/` to jedyne artefakty nieodtwarzalne.
- Kaggle: zapis do Kaggle Dataset (skrypt w notebooku, komórka 6) lub rclone.
- Colab: Google Drive + rsync w pętli.
- Lightning: persistent storage studia (domyślnie trwały).
- GCP: `gsutil rsync` / `rclone` do GCS; przy preemptible — hook preemption.
Konfiguracja rclone (remote + klucze) w sekretach platformy; komenda wzorcowa: `rclone copy runs/ remote:swistak/runs --min-age 5s`.

**Monitorowanie.** `/health` (liveness: uptime, liczba aktywnych runów) — sprawdzany z notebooka co iterację pętli oczekiwania i przez tunel z zewnątrz; `/metrics` (liczniki runów: total/finished/failed) — greppowalne, kompatybilne formatem z Prometheus text exposition po dodaniu prefiksów (TODO: prometheus-client). Logi aplikacji na stdout (uvicorn) → na Kaggle/Colab do `api.log`, na GCP do Cloud Logging przez agenta. Alarmy praktyczne: wzrost `runs_failed`, brak nowych rekordów w `episodes.jsonl` > 10 min podczas kampanii, `/health` nie odpowiada.

**Koszty.** Darmowe platformy: koszt = budżet czasowy (30 h/tydz. Kaggle); planuj partie seedów pod limit. GCP A100: ~3–4 USD/h — kampania 10 h ≈ 30–40 USD z kredytów; spot zmniejsza ~3×. LLM API (jeśli płatne zamiast darmowych tierów): budżet projektu ≤ 3500 USD wg DG 12 (szacunek 600–1700 USD przy ~830 epizodach); darmowe tiery Groq/Cerebras mogą pokryć znaczną część kalibracji F2 kosztem tempa.

---

## 6. Bezpieczeństwo i izolacja eksperymentów

- **Osobne katalogi runów:** `runs/<run_id>/` per run; żaden run nie pisze poza swój katalog. Analiza czyta wielu runów, nigdy nie modyfikuje.
- **Seed pinning:** seed instancji jest częścią konfiguracji eksperymentu i trafia do pierwszego rekordu JSONL; seedy mutacji są pochodną (`loop_seed = seed + k`, próby G1 `seed*1000+attempt`) — epizod jest w pełni reprodukowalny z jednej liczby (dla LLM: w trybie replay, DG 4.1).
- **Brak współdzielenia Φ między runami:** `LoopMemory` jest tworzone w `run_episode` i umiera z epizodem; sesje env (`/env/*`) są per `env_id` i nie widzą Φ. **Jedyny legalny kanał międzyepizodowy to grupa A7** (dok. towarzyszący 5.3): trwały sklep lekcji L_epi musi być wtedy (a) jawnie włączony flagą w konfiguracji eksperymentu, (b) odnotowany w pierwszym rekordzie JSONL, (c) przechowywany w pliku per *seria A7*, nigdy współdzielony z A0–A6. Każde inne „przetrwanie" pamięci między epizodami to przeciek unieważniający wyniki (test powtórki DG 10.3).
- **Izolacja wykonania narzędzi (docelowo S2/S3):** sandbox tool-use w osobnym procesie/kontenerze odtwarzany z checkpointu (DG 4.1); agent nigdy nie dostaje shella na hoście harnessu.
- **Siec:** w trybie produkcyjnym API bindowane do localhost + dostęp wyłącznie przez uwierzytelniony tunel; Model Server bez wystawienia publicznego.
- **Integralność:** każdy checkpoint ma hash (łańcuch `parent_hash`); pierwszy rekord JSONL zawiera `config_hash` — rozbieżność hashy przy replay = unieważnienie runu (DG 4.1e).

---

## 7. Jak podpinać inne eksperymenty — checklista nowego scenariusza

Scaffold jest zbudowany tak, że **nowy scenariusz to 3 kroki i zero zmian w usługach**. Przykład: dodanie S2 (tool-use) lub własnego środowiska.

1. **Środowisko + weryfikator** — nowy plik `app/harness/envs/<twoj>.py`:
   - podklasa `Env` (z `app/harness/base.py`) implementująca kontrakt DG 8.3b: `make_checkpoint(seed)`, `restore(cp)`, `reset_from_checkpoint(cp)`, `step(action)`, `state`, `mutate(cp, op, args, seed)`, `solvable(cp)`;
   - podklasa `Verifier`: `check(state, traj, t, cfg) -> Verdict` (klasy DG 5.3: FATAL/RECOV-*/TIMEOUT/STAGNATION), `progress(state) -> float w [0,1]`;
   - **twardy wymóg determinizmu (DG 4.1):** cała losowość przez seed; bez zegara ściennego i iteracji po nieposortowanych słownikach; test regresyjny `hash(run(C,A)) == hash(run(C,A))` przed rejestracją.
2. **Rejestracja** — dwa wpisy w istniejących plikach:
   - fabryka w `_make_env()` w `app/main.py` (3 linie: `if scenario == "S2": return ToolUseEnv(...), ToolUseVerifier()`);
   - operacje mutacji w `Mutator.OPS` w `app/harness/mutator.py` (mapowanie poziom → `(op, opis)`); dla scenariuszy bez solvera formalnego — test zastępczy G1 wg DG 6.3 (solver heurystyczny + kworum reguł).
3. **Konfiguracja + eksperyment** — dwa pliki YAML:
   - `configs/<SCENARIUSZ>.yaml` (parametry, budżety, progi R0–R4 `[START]`);
   - wpis w `experiments/registry.yaml` (sekcja 2).
   Opcjonalnie polityka agenta: nowa klasa w `app/agent/runtime.py` (kontrakt `act(obs, mem_view)`, `reflect(summary)`) — potrzebna tylko, gdy format akcji/obserwacji odbiega od istniejących.

**Walidacja przed kampanią:** (a) `python -m compileall app` przechodzi; (b) `bash scripts/smoke_test.sh` przechodzi; (c) run z `agent_type: mock`/skryptowym rozwiązuje scenariusz lub zawodzi z klasami błędów zgodnymi z DG 5.3; (d) JSONL zawiera wszystkie pola obowiązkowe DG 8.3a; (e) mutacje przechodzą G1 (odsetek odrzuceń < 20% `[START]`, DG 11).

---

## 8. Zgodność z DG — mapowanie komponentów

| Komponent scaffolda | Sekcja DG | Co realizuje |
|---|---|---|
| `app/harness/base.py` (Env, Verifier, Checkpoint, Verdict) | 8.3b, 4.1, 2.2 | kontrakt API środowiska; snapshot z seedem i t_c; werdykt z klasą błędu |
| `app/harness/checkpoints.py` (CheckpointStore) | 2.2, 4.2 (I2) | C0 niemutowalne; mutacje kumulowane na bieżącym C_k; łańcuch L_mut przez `parent_hash` |
| `app/harness/envs/gridworld.py` | 9.1, 5.1, 6.3 (G1) | scenariusz S1; pułapka=FATAL; **zaszumiony** sygnał „podejrzane" (`suspicious_prob`, szum deterministyczny per seed/t/pozycja); solver BFS jako G1 |
| `app/harness/mutator.py` (DetectionCounters, Mutator) | 2.2 (det_k), 6.2 (R0–R4), 6.3 | liczniki lock-in; kolejność R0→R4, pierwsza spełniona wygrywa; cooldown nie blokuje R0/R1; G1 przy każdej mutacji, fallback do poziomu 0; **G1b dla poziomów ≥ 2**: replay sekwencji ruchów ostatniej pętli na zmutowanym checkpoincie musi przegrać, inaczej odrzut i ponowne losowanie |
| `app/harness/episode_log.py` (EpisodeLogger) | 8.3a | JSONL: `episode_config` (pierwszy rekord, config_hash), `loop_start`, `loop_end` (k, τ, cls, counted, mutation z seedem, hash_checkpoint, det, phi_hash), `sys_event` (kod, liczba retry, counted=false), `episode_end` (censored+reason); identyfikator rekordów: **`episode_id`**; licznik: **`det.cooldown_left`**; flush per rekord |
| `app/agent/memory.py` (LoopMemory) | 2.2, 3.2, 3.4, 4.2 (I1/I3) | Φ_k=(k,τ,H_k,Q_k) poza snapshotem; format „PĘTLA i \| strategia \| błąd \| lekcja"; stała linia ostrzegawcza o możliwych mutacjach (G3); max 5 hipotez `[START]` |
| `app/agent/runtime.py` (run_episode) | 8.1, 7.1, 7.4, 8.3c, 8.3d | kontroler pętli: restore→loop→P_end przed k+=1 (off-by-one poprawiony w DG v1.1)→det→Ψ; koszty α·dt+β·i·dt; kara P_end = `p_fatal_base` + `p_fatal_step`·k (TIMEOUT ×`p_end_timeout_factor`); budżety k_max/τ_max z **kontrolą τ_max co `t_check` kroków wewnątrz pętli** (TIMEOUT, `cause="tau_max"`); **gałąź SYS**: pętla unieważniona (sys_event, counted=false, bez k/τ/kosztu), po 3 kolejnych unieważnieniach run kończy się statusem `aborted_sys`; cenzurowanie; cooldown tylko po mutacjach poziomu ≥ 2 |
| `app/agent/llm_client.py` | 4.1, 8.3c | temperature=0; przypięty model; seed inference jeśli wspierany; wymienny base_url; klucz API ze zmiennej `api_key_env`; **SYS-retry**: backoff wykładniczy 1s→2s→4s→…, max 5 prób, jitter, na błędy 5xx/timeout/transport; po wyczerpaniu prób `LLMSysError` (klasa SYS) |
| `app/main.py` (`/env/*`, `/runs`) | 8.3b | HTTP-owy odpowiednik kontraktu LoopEnv/Verifier z modelami pydantic; **middleware auth** `Authorization: Bearer <API_TOKEN>` (`/health` publiczne); **walidacja scenariusza przy POST /runs**: nieznany eksperyment → 404, nieobsługiwany scenariusz (stub) → 422 jeszcze przed zakolejkowaniem runu |
| `experiments/registry.yaml` + `configs/S1.yaml` | 7.4, 6.2, Aneks A | progi `[START]` jako dane, nie kod; grupa A0–A14 w konfiguracji |

**Świadome uproszczenia scaffolda (do domknięcia w F0–F2, nie blokują wtyczkowości):**
- R4 (eskalacja poziomu) — zaznaczone w kodzie jako TODO; R0–R3 pełne.
- **R3 zawsze wybiera poziom 1** (`recolor`): operacja poziomu 2d (wzmacnianie diagnostyki sygnałów, np. `suspicious_prob`) nie jest jeszcze w puli `Mutator.OPS` — deterministyczny wybór „2d przed 1" z DG 6.2 jest więc zdegenerowany do samego poziomu 1. Do dodania przy kalibracji F2.
- **R0 nie celuje w regiony sondowania**: mutator przekazuje stale `trap_index=0` zamiast identyfikować pułapki w miejscach szybkich FATAL agenta. Wymaga to agregacji pozycji FATAL w `DetectionCounters` — odroczone; gwarancja G1b częściowo kompensuje brak celowania (odrzuca mutacje nieunieważniające lock-in).
- G1b (test replay unieważnienia lock-in, DG 6.3) zaimplementowane **dla gridworld** jako replay sekwencji ruchów ostatniej pętli na zmutowanym checkpoincie; dla przyszłych scenariuszy (S2 tool-use) wymagany odpowiednik: replay zarejestrowanej sekwencji narzędzi.
- STAGNATION: weryfikator gridworld liczy TIMEOUT; okno W i licznik stagnacji są w konfiguracji i kontrakcie, pętla detekcji do dodania przy RL.
- Kara P_early (DG 7.1) i klasy RECOV z eskalacją (DG 5.3) — parametry w `configs/S1.yaml`, rozliczenie do implementacji wraz z agentem RL.
- Tryb replay par (prompt, odpowiedź) (DG 4.1e) i plik `llm_pairs.jsonl` — wymagane przed F1 dla agenta LLM.
- Limit mutacji 40% z przerwaniem `censored_mutation_limit` (DG 6.2/8.3e) — licznik jest (`mutations_so_far`), przerwanie do dodania w `run_episode`.
- Grupy A5 (yoked) i A6 (placebo Φ) — wymagają odtwarzacza sekwencji mutacji i podmieniacza Φ; oba to funkcje rejestru/runtime, nie infrastruktury.

---

## 9. Changelog wdrożeniowy

### v1.1 (runda recenzencka) — mapowanie uwaga → rozwiązanie

| # | Uwaga recenzencka | Rozwiązanie |
|---|---|---|
| 1 | Brak wymuszenia auth mimo opisu w §5 | Middleware Bearer w `app/main.py` (`API_TOKEN` ustawione → 401 bez tokenu, `/health` publiczne; brak zmiennej = tryb otwarty lokalny); `smoke_test.sh` startuje serwer z tokenem i wysyła nagłówek (test pozytywny + negatywny 401) |
| 2 | Brak procedury SYS (DG 8.3c) | `llm_client.py`: retry z backoffem wykładniczym 1s→2s→4s→…, max 5 prób, jitter, na 5xx/timeout/błędy transportu; wyczerpanie prób → `LLMSysError`. `run_episode`: gałąź SYS przed rozliczeniem kosztu/τ — `sys_event`, `loop_end` z `counted=false`, `continue`; po 3 kolejnych unieważnieniach run kończy się statusem `aborted_sys` (wykluczony z analizy) |
| 3 | Brak testu G1b (unieważnienie lock-in) | `Mutator.decide`: dla mutacji poziomu ≥ 2 w pętli prób G1 dodany test replay — sekwencja ruchów ostatniej pętli odtworzona na zmutowanym checkpoincie musi przegrać (inaczej odrzut i ponowne losowanie); implementacja dla gridworld, odpowiednik dla S2 opisany w §8 |
| 4 | τ_max sprawdzane tylko między pętlami | `run_episode`: co `t_check` kroków warunek `tau + t >= tau_max` kończy pętlę werdyktem TIMEOUT z `cause="tau_max"` (DG 8.3d) |
| 5 | Runbook GCP: zbędne `--accelerator`, brak obrazu z driverami | §4.4: usunięto `--accelerator` (GPU wbudowane w `a2-highgpu-1g`), dodano `--maintenance-policy=TERMINATE` i obraz z rodziny `deeplearning-platform-release` z `--metadata=install-nvidia-driver=True` |
| 6 | Cooldown po każdej mutacji | `run_episode`: `reset_after_mutation` dostaje cooldown tylko dla poziomu ≥ 2 (DG 6.2); mutacje poziomu 1 resetują liczniki bez blokady |
| 7 | Nieudokumentowane uproszczenia R3/R0 | §8: dopisano — R3 zawsze poziom 1 (brak operacji 2d w puli), R0 nie celuje w regiony sondowania (`trap_index=0`); obie pozycje na liście do F2 |
| 8 | Martwe klucze konfigu | Podłączone: `p_fatal_base`/`p_fatal_step`/`p_end_timeout_factor` → kara P_end w `runtime.py`; `suspicious_prob` → zaszumiony sygnał w `_obs` gridworld (deterministyczny szum per seed/t/pozycja); `api_key_env` → odczyt klucza w `build_client` |
| 9 | Rozjazd schematu logu z DG 8.3a | `episode_id` zamiast `run_id` w rekordach JSONL; `det.cooldown_left` zamiast `cooldown`; `loop_seed = seed + k` liczone dla k po inkrementacji; dokument zaktualizowany (§2.2, §6, §8) |
| 10 | Deprecated entrypoint vLLM w notebooku | Komórka 2 `scripts/kaggle_notebook.md`: `vllm serve <model> --max-model-len ...` zamiast `python -m vllm.entrypoints.openai.api_server` |
| — | Błąd scenariusza-stuba dopiero w wątku runu | `POST /runs` waliduje synchronicznie: nieznany eksperyment → 404, scenariusz bez fabryki (np. stub S2) → 422 przed zakolejkowaniem; ten sam kod 422 w `/env/make_checkpoint` |

Walidacja v1.1: `python -m compileall app` — OK; `bash scripts/smoke_test.sh` — OK (w tym asercje 401/422 i pola `episode_id` w JSONL).

---

*Koniec dokumentu wdrożeniowego.*
