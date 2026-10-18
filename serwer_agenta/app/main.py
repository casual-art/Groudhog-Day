"""Serwer FastAPI harnessu "Petla Swistaka".

Warstwy (dokument wdrożeniowy, sekcja 1):
  - Model Server  -- poza tym procesem (vLLM / Groq / Cerebras), klient w agent/llm_client.py
  - Agent Runtime -- app/agent (polityka + pamiec Phi + kontroler petli)
  - Harness       -- app/harness (srodowisko, checkpointy, mutacje, weryfikator, logi)

Endpointy:
  /health, /metrics            -- operacyjne
  /experiments                 -- rejestr YAML
  /runs (POST/GET)             -- uruchamianie i status runow
  /env/*                       -- kontrakt API srodowiska wg DG 8.3b
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .agent.llm_client import build_client
from .agent.runtime import LLMGridAgent, MockGridAgent, run_episode
from .harness.base import Checkpoint
from .harness.envs.gridworld import GridWorldEnv, GridWorldVerifier
from .harness.episode_log import EpisodeLogger
from .registry import build_run_config, get_experiment, load_registry

RUNS_DIR = Path(__file__).resolve().parent.parent / "runs"

app = FastAPI(title="Serwer agenta -- Petla Swistaka", version="0.1.0")
_STARTED = time.time()

# -- auth: Bearer API_TOKEN wymuszany, gdy zmienna jest ustawiona -------------
# Bez ustawionego API_TOKEN serwer dziala w trybie otwartym (lokalny
# localhost/notebook). /health pozostaje publiczne (proby liveness).
_PUBLIC_PATHS = {"/health"}


@app.middleware("http")
async def bearer_auth(request: Request, call_next):
    token = os.environ.get("API_TOKEN", "")
    if token and request.url.path not in _PUBLIC_PATHS:
        auth = request.headers.get("Authorization", "")
        if auth != f"Bearer {token}":
            return JSONResponse(status_code=401,
                                content={"detail": "brak lub bledny token "
                                                   "Authorization: Bearer"})
    return await call_next(request)

# -- stan w pamieci procesu (izolacja per run; brak wspoldzielenia Phi) ------
_runs: dict[str, dict[str, Any]] = {}       # run_id -> status/wynik
_env_sessions: dict[str, dict[str, Any]] = {}  # env_id -> (env, checkpoints)


# ===========================================================================
# Modele pydantic
# ===========================================================================

class RunRequest(BaseModel):
    experiment: str = Field(..., description="nazwa z rejestru experiments/registry.yaml")
    overrides: dict[str, Any] = Field(default_factory=dict,
                                      description="nadpisania progow [START], seeda itp.")
    background: bool = True


class RunStatus(BaseModel):
    run_id: str
    status: str                     # queued / running / finished / failed
    result: dict[str, Any] | None = None
    error: str | None = None


class MakeCheckpointRequest(BaseModel):
    scenario: str = "S1"
    seed: int = 42
    params: dict[str, Any] = Field(default_factory=dict)


class EnvRef(BaseModel):
    env_id: str


class StepRequest(BaseModel):
    env_id: str
    action: dict[str, Any]          # np. {"move": "down"} dla gridworld


class MutateRequest(BaseModel):
    env_id: str
    op: str                         # recolor / move_trap / add_wall
    args: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0
    checkpoint_id: str = "current"


# ===========================================================================
# Fabryki scenariuszy (rozszerzalne: patrz checklista w dok. wdrozeniowym)
# ===========================================================================

def _make_env(scenario: str, params: dict) -> tuple:
    if scenario == "S1":
        env = GridWorldEnv(size=params.get("size", 12),
                           n_traps=params.get("n_traps", 4),
                           t_max=params.get("t_max", 200),
                           suspicious_prob=params.get("suspicious_prob", 0.7))
        return env, GridWorldVerifier()
    raise HTTPException(422, f"nieobslugiwany scenariusz: {scenario} "
                             "(fabryka env nie ma wpisu; patrz checklista "
                             "w dok. wdrozeniowym, sekcja 7)")


# ===========================================================================
# Endpointy operacyjne
# ===========================================================================

@app.get("/health")
def health() -> dict:
    return {"status": "ok", "uptime_s": round(time.time() - _STARTED, 1),
            "runs_active": sum(1 for r in _runs.values()
                               if r["status"] in ("queued", "running"))}


@app.get("/metrics")
def metrics() -> dict:
    """Minimalne metryki tekstowe (kompatybilne z grepem / Prometheusem)."""
    return {"runs_total": len(_runs),
            "runs_finished": sum(1 for r in _runs.values()
                                 if r["status"] == "finished"),
            "runs_failed": sum(1 for r in _runs.values()
                               if r["status"] == "failed")}


@app.get("/experiments")
def experiments() -> dict:
    reg = load_registry()
    return {"experiments": [
        {"name": name, "scenario": e.get("scenario"), "group": e.get("group"),
         "agent_type": e.get("agent_type"), "description": e.get("description", "")}
        for name, e in reg.items()]}


# ===========================================================================
# Runy eksperymentow
# ===========================================================================

def _execute_run(run_id: str, exp_name: str, overrides: dict) -> None:
    rec = _runs[run_id]
    try:
        exp = get_experiment(exp_name)
        cfg = build_run_config(exp, overrides)
        run_dir = RUNS_DIR / run_id
        env, verifier = _make_env(exp["scenario"], cfg)
        if cfg.get("agent_type") == "llm":
            policy = LLMGridAgent(build_client(exp.get("model", {})))
        else:
            policy = MockGridAgent(env)
        logger = EpisodeLogger(run_dir, run_id,
                               config={"experiment": exp_name, **cfg})
        rec["status"] = "running"
        result = run_episode(env, verifier, policy, logger, cfg)
        logger.close()
        rec.update(status="finished", result=result)
    except Exception as exc:  # SYS: logowane, run oznaczony jako failed
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}")


@app.post("/runs", response_model=RunStatus, status_code=201)
def start_run(req: RunRequest) -> RunStatus:
    # Walidacja SYNCHRONICZNA, przed zakolejkowaniem: nieznany eksperyment
    # (404) i scenariusz bez implementacji fabryki (422, np. stub S2)
    # odrzucane juz przy POST, a nie w watku w tle.
    try:
        exp = get_experiment(req.experiment)
    except KeyError as exc:
        raise HTTPException(404, str(exc))
    _make_env(exp["scenario"], {})  # 422 dla nieobslugiwanych scenariuszy
    run_id = f"{req.experiment}-{uuid.uuid4().hex[:6]}"
    _runs[run_id] = {"status": "queued", "result": None, "error": None}
    if req.background:
        threading.Thread(target=_execute_run,
                         args=(run_id, req.experiment, req.overrides),
                         daemon=True).start()
    else:
        _execute_run(run_id, req.experiment, req.overrides)
    return RunStatus(run_id=run_id, **_runs[run_id])


@app.get("/runs/{run_id}", response_model=RunStatus)
def run_status(run_id: str) -> RunStatus:
    if run_id not in _runs:
        raise HTTPException(404, f"nieznany run_id: {run_id}")
    return RunStatus(run_id=run_id, **_runs[run_id])


@app.get("/runs")
def list_runs() -> dict:
    return {"runs": [{"run_id": rid, **{k: v for k, v in r.items()
                                        if k != "result"}}
                     for rid, r in _runs.items()]}


# ===========================================================================
# Kontrakt API srodowiska (DG 8.3b) -- sesje env per env_id
# ===========================================================================

@app.post("/env/make_checkpoint", status_code=201)
def env_make_checkpoint(req: MakeCheckpointRequest) -> dict:
    env, verifier = _make_env(req.scenario, req.params)
    cp = env.make_checkpoint(seed=req.seed)
    env_id = uuid.uuid4().hex[:8]
    _env_sessions[env_id] = {"env": env, "verifier": verifier,
                             "checkpoints": {"C0": cp}, "current": "C0"}
    env.restore(cp)
    return {"env_id": env_id, "checkpoint_id": "C0", "hash": cp.hash}


@app.post("/env/restore")
def env_restore(req: EnvRef, checkpoint_id: str = "current") -> dict:
    sess = _get_session(req.env_id)
    key = sess["current"] if checkpoint_id == "current" else checkpoint_id
    cp = sess["checkpoints"].get(key)
    if cp is None:
        raise HTTPException(404, f"nieznany checkpoint: {key}")
    sess["env"].restore(cp)  # I1: restore nie dotyka Phi (Phi jest w runtime)
    return {"restored": key, "hash": cp.hash}


@app.post("/env/step")
def env_step(req: StepRequest) -> dict:
    sess = _get_session(req.env_id)
    obs, reward, info = sess["env"].step(req.action)
    return {"obs": obs, "reward": reward, "info": info}


@app.post("/env/verify")
def env_verify(req: EnvRef, t: int = 0, t_max: int = 200) -> dict:
    sess = _get_session(req.env_id)
    v = sess["verifier"].check(sess["env"].state, [], t, {"t_max": t_max})
    return {"status": v.status, "cls": v.cls, "detail": v.detail,
            "progress": sess["verifier"].progress(sess["env"].state)}


@app.get("/env/state")
def env_state(env_id: str) -> dict:
    """Stan wewnetrzny -- tylko dla weryfikatora/kontrolera (DG 8.3b)."""
    sess = _get_session(env_id)
    return {"state": sess["env"].state,
            "progress": sess["verifier"].progress(sess["env"].state)}


@app.post("/env/mutate", status_code=201)
def env_mutate(req: MutateRequest) -> dict:
    """Mutacja Psi: nowa wersja checkpointu (I2), stara pozostaje."""
    sess = _get_session(req.env_id)
    key = sess["current"] if req.checkpoint_id == "current" else req.checkpoint_id
    cp = sess["checkpoints"].get(key)
    if cp is None:
        raise HTTPException(404, f"nieznany checkpoint: {key}")
    cp_new = sess["env"].mutate(cp, req.op, req.args, req.seed)
    if not sess["env"].solvable(cp_new):  # G1
        raise HTTPException(422, "mutacja odrzucona: nierozwiazywalna (G1)")
    new_id = f"C{len(sess['checkpoints'])}"
    sess["checkpoints"][new_id] = cp_new
    sess["current"] = new_id
    return {"checkpoint_id": new_id, "hash": cp_new.hash,
            "parent_hash": cp.hash}


def _get_session(env_id: str) -> dict:
    if env_id not in _env_sessions:
        raise HTTPException(404, f"nieznany env_id: {env_id}")
    return _env_sessions[env_id]


# ===========================================================================
# /chat -- minimalny endpoint rozmowy z agentem (klient terminalowy/Android)
# ===========================================================================
# Niezależny od harnessu eksperymentów: prosta rozmowa z modelem z historią
# per session_id (pamięć w RAM procesu; sesje czatu nie trafiają do runs/).

class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None      # brak -> nowa sesja
    system: str | None = None          # opcjonalny nadpisany prompt systemowy


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    turns: int


_chat_sessions: dict[str, list[dict[str, str]]] = {}
_CHAT_SYSTEM_DEFAULT = (
    "Jestes agentem testowym projektu Petla Swistaka. Odpowiadasz zwiezle, "
    "rzeczowo i po polsku. Jesli uzytkownik pyta o eksperymenty, wskaz "
    "endpointy /experiments i /runs."
)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    sid = req.session_id or uuid.uuid4().hex[:12]
    history = _chat_sessions.setdefault(sid, [])
    if not history:
        history.append({"role": "system",
                        "content": req.system or _CHAT_SYSTEM_DEFAULT})
    history.append({"role": "user", "content": req.message})
    client = build_client({"model": os.environ.get("LLM_MODEL", "local-model")})
    try:
        reply = client.chat(history)
    except Exception as exc:  # SYS po wyczerpaniu retry -> 503, nie 500
        history.pop()  # nie psuj historii nieudaną turą
        raise HTTPException(status_code=503, detail=f"LLM niedostepny: {exc}")
    history.append({"role": "assistant", "content": reply})
    return ChatResponse(session_id=sid, reply=reply, turns=len(history) // 2)


@app.delete("/chat/{session_id}")
def chat_reset(session_id: str) -> dict[str, Any]:
    """Wyczyszczenie pamięci sesji czatu (odpowiednik resetu epizodu)."""
    existed = _chat_sessions.pop(session_id, None) is not None
    return {"session_id": session_id, "cleared": existed}
