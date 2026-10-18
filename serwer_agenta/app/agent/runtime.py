"""Runtime agenta i kontroler petli (uproszczony DG 8.1).

Agent runtime to osobna warstwa: polityka (mock / LLM) + pamiec Phi.
Kontroler petli jest wlascicielem zegara i licznikow -- agent nie moze
ich ominac (DG 8.2).
"""
from __future__ import annotations

import re
from collections import deque
from typing import Any

from ..harness.base import Env, Verifier, Verdict
from ..harness.checkpoints import CheckpointStore
from ..harness.episode_log import EpisodeLogger
from ..harness.mutator import DetectionCounters, Mutator
from .llm_client import LLMClient, LLMSysError
from .memory import FailureRecord, LoopMemory


# ---------------------------------------------------------------------------
# Polityki agenta
# ---------------------------------------------------------------------------

class AgentPolicy:
    """Kontrakt polityki: act(obs, mem_view) -> akcja; reflect() -> lekcja."""

    def act(self, obs: dict, mem_view: str) -> dict:
        raise NotImplementedError

    def reflect(self, summary: str) -> str:
        return ""


class MockGridAgent(AgentPolicy):
    """Agent testowy bez GPU: BFS do celu z omijaniem pulapek znanych z Phi.

    Pokazuje, ze pamiec petli dziala: po kazdym FATAL pulapka trafia do Phi
    i kolejna petla ja omija. Do smoke testu i walidacji harnessu (F0).
    """

    def __init__(self, env: Env):
        self.env = env
        self.known_traps: set[tuple[int, int]] = set()

    def learn_from(self, memory: LoopMemory) -> None:
        for f in memory.failures:
            m = re.search(r"\[(\d+),\s*(\d+)\]", f.error)
            if m:
                self.known_traps.add((int(m.group(1)), int(m.group(2))))

    def act(self, obs: dict, mem_view: str) -> dict:
        size = obs["size"]
        start, goal = tuple(obs["pos"]), tuple(obs["goal"])
        seen, q = {start}, deque([(start, [])])
        while q:
            (r, c), path = q.popleft()
            if (r, c) == goal:
                return {"move": path[0]} if path else {"move": "down"}
            for name, (dr, dc) in {"down": (1, 0), "right": (0, 1),
                                   "up": (-1, 0), "left": (0, -1)}.items():
                nxt = (r + dr, c + dc)
                if (0 <= nxt[0] < size and 0 <= nxt[1] < size
                        and nxt not in seen and nxt not in self.known_traps):
                    seen.add(nxt)
                    q.append((nxt, path + [name]))
        return {"move": "down"}  # slepy zaulek -- wymus postep

    def reflect(self, summary: str) -> str:
        return f"ominac znane pulapki: {sorted(self.known_traps)}"


class LLMGridAgent(AgentPolicy):
    """Agent LLM: blok Phi w prompcie (DG 3.4), parsowanie ruchu z odpowiedzi."""

    def __init__(self, client: LLMClient):
        self.client = client

    def act(self, obs: dict, mem_view: str) -> dict:
        prompt = (
            f"{mem_view}\n\nJestes agentem w gridworld {obs['size']}x{obs['size']}."
            f" Pozycja: {obs['pos']}, cel: {obs['goal']}, krok: {obs['t']}."
            f" Podejrzane sasiedztwo: {obs['suspicious_adjacent']}."
            " Odpowiedz JEDNYM slowem: up / down / left / right.")
        text = self.client.chat([{"role": "user", "content": prompt}],
                                max_tokens=8)
        m = re.search(r"\b(up|down|left|right)\b", text.lower())
        return {"move": m.group(1)} if m else {"move": "down"}


# ---------------------------------------------------------------------------
# Kontroler petli
# ---------------------------------------------------------------------------

def strategy_class(traj: list) -> str:
    """Klasa strategii strat(sigma) dla gridworld: sekwencja regionow 2x2
    (DG 2.3 / dokument towarzyszacy 2.1 -- wersja uproszczona)."""
    regs = []
    for obs, _act, _r in traj:
        r, c = obs["pos"]
        reg = f"{r // 6}{c // 6}"
        if not regs or regs[-1] != reg:
            regs.append(reg)
    return "-".join(regs)


def run_episode(env: Env, verifier: Verifier, policy: AgentPolicy,
                logger: EpisodeLogger, cfg: dict) -> dict:
    """Uproszczony kontroler petli wg DG 8.1: restore -> loop -> settle ->
    detekcja -> mutacja. Phi poza snapshotem (I1), mutacje kumulowane (I2).
    Galaz SYS (DG 8.3c): petla uniewazniona (counted=false), nie rozliczana
    z k/tau/budzetu; po 3 kolejnych uniewaznieniach run = aborted_sys.
    tau_max sprawdzane co t_check krokow wewnatrz petli (DG 8.3d)."""
    store = CheckpointStore()
    C = store.create_base(env.make_checkpoint(seed=cfg["seed"]))
    memory = LoopMemory(max_hypotheses=cfg.get("max_hypotheses", 5))
    det = DetectionCounters()
    mutator = Mutator(env, cfg, verifier)  # verifier potrzebny do G1b (DG 6.3)
    k_max = cfg.get("k_max", 15)
    tau_max = cfg.get("tau_max", 3000)
    t_max = cfg.get("t_max", 200)
    t_check = cfg.get("t_check", 10)       # DG 8.3d
    cooldown = cfg.get("mutation_cooldown", 2)
    cost = 0.0
    alpha, beta = cfg.get("alpha", 0.01), cfg.get("beta", 0.002)
    # kara P_end z konfiguracji (DG 7.1): base + step * k, taniej za TIMEOUT
    p_fatal_base = cfg.get("p_fatal_base", 1.0)
    p_fatal_step = cfg.get("p_fatal_step", 0.25)
    p_end_timeout_factor = cfg.get("p_end_timeout_factor", 0.8)
    sys_streak = 0  # kolejne petle uniewaznione SYS (DG 8.3c)

    while memory.k < k_max and memory.tau < tau_max:
        i = memory.k  # 0-indeksowany numer petli (DG 8.1)
        logger.loop_start(k=i, hash_checkpoint=C.hash)
        obs = env.reset_from_checkpoint(C)        # restore; Phi nietkniete
        if isinstance(policy, MockGridAgent):
            policy.learn_from(memory)
        mem_view = memory.render(k_max=k_max, cost=cost)

        traj: list = []
        verdict = Verdict(status="TIMEOUT", cls="TIMEOUT", t_end=t_max)
        sys_error: LLMSysError | None = None
        for t in range(1, t_max + 1):
            try:
                action = policy.act(obs, mem_view)
            except LLMSysError as exc:
                sys_error = exc  # blad systemowy po wyczerpaniu retry
                break
            obs, r, _info = env.step(action)
            traj.append((obs, action, r))
            # DG 8.3d: kontrola tau_max WEWNATRZ petli, co t_check krokow
            if t % t_check == 0 and memory.tau + t >= tau_max:
                verdict = Verdict(
                    status="TIMEOUT", cls="TIMEOUT",
                    detail="budzet tau_max wyczerpany w trakcie petli",
                    t_end=t, cause="tau_max")
                break
            v = verifier.check(env.state, traj, t, {"t_max": t_max})
            if v.status != "CONTINUE":
                verdict = v
                break

        # -- galaz SYS: PRZED rozliczeniem kosztu i tau (DG 8.3c) -----------
        if sys_error is not None:
            logger.sys_event(code=sys_error.code,
                             retries=sys_error.retries, k=i)
            logger.loop_end(k=i, tau=memory.tau, dt=len(traj), cls="SYS",
                            t_end=len(traj), counted=False, cost_cum=cost,
                            mutation=None, hash_checkpoint=C.hash,
                            det=det.snapshot(), phi_hash=memory.hash)
            sys_streak += 1
            if sys_streak >= 3:
                logger.episode_end(success=False, loops=memory.k, cost=cost,
                                   censored=True, reason="aborted_sys")
                return {"success": False, "status": "aborted_sys",
                        "loops": memory.k, "cost": round(cost, 4),
                        "censored": True}
            continue  # petla uniewazniona: bez k+=1, bez tau, bez kosztu
        sys_streak = 0

        dt = verdict.t_end or len(traj)
        memory.tau += dt
        cost += alpha * dt + beta * i * dt

        if verdict.status == "OK":
            logger.loop_end(k=i, tau=memory.tau, dt=dt, cls="OK",
                            t_end=dt, counted=True, cost_cum=cost,
                            mutation=None, hash_checkpoint=C.hash,
                            det=det.snapshot(), phi_hash=memory.hash)
            logger.episode_end(success=True, loops=i + 1, cost=cost,
                               censored=False)
            return {"success": True, "loops": i + 1, "cost": round(cost, 4)}

        # rozliczenie porazki: P_end + zapis do Phi PRZED k+=1 (DG 8.1)
        p_fatal = p_fatal_base + p_fatal_step * i
        cost += (p_fatal if verdict.cls == "FATAL"
                 else p_end_timeout_factor * p_fatal)
        strat = strategy_class(traj)
        prog = verifier.progress(env.state)
        lesson = policy.reflect(verdict.detail)
        memory.append_failure(FailureRecord(
            loop=i, strategy=strat, error=verdict.detail,
            cls=verdict.cls or verdict.status, lesson=lesson))
        memory.k += 1

        # detekcja lock-in i mutacja (R0-R4, kumulacja na biezacym C)
        det.update(strat, verdict, prog, t_max)
        C_new, m_lvl, rec = mutator.decide(
            C, memory.k, det,
            loop_seed=cfg["seed"] + memory.k,  # k PO inkrementacji (DG 8.3a)
            last_actions=[a for (_o, a, _r) in traj])  # replay G1b
        if m_lvl > 0:
            store.commit_mutation(C_new, rec)
            # cooldown tylko po mutacji poziomu >= 2 (DG 6.2)
            det.reset_after_mutation(cooldown if m_lvl >= 2 else 0)
            C = C_new

        logger.loop_end(k=i, tau=memory.tau, dt=dt,
                        cls=verdict.cls or verdict.status, t_end=dt,
                        counted=True, cost_cum=cost, mutation=rec,
                        hash_checkpoint=C.hash, det=det.snapshot(),
                        phi_hash=memory.hash)

    logger.episode_end(success=False, loops=memory.k, cost=cost,
                       censored=True, reason="budget_exhausted")
    return {"success": False, "loops": memory.k, "cost": round(cost, 4),
            "censored": True}
