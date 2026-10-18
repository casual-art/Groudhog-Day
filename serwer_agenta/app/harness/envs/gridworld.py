"""Scenariusz S1: Gridworld z ukryta pulapka (DG 9.1).

Siatka N x N, start w rogu (0,0), cel w rogu (N-1,N-1), kilka ukrytych
pulapek (seedowanych). Wejscie na pulapke = FATAL (DG 5.3). Pola sasiednie
pulapki daja slaby sygnal "podejrzane". Solver referencyjny G1: BFS.
"""
from __future__ import annotations

import copy
import random
from collections import deque

from ..base import Checkpoint, Env, Verdict, Verifier

ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}


class GridWorldEnv(Env):
    """Deterministyczny gridworld; cala losowosc przez seed."""

    def __init__(self, size: int = 12, n_traps: int = 4, t_max: int = 200,
                 suspicious_prob: float = 0.7):
        self.size = size
        self.n_traps = n_traps
        self.t_max = t_max
        self.suspicious_prob = suspicious_prob
        self._state: dict = {}

    # -- generacja instancji ------------------------------------------------
    def _gen_state(self, seed: int) -> dict:
        rng = random.Random(seed)
        start, goal = (0, 0), (self.size - 1, self.size - 1)
        traps: set[tuple[int, int]] = set()
        while len(traps) < self.n_traps:
            p = (rng.randrange(self.size), rng.randrange(self.size))
            if p not in (start, goal):
                traps.add(p)
        state = {
            "pos": list(start), "goal": list(goal), "start": list(start),
            "traps": sorted([list(t) for t in traps]),
            "walls": [], "t": 0, "done": False,
            "seed": seed,  # ziarno instancji (deterministyczny szum obs.)
            "decor": rng.randrange(1000),  # cecha kosmetyczna (mutacja poz. 1)
        }
        return state

    # -- kontrakt Env (DG 8.3b) ----------------------------------------------
    def make_checkpoint(self, seed: int) -> Checkpoint:
        return Checkpoint(state=self._gen_state(seed), seed=seed)

    def restore(self, cp: Checkpoint) -> None:
        self._state = copy.deepcopy(cp.state)  # I1: bez dostepu do Phi

    def reset_from_checkpoint(self, cp: Checkpoint) -> dict:
        self.restore(cp)
        return self._obs()

    def step(self, action: dict) -> tuple[dict, float, dict]:
        s = self._state
        if s["done"]:
            return self._obs(), 0.0, {"note": "petla zakonczona"}
        move = action.get("move")
        if move not in ACTIONS:
            # akcja spoza formatu: odrzucona bez zmiany stanu (DG 8.3b)
            return self._obs(), 0.0, {"rejected": True}
        dr, dc = ACTIONS[move]
        r, c = s["pos"]
        nr, nc = r + dr, c + dc
        in_bounds = 0 <= nr < self.size and 0 <= nc < self.size
        is_wall = [nr, nc] in s["walls"]
        if in_bounds and not is_wall:
            s["pos"] = [nr, nc]
        s["t"] += 1
        return self._obs(), -0.01, {}

    @property
    def state(self) -> dict:
        return self._state

    def mutate(self, cp: Checkpoint, op: str, args: dict, seed: int) -> Checkpoint:
        """Mutacje S1 (DG 9.1). Zwraca NOWY checkpoint (I2, kumulacja 2.2)."""
        rng = random.Random(seed)
        st = copy.deepcopy(cp.state)
        if op == "recolor":            # poziom 1: kosmetyka
            st["decor"] = rng.randrange(1000)
        elif op == "move_trap":        # poziom 2: przesuniecie pulapki o 1 pole
            traps = [tuple(t) for t in st["traps"]]
            idx = args.get("trap_index", rng.randrange(len(traps)))
            r, c = traps[idx]
            for dr, dc in rng.sample(list(ACTIONS.values()), 4):
                nr, nc = r + dr, c + dc
                cand = (nr, nc)
                if (0 <= nr < self.size and 0 <= nc < self.size
                        and cand not in traps and cand not in
                        (tuple(st["start"]), tuple(st["goal"]))):
                    traps[idx] = cand
                    break
            st["traps"] = sorted([list(t) for t in traps])
        elif op == "add_wall":         # poziom 3: nowa sciana (topologia)
            for _ in range(50):
                w = [rng.randrange(self.size), rng.randrange(self.size)]
                if (w not in st["traps"] and w not in st["walls"]
                        and w not in (st["start"], st["goal"])):
                    st["walls"].append(w)
                    break
        else:
            raise ValueError(f"nieznana operacja mutacji: {op}")
        return Checkpoint(state=st, seed=seed, parent_hash=cp.hash)

    def solvable(self, cp: Checkpoint) -> bool:
        """G1: BFS z pelna wiedza musi znalezc sciezke start -> cel."""
        st = cp.state
        blocked = {tuple(t) for t in st["traps"]} | {tuple(w) for w in st["walls"]}
        start, goal = tuple(st["start"]), tuple(st["goal"])
        seen, q = {start}, deque([start])
        while q:
            r, c = q.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in ACTIONS.values():
                nxt = (r + dr, c + dc)
                if (0 <= nxt[0] < self.size and 0 <= nxt[1] < self.size
                        and nxt not in blocked and nxt not in seen):
                    seen.add(nxt)
                    q.append(nxt)
        return False

    # -- obserwacja ----------------------------------------------------------
    def _obs(self) -> dict:
        s = self._state
        r, c = s["pos"]
        adjacent = any(
            abs(tr - r) + abs(tc - c) == 1 for tr, tc in s["traps"]
        )
        # slaby, ZASZUMIONY sygnal (DG 9.1): suspicious_prob steruje
        # informatywnoscia; szum deterministyczny per (seed, t, pozycja),
        # by zachowac powtarzalnosc trajektorii (DG 4.1)
        rng = random.Random(f"{s.get('seed', 0)}:{s['t']}:{r}:{c}")
        suspicious = adjacent and rng.random() < self.suspicious_prob
        return {
            "pos": list(s["pos"]), "goal": list(s["goal"]), "t": s["t"],
            "size": self.size,
            "suspicious_adjacent": suspicious,
            "decor": s["decor"],
        }


class GridWorldVerifier(Verifier):
    """Weryfikator S1: pulapka = FATAL, cel = OK, budzet = TIMEOUT."""

    def check(self, state: dict, traj: list, t: int, cfg: dict) -> Verdict:
        t_max = cfg.get("t_max", 200)
        if list(state["pos"]) in [list(x) for x in state["traps"]]:
            state["done"] = True
            return Verdict(status="FATAL", cls="FATAL",
                           detail=f"wejscie na pulapke w polu {state['pos']}",
                           t_end=t)
        if state["pos"] == state["goal"]:
            state["done"] = True
            return Verdict(status="OK", cls=None, detail="cel osiagniety", t_end=t)
        if t >= t_max:
            state["done"] = True
            return Verdict(status="TIMEOUT", cls="TIMEOUT",
                           detail="budzet krokow petli wyczerpany", t_end=t)
        return Verdict(status="CONTINUE", t_end=t)

    def progress(self, state: dict) -> float:
        """Znormalizowana bliskosc celu: 1 na starcie ... 0 na celu."""
        r, c = state["pos"]
        gr, gc = state["goal"]
        d = abs(gr - r) + abs(gc - c)
        d_max = abs(gr - state["start"][0]) + abs(gc - state["start"][1]) or 1
        return 1.0 - d / d_max
