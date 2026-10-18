"""Kontroler mutacji Psi (DG 6): liczniki detekcji + reguly R0-R4.

Uproszczona, ale wierna semantycznie implementacja:
- liczniki det_k aktualizowane po kazdej zakonczonej petli (DG 2.2),
- reguly ewaluowane scisle w kolejnosci R0->R4, pierwsza spelniona wygrywa,
- cooldown blokuje R2/R3, nie blokuje R0/R1 (DG 6.2),
- kazda mutacja kumuluje sie na biezacym checkpoincie i wymaga G1 (solver),
- mutacje poziomu >= 2 dodatkowo wymagaja G1b: replay powtarzanej strategii
  na zmutowanym checkpoincie musi sie NIE powiesc (DG 6.3).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .base import Checkpoint, Env, Verdict, Verifier


@dataclass
class DetectionCounters:
    """Licznik detekcji det_k (DG 2.2) -- stan kontrolera, nie agenta.

    Nazwy pol zgodne ze schematem DG 8.3a (cooldown_left)."""
    same_strat_streak: int = 0
    same_error_streak: int = 0
    no_progress_loops: int = 0
    cooldown_left: int = 0
    mutations_so_far: int = 0
    fast_fatal: int = 0          # FATAL w pierwszych 10% budzetu (DG 7.3)
    counted_loops: int = 0
    best_progress: float = -1.0
    _last_strat: str | None = None
    _last_cls: str | None = None

    @property
    def fast_fatal_share(self) -> float:
        return self.fast_fatal / self.counted_loops if self.counted_loops else 0.0

    def snapshot(self) -> dict:
        d = asdict(self)
        d.pop("_last_strat", None)
        d.pop("_last_cls", None)
        d.pop("best_progress", None)
        d["fast_fatal_share"] = round(self.fast_fatal_share, 4)
        return d

    def update(self, strat: str, verdict: Verdict, prog: float, t_max: int) -> None:
        """Aktualizacja po zakonczonej, liczonej petli."""
        self.counted_loops += 1
        self.same_strat_streak = (
            self.same_strat_streak + 1 if strat == self._last_strat else 0
        )
        self._last_strat = strat
        cls = verdict.cls or verdict.status
        self.same_error_streak = (
            self.same_error_streak + 1 if cls == self._last_cls else 0
        )
        self._last_cls = cls
        if cls == "FATAL" and verdict.t_end < 0.1 * t_max:
            self.fast_fatal += 1
        if prog > self.best_progress + 0.01:  # epsilon jak w M8
            self.best_progress = prog
            self.no_progress_loops = 0
        else:
            self.no_progress_loops += 1
        if self.cooldown_left > 0:
            self.cooldown_left -= 1

    def reset_after_mutation(self, cooldown: int) -> None:
        self.mutations_so_far += 1
        self.cooldown_left = cooldown
        self.same_strat_streak = 0
        self.same_error_streak = 0
        self.no_progress_loops = 0
        self.best_progress = -1.0


class Mutator:
    """Polityka Psi: reguly R0-R4 (DG 6.2) + gwarancje G1 i G1b (DG 6.3)."""

    # poziom -> (operacja gridworld, opis)
    OPS = {1: ("recolor", "mutacja kosmetyczna"),
           2: ("move_trap", "przesuniecie pulapki o 1 pole"),
           3: ("add_wall", "nowa sciana zmieniajaca topologie")}

    def __init__(self, env: Env, cfg: dict, verifier: Verifier | None = None):
        self.env = env
        self.cfg = cfg  # progi [START] z konfiguracji eksperymentu
        self.verifier = verifier  # wymagany do testu replay G1b

    def _replay_fails(self, C_new: Checkpoint, last_actions: list[dict]) -> bool:
        """G1b (DG 6.3): replay powtarzanej strategii (sekwencji akcji
        ostatniej petli) na zmutowanym checkpoincie musi zakonczyc sie
        porazka (FATAL/TIMEOUT/STAGNATION). Sukces replay = mutacja
        bezcelowa (nie uniewaznia lock-in) -> odrzut.

        Uwaga: replay rusza wewnetrzny stan env; kontroler i tak rozpoczyna
        kolejna petle od reset_from_checkpoint, wiec jest to bezpieczne."""
        if not last_actions or self.verifier is None:
            return True  # brak trajektorii/weryfikatora -- test pomijalny
        t_max = self.cfg.get("t_max", 200)
        self.env.reset_from_checkpoint(C_new)
        status = "TIMEOUT"
        for t, act in enumerate(last_actions, start=1):
            self.env.step(act)
            v = self.verifier.check(self.env.state, [], t, {"t_max": t_max})
            if v.status != "CONTINUE":
                status = v.status
                break
        return status != "OK"

    def decide(self, C: Checkpoint, k: int, det: DetectionCounters,
               loop_seed: int,
               last_actions: list[dict] | None = None
               ) -> tuple[Checkpoint, int, dict | None]:
        """Ewaluacja R0->R4; zwraca (nowy_checkpoint, m_lvl, rekord|None)."""
        p = self.cfg.get("thresholds", {})
        lvl = 0
        # R0: straznik sondowania smiercia (ignoruje cooldown)
        if det.fast_fatal_share > p.get("r0_fast_fatal_share", 0.30):
            lvl = 2
        # R1: lock-in strukturalny
        elif det.same_strat_streak >= p.get("r1_same_strat", 5):
            lvl = 3
        # R2: lock-in lokalny (blokowany cooldownem)
        elif (det.same_strat_streak >= p.get("r2_same_strat", 3)
              and det.same_error_streak >= p.get("r2_same_error", 3)
              and det.cooldown_left == 0):
            lvl = 2
        # R3: brak postepu -- tania interwencja (blokowana cooldownem)
        elif (det.no_progress_loops >= p.get("r3_no_progress", 2)
              and k >= 2 and det.cooldown_left == 0):
            lvl = 1
        # R4 (eskalacja o 1) celowo uproszczona: patrz dokument wdrozeniowy.

        if lvl == 0:
            return C, 0, None

        op, desc = self.OPS[lvl]
        for attempt in range(10):  # G1/G1b: ponowne losowanie przy odrzucie
            seed = loop_seed * 1000 + attempt
            C_new = self.env.mutate(C, op, {"trap_index": 0}, seed)
            if not self.env.solvable(C_new):      # G1
                continue
            # G1b dotyczy wylacznie poziomow >= 2 (DG 6.3)
            if lvl >= 2 and not self._replay_fails(C_new, last_actions or []):
                continue
            rec = {"m_lvl": lvl, "seed": seed, "op": op, "opis": desc,
                   "hash_C_before": C.hash, "hash_C_after": C_new.hash}
            return C_new, lvl, rec
        return C, 0, None  # fallback do poziomu 0 (DG 6.3)
