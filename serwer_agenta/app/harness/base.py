"""Bazowe kontrakty harnessu -- implementacja DG v1.2 sekcja 8.3b.

LoopEnv / Verifier jako klasy bazowe (ABC). Nowy scenariusz = podklasa Env
+ weryfikator + wpis w rejestrze eksperymentow. Uslugi (API, runtime agenta)
pozostaja nietkniete.
"""
from __future__ import annotations

import copy
import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Verdict:
    """Wynik weryfikatora V (DG 5.1): status + klasa bledu + opis."""
    status: str = "CONTINUE"          # CONTINUE / OK / FATAL / TIMEOUT / STAGNATION / SYS
    cls: str | None = None            # klasa bledu wg DG 5.3 (FATAL, RECOV-*, ...)
    detail: str = ""
    t_end: int = 0                    # krok zakonczenia petli
    cause: str = ""                   # np. "tau_max" przy TIMEOUT z DG 8.3d


@dataclass
class Checkpoint:
    """Niemutowalny snapshot stanu srodowiska (DG 2.2, 4.1).

    Kazda mutacja tworzy NOWY checkpoint (inwariant I2, DG 4.2) --
    mutowanie "w miejscu" jest zabronione.
    """
    state: dict                       # gleboka kopia stanu symulacji
    seed: int                         # ziarno RNG snapshotu
    t: int = 0                        # znacznik czasu logicznego
    parent_hash: str | None = None    # hash poprzedniej wersji (lancuch mutacji)

    @property
    def hash(self) -> str:
        payload = json.dumps(self.state, sort_keys=True, default=str).encode()
        return hashlib.sha256(payload).hexdigest()[:12]


class Env(ABC):
    """Kontrakt srodowiska wg DG 8.3b.

    Determinizm: dla ustalonego checkpointu i sekwencji akcji trajektoria
    musi byc identyczna (DG 4.1). Zakaz zegara sciennego i RNG bez seeda.
    """

    @abstractmethod
    def make_checkpoint(self, seed: int) -> Checkpoint:
        """Nowy, niemutowalny snapshot C0 (DG 8.3b)."""

    @abstractmethod
    def restore(self, cp: Checkpoint) -> None:
        """Przywraca stan z checkpointu. NIE dotyka pamieci Phi (I1)."""

    @abstractmethod
    def reset_from_checkpoint(self, cp: Checkpoint) -> dict:
        """restore + obserwacja poczatkowa."""

    @abstractmethod
    def step(self, action: dict) -> tuple[dict, float, dict]:
        """Krok dynamiki: (obserwacja, nagroda krokowa, info)."""

    @property
    @abstractmethod
    def state(self) -> dict:
        """Stan wewnetrzny -- tylko dla weryfikatora i kontrolera."""

    @abstractmethod
    def mutate(self, cp: Checkpoint, op: str, args: dict, seed: int) -> Checkpoint:
        """Zwraca NOWA wersje checkpointu z nalozona mutacja (I2)."""

    @abstractmethod
    def solvable(self, cp: Checkpoint) -> bool:
        """G1 (DG 6.3): solver referencyjny potwierdza rozwiazywalnosc."""

    def snapshot_state(self) -> dict:
        """Gleboka kopia biezacego stanu (do checkpoint store)."""
        return copy.deepcopy(self.state)


class Verifier(ABC):
    """Kontrakt weryfikatora wg DG 8.3b / 5.1."""

    @abstractmethod
    def check(self, state: dict, traj: list, t: int, cfg: dict) -> Verdict:
        """Sprawdza ograniczenia twarde, cel, budzet; zwraca werdykt."""

    @abstractmethod
    def progress(self, state: dict) -> float:
        """prog(s) w [0,1] -- do detekcji stagnacji (DG 5.1) i metryki M3."""
