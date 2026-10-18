"""Checkpoint store: trzymanie wersji checkpointow per run (DG 4.1-4.2).

C0 jest niemutowalny i przechowywany przez caly epizod (audyt + G2).
Kazda wersja C_k ma osobny obiekt i hash; mutacje nigdy nie dzialaja
w miejscu (inwariant I2).
"""
from __future__ import annotations

from .base import Checkpoint


class CheckpointStore:
    """Prosty magazyn w pamieci procesu; w wariancie produkcyjnym mozna
    podmienic backend na dysk (pickle) bez zmiany kontraktu."""

    def __init__(self) -> None:
        self._base: Checkpoint | None = None       # C0 -- niemutowalny
        self._current: Checkpoint | None = None    # C_k -- operand mutacji
        self._chain: list[dict] = []               # lancuch mutacji L_mut

    def create_base(self, cp: Checkpoint) -> Checkpoint:
        self._base = cp
        self._current = cp
        self._chain = []
        return cp

    @property
    def base(self) -> Checkpoint:
        assert self._base is not None, "brak checkpointu bazowego"
        return self._base

    @property
    def current(self) -> Checkpoint:
        assert self._current is not None, "brak biezacego checkpointu"
        return self._current

    def commit_mutation(self, cp_new: Checkpoint, record: dict) -> None:
        """Kumulacja: C_{k+1} zastepuje biezacy, rekord do L_mut (DG 2.2)."""
        assert cp_new is not self._current, "I2: mutacja musi tworzyc nowy obiekt"
        self._current = cp_new
        self._chain.append(record)

    @property
    def mutation_chain(self) -> list[dict]:
        return list(self._chain)
