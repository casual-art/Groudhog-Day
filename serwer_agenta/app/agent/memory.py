"""Pamiec petli Phi_k (DG 2.2, 3.2) -- nalezy do agenta, nie do srodowiska.

Przechowywana POZA snapshotem srodowiska (inwariant I1): restore(C) nigdy
jej nie dotyka. Zapis tylko po zakonczeniu petli lub przez akcje
notuj_hipoteze (inwariant I3).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field


@dataclass
class FailureRecord:
    """Rekord nieudanej proby h_i = (sigma, e, cls) wg DG 2.2."""
    loop: int
    strategy: str          # klasa strategii strat(sigma)
    error: str             # opis zdarzenia bledu
    cls: str               # klasa bledu (DG 5.3)
    lesson: str = ""       # wypelnia agent (pole "lekcja", DG 3.2)


class LoopMemory:
    """Meta-stan Phi_k = (k, tau, H_k, Q_k)."""

    def __init__(self, max_hypotheses: int = 5):
        self.k = 0                       # licznik petli
        self.tau = 0                     # skumulowany czas logiczny
        self.failures: list[FailureRecord] = []
        self.hypotheses: list[str] = []  # Q_k, verbatim, max N ostatnich
        self.max_hypotheses = max_hypotheses

    def append_failure(self, rec: FailureRecord) -> None:
        self.failures.append(rec)

    def note_hypothesis(self, q: str) -> None:
        """Akcja specjalna agenta notuj_hipoteze (DG 8.3b)."""
        self.hypotheses.append(q)
        self.hypotheses = self.hypotheses[-self.max_hypotheses:]

    @property
    def hash(self) -> str:
        blob = json.dumps(
            {"k": self.k, "tau": self.tau,
             "h": [f.__dict__ for f in self.failures], "q": self.hypotheses},
            sort_keys=True, default=str).encode()
        return hashlib.sha256(blob).hexdigest()[:12]

    def render(self, k_max: int, cost: float = 0.0) -> str:
        """Blok kontekstu dla LLM-agenta (DG 3.4)."""
        lines = [
            "[META-STAN PETLI]",
            f"Petla: {self.k} / budzet: {k_max}. Laczny czas: {self.tau} krokow."
            f" Laczny koszt: {cost:.2f}.",
            "Historia prob:",
        ]
        for f in self.failures:
            lines.append(
                f"  PETLA {f.loop} | strategia: {f.strategy} | blad: {f.cls}"
                f" ({f.error}) | lekcja: {f.lesson or '-'}")
        lines.append(f"Twoje hipotezy: {json.dumps(self.hypotheses, ensure_ascii=False)}")
        # stala linia ostrzegawcza (DG 3.4, G3): mutacje mozliwe, nieujawnione
        lines.append("UWAGA: warunki srodowiska mogly sie minimalnie zmienic. "
                     "Zweryfikuj zalozenia observacja, zanim zaufasz pamieci.")
        return "\n".join(lines)
