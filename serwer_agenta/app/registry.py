"""Rejestr eksperymentow: wczytanie YAML + scalanie konfiguracji scenariusza.

Zasada wielokrotnego uzytku: nowy eksperyment = nowy wpis w registry.yaml
(+ ewentualnie nowy plik scenariusza w configs/). Kod uslug bez zmian.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = ROOT / "experiments" / "registry.yaml"
CONFIGS_DIR = ROOT / "configs"


def load_registry(path: Path | None = None) -> dict[str, dict]:
    data = yaml.safe_load((path or REGISTRY_PATH).read_text(encoding="utf-8"))
    return data.get("experiments", {})


def get_experiment(name: str) -> dict[str, Any]:
    reg = load_registry()
    if name not in reg:
        raise KeyError(f"nieznany eksperyment: {name}; dostepne: {sorted(reg)}")
    exp = dict(reg[name])
    # dolacz konfiguracje scenariusza z configs/ (progi [START], budzety)
    scenario_path = CONFIGS_DIR / f"{exp['scenario']}.yaml"
    if scenario_path.exists():
        exp["scenario_config"] = yaml.safe_load(
            scenario_path.read_text(encoding="utf-8"))
    return exp


def build_run_config(exp: dict[str, Any], overrides: dict | None = None) -> dict:
    """Plaska konfiguracja runu: scenariusz <- eksperyment <- overrides."""
    cfg: dict[str, Any] = {}
    cfg.update(exp.get("scenario_config") or {})
    for key in ("seed", "k_max", "tau_max", "t_max", "agent_type", "group"):
        if key in exp:
            cfg[key] = exp[key]
    cfg.setdefault("seed", 42)
    cfg.setdefault("agent_type", "mock")
    if overrides:
        cfg.update(overrides)
    return cfg
