"""Zapis logow epizodu JSONL wg schematu DG 8.3a.

Jeden plik na run: runs/<episode_id>/episodes.jsonl. Pierwszy rekord to
konfiguracja epizodu (seedy, wersje, hash konfiguracji); kolejne to
zdarzenia loop_start / loop_end / mutation / sys_event / episode_end.
Pole identyfikatora epizodu nosi nazwe `episode_id` (DG 8.3a); w tym
scaffoldzie jest tozsame z run_id warstwy API.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EpisodeLogger:
    """Strumieniowy zapis JSONL -- kazdy rekord flushowany natychmiast
    (przetrwanie naglego konca sesji na Kaggle/Colab)."""

    def __init__(self, run_dir: Path, episode_id: str, config: dict):
        self.episode_id = episode_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self._path = run_dir / "episodes.jsonl"
        self._fh = open(self._path, "a", encoding="utf-8")
        cfg_blob = json.dumps(config, sort_keys=True, default=str).encode()
        self.config_hash = hashlib.sha256(cfg_blob).hexdigest()[:12]
        # pierwszy rekord: konfiguracja epizodu (DG 8.3a)
        self._write({"event": "episode_config", "episode_id": episode_id,
                     "config": config, "config_hash": self.config_hash})

    def _write(self, rec: dict[str, Any]) -> None:
        rec.setdefault("ts", datetime.now(timezone.utc).isoformat())
        rec.setdefault("episode_id", self.episode_id)
        rec.setdefault("config_hash", self.config_hash)
        self._fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self._fh.flush()

    def loop_start(self, k: int, hash_checkpoint: str) -> None:
        self._write({"event": "loop_start", "k": k,
                     "hash_checkpoint": hash_checkpoint})

    def loop_end(self, k: int, tau: int, dt: int, cls: str, t_end: int,
                 counted: bool, cost_cum: float, mutation: dict | None,
                 hash_checkpoint: str, det: dict, phi_hash: str) -> None:
        """Obowiazkowe pola DG 8.3a: k, tau, cls, mutation, hash, det, counted."""
        self._write({"event": "loop_end", "k": k, "tau": tau, "dt": dt,
                     "cls": cls, "t_end": t_end, "counted": counted,
                     "cost_cum": round(cost_cum, 4),
                     "mutation": mutation, "hash_checkpoint": hash_checkpoint,
                     "det": det, "phi_hash": phi_hash})

    def sys_event(self, code: str, retries: int, k: int | None = None) -> None:
        rec = {"event": "sys_event", "code": code, "retries": retries,
               "counted": False}
        if k is not None:
            rec["k"] = k
        self._write(rec)

    def episode_end(self, success: bool, loops: int, cost: float,
                    censored: bool, reason: str = "") -> None:
        self._write({"event": "episode_end", "success": success,
                     "loops": loops, "cost_cum": round(cost, 4),
                     "censored": censored, "reason": reason})

    def close(self) -> None:
        self._fh.close()
