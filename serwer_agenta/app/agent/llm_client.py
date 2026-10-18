"""Klienci LLM dla runtime agenta.

Wymiennosc warstwy modelu: jeden kontrakt chat(), konfigurowalny base_url
-> lokalny vLLM (endpoint OpenAI-compatible), Groq, Cerebras albo stub
lokalny do testow bez GPU (smoke test / CI).
"""
from __future__ import annotations

import os
import random
import time
from typing import Any

import httpx

# Procedura SYS-retry wg DG 8.3c: backoff wykladniczy 1s -> 2s -> 4s -> ...,
# maks. 5 prob, jitter; dotyczy bledow transportowych i odpowiedzi 5xx.
SYS_MAX_ATTEMPTS = 5
SYS_BACKOFF_BASE_S = 1.0
SYS_JITTER_S = 0.5


class LLMSysError(RuntimeError):
    """Blad systemowy wywolania LLM (klasa SYS, DG 5.3) po wyczerpaniu
    procedury retry z DG 8.3c. Petla z tym bledem jest uniewazniana
    (counted=false), nie rozliczana z k/tau/budzetu."""

    def __init__(self, code: str, retries: int, detail: str = ""):
        super().__init__(f"SYS {code} po {retries} probach: {detail}")
        self.code = code
        self.retries = retries


def _is_retryable(exc: BaseException) -> bool:
    """Bledy transportowe / timeout / HTTP 5xx = SYS (retry); 4xx to blad
    zadania (nie retry, propagujemy jako zwykly wyjatek)."""
    if isinstance(exc, httpx.TimeoutException | httpx.TransportError):
        return True
    return (isinstance(exc, httpx.HTTPStatusError)
            and exc.response.status_code >= 500)


class LLMClient:
    """Kontrakt: chat(messages) -> tekst odpowiedzi."""

    def chat(self, messages: list[dict], **kw: Any) -> str:
        raise NotImplementedError


class MockLLMClient(LLMClient):
    """Stub lokalny: nie wymaga GPU ani sieci. Uzywany w smoke tescie
    i jako fallback, gdy nie skonfigurowano zadnego backendu."""

    def chat(self, messages: list[dict], **kw: Any) -> str:
        return "mock: brak skonfigurowanego modelu (ustaw LLM_BASE_URL)"


class OpenAICompatibleClient(LLMClient):
    """Klient OpenAI-compatible przez httpx (vLLM / Groq / Cerebras).

    Powtarzalnosc wg DG 4.1: temperatura 0, przypieta wersja modelu,
    seed inference jesli API wspiera.
    """

    def __init__(self, base_url: str, model: str, api_key: str = "none",
                 timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def chat(self, messages: list[dict], **kw: Any) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.0,          # DG 4.1(a)
            "max_tokens": kw.get("max_tokens", 512),
        }
        if kw.get("seed") is not None:
            payload["seed"] = kw["seed"]  # DG 4.1(c), jesli wspierane
        headers = {"Authorization": f"Bearer {self.api_key}"}
        url = f"{self.base_url}/v1/chat/completions"
        last_exc: BaseException | None = None
        for attempt in range(1, SYS_MAX_ATTEMPTS + 1):
            try:
                with httpx.Client(timeout=self.timeout) as client:
                    resp = client.post(url, json=payload, headers=headers)
                    resp.raise_for_status()
                    data = resp.json()
                # odpowiedz czesciowa (brak choices) odrzucana w calosci
                return data["choices"][0]["message"]["content"]
            except Exception as exc:
                if not _is_retryable(exc):
                    raise  # 4xx i bledy parsowania nie sa SYS-retry
                last_exc = exc
                if attempt < SYS_MAX_ATTEMPTS:
                    delay = SYS_BACKOFF_BASE_S * (2 ** (attempt - 1))
                    delay += random.uniform(0.0, SYS_JITTER_S)  # jitter
                    time.sleep(delay)
        code = (f"http_{last_exc.response.status_code}"
                if isinstance(last_exc, httpx.HTTPStatusError)
                else type(last_exc).__name__)
        raise LLMSysError(code=code, retries=SYS_MAX_ATTEMPTS,
                          detail=str(last_exc))


def build_client(model_cfg: dict) -> LLMClient:
    """Fabryka klienta z konfiguracji modelu eksperymentu.

    Priorytet: jawny base_url w YAML > zmienne srodowiskowe > stub.
    """
    base_url = (model_cfg.get("base_url")
                or os.environ.get("LLM_BASE_URL", ""))
    if not base_url:
        return MockLLMClient()
    # api_key_env: nazwa zmiennej srodowiskowej z kluczem (w YAML NIGDY
    # wartosc klucza -- patrz dok. wdrozeniowy, sekcja 5 "Auth")
    key_env = model_cfg.get("api_key_env") or "LLM_API_KEY"
    api_key = model_cfg.get("api_key") or os.environ.get(key_env, "none")
    return OpenAICompatibleClient(
        base_url=base_url,
        model=model_cfg.get("model", "local-model"),
        api_key=api_key,
    )
