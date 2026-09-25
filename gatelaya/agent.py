"""Laya agent protocol and lazy-loading script-routing agent."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Literal, Protocol, runtime_checkable

from .errors import LayaNotInstalledError

ENGLISH_CHECKPOINT = "convaiinnovations/laya"
MULTILINGUAL_CHECKPOINT = "convaiinnovations/laya-multilingual"

Bucket = Literal["english", "multilingual"]


def detect_bucket(text: str) -> Bucket:
    """Pick the checkpoint bucket: ASCII-only text routes to english, else multilingual."""
    if text and all(ord(ch) < 128 for ch in text):
        return "english"
    return "multilingual"


@runtime_checkable
class LayaAgent(Protocol):
    """Minimal prediction interface consumed by GateLayaGuardrail."""

    def predict(self, state: dict, questions: dict) -> dict:
        """Run one forward pass; returns Laya-shaped {"answers": {...}}."""
        ...


class LayaRouterAgent:
    """Loads english/multilingual Laya checkpoints lazily and routes by script.

    ``model_path`` overrides routing: one checkpoint (e.g. a fine-tune) serves
    both buckets.
    """

    def __init__(
        self,
        english_checkpoint: str = ENGLISH_CHECKPOINT,
        multilingual_checkpoint: str = MULTILINGUAL_CHECKPOINT,
        model_path: str | None = None,
    ) -> None:
        self.english_checkpoint = english_checkpoint
        self.multilingual_checkpoint = multilingual_checkpoint
        self.model_path = str(model_path) if model_path else None
        self._agents: dict[str, Any] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _laya() -> Any:
        try:
            import laya
        except ImportError as exc:
            raise LayaNotInstalledError() from exc
        return laya

    def route(self, state: dict) -> Bucket:
        """Choose the checkpoint bucket from state["text"]."""
        return detect_bucket(str(state.get("text") or ""))

    def _get_agent(self, bucket: Bucket) -> Any:
        with self._lock:
            key = "model_path" if self.model_path else bucket
            cached = self._agents.get(key)
            if cached is not None:
                return cached
            checkpoint = self.model_path or (
                self.english_checkpoint if bucket == "english" else self.multilingual_checkpoint
            )
            agent = self._laya().load(checkpoint)
            self._agents[key] = agent
            return agent

    def predict(self, state: dict, questions: dict) -> dict:
        """Run one forward pass on the routed checkpoint."""
        agent = self._get_agent(self.route(state))
        result = agent.predict(state, questions)
        if not isinstance(result, dict):
            raise TypeError(f"Laya returned {type(result).__name__}, expected dict")
        if "answers" in result:
            return result
        return {"answers": result}

    async def apredict(self, state: dict, questions: dict) -> dict:
        """Thread-offloaded predict so the event loop never blocks on inference."""
        return await asyncio.to_thread(self.predict, state, questions)
