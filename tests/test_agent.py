"""Tests for routing heuristics and the lazy-loading LayaRouterAgent."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from gatelaya.agent import LayaAgent, LayaRouterAgent, detect_bucket
from gatelaya.errors import LayaNotInstalledError

from .conftest import FakeAgent

# --------------------------------------------------------- detect_bucket


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("hello world 123", "english"),
        ("Simple ASCII question?", "english"),
        ("नमस्ते दुनिया", "multilingual"),  # Devanagari
        ("你好世界", "multilingual"),  # CJK
        ("مرحبا بالعالم", "multilingual"),  # Arabic
        ("Привет мир", "multilingual"),  # Cyrillic
        ("emoji \U0001F600 text", "multilingual"),  # non-ASCII emoji
        ("", "multilingual"),  # empty input: no ASCII-only proof possible
    ],
)
def test_detect_bucket_routes_by_script(text: str, expected: str) -> None:
    """Arrange/act: text in various scripts. Assert: checkpoint bucket picked."""
    assert detect_bucket(text) == expected


def test_detect_bucket_ascii_with_mixed_digits_and_punctuation() -> None:
    """Arrange/act: pure ASCII with digits/punctuation. Assert: english bucket."""
    assert detect_bucket("What's 2+2? (42)") == "english"


# ------------------------------------------------------- LayaRouterAgent


def test_router_routes_from_state_text() -> None:
    """Arrange: router agent. Act: route() with ASCII and Devanagari text.
    Assert: english vs multilingual buckets."""
    agent = LayaRouterAgent()
    assert agent.route({"text": "hello"}) == "english"
    assert agent.route({"text": "नमस्ते"}) == "multilingual"
    assert agent.route({}) == "multilingual"  # missing text -> "" -> multilingual


def test_router_predict_raises_friendly_laya_not_installed_error() -> None:
    """Arrange: laya is NOT installed in the test venv. Act: predict.
    Assert: LayaNotInstalledError with the friendly install hint — not a raw
    ImportError."""
    agent = LayaRouterAgent()
    # Act
    with pytest.raises(LayaNotInstalledError) as excinfo:
        agent.predict({"text": "hello"}, {"q": {"type": "noul", "instructions": "x"}})
    # Assert
    message = str(excinfo.value)
    assert "pip install" in message
    assert "laya" in message
    assert not isinstance(excinfo.value, ImportError)


def test_router_constructing_does_not_import_laya() -> None:
    """Arrange: clean module state. Act: construct LayaRouterAgent.
    Assert: laya was never imported (lazy loading)."""
    sys.modules.pop("laya", None)
    # Act
    agent = LayaRouterAgent()
    # Assert
    assert isinstance(agent, LayaRouterAgent)
    assert "laya" not in sys.modules
    assert agent._agents == {}  # no checkpoint loaded yet


def test_router_routes_before_touching_laya(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: _laya patched to fail the test if called. Act: route().
    Assert: routing is pure text inspection — laya never touched."""
    def _explode() -> Any:
        raise AssertionError("route() must not import laya")

    monkeypatch.setattr(LayaRouterAgent, "_laya", staticmethod(_explode))
    agent = LayaRouterAgent()
    # Act/Assert
    assert agent.route({"text": "hi"}) == "english"


# ------------------------------------------------- predict with fake laya


class _FakeLayaModel:
    """Stand-in for a loaded laya checkpoint."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.calls: list[tuple[dict, dict]] = []

    def predict(self, state: dict, questions: dict) -> Any:
        self.calls.append((state, questions))
        return self.result


class _FakeLayaModule:
    """Stand-in for the `laya` package (load() -> model)."""

    def __init__(self, result: Any) -> None:
        self.result = result
        self.loaded: list[str] = []

    def load(self, checkpoint: str) -> _FakeLayaModel:
        self.loaded.append(checkpoint)
        return _FakeLayaModel(self.result)


def _patch_laya(monkeypatch: pytest.MonkeyPatch, module: _FakeLayaModule) -> None:
    monkeypatch.setattr(LayaRouterAgent, "_laya", staticmethod(lambda: module))


def test_router_predict_wraps_bare_answer_dicts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: fake laya returning a bare answers dict (no 'answers' key).
    Act: predict. Assert: wrapped into the documented {'answers': {...}} shape."""
    fake_laya = _FakeLayaModule({"injection": {"noul": 0.4}})
    _patch_laya(monkeypatch, fake_laya)
    agent = LayaRouterAgent()
    # Act
    result = agent.predict({"text": "hello"}, {"injection": {"type": "noul", "instructions": "x"}})
    # Assert
    assert result == {"answers": {"injection": {"noul": 0.4}}}
    assert fake_laya.loaded == ["convaiinnovations/laya"]  # english checkpoint


def test_router_predict_passes_through_answer_shaped_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: fake laya already returning {'answers': {...}}. Act: predict.
    Assert: result unchanged."""
    fake_laya = _FakeLayaModule({"answers": {"toxicity": {"noul": 0.1}}})
    _patch_laya(monkeypatch, fake_laya)
    agent = LayaRouterAgent()
    # Act
    result = agent.predict({"text": "hi"}, {})
    # Assert
    assert result == {"answers": {"toxicity": {"noul": 0.1}}}


def test_router_predict_picks_multilingual_checkpoint(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: fake laya + Devanagari input. Act: predict.
    Assert: multilingual checkpoint loaded."""
    fake_laya = _FakeLayaModule({"answers": {}})
    _patch_laya(monkeypatch, fake_laya)
    agent = LayaRouterAgent()
    # Act
    agent.predict({"text": "नमस्ते"}, {})
    # Assert
    assert fake_laya.loaded == ["convaiinnovations/laya-multilingual"]


def test_router_predict_caches_loaded_checkpoints(monkeypatch: pytest.MonkeyPatch) -> None:
    """Arrange: fake laya. Act: two predicts in the same bucket.
    Assert: checkpoint loaded exactly once."""
    fake_laya = _FakeLayaModule({"answers": {}})
    _patch_laya(monkeypatch, fake_laya)
    agent = LayaRouterAgent()
    # Act
    agent.predict({"text": "one"}, {})
    agent.predict({"text": "two"}, {})
    # Assert
    assert fake_laya.loaded == ["convaiinnovations/laya"]


def test_router_predict_rejects_non_dict_laya_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Arrange: fake laya returning a non-dict. Act: predict.
    Assert: TypeError naming the actual type."""
    fake_laya = _FakeLayaModule(["not", "a", "dict"])
    _patch_laya(monkeypatch, fake_laya)
    agent = LayaRouterAgent()
    # Act/Assert
    with pytest.raises(TypeError, match="expected dict"):
        agent.predict({"text": "hi"}, {})


async def test_router_apredict_raises_laya_error_off_loop() -> None:
    """Arrange: laya missing. Act: async apredict.
    Assert: friendly LayaNotInstalledError propagates (thread-offloaded)."""
    agent = LayaRouterAgent()
    # Act/Assert
    with pytest.raises(LayaNotInstalledError):
        await agent.apredict({"text": "hi"}, {})


# ----------------------------------------------------------- protocol


def test_fake_agent_satisfies_laya_agent_protocol() -> None:
    """Arrange: FakeAgent. Act: runtime-checkable isinstance.
    Assert: satisfies the LayaAgent protocol the guardrail depends on."""
    assert isinstance(FakeAgent(), LayaAgent)
    assert isinstance(LayaRouterAgent(), LayaAgent)


def test_protocol_predict_returns_answers_shape() -> None:
    """Arrange: FakeAgent. Act: predict with questions. Assert: documented shape."""
    agent = FakeAgent({"injection": {"noul": 0.5}})
    result = agent.predict({"text": "hi"}, {"injection": {"type": "noul"}})
    assert set(result) == {"answers"}
    assert result["answers"]["injection"] == {"noul": 0.5}
