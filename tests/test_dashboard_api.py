"""Dashboard API tests (Phase 3): auth, stats, decisions, review, config, calibration.

Every test runs against a fresh sqlite+aiosqlite database (tmp_path) through an
ASGI transport — no network, no model downloads.
"""

from __future__ import annotations

import importlib.util
import json
from collections.abc import AsyncIterator
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx
import pytest

from gatelaya.audit import DecisionRecord, guardrail_decisions
from gatelaya.config import GateLayaConfig

from dashboard.db import get_engine, get_sessionmaker, init_db
from dashboard.main import create_app
from dashboard.settings import get_settings

NOW = datetime.now(timezone.utc)


# ------------------------------------------------------------------ helpers


def _record(**overrides: Any) -> DecisionRecord:
    """One audit record with sensible defaults (blocked injection in pre_call)."""
    fields: dict[str, Any] = dict(
        check="injection",
        language="english",
        probs={"raw": 0.99, "calibrated": 0.97},
        action_taken="block",
        blocked=True,
        masked=False,
        input_sha256="a" * 64,
        latency_ms=10.0,
        mode="pre_call",
        detail={"threshold": "0.9"},
    )
    fields.update(overrides)
    return DecisionRecord(**fields)


def _clear_caches() -> None:
    """Drop cached settings/engine/sessionmaker so env changes take effect."""
    get_settings.cache_clear()
    get_engine.cache_clear()
    get_sessionmaker.cache_clear()


async def _seed(records: list[DecisionRecord]) -> None:
    """Insert decision records directly into the dashboard database."""
    async with get_engine().begin() as conn:
        for rec in records:
            await conn.execute(guardrail_decisions.insert().values(**rec.model_dump()))


def _laya_installed() -> bool:
    """True when the optional laya package is importable."""
    return importlib.util.find_spec("laya") is not None


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the dashboard at a temp DB/config/calibration and clear caches."""
    monkeypatch.setenv("GATELAYA_DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path}/dashboard.db")
    monkeypatch.setenv("GATELAYA_CONFIG_PATH", str(tmp_path / "gatelaya.yaml"))
    monkeypatch.setenv("GATELAYA_CALIBRATION_PATH", str(tmp_path / "calibration.json"))
    monkeypatch.delenv("GATELAYA_DASHBOARD_TOKEN", raising=False)
    _clear_caches()
    return tmp_path


@pytest.fixture
async def client(env: Path) -> AsyncIterator[httpx.AsyncClient]:
    """ASGI test client on a fresh sqlite DB with token auth disabled."""
    await init_db()
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await get_engine().dispose()
    _clear_caches()


@pytest.fixture
async def auth_client(
    env: Path, monkeypatch: pytest.MonkeyPatch
) -> AsyncIterator[httpx.AsyncClient]:
    """Same as `client`, but GATELAYA_DASHBOARD_TOKEN is set."""
    monkeypatch.setenv("GATELAYA_DASHBOARD_TOKEN", "s3cret")
    get_settings.cache_clear()
    await init_db()
    transport = httpx.ASGITransport(app=create_app())
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
        yield c
    await get_engine().dispose()
    _clear_caches()


# ------------------------------------------------------------------- health


async def test_health_is_open(client: httpx.AsyncClient) -> None:
    """Health probe answers without any authentication."""
    resp = await client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


# -------------------------------------------------------------------- auth


async def test_token_auth_enforced_when_configured(auth_client: httpx.AsyncClient) -> None:
    """With a token set: health stays open, /api/* needs the exact bearer."""
    assert (await auth_client.get("/api/health")).status_code == 200

    assert (await auth_client.get("/api/stats")).status_code == 401
    wrong = await auth_client.get("/api/stats", headers={"Authorization": "Bearer nope"})
    assert wrong.status_code == 401
    assert (await auth_client.get("/api/decisions")).status_code == 401

    ok = await auth_client.get("/api/stats", headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200


# ------------------------------------------------------------------- stats


async def test_stats_counts_window_and_breakdowns(client: httpx.AsyncClient) -> None:
    """Stats aggregate only the trailing window; by_* maps group within it."""
    await _seed(
        [
            _record(
                check="pii",
                action_taken="mask",
                blocked=False,
                masked=True,
                latency_ms=10.0,
                mode="pre_call",
            ),
            _record(check="injection", action_taken="block", blocked=True, latency_ms=30.0),
            _record(
                check="toxicity",
                action_taken="flag",
                blocked=False,
                latency_ms=20.0,
                mode="post_call",
            ),
            _record(
                check="secret_leak",
                action_taken="allow",
                blocked=False,
                latency_ms=40.0,
                mode="post_call",
            ),
            # 30h old: outside the 24h default window, inside hours=48
            _record(
                check="pii",
                action_taken="mask",
                blocked=False,
                masked=True,
                latency_ms=99.0,
                timestamp=NOW - timedelta(hours=30),
            ),
        ]
    )
    data = (await client.get("/api/stats?hours=24")).json()
    assert data["total"] == 4
    assert data["blocked"] == 1
    assert data["flagged"] == 1
    assert data["masked"] == 1
    assert data["allowed"] == 1
    assert data["avg_latency_ms"] == pytest.approx(25.0)
    assert data["by_check"] == {"pii": 1, "injection": 1, "toxicity": 1, "secret_leak": 1}
    assert data["by_action"] == {"mask": 1, "block": 1, "flag": 1, "allow": 1}
    assert data["by_mode"] == {"pre_call": 2, "post_call": 2}
    assert data["window_hours"] == 24

    wide = (await client.get("/api/stats?hours=48")).json()
    assert wide["total"] == 5
    assert wide["window_hours"] == 48


# --------------------------------------------------------------- decisions


async def test_decisions_filters_and_ordering(client: httpx.AsyncClient) -> None:
    """Exact-match filters on check/mode/sha256 + time window; newest first."""
    sha_a, sha_b = "a" * 64, "b" * 64
    await _seed(
        [
            _record(
                id="rec-a",
                check="pii",
                mode="pre_call",
                input_sha256=sha_a,
                timestamp=NOW - timedelta(hours=1),
            ),
            _record(
                id="rec-b",
                check="injection",
                mode="post_call",
                input_sha256=sha_b,
                timestamp=NOW - timedelta(hours=2),
            ),
            _record(
                id="rec-c",
                check="pii",
                mode="post_call",
                input_sha256=sha_a,
                timestamp=NOW - timedelta(hours=3),
            ),
        ]
    )

    by_check = (await client.get("/api/decisions", params={"check": "pii"})).json()
    assert by_check["total"] == 2
    assert {item["check"] for item in by_check["items"]} == {"pii"}

    by_mode = (await client.get("/api/decisions", params={"mode": "post_call"})).json()
    assert by_mode["total"] == 2

    by_sha = (await client.get("/api/decisions", params={"sha256": sha_a})).json()
    assert by_sha["total"] == 2

    combined = (
        await client.get("/api/decisions", params={"check": "pii", "mode": "post_call"})
    ).json()
    assert combined["total"] == 1
    assert combined["items"][0]["id"] == "rec-c"

    since = (
        await client.get(
            "/api/decisions", params={"since": (NOW - timedelta(hours=2)).isoformat()}
        )
    ).json()
    assert since["total"] == 2  # rec-a + rec-b, rec-c too old

    ordered = (await client.get("/api/decisions")).json()["items"]
    assert [item["id"] for item in ordered] == ["rec-a", "rec-b", "rec-c"]
    first = ordered[0]
    assert first["detail"] == {"threshold": "0.9"}
    assert first["probs"] == {"raw": 0.99, "calibrated": 0.97}
    assert first["input_sha256"] == sha_a
    assert first["timestamp"].endswith(("Z", "+00:00"))  # UTC, never naive


async def test_decisions_pagination_and_limit_cap(client: httpx.AsyncClient) -> None:
    """limit is clamped to 200; offset pages through the full filtered set."""
    await _seed(
        [
            _record(id=f"rec-{i:04d}", timestamp=NOW - timedelta(seconds=i))
            for i in range(210)
        ]
    )
    capped = await client.get("/api/decisions", params={"limit": 1000})
    assert capped.status_code == 200
    body = capped.json()
    assert len(body["items"]) == 200
    assert body["total"] == 210

    page = await client.get("/api/decisions", params={"limit": 50, "offset": 200})
    assert len(page.json()["items"]) == 10
    assert page.json()["total"] == 210


async def test_decision_detail_and_404(client: httpx.AsyncClient) -> None:
    """Single decision fetch returns the full record or 404."""
    await _seed([_record(id="rec-1")])
    found = await client.get("/api/decisions/rec-1")
    assert found.status_code == 200
    assert found.json()["id"] == "rec-1"
    assert found.json()["action_taken"] == "block"

    missing = await client.get("/api/decisions/does-not-exist")
    assert missing.status_code == 404


# ------------------------------------------------------------------ review


async def test_review_pending_resolved_all(client: httpx.AsyncClient) -> None:
    """Queue logic: only flag decisions; pending = unlabeled, resolved = labeled."""
    await _seed(
        [
            _record(id="flag-1", action_taken="flag", blocked=False),
            _record(id="flag-2", action_taken="flag", blocked=False, mode="post_call"),
            _record(id="allow-1", action_taken="allow", blocked=False),
        ]
    )

    pending = (await client.get("/api/review")).json()
    assert pending["total"] == 2
    assert all(item["label"] is None for item in pending["items"])

    labeled = await client.post(
        "/api/review/flag-1/label",
        json={"label": "block", "note": "confirmed bad", "text": "evil prompt"},
    )
    assert labeled.status_code == 200

    pending2 = (await client.get("/api/review?status=pending")).json()
    assert pending2["total"] == 1
    assert pending2["items"][0]["decision"]["id"] == "flag-2"

    resolved = (await client.get("/api/review?status=resolved")).json()
    assert resolved["total"] == 1
    item = resolved["items"][0]
    assert item["decision"]["id"] == "flag-1"
    assert item["label"]["label"] == "block"
    assert item["label"]["note"] == "confirmed bad"
    assert item["label"]["text"] == "evil prompt"

    everything = (await client.get("/api/review?status=all")).json()
    assert everything["total"] == 2  # allow-1 never enters the queue


async def test_label_404_409_success_and_body_validation(client: httpx.AsyncClient) -> None:
    """Label errors: 404 unknown decision, 409 duplicate, 422 bad verdict."""
    await _seed(
        [
            _record(id="flag-1", action_taken="flag", blocked=False),
            _record(id="flag-2", action_taken="flag", blocked=False),
        ]
    )

    missing = await client.post("/api/review/nope/label", json={"label": "allow"})
    assert missing.status_code == 404

    first = await client.post("/api/review/flag-1/label", json={"label": "allow"})
    assert first.status_code == 200
    body = first.json()
    assert body["ok"] is True
    assert isinstance(body["id"], str) and body["id"]

    duplicate = await client.post("/api/review/flag-1/label", json={"label": "block"})
    assert duplicate.status_code == 409

    invalid = await client.post("/api/review/flag-2/label", json={"label": "explode"})
    assert invalid.status_code == 422


async def test_review_export_jsonl_shape(client: httpx.AsyncClient) -> None:
    """Export emits calibrate.py rows for labeled flags with text (allow->no)."""
    await _seed(
        [
            _record(
                id="flag-1",
                action_taken="flag",
                blocked=False,
                check="injection",
                timestamp=NOW - timedelta(hours=4),
            ),
            _record(
                id="flag-2",
                action_taken="flag",
                blocked=False,
                check="toxicity",
                timestamp=NOW - timedelta(hours=3),
            ),
            _record(
                id="flag-3",
                action_taken="flag",
                blocked=False,
                timestamp=NOW - timedelta(hours=2),
            ),
            _record(
                id="flag-4",
                action_taken="flag",
                blocked=False,
                check="pii",
                timestamp=NOW - timedelta(hours=1),
            ),
            _record(id="allow-1", action_taken="allow", blocked=False),
        ]
    )
    await client.post(
        "/api/review/flag-1/label", json={"label": "block", "text": "inject please"}
    )
    await client.post("/api/review/flag-2/label", json={"label": "allow", "text": "benign"})
    await client.post("/api/review/flag-3/label", json={"label": "mask"})  # no text -> skip
    await client.post(
        "/api/review/flag-4/label", json={"label": "mask", "text": "blur it"}
    )
    await client.post(
        "/api/review/allow-1/label", json={"label": "mask", "text": "not a flag"}
    )

    resp = await client.get("/api/review/export.jsonl")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    lines = [json.loads(line) for line in resp.text.splitlines()]
    assert lines == [
        {"text": "inject please", "label": "yes", "check": "injection", "question": "injection"},
        {"text": "benign", "label": "no", "check": "toxicity", "question": "toxicity"},
        {"text": "blur it", "label": "yes", "check": "pii", "question": "pii"},
    ]


# ------------------------------------------------------------------ config


async def test_config_get_default_put_writes_file(
    client: httpx.AsyncClient, env: Path
) -> None:
    """GET serves defaults until PUT patches and persists the YAML file."""
    cfg_path = env / "gatelaya.yaml"

    initial = (await client.get("/api/config")).json()
    assert initial["exists"] is False
    assert initial["path"] == str(cfg_path)
    assert initial["config"]["thresholds"]["injection"] == 0.9
    assert initial["config"]["fail_open"] is True

    resp = await client.put(
        "/api/config", json={"thresholds": {"pii": 0.7}, "fail_open": False}
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["ok"] is True
    assert body["hot_reload"] is True
    assert body["restart_required"] is False  # GATELAYA_HOT_RELOAD defaults on
    assert body["path"] == str(cfg_path)
    assert cfg_path.exists()

    after = (await client.get("/api/config")).json()
    assert after["exists"] is True
    assert after["config"]["thresholds"]["pii"] == 0.7
    assert after["config"]["thresholds"]["injection"] == 0.9  # untouched fields merge
    assert after["config"]["fail_open"] is False

    reloaded = GateLayaConfig.from_yaml(cfg_path)
    assert reloaded.threshold("pii") == pytest.approx(0.7)


async def test_config_put_reports_restart_when_hot_reload_disabled(
    client: httpx.AsyncClient, env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GATELAYA_HOT_RELOAD=0: PUT still succeeds but flags restart_required."""
    monkeypatch.setenv("GATELAYA_HOT_RELOAD", "0")
    resp = await client.put("/api/config", json={"thresholds": {"pii": 0.6}})
    assert resp.status_code == 200
    body = resp.json()
    assert body["hot_reload"] is False
    assert body["restart_required"] is True


async def test_config_invalid_patch_is_400(
    client: httpx.AsyncClient, env: Path
) -> None:
    """Invalid actions are rejected with 400 and nothing is written."""
    resp = await client.put("/api/config", json={"actions": {"pii": "explode"}})
    assert resp.status_code == 400
    assert "explode" in resp.json()["detail"]
    assert not (env / "gatelaya.yaml").exists()


async def test_config_unknown_field_is_422(client: httpx.AsyncClient) -> None:
    """The patch body forbids unknown keys."""
    resp = await client.put("/api/config", json={"totally_unknown": 1})
    assert resp.status_code == 422


# ------------------------------------------------------------- calibration


async def test_calibration_missing_returns_null(
    client: httpx.AsyncClient, env: Path
) -> None:
    """No calibration file yet -> temperatures null, exists false."""
    data = (await client.get("/api/calibration")).json()
    assert data == {
        "temperatures": None,
        "path": str(env / "calibration.json"),
        "exists": False,
    }


async def test_calibration_upload_parse_errors_are_400(client: httpx.AsyncClient) -> None:
    """Malformed JSONL and missing checks fail fast with line context."""
    bad_json = await client.post(
        "/api/calibration/upload",
        files={"file": ("data.jsonl", b"{oops\n", "application/json")},
    )
    assert bad_json.status_code == 400
    assert "line 1" in bad_json.json()["detail"]

    no_check = await client.post(
        "/api/calibration/upload",
        files={
            "file": (
                "data.jsonl",
                json.dumps({"text": "hi", "label": "no"}).encode(),
                "application/json",
            )
        },
    )
    assert no_check.status_code == 400
    assert "check" in no_check.json()["detail"]


async def test_calibration_upload_503_without_laya(
    client: httpx.AsyncClient, env: Path
) -> None:
    """Valid dataset reaches the lazy agent: 503 with install hint, no file."""
    if _laya_installed():
        pytest.skip("laya installed: upload would run a real model in this test")
    payload = json.dumps({"text": "hello", "label": "no"}).encode()
    resp = await client.post(
        "/api/calibration/upload",
        files={"file": ("data.jsonl", payload, "application/json")},
        data={"check": "injection"},
    )
    assert resp.status_code == 503
    assert "pip install laya" in resp.json()["detail"]
    assert not (env / "calibration.json").exists()
