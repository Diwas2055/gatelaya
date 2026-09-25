"""Phase 2 model routing: Laya typed decisions rewrite data["model"] pre-call."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from .agent import LayaAgent, LayaRouterAgent, detect_bucket
from .audit import AuditSink, DecisionRecord, InMemoryAuditSink, SqlAlchemyAuditSink
from .calibration import TemperatureMap, calibrated, load_temperature_map
from .errors import GuardrailConfigurationError
from .hotreload import build_watcher, env_flag, env_interval
from .policies import PolicySet, RoutingOverrides, reload_policies
from .questions import (
    COMPLEXITY_OPTIONS,
    TASK_OPTIONS,
    build_routing_questions,
    choice_probs,
    noul_probs,
)
from .scan import extract_scan_text

try:  # real litellm preferred
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:  # pragma: no cover - litellm-less fallback stub
    from ._litellm_stub import CustomGuardrail

logger = logging.getLogger("gatelaya")

TIER_NAMES: tuple[str, ...] = ("trivial", "moderate", "hard")
RouteReason = Literal[
    "sensitive", "tier", "low-confidence", "no-tier-match", "disabled", "agent-error"
]


class RoutingPolicy(BaseModel):
    """Tier, sensitivity, and confidence rules for the model router."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    tiers: dict[str, str] = Field(default_factory=dict)
    default_model: str | None = None
    sensitive_model: str | None = None
    sensitive_threshold: float = 0.85
    confidence_threshold: float = 0.70

    @field_validator("tiers")
    @classmethod
    def _validate_tiers(cls, value: dict[str, str]) -> dict[str, str]:
        unknown = set(value) - set(TIER_NAMES)
        if unknown:
            raise ValueError(f"unknown tier keys: {sorted(unknown)}; allowed: {list(TIER_NAMES)}")
        return value

    @field_validator("sensitive_threshold", "confidence_threshold")
    @classmethod
    def _validate_threshold(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {value}")
        return value

    @classmethod
    def from_yaml(cls, path: str | Path) -> RoutingPolicy:
        """Load a routing policy from a YAML file."""
        p = Path(path)
        try:
            raw = yaml.safe_load(p.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise GuardrailConfigurationError(f"config file not found: {p}") from exc
        except yaml.YAMLError as exc:
            raise GuardrailConfigurationError(f"invalid YAML in {p}: {exc}") from exc
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise GuardrailConfigurationError(f"config root in {p} must be a mapping")
        try:
            return cls(**raw)
        except PydanticValidationError as exc:
            raise GuardrailConfigurationError(f"invalid config in {p}: {exc}") from exc

    def to_yaml(self, path: str | Path) -> None:
        """Write this routing policy to a YAML file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False),
            encoding="utf-8",
        )


@dataclass
class _Signals:
    """Parsed routing answers before a tier decision is made."""

    task: str
    complexity: str
    sensitive: bool
    sensitive_p: float
    confidence: float
    confidence_raw: float


class RoutingDecision(BaseModel):
    """One routing outcome: which model won, from which signals, and why."""

    model_config = ConfigDict(extra="forbid")

    original_model: str
    chosen_model: str
    task: str
    complexity: str
    sensitive: bool
    sensitive_p: float
    confidence: float
    reason: RouteReason
    latency_ms: float


def _env_overrides(policy: RoutingPolicy) -> RoutingPolicy:
    """Apply GATELAYA_ROUTE_* environment overrides (litellm deployment path only)."""
    data = policy.model_dump()
    raw_enabled = os.getenv("GATELAYA_ROUTE_ENABLED")
    if raw_enabled is not None:
        data["enabled"] = raw_enabled.strip().lower() in ("1", "true", "yes", "on")
    raw_default = os.getenv("GATELAYA_ROUTE_DEFAULT")
    if raw_default is not None:
        data["default_model"] = raw_default or None
    raw_sensitive = os.getenv("GATELAYA_ROUTE_SENSITIVE")
    if raw_sensitive is not None:
        data["sensitive_model"] = raw_sensitive or None
    raw_sensitive_threshold = os.getenv("GATELAYA_ROUTE_SENSITIVE_THRESHOLD")
    if raw_sensitive_threshold is not None:
        data["sensitive_threshold"] = float(raw_sensitive_threshold)
    raw_confidence_threshold = os.getenv("GATELAYA_ROUTE_CONFIDENCE_THRESHOLD")
    if raw_confidence_threshold is not None:
        data["confidence_threshold"] = float(raw_confidence_threshold)
    raw_tiers = os.getenv("GATELAYA_ROUTE_TIERS")
    if raw_tiers is not None:
        try:
            tiers = json.loads(raw_tiers)
        except json.JSONDecodeError as exc:
            raise GuardrailConfigurationError(
                f"GATELAYA_ROUTE_TIERS must be a JSON object: {exc}"
            ) from exc
        if not isinstance(tiers, dict):
            raise GuardrailConfigurationError("GATELAYA_ROUTE_TIERS must be a JSON object")
        data["tiers"] = {str(key): str(value) for key, value in tiers.items()}
    return RoutingPolicy(**data)


def apply_routing_overrides(
    policy: RoutingPolicy, overrides: RoutingOverrides | None
) -> RoutingPolicy:
    """Return a new policy with per-tenant routing overrides applied (re-validated)."""
    if overrides is None:
        return policy
    updates = overrides.model_dump(exclude_none=True)
    if not updates:
        return policy
    return RoutingPolicy(**{**policy.model_dump(), **updates})


class GateLayaRouter(CustomGuardrail):
    """Pre-call LiteLLM guardrail that routes requests to tier models; never blocks."""

    def __init__(
        self,
        policy: RoutingPolicy | None = None,
        agent: LayaAgent | None = None,
        audit: AuditSink | None = None,
        temperatures: TemperatureMap | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the router; omitted collaborators default to env/lazy deployments.

        `hot_reload=False` disables mtime polling of routing/calibration/policy
        files (default: GATELAYA_HOT_RELOAD, on). Interval: GATELAYA_RELOAD_INTERVAL.
        """
        routing_path = kwargs.pop("routing_path", None) or os.getenv("GATELAYA_ROUTING_PATH")
        audit_url = kwargs.pop("audit_database_url", None) or os.getenv("GATELAYA_DATABASE_URL")
        hot_reload = kwargs.pop("hot_reload", None)

        self._routing_path = Path(routing_path) if routing_path else None
        self._policy_explicit = policy is not None
        if policy is None:
            policy = RoutingPolicy.from_yaml(routing_path) if routing_path else RoutingPolicy()
            policy = _env_overrides(policy)
        if agent is None:
            agent = LayaRouterAgent()
        if audit is None:
            audit = SqlAlchemyAuditSink(audit_url) if audit_url else InMemoryAuditSink()
        calibration_env = os.getenv("GATELAYA_CALIBRATION_PATH")
        self._calibration_path = Path(calibration_env) if calibration_env else None
        self._temperatures_explicit = temperatures is not None
        if temperatures is None and calibration_env:
            temperatures = load_temperature_map(calibration_env)

        self.policy = policy
        self.agent = agent
        self.audit = audit
        self.temperatures = temperatures
        self.policies: PolicySet | None = None
        self.policies, _ = reload_policies(self._policy_path(), self.policies)

        guardrail_name = kwargs.pop("guardrail_name", None) or "gatelaya-router"
        event_hook = kwargs.pop("event_hook", None) or ["pre_call"]
        default_on = kwargs.pop("default_on", True)
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            **kwargs,
        )

        enabled = env_flag("GATELAYA_HOT_RELOAD", True) if hot_reload is None else hot_reload
        interval = env_interval("GATELAYA_RELOAD_INTERVAL", 1.0)
        self._watcher = build_watcher(self._watch_paths(), enabled=enabled, interval=interval)

    # ------------------------------------------------------------------ helpers

    def _temperature(self, question_type: str, option_count: int | None = None) -> float:
        if self.temperatures is None:
            return 1.0
        return self.temperatures.temperature_for(question_type, option_count)

    # ---------------------------------------------------------------- hot reload

    def _policy_path(self) -> Path | None:
        """Per-key policy YAML from GATELAYA_POLICY_PATH (unset = no policies)."""
        path = os.getenv("GATELAYA_POLICY_PATH")
        return Path(path) if path else None

    def _watch_paths(self) -> list[Path]:
        """Files whose changes trigger a hot reload (routing, calibration, policies)."""
        paths: list[Path] = []
        if not self._policy_explicit and self._routing_path is not None:
            paths.append(self._routing_path)
        if not self._temperatures_explicit and self._calibration_path is not None:
            paths.append(self._calibration_path)
        policy_path = self._policy_path()
        if policy_path is not None:
            paths.append(policy_path)
        return paths

    def _reload_state(self) -> bool:
        """Reload routing policy/temperatures/policies; False when any failed."""
        if not self._policy_explicit:
            try:
                if self._routing_path is not None:
                    self.policy = _env_overrides(RoutingPolicy.from_yaml(self._routing_path))
                else:
                    self.policy = _env_overrides(RoutingPolicy())
            except Exception as exc:
                logger.warning("GateLaya: routing reload failed, keeping previous: %s", exc)
                return False
        ok = True
        if not self._temperatures_explicit and self._calibration_path is not None:
            try:
                self.temperatures = load_temperature_map(self._calibration_path)
            except Exception as exc:
                logger.warning("GateLaya: calibration reload failed, keeping previous: %s", exc)
                ok = False
        policies, policies_ok = reload_policies(self._policy_path(), self.policies)
        self.policies = policies
        return policies_ok and ok

    def _maybe_reload(self) -> None:
        """Swap routing/temperatures/policies in place when their files change."""
        watcher = self._watcher
        if watcher is None:
            return
        paths = self._watch_paths()
        if not watcher.poll(paths):
            return
        if self._reload_state():
            watcher.commit()

    def _resolve_call(self, user_api_key_dict: Any) -> tuple[RoutingPolicy, str | None]:
        """Hot-reload check + per-key policy; returns (effective policy, audit label)."""
        self._maybe_reload()
        if not self.policies:
            return self.policy, None
        tenant, label = self.policies.resolve(user_api_key_dict)
        if tenant is None or tenant.overrides.routing is None:
            return self.policy, label
        return apply_routing_overrides(self.policy, tenant.overrides.routing), label

    async def _predict(self, text: str) -> tuple[dict[str, Any], float, Exception | None]:
        """Run the agent once for the routing questions; returns (answers, latency, error)."""
        questions = build_routing_questions()
        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(self.agent.predict, {"text": text}, questions)
        except Exception as exc:  # agent/model failures must never crash the proxy
            return {}, (time.perf_counter() - start) * 1000.0, exc
        latency_ms = (time.perf_counter() - start) * 1000.0
        if isinstance(result, dict):
            raw_answers = result.get("answers")
            if isinstance(raw_answers, dict):
                return raw_answers, latency_ms, None
            logger.warning(
                "GateLaya router agent returned %s without answers, treating as no answers",
                type(result).__name__,
            )
        return {}, latency_ms, None

    def _signals(
        self, answers: dict[str, Any], policy: RoutingPolicy | None = None
    ) -> _Signals:
        """Parse routing answers into calibrated sensitivity/confidence signals."""
        policy = policy or self.policy
        sensitive_raw = noul_probs(answers["sensitive"])[1] if "sensitive" in answers else 0.0
        sensitive_p = calibrated([1.0 - sensitive_raw, sensitive_raw], self._temperature("noul"))[1]

        complexity_names = list(COMPLEXITY_OPTIONS)
        if "complexity" in answers:
            raw_probs = choice_probs(answers["complexity"], complexity_names)
            cal_probs = calibrated(raw_probs, self._temperature("choice", len(complexity_names)))
            idx = max(range(len(raw_probs)), key=raw_probs.__getitem__)
            complexity = complexity_names[idx]
            confidence_raw, confidence = raw_probs[idx], cal_probs[idx]
        else:
            complexity, confidence_raw, confidence = "unknown", 0.0, 0.0

        task_names = list(TASK_OPTIONS)
        if "task" in answers:
            task_probs = choice_probs(answers["task"], task_names)
            task = task_names[max(range(len(task_probs)), key=task_probs.__getitem__)]
        else:
            task = "unknown"

        return _Signals(
            task=task,
            complexity=complexity,
            sensitive=sensitive_p >= policy.sensitive_threshold,
            sensitive_p=sensitive_p,
            confidence=confidence,
            confidence_raw=confidence_raw,
        )

    def _decide(
        self, original: str, signals: _Signals, policy: RoutingPolicy | None = None
    ) -> tuple[str, RouteReason]:
        """Pick the target model: sensitive > low-confidence > tier lookup."""
        policy = policy or self.policy
        if signals.sensitive and policy.sensitive_model:
            return policy.sensitive_model, "sensitive"
        if signals.confidence < policy.confidence_threshold:
            return policy.default_model or original, "low-confidence"
        tier_model = policy.tiers.get(signals.complexity, "")
        if tier_model:
            return tier_model, "tier"
        return original, "no-tier-match"

    async def _audit_record(self, rec: DecisionRecord) -> None:
        """Persist one decision; audit failures never break the request."""
        try:
            await self.audit.record(rec)
        except Exception:
            logger.warning("GateLaya audit sink failed", exc_info=True)

    async def _record_agent_error(
        self, exc: Exception | None, *, original: str, language: str,
        text_hash: str, latency_ms: float, data: dict,
    ) -> dict:
        """Keep the original model, stash an agent-error decision, audit routing_error."""
        logger.warning("GateLaya routing agent failed: %s", exc)
        decision = RoutingDecision(
            original_model=original,
            chosen_model=original,
            task="unknown",
            complexity="unknown",
            sensitive=False,
            sensitive_p=0.0,
            confidence=0.0,
            reason="agent-error",
            latency_ms=latency_ms,
        )
        data["gatelaya_routing"] = decision.model_dump()
        await self._audit_record(
            DecisionRecord(
                check="routing_error",
                language=language,
                probs={},
                action_taken=original,
                blocked=False,
                masked=False,
                input_sha256=text_hash,
                latency_ms=latency_ms,
                mode="route",
                detail={"reason": "agent-error"},
            )
        )
        return data

    # -------------------------------------------------------------------- hooks

    async def async_pre_call_hook(
        self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str
    ) -> dict | None:
        """Rewrite data['model'] from Laya routing signals; never blocks the request."""
        policy, policy_label = self._resolve_call(user_api_key_dict)
        if not policy.enabled:
            return None
        scan_text, _ = extract_scan_text(data.get("messages"))
        if not scan_text:
            return None

        original = str(data.get("model") or "")
        language = detect_bucket(scan_text)
        text_hash = hashlib.sha256(scan_text.encode("utf-8")).hexdigest()

        answers, latency_ms, error = await self._predict(scan_text)
        signals: _Signals | None = None
        if error is None:
            try:
                signals = self._signals(answers, policy)
            except Exception as exc:  # malformed answers must fail open, not crash the proxy
                error = exc
        if error is not None or signals is None:
            return await self._record_agent_error(
                error, original=original, language=language,
                text_hash=text_hash, latency_ms=latency_ms, data=data,
            )

        chosen, reason = self._decide(original, signals, policy)
        decision = RoutingDecision(
            original_model=original,
            chosen_model=chosen,
            task=signals.task,
            complexity=signals.complexity,
            sensitive=signals.sensitive,
            sensitive_p=signals.sensitive_p,
            confidence=signals.confidence,
            reason=reason,
            latency_ms=latency_ms,
        )
        detail: dict[str, str] = {
            "task": signals.task,
            "complexity": signals.complexity,
            "sensitive": str(signals.sensitive).lower(),
            "sensitive_p": f"{signals.sensitive_p:.4f}",
            "reason": reason,
        }
        if policy_label:
            detail["policy"] = policy_label
        data["model"] = chosen
        data["gatelaya_routing"] = decision.model_dump()
        await self._audit_record(
            DecisionRecord(
                check="routing",
                language=language,
                probs={"raw": signals.confidence_raw, "calibrated": signals.confidence},
                action_taken=chosen,
                blocked=False,
                masked=False,
                input_sha256=text_hash,
                latency_ms=latency_ms,
                mode="route",
                detail=detail,
            )
        )
        return data
