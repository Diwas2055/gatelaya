"""GateLaya CustomGuardrail: Laya-powered pre/post call hooks for LiteLLM Proxy."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import os
import re
import time
from collections.abc import AsyncGenerator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import HTTPException

from .agent import LayaAgent, LayaRouterAgent, detect_bucket
from .audit import AuditSink, DecisionRecord, InMemoryAuditSink, SqlAlchemyAuditSink
from .calibration import TemperatureMap, calibrated, load_temperature_map
from .config import ALL_CHECKS, GateLayaConfig
from .hotreload import build_watcher, env_flag, env_interval
from .policies import PolicySet, apply_overrides, reload_policies
from .questions import PII_TYPE_OPTIONS, build_questions, noul_probs
from .scan import content_to_text as _content_to_text
from .scan import extract_scan_text

try:  # real litellm preferred
    from litellm.integrations.custom_guardrail import CustomGuardrail
except ImportError:  # pragma: no cover - litellm-less fallback stub
    from ._litellm_stub import CustomGuardrail

logger = logging.getLogger("gatelaya")

CHECK_LABELS = {
    "pii": "PII",
    "injection": "prompt injection",
    "toxicity": "toxic content",
    "secret_leak": "secret leak",
}
#: Checks run per hook (pre and post) filtered by the config's enabled_checks.
CHECK_ORDER = ("pii", "injection", "toxicity", "secret_leak")
REDACTION_TOKEN = "[REDACTED:pii]"

_PII_PATTERNS: tuple[re.Pattern[str], ...] = (
    # SSN before credit-card so 000-00-0000 shapes stay intact
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"),
    re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    re.compile(r"\+?\b\d[\d\s().-]{7,14}\b"),
)


@dataclass
class _Outcome:
    """Per-check evaluation result before actions are applied."""

    check: str
    raw_p: float
    cal_p: float
    threshold: float
    decision: str  # allow | mask | block | flag
    masked: bool = False
    pii_type: str | None = None


@dataclass
class _Delta:
    """One text-bearing delta inside a buffered stream chunk."""

    index: int  # choice index
    kind: str  # content | text | delta (field name carrying the text)
    holder: Any  # dict or object owning the field
    current: str  # current text value


def _redact_pii(text: str) -> tuple[str, int]:
    """Replace common PII spans with a redaction token; returns (text, redactions)."""
    spans: list[tuple[int, int]] = []
    for pattern in _PII_PATTERNS:
        spans.extend(match.span() for match in pattern.finditer(text))
    if not spans:
        return text, 0
    spans.sort()
    merged: list[tuple[int, int]] = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    pieces: list[str] = []
    cursor = 0
    for start, end in merged:
        pieces.append(text[cursor:start])
        pieces.append(REDACTION_TOKEN)
        cursor = end
    pieces.append(text[cursor:])
    return "".join(pieces), len(merged)


def _choices_of(obj: Any) -> list[Any]:
    """Return the `choices` list of a response or stream chunk (dict or object)."""
    if isinstance(obj, dict):
        choices = obj.get("choices")
        return list(choices) if isinstance(choices, list) else []
    choices = getattr(obj, "choices", None)
    return list(choices) if isinstance(choices, (list, tuple)) else []


def _choice_texts(choice: Any) -> tuple[str, str]:
    """Flattened (content, reasoning_content) of one response choice."""
    if isinstance(choice, dict):
        message = choice.get("message")
        if isinstance(message, dict):
            return (
                _content_to_text(message.get("content")),
                _content_to_text(message.get("reasoning_content")),
            )
        if message is not None:
            return (
                _content_to_text(getattr(message, "content", None)),
                _content_to_text(getattr(message, "reasoning_content", None)),
            )
        return _content_to_text(choice.get("text")), ""
    message = getattr(choice, "message", None)
    if message is not None:
        return (
            _content_to_text(getattr(message, "content", None)),
            _content_to_text(getattr(message, "reasoning_content", None)),
        )
    return _content_to_text(getattr(choice, "text", None)), ""


def _set_choice_texts(choice: Any, content: str, reasoning: str) -> bool:
    """Write redacted content/reasoning back onto one choice; False = unsupported."""
    try:
        if isinstance(choice, dict):
            message = choice.get("message")
            if not isinstance(message, dict):
                return False
            if reasoning or "reasoning_content" in message:
                message["reasoning_content"] = reasoning
            message["content"] = content
            return True
        message = getattr(choice, "message", None)
        if message is None:
            return False
        if reasoning or getattr(message, "reasoning_content", None) is not None:
            setattr(message, "reasoning_content", reasoning)
        setattr(message, "content", content)
        return True
    except Exception:
        return False


def _choice_index(choice: Any, position: int) -> int:
    """Choice index from a stream chunk entry, falling back to list position."""
    if isinstance(choice, dict):
        index = choice.get("index")
    else:
        index = getattr(choice, "index", None)
    return index if isinstance(index, int) else position


def _chunk_deltas(chunk: Any) -> list[_Delta]:
    """Text-bearing deltas of one stream chunk (dict or ModelResponseStream shape)."""
    deltas: list[_Delta] = []
    choices = _choices_of(chunk)
    if choices:
        for position, choice in enumerate(choices):
            index = _choice_index(choice, position)
            if isinstance(choice, dict):
                delta = choice.get("delta")
                if isinstance(delta, dict):
                    text = delta.get("content")
                    if isinstance(text, str):
                        deltas.append(_Delta(index, "content", delta, text))
                elif delta is None and isinstance(choice.get("text"), str):
                    deltas.append(_Delta(index, "text", choice, choice["text"]))
            else:
                delta = getattr(choice, "delta", None)
                if delta is not None:
                    text = getattr(delta, "content", None)
                    if isinstance(text, str):
                        deltas.append(_Delta(index, "content", delta, text))
                else:
                    text = getattr(choice, "text", None)
                    if isinstance(text, str):
                        deltas.append(_Delta(index, "text", choice, text))
        return deltas
    # legacy/bare chunk shapes: {"delta": "text"} or {"delta": {"content": "text"}}
    if isinstance(chunk, dict):
        delta = chunk.get("delta")
        if isinstance(delta, str):
            deltas.append(_Delta(0, "delta", chunk, delta))
        elif isinstance(delta, dict) and isinstance(delta.get("content"), str):
            deltas.append(_Delta(0, "content", delta, delta["content"]))
    else:
        delta = getattr(chunk, "delta", None)
        if isinstance(delta, str):
            deltas.append(_Delta(0, "delta", chunk, delta))
        elif delta is not None and isinstance(getattr(delta, "content", None), str):
            deltas.append(_Delta(0, "content", delta, delta.content))
    return deltas


def _assemble_stream(chunks: list[Any]) -> dict[int, str]:
    """Concatenate delta text per choice index across buffered chunks."""
    texts: dict[int, str] = {}
    for chunk in chunks:
        for delta in _chunk_deltas(chunk):
            if delta.current:
                texts[delta.index] = texts.get(delta.index, "") + delta.current
    return texts


def _set_delta_text(delta: _Delta, text: str) -> bool:
    """Write new text onto one delta; False when the shape cannot be written."""
    try:
        if isinstance(delta.holder, dict):
            delta.holder[delta.kind] = text
        else:
            setattr(delta.holder, delta.kind, text)
        return True
    except Exception:
        return False


def _rewrite_stream(chunks: list[Any], redacted: dict[int, str]) -> bool:
    """Give each choice's first text delta the redacted text, clear later ones."""
    emitted: set[int] = set()
    for chunk in chunks:
        for delta in _chunk_deltas(chunk):
            if not delta.current or delta.index not in redacted:
                continue
            target = redacted[delta.index] if delta.index not in emitted else ""
            if not _set_delta_text(delta, target):
                return False
            emitted.add(delta.index)
    return True


def _env_overrides(config: GateLayaConfig) -> GateLayaConfig:
    """Apply GATELAYA_* environment overrides (litellm deployment path only)."""
    data = config.model_dump()
    for check in ALL_CHECKS:
        raw_threshold = os.getenv(f"GATELAYA_THRESHOLD_{check.upper()}")
        if raw_threshold is not None:
            data["thresholds"][check] = float(raw_threshold)
        raw_action = os.getenv(f"GATELAYA_ACTION_{check.upper()}")
        if raw_action is not None:
            data["actions"][check] = raw_action
    raw_fail_open = os.getenv("GATELAYA_FAIL_OPEN")
    if raw_fail_open is not None:
        data["fail_open"] = raw_fail_open.strip().lower() in ("1", "true", "yes", "on")
    raw_checks = os.getenv("GATELAYA_ENABLED_CHECKS")
    if raw_checks is not None:
        data["enabled_checks"] = [c.strip() for c in raw_checks.split(",") if c.strip()]
    raw_audit = os.getenv("GATELAYA_AUDIT_ENABLED")
    if raw_audit is not None:
        data["audit_enabled"] = raw_audit.strip().lower() in ("1", "true", "yes", "on")
    raw_calibration = os.getenv("GATELAYA_CALIBRATION_PATH")
    if raw_calibration is not None:
        data["calibration_path"] = raw_calibration or None
    raw_model = os.getenv("GATELAYA_MODEL_PATH")
    if raw_model is not None:
        data["model_path"] = raw_model or None
    raw_policy = os.getenv("GATELAYA_POLICY_PATH")
    if raw_policy is not None:
        data["policy_path"] = raw_policy or None
    return GateLayaConfig(**data)


def _policy_detail(policy: str | None) -> dict[str, str] | None:
    """Audit `detail` payload naming the matched tenant policy (None = global)."""
    return {"policy": policy} if policy else None


class GateLayaGuardrail(CustomGuardrail):
    """Laya-powered LiteLLM guardrail: block/mask/flag PII, injection, toxicity, secrets."""

    def __init__(
        self,
        config: GateLayaConfig | None = None,
        agent: LayaAgent | None = None,
        audit: AuditSink | None = None,
        temperatures: TemperatureMap | None = None,
        **kwargs: Any,
    ) -> None:
        """Build the guardrail; omitted collaborators default to env/lazy deployments.

        `hot_reload=False` disables mtime polling of config/calibration/policy
        files (default: GATELAYA_HOT_RELOAD, on). Interval: GATELAYA_RELOAD_INTERVAL.
        """
        config_path = kwargs.pop("config_path", None) or os.getenv("GATELAYA_CONFIG_PATH")
        thresholds_override = kwargs.pop("thresholds", None)
        actions_override = kwargs.pop("actions", None)
        audit_url = kwargs.pop("audit_database_url", None) or os.getenv("GATELAYA_DATABASE_URL")
        hot_reload = kwargs.pop("hot_reload", None)

        if config is None:
            config = GateLayaConfig.from_yaml(config_path) if config_path else GateLayaConfig()
            config = _env_overrides(config)
        if isinstance(thresholds_override, dict):
            config = config.model_copy(
                update={"thresholds": {**config.thresholds, **thresholds_override}}
            )
        if isinstance(actions_override, dict):
            config = config.model_copy(
                update={"actions": {**config.actions, **actions_override}}
            )

        if agent is None:
            agent = LayaRouterAgent(
                english_checkpoint=config.english_checkpoint,
                multilingual_checkpoint=config.multilingual_checkpoint,
                model_path=str(config.model_path) if config.model_path else None,
            )
        if audit is None:
            audit = SqlAlchemyAuditSink(audit_url) if audit_url else InMemoryAuditSink()
        temperatures_explicit = temperatures is not None
        if temperatures is None and config.calibration_path is not None:
            temperatures = load_temperature_map(config.calibration_path)

        self.config = config
        self.agent = agent
        self.audit = audit
        self.temperatures = temperatures
        self.policies: PolicySet | None = None
        self._stream_logged = False
        self._config_path = Path(config_path) if config_path else None
        self._thresholds_override = thresholds_override if isinstance(thresholds_override, dict) else None
        self._actions_override = actions_override if isinstance(actions_override, dict) else None
        self._temperatures_explicit = temperatures_explicit
        self.policies, _ = reload_policies(self._policy_path(), self.policies)

        guardrail_name = kwargs.pop("guardrail_name", None) or "gatelaya"
        event_hook = kwargs.pop("event_hook", None)
        self._event_hook_explicit = event_hook is not None
        event_hook = event_hook or list(config.mode)
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

    def _noul_temperature(self, check: str) -> float:
        """Temperature for a check's noul answer (per-check entry, else `default`)."""
        if self.temperatures is None:
            return 1.0
        return self.temperatures.noul_temperature(check)

    # ---------------------------------------------------------------- hot reload

    def _policy_path(self) -> Path | None:
        """Per-key policy YAML: config field, falling back to GATELAYA_POLICY_PATH."""
        path = self.config.policy_path or os.getenv("GATELAYA_POLICY_PATH")
        return Path(path) if path else None

    def _watch_paths(self) -> list[Path]:
        """Files whose changes trigger a hot reload (config, calibration, policies)."""
        paths: list[Path] = []
        if self._config_path is not None:
            paths.append(self._config_path)
        if self.config.calibration_path is not None:
            paths.append(Path(self.config.calibration_path))
        policy_path = self._policy_path()
        if policy_path is not None:
            paths.append(policy_path)
        return paths

    def _load_config_file(self) -> GateLayaConfig:
        """Re-read the config YAML (env overrides + constructor overrides re-applied)."""
        assert self._config_path is not None
        cfg = _env_overrides(GateLayaConfig.from_yaml(self._config_path))
        if self._thresholds_override is not None:
            cfg = cfg.model_copy(
                update={"thresholds": {**cfg.thresholds, **self._thresholds_override}}
            )
        if self._actions_override is not None:
            cfg = cfg.model_copy(update={"actions": {**cfg.actions, **self._actions_override}})
        return cfg

    def _reload_state(self) -> bool:
        """Reload config/temperatures/policies; False when any reload failed."""
        if self._config_path is not None:
            try:
                cfg = self._load_config_file()
            except Exception as exc:
                logger.warning("GateLaya: config reload failed, keeping previous: %s", exc)
                return False
            self.config = cfg
            if not self._event_hook_explicit:
                self.event_hook = list(cfg.mode)
        ok = True
        if not self._temperatures_explicit and self.config.calibration_path is not None:
            try:
                self.temperatures = load_temperature_map(self.config.calibration_path)
            except Exception as exc:
                logger.warning("GateLaya: calibration reload failed, keeping previous: %s", exc)
                ok = False
        policies, policies_ok = reload_policies(self._policy_path(), self.policies)
        self.policies = policies
        return policies_ok and ok

    def _maybe_reload(self) -> None:
        """Swap config/temperatures/policies in place when their files change.

        Failed reloads keep the previous state and are retried on the next poll
        (mtimes advance only after a fully successful cycle).
        """
        watcher = self._watcher
        if watcher is None:
            return
        paths = self._watch_paths()
        if not watcher.poll(paths):
            return
        if self._reload_state():
            watcher.commit()

    def _resolve_call(self, user_api_key_dict: Any) -> tuple[GateLayaConfig, str | None]:
        """Hot-reload check + per-key policy; returns (effective config, audit label)."""
        self._maybe_reload()
        if not self.policies:
            return self.config, None
        tenant, label = self.policies.resolve(user_api_key_dict)
        if tenant is None:
            return self.config, None
        return apply_overrides(self.config, tenant.overrides), label

    # --------------------------------------------------------------- evaluation

    async def _predict(
        self, text: str, checks: list[str], cfg: GateLayaConfig
    ) -> tuple[list[_Outcome] | None, float, Exception | None]:
        """Run the agent once for all checks; returns (outcomes, latency_ms, error)."""
        questions, _ = build_questions(checks)
        start = time.perf_counter()
        try:
            result = await asyncio.to_thread(self.agent.predict, {"text": text}, questions)
        except Exception as exc:  # agent/model failures must never crash the proxy
            return None, (time.perf_counter() - start) * 1000.0, exc
        latency_ms = (time.perf_counter() - start) * 1000.0

        answers: dict[str, Any] = {}
        if isinstance(result, dict):
            raw_answers = result.get("answers")
            answers = raw_answers if isinstance(raw_answers, dict) else {}
        else:
            logger.warning("GateLaya agent returned %s, treating as no answers", type(result).__name__)

        try:
            outcomes = [self._outcome(check, answers, questions, cfg) for check in checks]
        except Exception as exc:  # malformed answers must fail open, not crash the proxy
            logger.warning("GateLaya: could not evaluate agent answers: %s", exc)
            return None, latency_ms, exc
        return outcomes, latency_ms, None

    def _outcome(
        self, check: str, answers: dict[str, Any], questions: dict[str, Any], cfg: GateLayaConfig
    ) -> _Outcome:
        """Evaluate one check's answer against its threshold and configured action."""
        threshold = cfg.threshold(check)
        configured = cfg.action(check)
        answer = answers.get(check)
        if answer is None:
            logger.warning("GateLaya: no answer for check %r; treating as allow", check)
            vector = [1.0, 0.0]
        else:
            vector = noul_probs(answer)
        cal = calibrated(vector, self._noul_temperature(check))
        raw_p, cal_p = vector[1], cal[1]
        decision = "allow" if cal_p < threshold else configured

        pii_type: str | None = None
        if check == "pii" and "pii_type" in answers:
            pii_type = self._pii_type_label(answers["pii_type"])
        return _Outcome(
            check=check,
            raw_p=raw_p,
            cal_p=cal_p,
            threshold=threshold,
            decision=decision,
            pii_type=pii_type,
        )

    @staticmethod
    def _pii_type_label(answer: Any) -> str | None:
        """Reduce a pii_type choice answer to an option label (or None)."""
        options = list(PII_TYPE_OPTIONS)
        if isinstance(answer, dict):
            selected = answer.get("choice")
            return str(selected) if selected is not None else None
        if isinstance(answer, str):
            return answer if answer in options else None
        if isinstance(answer, int) and 0 <= answer < len(options):
            return options[answer]
        return None

    # ------------------------------------------------------------------- audit

    async def _audit_record(self, rec: DecisionRecord) -> None:
        """Persist one decision; audit failures never break the request."""
        if not self.config.audit_enabled:
            return
        try:
            await self.audit.record(rec)
        except Exception:
            logger.warning("GateLaya audit sink failed", exc_info=True)

    async def _record_outcomes(
        self, outcomes: list[_Outcome], *, language: str, text_hash: str,
        latency_ms: float, mode: str, detail: dict[str, str] | None = None,
    ) -> None:
        for outcome in outcomes:
            await self._audit_record(
                DecisionRecord(
                    check=outcome.check,
                    language=language,
                    probs={"raw": outcome.raw_p, "calibrated": outcome.cal_p},
                    action_taken=outcome.decision,
                    blocked=outcome.decision == "block",
                    masked=outcome.masked,
                    input_sha256=text_hash,
                    latency_ms=latency_ms,
                    mode=mode,
                    detail=detail or {},
                )
            )

    async def _record_failure(
        self, exc: Exception, *, text: str, latency_ms: float, mode: str, blocked: bool,
        detail: dict[str, str] | None = None,
    ) -> None:
        """Record an agent-failure decision (fail-open allow or fail-closed block)."""
        logger.warning("GateLaya agent failed (%s): %s", mode, exc)
        await self._audit_record(
            DecisionRecord(
                check="agent_error",
                language=detect_bucket(text),
                probs={},
                action_taken="block" if blocked else "allow",
                blocked=blocked,
                masked=False,
                input_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                latency_ms=latency_ms,
                mode=mode,
                detail=detail or {},
            )
        )

    @staticmethod
    def _decision_meta(
        outcomes: list[_Outcome], *, language: str, text_hash: str, policy: str | None = None
    ) -> dict[str, Any]:
        """Build the data['gatelaya'] decision metadata payload."""
        meta: dict[str, Any] = {
            "language": language,
            "input_sha256": text_hash,
            "decisions": [
                {
                    "check": o.check,
                    "raw": o.raw_p,
                    "calibrated": o.cal_p,
                    "threshold": o.threshold,
                    "action": o.decision,
                    "blocked": o.decision == "block",
                    "masked": o.masked,
                    "pii_type": o.pii_type,
                }
                for o in outcomes
            ],
        }
        if policy:
            meta["policy"] = policy
        return meta

    @staticmethod
    def _block_message(outcome: _Outcome, *, in_response: bool) -> str:
        label = CHECK_LABELS.get(outcome.check, outcome.check)
        where = " in response" if in_response else ""
        return f"GateLaya: {label} detected{where} (p={outcome.cal_p:.2f})"

    # --------------------------------------------------------------- actions

    def _apply_mask(self, messages: list[Any], last_user_idx: int | None, pii_type: str | None) -> int:
        """Regex-redact PII spans; optionally replace the last user message wholesale."""
        redactions = 0
        for msg in messages:
            if not isinstance(msg, dict):
                continue
            text = _content_to_text(msg.get("content"))
            if not text:
                continue
            redacted, count = _redact_pii(text)
            if count:
                msg["content"] = redacted
                redactions += count
        # v1 limitation: when the model names a PII type and regex found anything,
        # the entire last user message collapses to a single redaction token.
        if redactions and pii_type and pii_type != "none" and last_user_idx is not None:
            target = messages[last_user_idx]
            if isinstance(target, dict):
                target["content"] = REDACTION_TOKEN
        return redactions

    def _apply_actions(
        self, outcomes: list[_Outcome], *, messages: list[Any], last_user_idx: int | None
    ) -> bool:
        """Downgrade no-op masks to flags; returns True when a mask changed content."""
        mask_outcomes = [o for o in outcomes if o.decision == "mask"]
        if not mask_outcomes:
            return False
        pii_type = next((o.pii_type for o in mask_outcomes if o.check == "pii"), None)
        redactions = self._apply_mask(messages, last_user_idx, pii_type)
        if redactions == 0:
            logger.debug("GateLaya: mask action found no regex matches; downgrading to flag")
            for outcome in mask_outcomes:
                outcome.decision = "flag"
            return False
        for outcome in mask_outcomes:
            outcome.masked = True
        return True

    def _apply_response_mask(
        self, outcomes: list[_Outcome], choice: Any, content: str, reasoning: str
    ) -> bool:
        """Redact PII spans in a response choice; no-op masks downgrade to flag."""
        mask_outcomes = [o for o in outcomes if o.decision == "mask"]
        if not mask_outcomes:
            return False
        redacted_content, content_count = _redact_pii(content)
        redacted_reasoning, reasoning_count = _redact_pii(reasoning)
        if content_count + reasoning_count == 0:
            logger.debug("GateLaya: response mask found no regex matches; downgrading to flag")
            for outcome in mask_outcomes:
                outcome.decision = "flag"
            return False
        if not _set_choice_texts(choice, redacted_content, redacted_reasoning):
            logger.debug("GateLaya: response choice not maskable; downgrading to flag")
            for outcome in mask_outcomes:
                outcome.decision = "flag"
            return False
        for outcome in mask_outcomes:
            outcome.masked = True
        return True

    # -------------------------------------------------------------------- hooks

    async def async_pre_call_hook(
        self, user_api_key_dict: Any, cache: Any, data: dict, call_type: str
    ) -> Exception | str | dict | None:
        """Scan the request prompt; return a block string, masked data, or None."""
        cfg, policy = self._resolve_call(user_api_key_dict)
        messages = data.get("messages")
        scan_text, last_user_idx = extract_scan_text(messages)
        if not scan_text:
            return None

        checks = [c for c in CHECK_ORDER if c in cfg.enabled_checks]
        if not checks:
            return None

        language = detect_bucket(scan_text)
        text_hash = hashlib.sha256(scan_text.encode("utf-8")).hexdigest()
        outcomes, latency_ms, error = await self._predict(scan_text, checks, cfg)
        detail = _policy_detail(policy)

        if error is not None:
            if cfg.fail_open:
                await self._record_failure(
                    error, text=scan_text, latency_ms=latency_ms, mode="pre_call",
                    blocked=False, detail=detail,
                )
                return None
            await self._record_failure(
                error, text=scan_text, latency_ms=latency_ms, mode="pre_call",
                blocked=True, detail=detail,
            )
            return "GateLaya: internal guardrail error (fail-closed)"

        assert outcomes is not None
        blocked = next((o for o in outcomes if o.decision == "block"), None)
        masked = False
        if blocked is None:
            masked = self._apply_actions(
                outcomes, messages=messages, last_user_idx=last_user_idx
            )
        await self._record_outcomes(
            outcomes, language=language, text_hash=text_hash,
            latency_ms=latency_ms, mode="pre_call", detail=detail,
        )

        if blocked is not None:
            data["gatelaya"] = self._decision_meta(
                outcomes, language=language, text_hash=text_hash, policy=policy
            )
            return self._block_message(blocked, in_response=False)

        if any(o.decision != "allow" for o in outcomes):
            data["gatelaya"] = self._decision_meta(
                outcomes, language=language, text_hash=text_hash, policy=policy
            )
        if masked:
            return data
        return None

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: Any, response: Any
    ) -> Any:
        """Scan assistant output (every choice, content + reasoning); raise 400 on block.

        Returns the response when a mask rewrote it; None otherwise (litellm swaps
        the return value in only when it is not None).
        """
        cfg, policy = self._resolve_call(user_api_key_dict)
        checks = [c for c in CHECK_ORDER if c in cfg.enabled_checks]
        if not checks:
            return None

        detail = _policy_detail(policy)
        outcomes_all: list[_Outcome] = []
        blocked: _Outcome | None = None
        any_masked = False
        language = ""
        text_hash = ""
        scanned = 0
        for choice in _choices_of(response):
            content, reasoning = _choice_texts(choice)
            text = "\n".join(part for part in (content, reasoning) if part).strip()
            if not text:
                continue
            scanned += 1
            choice_language = detect_bucket(text)
            choice_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            if scanned == 1:
                language, text_hash = choice_language, choice_hash
            outcomes, latency_ms, error = await self._predict(text, checks, cfg)

            if error is not None:
                if cfg.fail_open:
                    await self._record_failure(
                        error, text=text, latency_ms=latency_ms, mode="post_call",
                        blocked=False, detail=detail,
                    )
                    continue
                await self._record_failure(
                    error, text=text, latency_ms=latency_ms, mode="post_call",
                    blocked=True, detail=detail,
                )
                raise HTTPException(
                    status_code=400, detail="GateLaya: internal guardrail error (fail-closed)"
                )

            assert outcomes is not None
            choice_blocked = next((o for o in outcomes if o.decision == "block"), None)
            masked = False
            if choice_blocked is None:
                masked = self._apply_response_mask(outcomes, choice, content, reasoning)
            await self._record_outcomes(
                outcomes, language=choice_language, text_hash=choice_hash,
                latency_ms=latency_ms, mode="post_call", detail=detail,
            )
            outcomes_all.extend(outcomes)
            any_masked = any_masked or masked
            if choice_blocked is not None and blocked is None:
                blocked = choice_blocked

        if scanned == 0:
            return None
        meta = self._decision_meta(
            outcomes_all, language=language, text_hash=text_hash, policy=policy
        )
        if blocked is not None:
            data["gatelaya"] = meta
            raise HTTPException(status_code=400, detail=self._block_message(blocked, in_response=True))

        if any(o.decision != "allow" for o in outcomes_all):
            data["gatelaya"] = meta
            self._attach_response_meta(response, meta)
        return response if any_masked else None

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> AsyncGenerator[Any, None]:
        """Buffer the stream, scan the assembled assistant text, then deliver it.

        Blocks raise before any chunk is yielded; masks rewrite the buffered
        deltas; clean chunks are yielded as the original objects (identity kept).
        """
        cfg, policy = self._resolve_call(user_api_key_dict)
        checks = [c for c in CHECK_ORDER if c in cfg.enabled_checks]
        if not checks:
            async for chunk in response:
                yield chunk
            return
        if not self._stream_logged:
            logger.info(
                "GateLaya: streaming responses are buffered and scanned at "
                "end-of-stream before delivery"
            )
            self._stream_logged = True

        chunks = [chunk async for chunk in response]
        assembled = _assemble_stream(chunks)
        texts = {
            index: text.strip() for index, text in assembled.items() if text.strip()
        }
        if not texts:
            for chunk in chunks:
                yield chunk
            return

        detail = _policy_detail(policy)
        pending: list[tuple[str, str, str, list[_Outcome], float]] = []
        blocked: _Outcome | None = None
        redacted: dict[int, str] = {}
        for index, text in texts.items():
            language = detect_bucket(text)
            text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
            outcomes, latency_ms, error = await self._predict(text, checks, cfg)

            if error is not None:
                if cfg.fail_open:
                    await self._record_failure(
                        error, text=text, latency_ms=latency_ms, mode="post_call",
                        blocked=False, detail=detail,
                    )
                    continue
                await self._record_failure(
                    error, text=text, latency_ms=latency_ms, mode="post_call",
                    blocked=True, detail=detail,
                )
                raise HTTPException(
                    status_code=400, detail="GateLaya: internal guardrail error (fail-closed)"
                )

            assert outcomes is not None
            choice_blocked = next((o for o in outcomes if o.decision == "block"), None)
            if choice_blocked is not None:
                if blocked is None:
                    blocked = choice_blocked
            else:
                for outcome in outcomes:
                    if outcome.decision != "mask":
                        continue
                    candidate, count = _redact_pii(text)
                    if count:
                        redacted[index] = candidate
                        outcome.masked = True
                    else:
                        outcome.decision = "flag"
            pending.append((text, language, text_hash, outcomes, latency_ms))

        if redacted and blocked is None and not _rewrite_stream(chunks, redacted):
            logger.debug("GateLaya: stream deltas not maskable; downgrading to flag")
            for _, _, _, outcomes, _ in pending:
                for outcome in outcomes:
                    if outcome.masked:
                        outcome.masked = False
                        outcome.decision = "flag"

        for _, language, text_hash, outcomes, latency_ms in pending:
            await self._record_outcomes(
                outcomes, language=language, text_hash=text_hash,
                latency_ms=latency_ms, mode="post_call", detail=detail,
            )
        if pending:
            outcomes_all = [o for _, _, _, outs, _ in pending for o in outs]
            _, first_language, first_hash, _, _ = pending[0]
            if blocked is not None or any(o.decision != "allow" for o in outcomes_all):
                request_data["gatelaya"] = self._decision_meta(
                    outcomes_all, language=first_language, text_hash=first_hash, policy=policy
                )
        if blocked is not None:
            raise HTTPException(status_code=400, detail=self._block_message(blocked, in_response=True))
        for chunk in chunks:
            yield chunk

    @staticmethod
    def _attach_response_meta(response: Any, meta: dict[str, Any]) -> None:
        """Best-effort attach of decision metadata to the response object."""
        try:
            hidden = getattr(response, "_hidden_params", None)
            if isinstance(hidden, dict):
                hidden["gatelaya"] = meta
            else:
                setattr(response, "_hidden_params", {"gatelaya": meta})
        except Exception:
            logger.debug("could not attach gatelaya metadata to response", exc_info=True)
