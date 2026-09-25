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
from typing import Any

from fastapi import HTTPException

from .agent import LayaAgent, LayaRouterAgent, detect_bucket
from .audit import AuditSink, DecisionRecord, InMemoryAuditSink, SqlAlchemyAuditSink
from .calibration import TemperatureMap, calibrated, load_temperature_map
from .config import ALL_CHECKS, GateLayaConfig
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
PRE_CHECK_ORDER = ("pii", "injection", "toxicity", "secret_leak")
POST_CHECK_ORDER = ("secret_leak", "toxicity")
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


def _response_text(response: Any) -> str:
    """Extract text from a LiteLLM/OpenAI-style response object or dict."""
    if isinstance(response, dict):
        choices = response.get("choices") or []
        if not choices:
            return ""
        first = choices[0]
        if isinstance(first, dict):
            if "message" in first:
                message = first.get("message") or {}
                content = message.get("content") if isinstance(message, dict) else None
            else:
                content = first.get("text")
        else:
            try:
                content = first.message.content
            except AttributeError:
                content = getattr(first, "text", None)
        return _content_to_text(content)
    try:
        content = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError):
        try:
            return _content_to_text(response.choices[0].text)
        except (AttributeError, IndexError, TypeError):
            return ""
    return _content_to_text(content)


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
    return GateLayaConfig(**data)


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
        """Build the guardrail; omitted collaborators default to env/lazy deployments."""
        config_path = kwargs.pop("config_path", None) or os.getenv("GATELAYA_CONFIG_PATH")
        thresholds_override = kwargs.pop("thresholds", None)
        actions_override = kwargs.pop("actions", None)
        audit_url = kwargs.pop("audit_database_url", None) or os.getenv("GATELAYA_DATABASE_URL")

        env_path = config_path or os.getenv("GATELAYA_CONFIG_PATH")
        if config is None:
            config = GateLayaConfig.from_yaml(env_path) if env_path else GateLayaConfig()
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
            )
        if audit is None:
            audit = SqlAlchemyAuditSink(audit_url) if audit_url else InMemoryAuditSink()
        if temperatures is None and config.calibration_path is not None:
            temperatures = load_temperature_map(config.calibration_path)

        self.config = config
        self.agent = agent
        self.audit = audit
        self.temperatures = temperatures
        self._stream_logged = False

        guardrail_name = kwargs.pop("guardrail_name", None) or "gatelaya"
        event_hook = kwargs.pop("event_hook", None) or list(config.mode)
        default_on = kwargs.pop("default_on", True)
        super().__init__(
            guardrail_name=guardrail_name,
            event_hook=event_hook,
            default_on=default_on,
            **kwargs,
        )

    # ------------------------------------------------------------------ helpers

    def _temperature(self, question_type: str, option_count: int | None = None) -> float:
        if self.temperatures is None:
            return 1.0
        return self.temperatures.temperature_for(question_type, option_count)

    async def _predict(
        self, text: str, checks: list[str]
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
            outcomes = [self._outcome(check, answers, questions) for check in checks]
        except Exception as exc:  # malformed answers must fail open, not crash the proxy
            logger.warning("GateLaya: could not evaluate agent answers: %s", exc)
            return None, latency_ms, exc
        return outcomes, latency_ms, None

    def _outcome(self, check: str, answers: dict[str, Any], questions: dict[str, Any]) -> _Outcome:
        """Evaluate one check's answer against its threshold and configured action."""
        threshold = self.config.threshold(check)
        configured = self.config.action(check)
        answer = answers.get(check)
        if answer is None:
            logger.warning("GateLaya: no answer for check %r; treating as allow", check)
            vector = [1.0, 0.0]
        else:
            vector = noul_probs(answer)
        cal = calibrated(vector, self._temperature("noul"))
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
        latency_ms: float, mode: str,
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
                )
            )

    async def _record_failure(
        self, exc: Exception, *, text: str, latency_ms: float, mode: str, blocked: bool
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
            )
        )

    @staticmethod
    def _decision_meta(
        outcomes: list[_Outcome], *, language: str, text_hash: str
    ) -> dict[str, Any]:
        """Build the data['gatelaya'] decision metadata payload."""
        return {
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

    @staticmethod
    def _block_message(outcome: _Outcome, *, in_response: bool) -> str:
        label = CHECK_LABELS.get(outcome.check, outcome.check)
        where = " in response" if in_response else ""
        return f"GateLaya: {label} detected{where} (p={outcome.cal_p:.2f})"

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
        self,
        outcomes: list[_Outcome],
        *,
        messages: list[Any] | None = None,
        last_user_idx: int | None = None,
        allow_mask: bool,
    ) -> bool:
        """Downgrade no-op masks to flags; returns True when a mask changed content."""
        mask_outcomes = [o for o in outcomes if o.decision == "mask"]
        if not mask_outcomes or not allow_mask:
            # mask outside the pre-call PII path degrades to flag (documented v1 behavior)
            for outcome in mask_outcomes:
                outcome.decision = "flag"
            return False
        pii_type = next((o.pii_type for o in mask_outcomes if o.check == "pii"), None)
        redactions = self._apply_mask(messages or [], last_user_idx, pii_type)
        if redactions == 0:
            logger.debug("GateLaya: mask action found no regex matches; downgrading to flag")
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
        messages = data.get("messages")
        scan_text, last_user_idx = extract_scan_text(messages)
        if not scan_text:
            return None

        checks = [c for c in PRE_CHECK_ORDER if c in self.config.enabled_checks]
        if not checks:
            return None

        language = detect_bucket(scan_text)
        text_hash = hashlib.sha256(scan_text.encode("utf-8")).hexdigest()
        outcomes, latency_ms, error = await self._predict(scan_text, checks)

        if error is not None:
            if self.config.fail_open:
                await self._record_failure(
                    error, text=scan_text, latency_ms=latency_ms, mode="pre_call", blocked=False
                )
                return None
            await self._record_failure(
                error, text=scan_text, latency_ms=latency_ms, mode="pre_call", blocked=True
            )
            return "GateLaya: internal guardrail error (fail-closed)"

        assert outcomes is not None
        blocked = next((o for o in outcomes if o.decision == "block"), None)
        masked = False
        if blocked is None:
            masked = self._apply_actions(
                outcomes, messages=messages, last_user_idx=last_user_idx, allow_mask=True
            )
        await self._record_outcomes(
            outcomes, language=language, text_hash=text_hash,
            latency_ms=latency_ms, mode="pre_call",
        )

        if blocked is not None:
            data["gatelaya"] = self._decision_meta(outcomes, language=language, text_hash=text_hash)
            return self._block_message(blocked, in_response=False)

        if any(o.decision != "allow" for o in outcomes):
            data["gatelaya"] = self._decision_meta(outcomes, language=language, text_hash=text_hash)
        if masked:
            return data
        return None

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: Any, response: Any
    ) -> Any:
        """Scan the model response for secrets/toxicity; raise 400 on block."""
        text = _response_text(response)
        if not text.strip():
            return None

        checks = [c for c in POST_CHECK_ORDER if c in self.config.enabled_checks]
        if not checks:
            return None

        language = detect_bucket(text)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        outcomes, latency_ms, error = await self._predict(text, checks)

        if error is not None:
            if self.config.fail_open:
                await self._record_failure(
                    error, text=text, latency_ms=latency_ms, mode="post_call", blocked=False
                )
                return None
            await self._record_failure(
                error, text=text, latency_ms=latency_ms, mode="post_call", blocked=True
            )
            raise HTTPException(
                status_code=400, detail="GateLaya: internal guardrail error (fail-closed)"
            )

        assert outcomes is not None
        # v1: response masking has no secret-redaction patterns; mask degrades to flag
        self._apply_actions(outcomes, allow_mask=False)
        await self._record_outcomes(
            outcomes, language=language, text_hash=text_hash,
            latency_ms=latency_ms, mode="post_call",
        )

        blocked = next((o for o in outcomes if o.decision == "block"), None)
        if blocked is not None:
            data["gatelaya"] = self._decision_meta(outcomes, language=language, text_hash=text_hash)
            raise HTTPException(status_code=400, detail=self._block_message(blocked, in_response=True))

        if any(o.decision != "allow" for o in outcomes):
            meta = self._decision_meta(outcomes, language=language, text_hash=text_hash)
            data["gatelaya"] = meta
            self._attach_response_meta(response, meta)
        return None

    async def async_post_call_streaming_iterator_hook(
        self, user_api_key_dict: Any, response: Any, request_data: dict
    ) -> AsyncGenerator[Any, None]:
        """v1: pass streaming chunks through unchanged (streaming checks are audit-only)."""
        if not self._stream_logged:
            logger.info(
                "GateLaya: streaming responses pass through unenforced in v1; "
                "streaming checks are audit-only (documented limitation)"
            )
            self._stream_logged = True
        async for chunk in response:
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
