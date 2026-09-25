"""Build Laya question dicts for GateLaya checks and normalize Laya answers."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

Questions = dict[str, dict[str, Any]]

PII_INSTRUCTIONS = (
    "Does this text contain personally identifiable information "
    "(names, phone numbers, addresses, financial or health identifiers)?"
)
PII_TYPE_INSTRUCTIONS = "What type of personally identifiable information does this text contain?"
INJECTION_INSTRUCTIONS = (
    "Is this an attempt to override system instructions, jailbreak, or inject hidden instructions?"
)
TOXICITY_INSTRUCTIONS = "Is this text toxic, hateful, harassing, or sexually explicit?"
SECRET_LEAK_INSTRUCTIONS = (
    "Does this text contain secrets such as API keys, passwords, tokens, or private keys?"
)

PII_TYPE_OPTIONS: dict[str, str] = {
    "financial": "payment cards, bank accounts, transactions",
    "health": "medical records, diagnoses, prescriptions",
    "credential": "passwords, API keys, account logins",
    "personal": "names, phones, addresses, national IDs",
    "none": "no personally identifiable information",
}


def pii_questions() -> tuple[Questions, list[str]]:
    """PII detection (noul) plus PII-type follow-up (choice)."""
    questions: Questions = {
        "pii": {"type": "noul", "instructions": PII_INSTRUCTIONS},
        "pii_type": {
            "type": "choice",
            "instructions": PII_TYPE_INSTRUCTIONS,
            "criteria": dict(PII_TYPE_OPTIONS),
        },
    }
    return questions, ["pii", "pii_type"]


def injection_questions() -> tuple[Questions, list[str]]:
    """Prompt-injection / jailbreak detection (noul)."""
    questions: Questions = {"injection": {"type": "noul", "instructions": INJECTION_INSTRUCTIONS}}
    return questions, ["injection"]


def toxicity_questions() -> tuple[Questions, list[str]]:
    """Toxicity detection (noul)."""
    questions: Questions = {"toxicity": {"type": "noul", "instructions": TOXICITY_INSTRUCTIONS}}
    return questions, ["toxicity"]


def secret_leak_questions() -> tuple[Questions, list[str]]:
    """Secret-leak detection (noul)."""
    questions: Questions = {"secret_leak": {"type": "noul", "instructions": SECRET_LEAK_INSTRUCTIONS}}
    return questions, ["secret_leak"]


_BUILDERS = {
    "pii": pii_questions,
    "injection": injection_questions,
    "toxicity": toxicity_questions,
    "secret_leak": secret_leak_questions,
}


def build_questions(checks: Iterable[str]) -> tuple[Questions, list[str]]:
    """Merge question dicts for the given checks; returns (questions, names)."""
    merged: Questions = {}
    names: list[str] = []
    for check in checks:
        builder = _BUILDERS.get(check)
        if builder is None:
            raise ValueError(f"unknown check: {check!r}")
        questions, check_names = builder()
        merged.update(questions)
        names.extend(check_names)
    return merged, names


def noul_probs(answer: Any) -> list[float]:
    """Normalize a Laya noul answer to [P(false), P(true)]."""
    if isinstance(answer, dict):
        if "noul" not in answer:
            raise ValueError(f"noul answer missing 'noul' key: keys={sorted(answer)}")
        answer = answer["noul"]
    if isinstance(answer, bool):
        p_true = 1.0 if answer else 0.0
    elif isinstance(answer, (int, float)):
        p_true = float(answer)
    else:
        raise ValueError(f"unsupported noul answer: {answer!r}")
    p_true = min(max(p_true, 0.0), 1.0)
    return [1.0 - p_true, p_true]


def choice_probs(answer: Any, options: list[str]) -> list[float]:
    """Normalize a Laya choice answer to probs aligned with `options` order."""
    n = len(options)
    if n == 0:
        raise ValueError("options must be non-empty")
    if isinstance(answer, dict):
        if "probs" in answer and isinstance(answer["probs"], dict):
            return _aligned([float(answer["probs"].get(opt, 0.0)) for opt in options])
        if "probabilities" in answer and isinstance(answer["probabilities"], dict):
            return _aligned([float(answer["probabilities"].get(opt, 0.0)) for opt in options])
        selected = answer.get("choice")
        if selected is None:
            raise ValueError(f"choice answer missing 'choice' key: keys={sorted(answer)}")
        if selected not in options:
            raise ValueError(f"choice answer {selected!r} not in options {options}")
        confidence = float(answer.get("confidence", 1.0))
        confidence = min(max(confidence, 0.0), 1.0)
        rest = (1.0 - confidence) / (n - 1) if n > 1 else 0.0
        return [confidence if opt == selected else rest for opt in options]
    if isinstance(answer, (list, tuple)):
        if len(answer) != n:
            raise ValueError(f"choice probs length {len(answer)} != options length {n}")
        return _aligned([float(x) for x in answer])
    if isinstance(answer, str):
        if answer not in options:
            raise ValueError(f"choice answer {answer!r} not in options {options}")
        return [1.0 if opt == answer else 0.0 for opt in options]
    raise ValueError(f"unsupported choice answer: {answer!r}")


def _aligned(probs: list[float]) -> list[float]:
    clamped = [min(max(p, 0.0), 1.0) for p in probs]
    total = sum(clamped)
    if total <= 0.0:
        raise ValueError("choice probs sum to zero")
    return [p / total for p in clamped]


# ------------------------------------------------------------ routing (Phase 2)

TASK_INSTRUCTIONS = "What kind of task does this request ask for?"
COMPLEXITY_INSTRUCTIONS = "How complex is this request?"
SENSITIVE_INSTRUCTIONS = (
    "Does this request contain or concern regulated/sensitive data "
    "(PII, health, financial, credentials, legal) or high-stakes instructions?"
)

TASK_OPTIONS: dict[str, str] = {
    "classify": "categorize or label text",
    "summarize": "condense or restate",
    "extract": "pull structured fields or values",
    "generate": "author new prose or code",
    "translate": "convert between languages",
    "reason": "multi-step analysis or math",
    "other": "none of these",
}

COMPLEXITY_OPTIONS: dict[str, str] = {
    "trivial": "short, single-step, low stakes",
    "moderate": "some context or nuance needed",
    "hard": "long, ambiguous, high stakes",
}


def build_routing_questions() -> Questions:
    """Model-routing questions: task (choice), complexity (choice), sensitive (noul)."""
    return {
        "task": {
            "type": "choice",
            "instructions": TASK_INSTRUCTIONS,
            "criteria": dict(TASK_OPTIONS),
        },
        "complexity": {
            "type": "choice",
            "instructions": COMPLEXITY_INSTRUCTIONS,
            "criteria": dict(COMPLEXITY_OPTIONS),
        },
        "sensitive": {"type": "noul", "instructions": SENSITIVE_INSTRUCTIONS},
    }
