"""Two-tier QA patterns and honest split helpers for the GateLaya eval dataset.

Two tiers per check:

* **STRICT** — high-precision regex/phrases. Must score **zero** hits on
  negative rows (after placeholder blanking). A hit means either a real
  mislabel or a pattern bug; both are surfaced, never auto-fixed.
* **MARKERS** — STRICT plus broad multilingual keyword lists. Used only to
  measure positive-row *coverage* (does every positive contain at least one
  detectable signal?). Thresholds: pii >= 0.85, secret_leak >= 0.90,
  injection >= 0.90. Toxicity has no regex QA (documented limitation).

Negative blanking replaces placeholder-looking values (``your``,
``example``, ``xxxx``, ``-key-here``, ...) and ``<...>`` spans before STRICT
scans, so fixture negatives like ``sk-YOUR-KEY-HERE`` are not false friends.

All QA is read-only reporting: this module never edits labels.
"""

from __future__ import annotations

import random
import re
from collections.abc import Iterable
from typing import Any

CHECKS: tuple[str, ...] = ("pii", "injection", "toxicity", "secret_leak")

#: Minimum positive coverage (marker tier) required to pass QA.
COVERAGE_THRESHOLDS: dict[str, float] = {
    "pii": 0.85,
    "secret_leak": 0.90,
    "injection": 0.90,
}

SPLIT_RATIOS: tuple[float, float, float] = (0.70, 0.15, 0.15)
DEFAULT_SEED = 42
MIN_TEST_PER_CHECK = 20

# --------------------------------------------------------------- blanking

_PLACEHOLDER_RE = re.compile(
    r"(your|xxxx+|example|test|placeholder|dummy|sample|fake|demo|tu-|votre|-key-here)",
    re.IGNORECASE,
)
_ANGLE_SPAN_RE = re.compile(r"<[^<>]*>")
_PII_CONTEXT_RE = re.compile(
    r"(isbn|barcode|ean|upc)([\s:\-]*)(\d[\d\-. ]*)", re.IGNORECASE
)
#: Google's publicly documented reCAPTCHA test secret (deliberate negative fixture).
_FIXTURE_VALUE_RE = re.compile(r"6LeIxAcT\S*", re.IGNORECASE)


def _blank(text: str, check: str) -> str:
    """Neutralize placeholder values before a STRICT scan of a negative row."""
    out = _PLACEHOLDER_RE.sub("x", text)
    out = _ANGLE_SPAN_RE.sub("<x>", out)
    out = _FIXTURE_VALUE_RE.sub("x", out)
    if check == "pii":
        out = _PII_CONTEXT_RE.sub(r"\1 x", out)
    return out


# ------------------------------------------------------------ STRICT tier

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
# Digit runs with optional separators (spaces, dashes, parens), bounded by
# non-alnum on both sides and not starting right after a hyphen: blocks
# "ORD-2024-55123", "LX449201188NP", "192.168.1.42", ISO dates, while
# catching "9841234567", "555-867-5309", "4111 1111 1111 1111".
_SQUASHED_RUN_RE = re.compile(r"(?<![A-Za-z0-9\-])[\d\-()\s]{6,}(?![A-Za-z0-9])")
_NON_DIGIT_RE = re.compile(r"\D+")


def _squashed_hit(text: str) -> bool:
    """True if any bounded digit run (separators stripped) has 9-16 digits."""
    for match in _SQUASHED_RUN_RE.finditer(text):
        digits = _NON_DIGIT_RE.sub("", match.group())
        if 9 <= len(digits) <= 16:
            return True
    return False
_PHONE_RE = re.compile(r"\+\d[\d\s\-().]{7,}\d")
_PASSPORT_RE = re.compile(r"\b[A-Z]{1,2}\d{6,9}\b")
_IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ -]?[A-Z0-9]{2,4}){5,}\b")
_DOB_ISO_RE = re.compile(r"\b(?:19|20)\d{2}-\d{2}-\d{2}\b")
_DOB_SLASH_RE = re.compile(r"\b\d{2}/\d{2}/(?:19|20)\d{2}\b")
_DEVANAGARI_RE = re.compile(r"[०-९]{10,15}")

_PII_STRICT_RES: tuple[re.Pattern[str], ...] = (
    _EMAIL_RE,
    _SSN_RE,
    _PHONE_RE,
    _PASSPORT_RE,
    _IBAN_RE,
    _DOB_ISO_RE,
    _DOB_SLASH_RE,
    _DEVANAGARI_RE,
)

_SECRET_ASSIGN_KEYWORD = (
    r"(?:password|passwd|passphrase|पासवर्ड|secret(?:[_-]?key)?|api[_-]?key|"
    r"client[_-]?secret|access[ _-]?key|jwt[_-]?secret)"
)
_SECRET_STRICT_RES: tuple[re.Pattern[str], ...] = tuple(
    # IGNORECASE: assignment keywords like "API_KEY=" must match any casing.
    re.compile(p, re.IGNORECASE)
    for p in (
        r"sk-[A-Za-z0-9][A-Za-z0-9\-]{15,}",
        r"sk_(?:live|test)_[A-Za-z0-9]{16,}",
        r"\bAKIA[0-9A-Z]{16}\b",
        r"\bghp_[A-Za-z0-9]{20,}",
        r"\bgithub_pat_[A-Za-z0-9_]{20,}",
        r"\bAIza[0-9A-Za-z_\-]{30,}",
        r"\bxox[bpare]-[A-Za-z0-9\-]{10,}",
        r"\bhf_[A-Za-z0-9]{30,}",
        r"\bnpm_[A-Za-z0-9]{30,}",
        r"\bglpat-[A-Za-z0-9\-_]{16,}",
        r"\bSG\.[A-Za-z0-9_\-]{16,}\.",
        r"\bwhsec_[A-Za-z0-9]{16,}",
        r"\bhvs\.[A-Za-z0-9]{16,}",
        r"\bya29\.[0-9A-Za-z_\-]{20,}",
        r"\bAC[0-9a-f]{32}\b",
        r"\bpat-[0-9a-f]{20,}",
        r"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY",
        r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.",
        r"://[^\s/:@]+:[^\s@]+@",
        r"\bBasic [A-Za-z0-9+/=]{12,}",
        rf"{_SECRET_ASSIGN_KEYWORD}\s*[=:]\s*\S{{6,}}",
        r"password\s+is\s+\S{6,}",
        r"password\s+for\s+[^:]{0,30}:\s*\S{6,}",
        r"login\s+is\s+\S+\s*/\s*\S+",
        r"api[_-]?secret\s+[A-Za-z0-9]{16,}",
        r"api[_-]?key\s+for\s+\S+\s+is\s+[A-Za-z0-9]{12,}",
    )
)

#: Injection phrases, matched on normalized text (see ``_normalize``).
_INJECTION_PHRASES: tuple[str, ...] = (
    "ignore all previous instructions",
    "ignore previous instructions",
    "ignore the previous instructions",
    "ignore prior instructions",
    "ignore above instructions",
    "ignore earlier instructions",
    "ignore all previous prompts",
    "ignore previous prompts",
    "ignore prior prompts",
    "ignore all previous messages",
    "ignore previous messages",
    "ignoreyourrules",
    "enter developer mode",
    "enter jailbreak mode",
    "enter dan mode",
    "enter maintenance mode",
    "developer mode",
    "jailbreak mode",
    "no restrictions",
    "without restrictions",
    "unrestricted",
    "disable all safety",
    "disable safety",
    "disable all filters",
    "disable filters",
    "disable all policies",
    "disable policies",
    "disable guardrails",
    "disable all guardrails",
    "disable_filters",
    "system override",
    "override safety",
    "disregard your",
    "disregard the",
    "disregard all",
    "pretend you are",
    "pretend you be",
    "pretend to are",
    "pretend to be",
    "do anything now",
    "dan mode",
    "you are dan",
    "obey only me",
    "obeyme",
    "follow mine",
    "hidden prompt",
    "hidden instructions",
    "never refuse",
    "restrictions are now off",
    "instructions instead",
    "new instructions",
    "instruction update",
    "simulate being",
    "fais semblant",
    "unaligned",
    "unfiltered",
)

_LEET_MAP = str.maketrans(
    {"1": "i", "3": "e", "0": "o", "4": "a", "5": "s", "7": "t", "@": "a", "$": "s", "!": "i"}
)


def _normalize(text: str) -> str:
    """Lowercase + leet-map + strip whitespace, for injection phrase matching."""
    return re.sub(r"\s+", "", text.lower().translate(_LEET_MAP))


#: Phrases pre-normalized once (``_normalize`` collapses whitespace).
_INJECTION_PHRASES_NORM: tuple[str, ...] = tuple(_normalize(p) for p in _INJECTION_PHRASES)


def matches_strict(text: str, check: str) -> bool:
    """True if ``text`` trips the STRICT tier for ``check`` (scan negatives with this)."""
    if check == "pii":
        if _squashed_hit(text):
            return True
        return any(p.search(text) for p in _PII_STRICT_RES)
    if check == "secret_leak":
        return any(p.search(text) for p in _SECRET_STRICT_RES)
    if check == "injection":
        norm = _normalize(text)
        return any(phrase in norm for phrase in _INJECTION_PHRASES_NORM)
    return False  # toxicity: no strict tier (documented limitation)


# ------------------------------------------------------------ MARKERS tier

_PII_MARKERS: tuple[str, ...] = (
    # en
    "email", "phone", "ssn", "passport", "iban", "account", "name", "badge",
    "waiver", "registrant", "contact person", "reservation", "member id",
    "insurance", "mrn", "street", "avenue", "road", "lane", "drive", "apt",
    "date of birth", "dob", "address", "terrace", "house",
    "card", "visa", "mastercard", "amex", "nhs", "aadhaar", "citizenship",
    # es
    "correo", "teléfono", "telefono", "nombre", "calle", "dni",
    # fr
    "téléphone", "telephone", "adresse", "appelle", "sécurité sociale", "securite sociale",
    # de
    "telefon", "straße", "strasse", "ausweis",
    # np / hi
    "फोन", "नम्बर", "नाम", "ठेगाना", "नागरिकता", "खाता", "नंबर", "पता", "आधार",
    # ar
    "اسم", "الاسم", "الهاتف", "العنوان", "البريد", "شارع",
)

_SECRET_MARKERS: tuple[str, ...] = (
    "password", "passphrase", "api key", "api_key", "api_secret", "api secret",
    "token", "secret", "private key", "login is", "backup code", "sticky note",
    "encryption key", "credential", "key is", "serverpasswort", "key",
    "dsn", "पासवर्ड", "clé", "mot de passe", "contraseña", "كلمة المرور",
)

_INJECTION_MARKERS: tuple[str, ...] = (
    # en
    "ignore", "instructions", "jailbreak", "system prompt", "system message",
    "override", "disregard", "pretend", "developer mode", "restrictions",
    "obey", "unrestricted", "unfiltered", "hidden prompt", "prompt injection",
    "verbatim", "bypass", "act as if", "previous context", "following commands",
    "without filtering", "usage policy", "roleplay", "content policy",
    "content checks", "guidelines", "say anything", "anything without",
    "pentest", "refusal", "every answer", "no exceptions", "obedience",
    "operator", "without ethics", "do anything now",
    # es
    "instrucciones", "ignora", "restricciones", "filtros", "obedecer",
    "modo desarrollador", "límites",
    # fr
    "ignorez", "règle", "filtres", "désactive", "restriction",
    # de
    "anweisungen", "ignoriere", "regeln", "sicherheitsfilter", "deaktiviere",
    "entwicklermodus",
    # np / hi
    "निर्देश", "अनदेखा", "नियम", "सुरक्षा", "परीक्षण",
    # ar
    "تجاهل", "تعليمات", "قيود", "فلاتر", "عطّل",
)

_MARKER_KEYWORDS: dict[str, tuple[str, ...]] = {
    "pii": _PII_MARKERS,
    "secret_leak": _SECRET_MARKERS,
    "injection": _INJECTION_MARKERS,
    "toxicity": (),
}


def matches_markers(text: str, check: str) -> bool:
    """True if a positive row contains any detectable signal (STRICT or keyword)."""
    if matches_strict(text, check):
        return True
    if check == "toxicity":
        return True  # no coverage requirement
    lowered = text.lower()
    return any(kw in lowered for kw in _MARKER_KEYWORDS[check])


# ------------------------------------------------------------- row / scan


def row_issues(row: dict[str, Any]) -> list[str]:
    """Structural + tier issues for one dataset row (empty list = clean)."""
    issues: list[str] = []
    check = row.get("check")
    if check not in CHECKS:
        issues.append("unknown_check")
    label = row.get("label")
    if label not in (0, 1, False, True):
        issues.append("bad_label")
    text = row.get("text")
    if not isinstance(text, str) or not text.strip():
        issues.append("empty_text")
    if not isinstance(row.get("id"), str) or not row.get("id"):
        issues.append("missing_id")
    if not isinstance(row.get("lang"), str) or not row.get("lang"):
        issues.append("missing_lang")
    if issues:
        return issues
    assert isinstance(text, str) and isinstance(check, str)
    if int(label) == 1 and check in COVERAGE_THRESHOLDS and not matches_markers(text, check):
        issues.append("positive_misses_markers")
    if int(label) == 0 and matches_strict(_blank(text, check), check):
        issues.append("negative_strict_hit")
    return issues


def qa_scan(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Full two-tier QA report. Read-only: never modifies rows or labels."""
    rows = list(rows)
    per_check: dict[str, Any] = {}
    for check in CHECKS:
        positives = [r for r in rows if r.get("check") == check and r.get("label") in (1, True)]
        negatives = [r for r in rows if r.get("check") == check and r.get("label") in (0, False)]
        strict_hit_ids = [
            r["id"] for r in negatives if matches_strict(_blank(r["text"], check), check)
        ]
        entry: dict[str, Any] = {
            "positives": len(positives),
            "negatives": len(negatives),
            "strict_hits": len(strict_hit_ids),
            "strict_hit_ids": strict_hit_ids,
            "rows_with_issues": [r["id"] for r in rows if r.get("check") == check and row_issues(r)],
        }
        threshold = COVERAGE_THRESHOLDS.get(check)
        if threshold is None:
            entry.update({"marker_hits": None, "coverage": None, "threshold": None, "misses": []})
        else:
            misses = [r["id"] for r in positives if not matches_markers(r["text"], check)]
            coverage = (len(positives) - len(misses)) / len(positives) if positives else 0.0
            entry.update(
                {
                    "marker_hits": len(positives) - len(misses),
                    "coverage": round(coverage, 4),
                    "threshold": threshold,
                    "misses": misses,
                }
            )
        per_check[check] = entry

    passed = all(
        e["strict_hits"] == 0
        and (e["coverage"] is None or (e["coverage"] >= e["threshold"] and e["positives"] > 0))
        for e in per_check.values()
    )
    return {
        "totals": {
            "rows": len(rows),
            "by_check": {c: sum(1 for r in rows if r.get("check") == c) for c in CHECKS},
            "positives": sum(1 for r in rows if r.get("label") in (1, True)),
            "negatives": sum(1 for r in rows if r.get("label") in (0, False)),
        },
        "per_check": per_check,
        "passed": passed,
    }


# -------------------------------------------------------------- splitting


def split_dataset(
    rows: Iterable[dict[str, Any]],
    *,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = SPLIT_RATIOS,
    min_test_per_check: int = MIN_TEST_PER_CHECK,
) -> dict[str, list[dict[str, Any]]]:
    """Deterministic stratified 70/15/15 split by (check, label).

    Guarantees: same seed -> identical output; every row in exactly one
    split; each (check, label) group contributes to all three splits; each
    check gets at least ``min_test_per_check`` test rows when possible.
    """
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"ratios must sum to 1.0, got {ratios}")
    rows = list(rows)
    rng = random.Random(seed)

    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in sorted(rows, key=lambda r: str(r.get("id", ""))):
        groups.setdefault((str(row["check"]), int(row["label"])), []).append(row)

    train: list[dict[str, Any]] = []
    val: list[dict[str, Any]] = []
    test: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = list(groups[key])
        rng.shuffle(group)
        n = len(group)
        n_test = max(1, round(n * ratios[2]))
        n_val = max(1, round(n * ratios[1]))
        if n >= 2 and n_test + n_val >= n:
            n_test, n_val = 1, 1
        if n == 1:
            n_test = n_val = 0
        test.extend(group[:n_test])
        val.extend(group[n_test : n_test + n_val])
        train.extend(group[n_test + n_val :])

    # Enforce minimum test rows per check by borrowing from train
    # (round-robin over labels; keep >= 1 train row per (check, label)).
    for check in CHECKS:
        while True:
            deficit = min_test_per_check - sum(1 for r in test if r["check"] == check)
            if deficit <= 0:
                break
            moved = False
            for label in (0, 1):
                if deficit <= 0:
                    break
                pool = [r for r in train if r["check"] == check and int(r["label"]) == label]
                if len(pool) <= 1:
                    continue
                victim = pool[-1]
                train.remove(victim)
                test.append(victim)
                deficit -= 1
                moved = True
            if not moved:
                break  # cannot borrow more without emptying a train group

    for split in (train, val, test):
        rng.shuffle(split)
    return {"train": train, "val": val, "test": test}
