"""Tests for question building and Laya answer normalizers (questions.py)."""

from __future__ import annotations

from typing import Any

import pytest

from gatelaya.questions import (
    PII_TYPE_OPTIONS,
    build_questions,
    choice_probs,
    injection_questions,
    noul_probs,
    pii_questions,
    secret_leak_questions,
    toxicity_questions,
)

ALL_CHECKS = ["pii", "injection", "toxicity", "secret_leak"]
PII_OPTS = list(PII_TYPE_OPTIONS)


# ------------------------------------------------------------ build_questions


def test_build_questions_all_checks() -> None:
    """Arrange/act: all checks. Assert: every question present, pii_type included."""
    questions, names = build_questions(ALL_CHECKS)
    assert set(questions) == {"pii", "pii_type", "injection", "toxicity", "secret_leak"}
    assert set(names) == set(questions)


def test_build_questions_types_are_noul_or_choice() -> None:
    """Arrange/act: all checks. Assert: type field within the allowed set."""
    questions, _ = build_questions(ALL_CHECKS)
    for name, question in questions.items():
        assert question["type"] in {"noul", "choice"}, f"{name}: {question['type']}"
        assert question.get("instructions"), f"{name} missing instructions"


def test_build_questions_only_requested_checks() -> None:
    """Arrange/act: subset ['toxicity']. Assert: no other checks leak in."""
    questions, names = build_questions(["toxicity"])
    assert set(questions) == {"toxicity"}
    assert names == ["toxicity"]


def test_build_questions_disabled_checks_absent() -> None:
    """Arrange/act: injection only (e.g. enabled_checks restriction).
    Assert: pii/toxicity/secret_leak questions absent."""
    questions, _ = build_questions(["injection"])
    for absent in ["pii", "pii_type", "toxicity", "secret_leak"]:
        assert absent not in questions


def test_build_questions_choice_option_count_within_limit() -> None:
    """Arrange/act: checks including pii_type. Assert: choice <= 20 options (PRODUCT constraint)."""
    questions, _ = build_questions(["pii"])
    criteria = questions["pii_type"]["criteria"]
    assert questions["pii_type"]["type"] == "choice"
    assert len(criteria) <= 20
    assert set(criteria) == set(PII_TYPE_OPTIONS)


def test_build_questions_unknown_check_raises() -> None:
    """Arrange/act: unknown check name. Assert: ValueError."""
    with pytest.raises(ValueError, match="unknown check"):
        build_questions(["hallucination"])


def test_build_questions_empty_input() -> None:
    """Arrange/act: no checks. Assert: empty questions and names."""
    questions, names = build_questions([])
    assert questions == {}
    assert names == []


def test_individual_builders_return_expected_shape() -> None:
    """Arrange/act: each builder. Assert: name list matches question keys."""
    for builder, expected in [
        (pii_questions, ["pii", "pii_type"]),
        (injection_questions, ["injection"]),
        (toxicity_questions, ["toxicity"]),
        (secret_leak_questions, ["secret_leak"]),
    ]:
        questions, names = builder()
        assert sorted(names) == sorted(questions)
        assert sorted(names) == sorted(expected)


# ---------------------------------------------------------------- noul_probs


def test_noul_probs_dict_answer() -> None:
    """Arrange/act: {'noul': 0.7}. Assert: [P(false), P(true)]."""
    assert noul_probs({"noul": 0.7}) == pytest.approx([0.3, 0.7])


def test_noul_probs_bool_and_number_answers() -> None:
    """Arrange/act: bool/int/float shapes. Assert: normalized pairs."""
    assert noul_probs(True) == [0.0, 1.0]
    assert noul_probs(False) == [1.0, 0.0]
    assert noul_probs(1) == [0.0, 1.0]
    assert noul_probs(0.25) == pytest.approx([0.75, 0.25])


def test_noul_probs_clamps_out_of_range() -> None:
    """Arrange/act: values outside [0,1]. Assert: clamped, still sums to 1."""
    assert noul_probs(1.5) == pytest.approx([0.0, 1.0])
    assert noul_probs(-0.2) == pytest.approx([1.0, 0.0])


@pytest.mark.parametrize(
    "answer",
    [
        {"confidence": 0.9},  # dict missing 'noul' key
        "yes",  # unsupported type
        None,  # unsupported type
        {"noul": "high"},  # non-numeric noul value
    ],
)
def test_noul_probs_garbage_raises_value_error(answer: Any) -> None:
    """Arrange/act: garbage input. Assert: ValueError with useful message."""
    with pytest.raises(ValueError):
        noul_probs(answer)


# -------------------------------------------------------------- choice_probs


def test_choice_probs_selected_with_confidence() -> None:
    """Arrange/act: {'choice': 'personal', 'confidence': 0.8}.
    Assert: selected gets confidence, rest share remainder, sums to 1."""
    probs = choice_probs({"choice": "personal", "confidence": 0.8}, PII_OPTS)
    assert probs[PII_OPTS.index("personal")] == pytest.approx(0.8)
    assert sum(probs) == pytest.approx(1.0)
    rest = 0.2 / (len(PII_OPTS) - 1)
    for idx, opt in enumerate(PII_OPTS):
        if opt != "personal":
            assert probs[idx] == pytest.approx(rest)


def test_choice_probs_selected_defaults_confidence_to_one() -> None:
    """Arrange/act: choice without confidence. Assert: one-hot."""
    probs = choice_probs({"choice": "none"}, PII_OPTS)
    assert probs == [1.0 if o == "none" else 0.0 for o in PII_OPTS]


def test_choice_probs_bare_string_one_hot() -> None:
    """Arrange/act: plain label string. Assert: one-hot vector."""
    probs = choice_probs("financial", PII_OPTS)
    assert probs[PII_OPTS.index("financial")] == 1.0
    assert sum(probs) == 1.0


def test_choice_probs_probs_dict_normalized() -> None:
    """Arrange/act: {'probs': {...}} with out-of-range weights.
    Assert: entries clamped to [0,1] first, then renormalized — option order
    preserved."""
    probs = choice_probs({"probs": {"personal": 3.0, "none": 1.0}}, PII_OPTS)
    # 3.0 clamps to 1.0 -> [1.0, 1.0] -> equal split after normalization
    assert sum(probs) == pytest.approx(1.0)
    assert probs[PII_OPTS.index("personal")] == pytest.approx(0.5)
    assert probs[PII_OPTS.index("none")] == pytest.approx(0.5)
    assert probs[PII_OPTS.index("financial")] == pytest.approx(0.0)


def test_choice_probs_probabilities_key_alias() -> None:
    """Arrange/act: {'probabilities': {...}} alias. Assert: same alignment."""
    probs = choice_probs({"probabilities": {"health": 2.0, "financial": 2.0}}, PII_OPTS)
    assert sum(probs) == pytest.approx(1.0)
    assert probs[PII_OPTS.index("health")] == pytest.approx(0.5)


def test_choice_probs_list_and_tuple_input() -> None:
    """Arrange/act: raw list/tuple of probs (already in [0,1]).
    Assert: aligned + normalized to sum 1 in option order."""
    opts = ["a", "b", "c"]
    assert choice_probs([0.2, 0.2, 0.4], opts) == pytest.approx([0.25, 0.25, 0.5])
    assert choice_probs((0.4, 0.2, 0.2), opts) == pytest.approx([0.5, 0.25, 0.25])


def test_choice_probs_rejects_bad_lengths_and_labels() -> None:
    """Arrange/act: wrong-length list, unknown label, empty options.
    Assert: ValueError each time."""
    opts = ["a", "b"]
    with pytest.raises(ValueError, match="length"):
        choice_probs([0.5, 0.3, 0.2], opts)
    with pytest.raises(ValueError, match="not in options"):
        choice_probs("z", opts)
    with pytest.raises(ValueError, match="not in options"):
        choice_probs({"choice": "z"}, opts)
    with pytest.raises(ValueError, match="non-empty"):
        choice_probs("a", [])


def test_choice_probs_garbage_raises_value_error() -> None:
    """Arrange/act: unsupported answer shapes. Assert: ValueError."""
    opts = ["a", "b"]
    with pytest.raises(ValueError, match="unsupported choice"):
        choice_probs(42, opts)
    with pytest.raises(ValueError, match="missing 'choice'"):
        choice_probs({"confidence": 0.5}, opts)
    with pytest.raises(ValueError, match="sum to zero"):
        choice_probs({"probs": {"a": 0.0, "b": 0.0}}, opts)
