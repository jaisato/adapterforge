"""Tests de las métricas propias (exact-match, ROUGE-L, extracción JSON)."""

from __future__ import annotations

import pytest

from adapterforge.eval.metrics import (
    contains_match,
    exact_match,
    extract_first_json_object,
    field_exact_match,
    normalize_text,
    rouge_l,
)


def test_normalize_and_exact_match():
    assert normalize_text("  Tres  MESES. ") == "tres meses"
    assert exact_match("Tres Meses", "tres meses")
    assert exact_match("días hábiles", "dias habiles")  # sin tildes
    assert not exact_match("tres meses", "seis meses")


def test_contains_match():
    assert contains_match("El plazo es de tres meses desde la publicación.", "tres meses")
    assert not contains_match("El plazo es de seis meses.", "tres meses")
    assert not contains_match("respuesta", "")


def test_rouge_l_known_values():
    # pred: a b c d / ref: a c d -> LCS=3, P=3/4, R=3/3, F1=2*0.75/1.75
    score = rouge_l("a b c d", "a c d")
    assert score == pytest.approx(2 * 0.75 * 1.0 / 1.75)
    assert rouge_l("igual igual igual", "igual igual igual") == pytest.approx(1.0)
    assert rouge_l("nada que ver aquí", "totalmente distinto texto") == 0.0
    assert rouge_l("", "referencia") == 0.0


def test_extract_first_json_object_from_noisy_text():
    text = 'Claro, aquí tienes:\n```json\n{"plazo": "tres meses", "n": 3}\n``` ¡Listo!'
    parsed = extract_first_json_object(text)
    assert parsed == {"plazo": "tres meses", "n": 3}
    assert extract_first_json_object("sin json alguno") is None
    # JSON roto seguido de JSON válido: encuentra el válido.
    assert extract_first_json_object('{rota} {"ok": true}') == {"ok": True}


def test_field_exact_match_with_json_and_fallback():
    expected = {"plazo": "tres meses", "organo": "SEPE"}
    perfect = '{"plazo": "Tres Meses", "organo": "sepe"}'
    score, detail = field_exact_match(perfect, expected)
    assert score == 1.0
    assert detail == {"plazo": True, "organo": True}
    # Sin JSON: red de seguridad por containment en la prosa.
    prose = "El plazo es de tres meses y lo gestiona el SEPE."
    score_prose, _ = field_exact_match(prose, expected)
    assert score_prose == 1.0
    # Fallo parcial.
    partial = '{"plazo": "seis meses", "organo": "SEPE"}'
    score_partial, detail_partial = field_exact_match(partial, expected)
    assert score_partial == 0.5
    assert detail_partial["plazo"] is False
