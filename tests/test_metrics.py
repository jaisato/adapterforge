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
    strip_response_artifacts,
)


def test_normalize_and_exact_match():
    # Los numerales se canonicalizan a dígitos («tres» -> «3»).
    assert normalize_text("  Tres  MESES. ") == "3 meses"
    assert exact_match("Tres Meses", "tres meses")
    assert exact_match("días hábiles", "dias habiles")  # sin tildes
    assert not exact_match("tres meses", "seis meses")


def test_numeric_canonicalization():
    # Separadores de miles: «7.500» y «7500» son el mismo número.
    assert exact_match("7.500 euros", "7500 euros")
    assert contains_match("multa de hasta 7.500 euros por centro de trabajo", "7500 euros")
    # Numerales en castellano -> dígitos.
    assert exact_match("ocho millones de euros", "8.000.000 de euros")
    assert contains_match("los municipios de más de veinte mil habitantes", "20000 habitantes")
    assert exact_match("treinta y cinco días", "35 dias")
    assert exact_match("mil millones", "1000000000")
    assert exact_match("seiscientos euros mensuales", "600 euros mensuales")
    # El decimal sobrevive a la normalización y casa entre convenios.
    assert exact_match("un incremento del 3,5 %", "un incremento del 3.5%")
    # Números distintos siguen sin casar.
    assert not exact_match("7.500 euros", "7400 euros")


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


def test_field_match_accepts_terser_but_numerically_equal_values():
    # eb-004 real: el adapter responde el número pelado y el documento dice
    # «7.500 euros» — mismo número, acierto.
    score, _ = field_exact_match('{"multa_maxima": 7500}', {"multa_maxima": "7.500 euros"})
    assert score == 1.0
    # eb-007 real: dígito vs numeral en castellano.
    score, detail = field_exact_match(
        '{"umbral": "8 millones", "plazo": "12 meses"}',
        {"umbral": "ocho millones de euros", "plazo": "doce meses"},
    )
    assert score == 1.0
    assert detail == {"umbral": True, "plazo": True}
    # Una fecha incompleta NO cuenta: faltan componentes numéricos.
    score, _ = field_exact_match(
        '{"fecha_limite": "2027"}', {"fecha_limite": "1 de enero de 2027"}
    )
    assert score == 0.0
    # Texto sin números no gana por containment inverso.
    score, _ = field_exact_match('{"cuantia": "euros"}', {"cuantia": "600 euros"})
    assert score == 0.0


def test_compound_numerals_with_connector():
    # Centenas/millares con conector «y» (bug de la primera versión).
    assert normalize_text("ciento treinta y cinco") == "135"
    assert contains_match("el plazo es de 365 días", "trescientos sesenta y cinco días")
    assert normalize_text("mil novecientos ochenta y cinco") == "1985"
    # Compuesto anidado real del dataset: «... mil ... millones».
    assert (
        normalize_text("cuarenta y ocho mil ochocientos setenta y un millones")
        == "48871000000"
    )


def test_bare_millones_is_not_a_number():
    # «varios millones» no es un número concreto: no debe casar con «un millón».
    assert normalize_text("varios millones de euros") == "varios millones de euros"
    assert not contains_match("la sanción asciende a varios millones de euros",
                              "un millón de euros")
    # Con cuantificador sí se convierte.
    assert exact_match("un millón de euros", "1.000.000 de euros")


def test_independent_numerals_do_not_merge():
    # Enumeraciones: cada numeral queda separado, no se suman.
    assert normalize_text("los apartados uno, dos y tres") == "los apartados 1 2 y 3"
    assert contains_match("los apartados 1, 2 y 3", "apartados uno, dos y tres")
    # Cruce de frase: el número de apartado no se funde con la duración.
    assert contains_match("según el apartado dos. Tres meses después", "tres meses")


def test_time_notation_dot_vs_colon():
    assert exact_match("a las 9.30 horas", "a las 9:30 horas")


def test_strip_response_artifacts():
    garbage = '</tool_call>\n\n<tool_call>\n\n{"multa_maxima": 7500}'
    assert strip_response_artifacts(garbage) == '{"multa_maxima": 7500}'
    assert strip_response_artifacts("<think>razonando...</think>La respuesta") == "La respuesta"
    assert strip_response_artifacts("texto normal") == "texto normal"
    # <think> sin cerrar = razonamiento truncado: no hay respuesta que puntuar.
    assert strip_response_artifacts("<think>El plazo creo que son 15 días...") == ""
