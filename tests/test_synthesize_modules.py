"""Tests del paquete ``adapterforge.data.synthesize`` tras dividirlo por módulos.

Cubren lo que el reparto hace visible como unidad: la superficie pública
reexportada por el paquete, el punto de entrada ``python -m``, el parseo de
``--balance`` (cli), los helpers de redacción (text) y el contrato del
teacher online con un cliente simulado (online), sin GPU ni red.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from adapterforge.data import synthesize
from adapterforge.data.extract import RawDocument
from adapterforge.data.synthesize import cli, online, text
from adapterforge.data.synthesize.examples import DEFAULT_BALANCE


def _make_doc(doc_id: str = "BOE-A-2026-99999", rango: str = "Real Decreto") -> RawDocument:
    return RawDocument(
        id=doc_id,
        rango=rango,
        titulo=f"{rango} 999/2026, de prueba para tests unitarios",
        organo_emisor="Ministerio de Pruebas",
        fecha_publicacion="2026-06-01",
        materia=["pruebas"],
        url="https://boe.es/eli/9999",
        texto=(
            "Artículo 1. Objeto. La presente norma regula las pruebas unitarias. "
            "Artículo 2. Plazos. Las solicitudes se presentarán en el plazo de "
            "tres meses. " + "Texto de relleno normativo. " * 10
        ),
    )


# ---------------------------------------------------------------------------
# Superficie pública del paquete
# ---------------------------------------------------------------------------


def test_package_reexports_resolve_to_submodule_objects():
    """Cada nombre de ``__all__`` es el MISMO objeto que define su submódulo."""
    assert len(set(synthesize.__all__)) == len(synthesize.__all__)
    for name in synthesize.__all__:
        obj = getattr(synthesize, name)
        module_name = getattr(obj, "__module__", None)
        if module_name is None:  # constantes (tuplas/dicts/ints) sin __module__
            continue
        assert module_name.startswith("adapterforge.data.synthesize."), (name, module_name)
        submodule = sys.modules[module_name]
        assert getattr(submodule, name) is obj


def test_package_keeps_private_names_used_by_existing_tests():
    for name in ("_amount_value", "_first_sentence", "_future_date", "_trim_purpose"):
        assert callable(getattr(synthesize, name))


def test_main_module_entry_point_prints_help():
    result = subprocess.run(
        [sys.executable, "-m", "adapterforge.data.synthesize", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage: python -m adapterforge.data.synthesize" in result.stdout
    assert "--offline" in result.stdout and "--dpo" in result.stdout


# ---------------------------------------------------------------------------
# cli: --balance
# ---------------------------------------------------------------------------


def test_parse_balance_keywords_and_specs():
    assert cli._parse_balance("none") is None
    default = cli._parse_balance("default")
    assert default == DEFAULT_BALANCE
    assert default is not DEFAULT_BALANCE  # copia: el CLI no muta la constante
    assert cli._parse_balance("extraccion=0.5, qa=0.5") == {"extraccion": 0.5, "qa": 0.5}


def test_parse_balance_rejects_non_numeric_weights():
    with pytest.raises(ValueError):
        cli._parse_balance("qa=mucho")


# ---------------------------------------------------------------------------
# text: redacción en castellano
# ---------------------------------------------------------------------------


def test_format_date_es():
    assert text.format_date_es("2026-03-18") == "18 de marzo de 2026"
    assert text.format_date_es("2027-01-05") == "5 de enero de 2027"


def test_short_ref_uses_grammatical_gender_of_the_rank():
    masculine = _make_doc(rango="Real Decreto")
    feminine = _make_doc(rango="Ley")
    assert text._short_ref(masculine) == "el Real Decreto 999/2026"
    assert text._short_ref(masculine, preceded_by_de=True) == "del Real Decreto 999/2026"
    assert text._short_ref(feminine) == "la Ley 999/2026"
    assert text._short_ref(feminine, preceded_by_de=True) == "de la Ley 999/2026"


def test_organ_with_article_only_for_known_first_words():
    assert text._organ_with_article("Ministerio de Pruebas") == "el Ministerio de Pruebas"
    assert text._organ_with_article("Agencia Estatal X") == "la Agencia Estatal X"
    assert text._organ_with_article("Consejería de Y") == "Consejería de Y"
    assert text._organ_with_article("") == ""


def test_ends_sentence_accepts_closing_punctuation_only():
    assert text._ends_sentence("Frase completa.")
    assert text._ends_sentence("¿Pregunta?  ")
    assert text._ends_sentence("Cita [BOE-A-2026-1]")
    assert not text._ends_sentence("Frase cortada,")
    assert not text._ends_sentence("   ")


def test_simplify_text_applies_longest_expressions_first():
    plain = text.simplify_text("Las empresas estarán obligadas a presentar la memoria.")
    assert plain == "Las empresas tienen que presentar la memoria."
    assert text.simplify_text("deberán hacerlo en el plazo de un mes") == (
        "tienen que hacerlo en un máximo de un mes"
    )


# ---------------------------------------------------------------------------
# online: contrato del teacher con cliente simulado (sin red)
# ---------------------------------------------------------------------------


def _write_templates(prompts_dir):
    (prompts_dir / "qa.md").write_text(
        "Documento {{id}} ({{titulo_corto}})\n{{texto}}\nPREGUNTA:/RESPUESTA:",
        encoding="utf-8",
    )
    (prompts_dir / "resumen.md").write_text("Resume {{titulo}}:\n{{texto}}", encoding="utf-8")
    (prompts_dir / "extraccion.md").write_text("Extrae de {{texto}}", encoding="utf-8")


def test_synthesize_online_parses_qa_and_discards_contract_violations(tmp_path, monkeypatch):
    import adapterforge.llm as llm

    _write_templates(tmp_path)
    docs = [_make_doc("BOE-A-2026-00001"), _make_doc("BOE-A-2026-00002")]
    seen_prompts: list[str] = []

    def fake_chat_complete(messages, model, **kwargs):
        prompt = messages[0]["content"]
        seen_prompts.append(prompt)
        assert model == "stub/teacher" and kwargs["backend"] == "openai"
        if "BOE-A-2026-00002" in prompt and "PREGUNTA" in prompt:
            return "texto libre sin los bloques esperados"  # contrato roto
        if "PREGUNTA" in prompt:
            return (
                "**PREGUNTA:** ¿Qué regula?\n"
                "**RESPUESTA:** Regula pruebas [BOE-A-2026-00001, art. 1]."
            )
        return "Resumen fiel del documento."

    monkeypatch.setattr(llm, "chat_complete", fake_chat_complete)
    examples = online.synthesize_online(
        docs,
        ("qa", "resumen"),
        model="stub/teacher",
        backend="openai",
        api_base="http://localhost:1/v1",
        api_key="x",
        prompts_dir=tmp_path,
    )

    by_id = {example.id: example for example in examples}
    assert set(by_id) == {
        "BOE-A-2026-00001-qa",
        "BOE-A-2026-00001-resumen",
        "BOE-A-2026-00002-resumen",
    }
    qa = by_id["BOE-A-2026-00001-qa"]
    assert qa.instruction == "¿Qué regula?"
    assert qa.response == "Regula pruebas [BOE-A-2026-00001, art. 1]."
    assert qa.source == "stub/teacher"
    # La ventana guardada como input es exactamente la que leyó el teacher.
    assert any(qa.input in prompt for prompt in seen_prompts)
    resumen = by_id["BOE-A-2026-00001-resumen"]
    assert resumen.instruction in synthesize.prompts._RESUMEN_INSTRUCTIONS_ESTRUCTURADO
    assert len(seen_prompts) == 4


def test_synthesize_online_skips_llm_errors_and_tasks_without_instruction(tmp_path, monkeypatch):
    import adapterforge.llm as llm

    _write_templates(tmp_path)
    docs = [_make_doc("BOE-A-2026-00001"), _make_doc("BOE-A-2026-00002")]
    calls = 0

    def fake_chat_complete(messages, model, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise llm.LlmError("timeout simulado")
        return "Respuesta válida del teacher."

    monkeypatch.setattr(llm, "chat_complete", fake_chat_complete)
    examples = online.synthesize_online(
        docs,
        ("resumen", "extraccion"),
        model="stub/teacher",
        backend="litellm",
        api_base=None,
        api_key=None,
        prompts_dir=tmp_path,
    )
    # resumen: el primer doc falla (se omite), el segundo se emite;
    # extraccion: sin pool de instrucción online, todas sus salidas se descartan.
    assert [example.id for example in examples] == ["BOE-A-2026-00002-resumen"]
    assert calls == 4
