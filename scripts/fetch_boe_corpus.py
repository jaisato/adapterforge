#!/usr/bin/env python3
"""Descarga corpus REAL del BOE (API de datos abiertos) en formato RawDocument.

Implementa el ``TODO(jairo)`` de ``data/extract.py``: mini conector BOE
standalone. Descarga sumarios diarios por rango de fechas (y/o documentos
concretos por identificador), extrae los metadatos y el texto articulado de
cada disposición, y escribe un JSONL directamente consumible por
``adapterforge.data.extract``.

Solo conserva disposiciones *articuladas* (con "Artículo 1." en el texto):
resoluciones y anuncios sin articulado se descartan — es un filtro de calidad
natural para el dataset instruccional.

Idempotente: si el fichero de salida existe, se conservan los documentos ya
descargados y solo se añaden los nuevos (merge por identificador).

Uso::

    # Un rango de fechas (sección 1 = Disposiciones generales)
    python scripts/fetch_boe_corpus.py --from 2025-01-01 --to 2025-12-31 \
        --output data/raw/corpus_boe.jsonl

    # Normas concretas por identificador (leyes "clásicas")
    python scripts/fetch_boe_corpus.py \
        --ids BOE-A-2015-10565,BOE-A-2015-10566,BOE-A-2015-11430 \
        --output data/raw/corpus_boe.jsonl

TODO(jairo): variante que lea el corpus ya ingerido por NormaRAG
(Postgres/MinIO) y conector de legislación consolidada del BOE.
"""

from __future__ import annotations

import argparse
import sys
import time
import xml.etree.ElementTree as ET
from datetime import date, timedelta
from pathlib import Path

import httpx

# Permite ejecutar el script sin instalar el paquete (repo recién clonado).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from adapterforge.data.extract import RawDocument  # noqa: E402
from adapterforge.jsonl import read_jsonl, write_jsonl  # noqa: E402

BOE_BASE = "https://www.boe.es"
SUMARIO_URL = BOE_BASE + "/datosabiertos/api/boe/sumario/{fecha}"
DOC_XML_URL = BOE_BASE + "/diario_boe/xml.php?id={boe_id}"
DOC_HTML_URL = BOE_BASE + "/diario_boe/txt.php?id={boe_id}"

RETRIES = 3
RETRY_SLEEP = 2.0


def _get(client: httpx.Client, url: str, **kwargs) -> httpx.Response | None:
    """GET con reintentos simples; None si 404 (día sin boletín / doc ausente)."""
    for attempt in range(1, RETRIES + 1):
        try:
            response = client.get(url, **kwargs)
        except httpx.HTTPError as exc:
            if attempt == RETRIES:
                print(f"  [aviso] error de red en {url}: {exc}", file=sys.stderr)
                return None
            time.sleep(RETRY_SLEEP * attempt)
            continue
        if response.status_code == 404:
            return None
        if response.status_code >= 500 and attempt < RETRIES:
            time.sleep(RETRY_SLEEP * attempt)
            continue
        if response.status_code != 200:
            print(f"  [aviso] HTTP {response.status_code} en {url}", file=sys.stderr)
            return None
        return response
    return None


def _collect_ids(node: object, acc: list[str]) -> None:
    """Recoge recursivamente todos los 'identificador' BOE-A-* de un subárbol JSON."""
    if isinstance(node, dict):
        identificador = node.get("identificador")
        if isinstance(identificador, str) and identificador.startswith("BOE-A-"):
            acc.append(identificador)
        for value in node.values():
            _collect_ids(value, acc)
    elif isinstance(node, list):
        for value in node:
            _collect_ids(value, acc)


def ids_del_sumario(client: httpx.Client, fecha: date, seccion: str) -> list[str]:
    """Identificadores de la sección pedida en el sumario del día (o [])."""
    url = SUMARIO_URL.format(fecha=fecha.strftime("%Y%m%d"))
    response = _get(client, url, headers={"Accept": "application/json"})
    if response is None:
        return []
    try:
        data = response.json()
    except ValueError:
        return []
    # La API devuelve dict o lista según el día (lección heredada de NormaRAG):
    # navegamos con tolerancia hasta las secciones y filtramos por código.
    diarios = (((data or {}).get("data") or {}).get("sumario") or {}).get("diario") or []
    if isinstance(diarios, dict):
        diarios = [diarios]
    ids: list[str] = []
    for diario in diarios:
        secciones = diario.get("seccion") or []
        if isinstance(secciones, dict):
            secciones = [secciones]
        for sec in secciones:
            if str(sec.get("codigo", "")) != str(seccion):
                continue
            _collect_ids(sec, ids)
    # Preserva orden y elimina duplicados.
    return list(dict.fromkeys(ids))


def _clean(value: str) -> str:
    """Normaliza espacios duros del BOE (NBSP \\xa0 y NNBSP \\u202f).

    El BOE escribe 'Artículo\\xa01.' con espacio duro; sin normalizar, ni el
    marcador literal ni los regex de ``extract.py`` de los consumidores del
    corpus se comportan de forma predecible (bug real detectado 09-08-2026).
    """
    return value.replace("\xa0", " ").replace(" ", " ")


def _texto_articulado(root: ET.Element) -> str:
    """Concatena los párrafos del <texto> HIJO DIRECTO de <documento>.

    Ojo: no usar ``.//texto`` — el bloque <analisis><referencias> contiene
    nodos <texto> propios y el descendiente-cualquiera devolvería el primero
    de ellos (bug real detectado contra la API el 09-08-2026).
    """
    texto_node = root.find("texto")
    if texto_node is None:
        return ""
    parts: list[str] = []
    for p in texto_node.iter("p"):
        content = _clean("".join(p.itertext())).strip()
        if content:
            parts.append(content)
    return "\n".join(parts)


def _rango_desde_titulo(titulo: str) -> str:
    lowered = titulo.lower()
    for candidato in (
        "ley orgánica", "real decreto-ley", "real decreto legislativo",
        "real decreto", "ley", "orden", "resolución", "circular",
    ):
        if lowered.startswith(candidato):
            return candidato.title()
    return "Disposición"


def descargar_documento(client: httpx.Client, boe_id: str) -> dict | None:
    """Descarga y transforma un documento BOE a dict RawDocument (o None si no vale)."""
    response = _get(client, DOC_XML_URL.format(boe_id=boe_id))
    if response is None:
        return None
    try:
        root = ET.fromstring(response.content)
    except ET.ParseError:
        print(f"  [aviso] XML inválido: {boe_id}", file=sys.stderr)
        return None

    def _meta(tag: str) -> str:
        node = root.find(f"metadatos/{tag}")  # ruta directa: sin sorpresas de .//
        return _clean(node.text).strip() if node is not None and node.text else ""

    texto = _texto_articulado(root)
    if "Artículo 1." not in texto or len(texto) < 200:
        return None  # no articulada: fuera del dataset

    titulo = _meta("titulo") or boe_id
    if len(titulo) < 10:
        return None
    fecha_raw = _meta("fecha_publicacion")  # AAAAMMDD
    if len(fecha_raw) != 8 or not fecha_raw.isdigit():
        return None
    materias = [
        (m.text or "").strip()
        for m in root.findall("analisis/materias/materia")
        if (m.text or "").strip()
    ]
    record = {
        "id": boe_id,
        "rango": (_meta("rango") or _rango_desde_titulo(titulo))[:60],
        "titulo": titulo[:400],
        "organo_emisor": (_meta("departamento") or "Administración General del Estado")[:200],
        "fecha_publicacion": f"{fecha_raw[:4]}-{fecha_raw[4:6]}-{fecha_raw[6:]}",
        "materia": (materias or ["administración pública"])[:10],
        "url": DOC_HTML_URL.format(boe_id=boe_id),
        "texto": texto,
    }
    try:
        RawDocument.model_validate(record)
    except Exception as exc:  # doc raro: se descarta con contexto, sin romper el lote
        print(f"  [aviso] {boe_id} no pasa el esquema: {exc}", file=sys.stderr)
        return None
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from", dest="date_from", help="AAAA-MM-DD")
    parser.add_argument("--to", dest="date_to", help="AAAA-MM-DD (incluida)")
    parser.add_argument("--ids", default="", help="Identificadores BOE-A-... separados por comas")
    parser.add_argument("--seccion", default="1", help="Sección del sumario (1 = Disp. generales)")
    parser.add_argument("--output", default="data/raw/corpus_boe.jsonl")
    parser.add_argument("--sleep", type=float, default=0.3, help="Pausa entre documentos (s)")
    parser.add_argument("--limit", type=int, default=None, help="Máx. docs nuevos (para pruebas)")
    args = parser.parse_args(argv)

    if not args.ids and not (args.date_from and args.date_to):
        parser.error("indica --from/--to, --ids, o ambos")

    output = Path(args.output)
    existing: dict[str, dict] = {}
    if output.exists():
        existing = {rec["id"]: rec for rec in read_jsonl(output)}
        print(f"[fetch] {len(existing)} documentos ya presentes en {output}")

    wanted: list[str] = [i.strip() for i in args.ids.split(",") if i.strip()]
    with httpx.Client(timeout=30.0, headers={"User-Agent": "adapterforge-corpus/0.1"}) as client:
        if args.date_from and args.date_to:
            current = date.fromisoformat(args.date_from)
            last = date.fromisoformat(args.date_to)
            while current <= last:
                day_ids = ids_del_sumario(client, current, args.seccion)
                if day_ids:
                    print(f"[fetch] {current}: {len(day_ids)} items en sección {args.seccion}")
                wanted.extend(day_ids)
                current += timedelta(days=1)

        nuevos = 0
        descartados = 0
        for boe_id in dict.fromkeys(wanted):  # dedup preservando orden
            if boe_id in existing:
                continue
            if args.limit is not None and nuevos >= args.limit:
                break
            record = descargar_documento(client, boe_id)
            if record is None:
                descartados += 1
            else:
                existing[boe_id] = record
                nuevos += 1
                if nuevos % 25 == 0:
                    print(f"[fetch] {nuevos} nuevos (último: {boe_id})")
            time.sleep(args.sleep)

    count = write_jsonl(output, list(existing.values()))
    print(
        f"[fetch] fin: {nuevos} nuevos, {descartados} descartados (no articulados), "
        f"{count} documentos totales -> {output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())