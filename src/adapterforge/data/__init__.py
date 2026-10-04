"""Pipeline de dataset instruccional (solo dependencias core).

Etapas: ``extract`` -> ``synthesize`` -> ``filter`` -> ``format``. Cada etapa
es un CLI independiente (``python -m adapterforge.data.<etapa>``) que lee y
escribe JSONL, de modo que el pipeline es inspeccionable paso a paso.
``pull`` descarga el corpus BOE y los datasets ``juristune-v*`` generados
(no versionados) desde el dataset privado de Hugging Face Hub.
"""
