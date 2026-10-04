"""CLI de la síntesis (``python -m adapterforge.data.synthesize``).

Lee el corpus de ``extract``, elige modo (offline / online / ``--dpo``),
aplica el gate de presupuesto de tokens (ningún dataset que no quepa en
``seq_len``) y escribe el JSONL de ejemplos con un resumen por tarea.
"""

from __future__ import annotations

import argparse

from adapterforge.data.budget import TokenCounter, assert_token_budget
from adapterforge.data.extract import load_corpus
from adapterforge.data.synthesize.dpo import build_dpo_offline
from adapterforge.data.synthesize.examples import DEFAULT_BALANCE, DEFAULT_WINDOW_TOKENS, SFT_TASKS
from adapterforge.data.synthesize.offline import build_sft_offline
from adapterforge.data.synthesize.online import synthesize_online
from adapterforge.jsonl import write_jsonl
from adapterforge.settings import get_settings


def _parse_balance(spec: str) -> dict[str, float] | None:
    """'extraccion=0.3,qa=0.25' -> dict; 'none' -> None; 'default' -> DEFAULT."""
    if spec == "none":
        return None
    if spec == "default":
        return dict(DEFAULT_BALANCE)
    weights: dict[str, float] = {}
    for part in spec.split(","):
        task, _, value = part.partition("=")
        weights[task.strip()] = float(value)
    return weights


def main(argv: list[str] | None = None) -> int:
    # ``prog`` explícito: al ser un paquete, argparse mostraría «__main__.py».
    parser = argparse.ArgumentParser(
        prog="python -m adapterforge.data.synthesize",
        description="Síntesis de dataset instruccional",
    )
    parser.add_argument("--input", required=True, help="Corpus JSONL (salida de extract)")
    parser.add_argument("--output", required=True, help="JSONL de ejemplos generados")
    parser.add_argument(
        "--offline", action="store_true", help="Sin LLM: plantillas determinísticas"
    )
    parser.add_argument("--dpo", action="store_true", help="Genera pares chosen/rejected (offline)")
    parser.add_argument("--tasks", default=",".join(SFT_TASKS), help="Tareas separadas por comas")
    parser.add_argument("--max-per-task", type=int, default=None)
    parser.add_argument("--n-pairs", type=int, default=15, help="Nº de pares DPO")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--balance",
        default="default",
        help="Pesos por tarea 'tarea=w,...', 'default' o 'none' (solo offline)",
    )
    parser.add_argument(
        "--token-budget", type=int, default=DEFAULT_WINDOW_TOKENS,
        help="Presupuesto de tokens de la ventana de documento",
    )
    parser.add_argument("--seq-len", type=int, default=2048, help="Para el gate de presupuesto")
    parser.add_argument(
        "--max-over-frac", type=float, default=0.02,
        help="Fracción máxima de ejemplos que pueden exceder seq-len",
    )
    parser.add_argument("--tokenizer", default=None, help="Ruta a tokenizer.json (opcional)")
    parser.add_argument("--model", default=None, help="Teacher (default: AF_TEACHER_MODEL)")
    parser.add_argument("--backend", default="litellm", choices=["litellm", "openai"])
    parser.add_argument("--api-base", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--prompts-dir", default="prompts")
    args = parser.parse_args(argv)

    settings = get_settings()
    docs = load_corpus(args.input)
    if args.max_docs is not None:
        docs = docs[: args.max_docs]
    tasks = tuple(task.strip() for task in args.tasks.split(",") if task.strip())
    counter = TokenCounter(args.tokenizer)
    print(f"[synthesize] contador de tokens: {counter.describe()}")

    if args.dpo:
        if not args.offline:
            raise SystemExit(
                "La generación DPO online no está en el alcance base; usa --offline. "
                "TODO(jairo): pares de preferencia online (teacher fuerte vs débil)."
            )
        pairs = build_dpo_offline(
            docs, seed=args.seed, n_pairs=args.n_pairs,
            counter=counter, token_budget=args.token_budget,
        )
        count = write_jsonl(args.output, (pair.model_dump() for pair in pairs))
        print(f"[synthesize] {count} pares DPO offline -> {args.output}")
        return 0

    if args.offline:
        examples = build_sft_offline(
            docs, tasks=tasks, seed=args.seed, max_per_task=args.max_per_task,
            counter=counter, token_budget=args.token_budget,
            balance=_parse_balance(args.balance),
        )
    else:
        examples = synthesize_online(
            docs,
            tasks,
            model=args.model or settings.teacher_model,
            backend=args.backend,
            api_base=args.api_base or settings.api_base,
            api_key=args.api_key or settings.api_key,
            prompts_dir=args.prompts_dir,
            counter=counter,
            token_budget=args.token_budget,
            seed=args.seed,
        )

    # Gate de presupuesto: nunca más un dataset que no cabe en seq_len.
    from adapterforge.data.format import SYSTEM_PROMPT  # import local: sin ciclo

    report = assert_token_budget(
        examples, counter, seq_len=args.seq_len,
        max_over_frac=args.max_over_frac, system_prompt=SYSTEM_PROMPT,
    )
    composition: dict[str, int] = {}
    for example in examples:
        composition[example.task] = composition.get(example.task, 0) + 1
    count = write_jsonl(args.output, (example.model_dump() for example in examples))
    mode = "offline" if args.offline else f"online ({args.model or settings.teacher_model})"
    print(f"[synthesize] {count} ejemplos SFT {mode} -> {args.output}")
    print(f"[synthesize] composición por tarea: {dict(sorted(composition.items()))}")
    print(
        f"[synthesize] presupuesto: p50={report['p50']} p95={report.get('p95')} "
        f"max={report['max']} tokens · {report['over']} exceden seq_len={args.seq_len}"
    )
    return 0
