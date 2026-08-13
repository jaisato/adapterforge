"""Sube la carpeta space/ al Hugging Face Space de demo (Jaisato).

Lo usa .github/workflows/deploy-hf.yml con el secret HF_TOKEN. También puede
ejecutarse en local:

    HF_TOKEN=hf_xxx python scripts/deploy_space.py --repo-id Jaisato/adapterforge-demo
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Despliegue del Space de demo")
    parser.add_argument("--repo-id", default="Jaisato/adapterforge-demo")
    parser.add_argument("--space-dir", default="space")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("[deploy_space] FALLO: define HF_TOKEN (con permiso de escritura)")
        return 1
    space_dir = Path(args.space_dir)
    if not space_dir.is_dir():
        print(f"[deploy_space] FALLO: no existe {space_dir}")
        return 1

    from huggingface_hub import HfApi  # perezoso

    api = HfApi(token=token)
    api.create_repo(repo_id=args.repo_id, repo_type="space", space_sdk="gradio", exist_ok=True)
    api.upload_folder(
        folder_path=str(space_dir),
        repo_id=args.repo_id,
        repo_type="space",
        commit_message="Despliegue automático del Space de demo",
    )
    print(f"[deploy_space] OK: https://huggingface.co/spaces/{args.repo_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
