"""Assemble le dossier dist/space/ à déposer tel quel dans un Space Hugging Face (SDK Gradio).

    python build_space.py
    # puis, avec le CLI Hugging Face :
    huggingface-cli upload <votre-compte>/timestral-m5 dist/space . --repo-type space

Le Space télécharge les données M5 au démarrage (~50 Mo) et les poids Timestral au premier usage.
"""

from __future__ import annotations

import shutil
from pathlib import Path

ROOT = Path(__file__).parent
FILES = ["app.py", "download_data.py", "run_baselines.py"]
DIRS = ["m5bench", "app_assets"]


def main() -> None:
    dist = ROOT / "dist" / "space"
    if dist.exists():
        shutil.rmtree(dist)
    dist.mkdir(parents=True)
    for f in FILES:
        shutil.copy(ROOT / f, dist / f)
    for d in DIRS:
        shutil.copytree(ROOT / d, dist / d, ignore=shutil.ignore_patterns("__pycache__"))
    for f in ("README.md", "requirements.txt"):
        shutil.copy(ROOT / "space" / f, dist / f)
    print(f"{dist} prêt ({sum(1 for _ in dist.rglob('*') if _.is_file())} fichiers)")


if __name__ == "__main__":
    main()
