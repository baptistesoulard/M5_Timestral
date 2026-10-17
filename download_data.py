"""Télécharge le jeu M5 complet (copie publique Nixtla des fichiers Kaggle, test inclus)
et les prévisions des meilleures équipes du concours, qui servent de points de repère.

    python download_data.py            # données + top 10
    python download_data.py --top 3

Alternative : télécharger les fichiers depuis Kaggle (concours « M5 Forecasting - Accuracy »)
et les déposer dans data/ ; sales_test_evaluation.csv est alors à prendre chez Nixtla.
"""

from __future__ import annotations

import argparse
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://github.com/Nixtla/m5-forecasts/raw/main"
TEAMS = ["0001 YJ_STU", "0002 Matthias", "0003 mf", "0004 monsaraida", "0005 Alan Lahoud",
         "0006 wyzJack_STU", "0007 RandomLearner", "0008 SHJ", "0009 gest 2", "0010 DenisKokosinskiy_STU"]
FILES = ["calendar.csv", "sell_prices.csv", "sales_train_evaluation.csv", "sales_test_evaluation.csv",
         "weights_evaluation.csv"]


def fetch(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"  déjà présent : {dest}")
        return
    print(f"  téléchargement : {dest.name}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--top", type=int, default=10, help="nombre de soumissions du classement à récupérer")
    args = ap.parse_args()
    data = Path(args.data)

    if not all((data / f).exists() for f in FILES):
        archive = data / "m5.zip"
        fetch(f"{BASE}/datasets/m5.zip", archive)
        with zipfile.ZipFile(archive) as z:
            for f in FILES:
                z.extract(f, data)
        archive.unlink()
    for team in TEAMS[: args.top]:
        fetch(f"{BASE}/forecasts/{urllib.parse.quote(team)}.zip", data / "reference" / f"{team}.zip")
    print("ok")


if __name__ == "__main__":
    main()
