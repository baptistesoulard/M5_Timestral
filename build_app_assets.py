"""Prépare les fichiers légers embarqués dans l'appli (dossier app_assets/) :

- reference_forecasts.npz : prévisions article × magasin des méthodes de comparaison
  (naïf saisonnier, TSB, AutoETS, vainqueur du concours) ;
- scores.csv / by_level.csv : classement complet issu de evaluate.py.

À relancer après de nouveaux runs complets (run_timestral.py puis evaluate.py) pour que
le classement de l'appli les affiche.

    python build_app_assets.py
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from evaluate import reference_submissions
from m5bench.data import load_m5
from m5bench.store import load_forecast

METHODS = {"snaive": "Naïf saisonnier", "tsb": "TSB (intermittent)", "ets": "AutoETS saisonnier"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--forecasts", default="forecasts")
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="app_assets")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    data = load_m5(args.data)
    arrays, labels = {}, {}
    for key, label in METHODS.items():
        path = Path(args.forecasts) / f"{key}.npz"
        if path.exists():
            _, a = load_forecast(path)
            arrays[key] = a["point_bottom"]
            labels[key] = label
        else:
            print(f"  {path} absent, ignoré")
    ref_dir = Path(args.data) / "reference"
    if ref_dir.exists():
        refs = reference_submissions(ref_dir, data)
        winner = [k for k in refs if k.startswith("M5 #1 ")]
        if winner:
            arrays["winner"] = refs[winner[0]]
            labels["winner"] = f"Vainqueur M5 ({winner[0].split(' ', 2)[2]})"

    np.savez_compressed(out / "reference_forecasts.npz", labels=np.array(json.dumps(labels, ensure_ascii=False)),
                        **{k: v.astype(np.float16) for k, v in arrays.items()})
    for f in ("scores.csv", "by_level.csv"):
        if (Path(args.results) / f).exists():
            shutil.copy(Path(args.results) / f, out / f)
    print(f"{out} prêt : {', '.join(labels.values())}")


if __name__ == "__main__":
    main()
