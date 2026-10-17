"""Évalue toutes les prévisions de forecasts/ sur la période de test officielle (d_1942 – d_1969).

Écrit results/scores.csv (score total par méthode), results/by_level.csv (détail des 12 niveaux)
et results/scores.md (tableau lisible). Les soumissions du top 10 du concours placées dans
data/reference/*.zip (voir download_data.py) servent de points de repère.

    python evaluate.py
"""

from __future__ import annotations

import argparse
import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from m5bench.data import load_m5
from m5bench.metrics import M5Evaluator
from m5bench.store import load_forecast


def reference_submissions(ref_dir: Path, data) -> dict[str, np.ndarray]:
    out = {}
    key = data.bottom_ids["item_id"] + "_" + data.bottom_ids["store_id"]
    for z in sorted(ref_dir.glob("*.zip")):
        with zipfile.ZipFile(z) as f:
            name = [n for n in f.namelist() if n.endswith(".csv")][0]
            df = pd.read_csv(io.BytesIO(f.read(name)))
        df = df[df["id"].str.endswith("_evaluation")]
        df.index = df["id"].str.replace("_evaluation", "", regex=False)
        rank, _, team = z.stem.partition(" ")
        out[f"M5 #{int(rank)} {team}"] = df.loc[key, [f"F{i}" for i in range(1, 29)]].to_numpy(dtype=np.float32)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--forecasts", default="forecasts")
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    data = load_m5(args.data)
    ev = M5Evaluator(data)
    totals, details = [], []

    def add(method: str, family: str, variant: str, mode: str, table: pd.DataFrame, metric: str, seconds=None):
        tot = float(table.loc[table["level"] == "Total", metric].iloc[0])
        totals.append({"méthode": method, "famille": family, "variante": variant, "agrégation": mode,
                       "métrique": metric, "score": tot, "durée_s": seconds})
        details.append(table.assign(méthode=method, variante=variant, agrégation=mode, métrique=metric)
                       .rename(columns={metric: "score"}))

    for path in sorted(Path(args.forecasts).glob("*.npz")):
        meta, arrays = load_forecast(path)
        method, family = meta.get("method", path.stem), meta.get("family", "")
        for k, arr in arrays.items():
            if np.isnan(arr).any():
                print(f"  {path.name}:{k} incomplet, ignoré")
                continue
            variant = k.split("_", 2)[2] if k.count("_") >= 2 else ""
            if k.startswith("point_bottom"):
                add(method, family, variant, "bottom-up", ev.wrmsse(arr), "wrmsse", meta.get("seconds"))
            elif k.startswith("point_all"):
                add(method, family, variant, "direct", ev.wrmsse(arr), "wrmsse", meta.get("seconds"))
            elif k == "quantiles_all":
                add(method, family, "", "direct", ev.wspl(arr), "wspl", meta.get("seconds"))

    ref_dir = Path(args.data) / "reference"
    if ref_dir.exists():
        for name, arr in reference_submissions(ref_dir, data).items():
            add(name, "concours M5", "", "bottom-up", ev.wrmsse(arr), "wrmsse")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scores = pd.DataFrame(totals).sort_values(["métrique", "score"]).reset_index(drop=True)
    scores.to_csv(out / "scores.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(out / "by_level.csv", index=False)

    lines = []
    for metric, title in [("wrmsse", "Précision — WRMSSE (plus bas = meilleur)"),
                          ("wspl", "Incertitude — WSPL (plus bas = meilleur)")]:
        s = scores[scores["métrique"] == metric]
        if s.empty:
            continue
        lines += [f"## {title}", "", "| méthode | variante | agrégation | score |", "|---|---|---|---|"]
        lines += [f"| {r.méthode} | {r.variante} | {r.agrégation} | {r.score:.4f} |" for r in s.itertuples()]
        lines.append("")
    (out / "scores.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
