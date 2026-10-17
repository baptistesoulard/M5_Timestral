"""Benchmarks statistiques du M5, calculés au niveau article × magasin puis agrégés (bottom-up).

Rapides (numpy) : naïf, naïf saisonnier 7 jours, moyenne mobile 28 jours, quantiles empiriques.
Avec --stats (statsforecast) : SES, Croston SBA, TSB, ADIDA, iMAPA ; --ets ajoute AutoETS (lent).

    python run_baselines.py --data data --out forecasts
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from m5bench.data import HORIZON, load_m5, mask_leading_zeros
from m5bench.metrics import M5_QUANTILES
from m5bench.store import save_forecast


def simple_baselines(train: np.ndarray) -> dict[str, np.ndarray]:
    last = train[:, -1:]
    return {
        "naive": np.repeat(last, HORIZON, axis=1),
        "snaive": np.tile(train[:, -7:], HORIZON // 7),
        "ma28": np.repeat(train[:, -28:].mean(axis=1, keepdims=True), HORIZON, axis=1),
    }


def empirical_weekday_quantiles(history_all: np.ndarray, weeks: int = 26) -> np.ndarray:
    """Quantiles empiriques par jour de semaine sur les `weeks` dernières semaines, pour les 42 840 séries.
    Référence simple pour la piste incertitude (WSPL)."""
    n, t = history_all.shape
    out = np.empty((n, HORIZON, len(M5_QUANTILES)), dtype=np.float32)
    for h in range(7):
        # le jour h de l'horizon a le même jour de semaine que t - 7 + h, t - 14 + h, …
        cols = [t - 7 * k + h for k in range(1, weeks + 1)]
        q = np.quantile(history_all[:, cols], M5_QUANTILES, axis=1).T  # (n, 9)
        out[:, h::7, :] = q[:, None, :]
    return out


def stats_baselines(train: np.ndarray, with_ets: bool, n_jobs: int) -> dict[str, np.ndarray]:
    import pandas as pd
    from statsforecast import StatsForecast
    from statsforecast.models import ADIDA, IMAPA, TSB, AutoETS, CrostonSBA, SimpleExponentialSmoothingOptimized

    models = [SimpleExponentialSmoothingOptimized(alias="ses"), CrostonSBA(alias="sba"),
              TSB(alpha_d=0.1, alpha_p=0.1, alias="tsb"), ADIDA(alias="adida"), IMAPA(alias="imapa")]
    if with_ets:
        models.append(AutoETS(season_length=7, alias="ets"))
    masked = mask_leading_zeros(train)  # les modèles ne voient l'historique qu'à partir du lancement
    n, t = masked.shape
    out = {m.alias: np.zeros((n, HORIZON), dtype=np.float32) for m in models}
    chunk = 3000  # par paquets, pour limiter la mémoire
    for s in range(0, n, chunk):
        part = masked[s : s + chunk]
        k = len(part)
        df = pd.DataFrame({"unique_id": np.repeat(np.arange(s, s + k), t), "ds": np.tile(np.arange(t), k),
                           "y": part.ravel()}).dropna()
        fc = StatsForecast(models=models, freq=1, n_jobs=n_jobs).forecast(df=df, h=HORIZON)
        fc = fc.reset_index() if "unique_id" not in fc.columns else fc
        for m in models:
            arr = fc.pivot(index="unique_id", columns="ds", values=m.alias).reindex(np.arange(s, s + k)).to_numpy()
            out[m.alias][s : s + k] = np.nan_to_num(arr, nan=0.0)
        print(f"\r    statsforecast {s + k}/{n}", end="", flush=True)
    print()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="forecasts")
    ap.add_argument("--stats", action="store_true", help="ajoute SES, SBA, TSB, ADIDA, iMAPA (statsforecast)")
    ap.add_argument("--ets", action="store_true", help="ajoute AutoETS saisonnier 7 jours (plusieurs dizaines de minutes)")
    ap.add_argument("--n-jobs", type=int, default=-1)
    args = ap.parse_args()

    data = load_m5(args.data)
    t0 = time.time()
    for name, fc in simple_baselines(data.train).items():
        save_forecast(f"{args.out}/{name}.npz", {"method": name, "family": "benchmark"}, point_bottom=fc)
        print(f"{name:>8} enregistré")

    q = empirical_weekday_quantiles(data.aggregate(data.train))
    save_forecast(f"{args.out}/empirical_quantiles.npz", {"method": "quantiles empiriques jour de semaine (26 sem.)",
                                                          "family": "benchmark"}, quantiles_all=q)
    print("quantiles empiriques enregistrés")

    if args.stats or args.ets:
        t1 = time.time()
        for name, fc in stats_baselines(data.train, args.ets, args.n_jobs).items():
            save_forecast(f"{args.out}/{name}.npz", {"method": name, "family": "benchmark"}, point_bottom=fc)
            print(f"{name:>8} enregistré")
        print(f"statsforecast : {time.time() - t1:.0f} s")
    print(f"terminé en {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
