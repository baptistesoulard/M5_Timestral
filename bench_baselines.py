"""Statistical baselines computed in GitHub Actions (too slow for a laptop), saved as forecast shards.

    python bench_baselines.py --bench m5  --method arima --shard 0 --nshards 32   # AutoARIMA, weekly season
    python bench_baselines.py --bench vn1 --method stats                          # quick methods on VN1
    python bench_baselines.py --bench vn1 --method arima --shard 0 --nshards 16   # AutoARIMA, yearly season

Each shard is written to parts/baseline__<bench>__<method>__<shard>of<n>.npz with the bottom-level series
indices, one point forecast per method and the run time.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def statsforecast_run(y: np.ndarray, horizon: int, models, n_jobs: int = -1) -> dict[str, np.ndarray]:
    """y: (n, T) with NaN before the series starts. Returns {alias: (n, horizon)}."""
    from statsforecast import StatsForecast
    from statsforecast.models import Naive

    n, t = y.shape
    df = pd.DataFrame({"unique_id": np.repeat(np.arange(n), t), "ds": np.tile(np.arange(t), n),
                       "y": y.ravel()}).dropna()
    empty = np.setdiff1d(np.arange(n), df["unique_id"].unique())
    fc = StatsForecast(models=models, freq=1, n_jobs=n_jobs, fallback_model=Naive()).forecast(df=df, h=horizon)
    fc = fc.reset_index() if "unique_id" not in fc.columns else fc
    out = {}
    for m in models:
        arr = fc.pivot(index="unique_id", columns="ds", values=m.alias).reindex(np.arange(n)).to_numpy()
        arr[empty] = 0.0
        out[m.alias] = np.nan_to_num(np.maximum(arr, 0), nan=0.0).astype(np.float32)
    return out


def main() -> None:
    from statsforecast.models import (ADIDA, IMAPA, TSB, AutoARIMA, AutoETS, CrostonSBA, SeasonalNaive,
                                      SimpleExponentialSmoothingOptimized)

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", choices=["m5", "vn1"], required=True)
    ap.add_argument("--method", choices=["arima", "stats"], required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="parts")
    args = ap.parse_args()

    if args.bench == "m5":
        from m5bench.data import load_m5, mask_leading_zeros

        data = load_m5(args.data)
        y_all, horizon, season = mask_leading_zeros(data.train), 28, 7
    else:
        from vn1bench.data import HORIZON, load_vn1, mask_leading_zeros

        data = load_vn1(Path(args.data) / "vn1")
        y_all, horizon, season = mask_leading_zeros(data.train), HORIZON, 52
    idx = np.arange(len(y_all))[args.shard :: args.nshards]
    if args.method == "arima":
        models = [AutoARIMA(season_length=season, alias="arima")]
    else:
        models = [SeasonalNaive(season_length=season, alias="snaive"),
                  SimpleExponentialSmoothingOptimized(alias="ses"), CrostonSBA(alias="sba"),
                  TSB(alpha_d=0.1, alpha_p=0.1, alias="tsb"), ADIDA(alias="adida"), IMAPA(alias="imapa"),
                  AutoETS(season_length=season, alias="ets")]
    print(f"{args.bench} {args.method}: {len(idx)} series, shard {args.shard}/{args.nshards}", flush=True)
    t0 = time.time()
    out = {}
    for s in range(0, len(idx), 1000):  # in blocks, to keep memory low and show progress
        part = statsforecast_run(y_all[idx[s : s + 1000]], horizon, models)
        for k, v in part.items():
            out.setdefault(k, []).append(v)
        print(f"  {min(s + 1000, len(idx))}/{len(idx)} · {time.time() - t0:.0f} s", flush=True)
    seconds = time.time() - t0
    meta = {"kind": "baseline", "bench": args.bench, "method": args.method, "shard": args.shard,
            "nshards": args.nshards, "seconds": seconds, "series": int(len(idx))}
    Path(args.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(args.out) / f"baseline__{args.bench}__{args.method}__{args.shard:02d}of{args.nshards:02d}.npz",
                        idx=idx, meta=np.array(json.dumps(meta)), **{k: np.concatenate(v) for k, v in out.items()})
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
