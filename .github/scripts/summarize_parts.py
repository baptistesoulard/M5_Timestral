"""Speed and sanity summary of the forecast shards (used for probe runs and in every report)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from m5bench.data import load_m5  # noqa: E402
from m5bench.metrics import M5Evaluator  # noqa: E402
from m5bench.store import mean_from_quantiles  # noqa: E402

parts = sorted(Path(sys.argv[1] if len(sys.argv) > 1 else "parts").rglob("*.npz"))
if not parts:
    print("No forecast shard found.")
    sys.exit(0)
data = load_m5("data")
ev = M5Evaluator(data)
ref = {}
with np.load("app_assets/reference_forecasts.npz") as z:
    for k in ("snaive", "ets", "winner"):
        if k in z.files:
            ref[k] = data.aggregate(z[k].astype(np.float32))
levels_of = data.series["level"].to_numpy()
print("| shard | series | ms/series | est. full run (min, 1 job) | finite | RMSSE median | RMSSE mean | ETS | sNaive | winner |")
print("|---|---|---|---|---|---|---|---|---|---|")
for p in parts:
    with np.load(p) as z:
        meta = json.loads(str(z["meta"]))
        if meta.get("kind", "timestral") != "timestral" or meta.get("bench", "m5") != "m5":
            print(f"| {p.stem} | {meta.get('series')} | {1000 * meta['seconds'] / max(meta.get('series') or 1, 1):.0f} | | | | | | | |")
            continue
        idx, q, lv = z["idx"], z["q"], list(z["levels"])
    med = q[..., lv.index(0.5)]
    mean = mean_from_quantiles(q, lv)
    r = lambda f: float(np.nanmean(ev.rmsse(f, idx)))  # noqa: E731
    est = meta["ms_per_series"] * len(levels_of) / 1000 / 60
    print(f"| {p.stem} | {meta['series']} | {meta['ms_per_series']:.0f} | {est:.0f} | {np.isfinite(q).mean():.3f} | "
          f"{r(med):.3f} | {r(mean):.3f} | " + " | ".join(f"{r(ref[k][idx]):.3f}" for k in ("ets", "snaive", "winner")) + " |")
