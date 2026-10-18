"""VN1 part of the static website: scores every method with the official VN1 score and writes public/data/vn1/.

Called by `export_static.py build` when the VN1 data is present in data/vn1.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from vn1bench.data import HORIZON, load_vn1, reference_forecasts, vn1_score

ROOT = Path(__file__).parent
# methods drawn on the chart (comparison checkboxes), in this order, when available
CHART_REFS = ("naive", "ma12", "snaive", "ets", "tsb", "arima", "vn1_1st")
FAMILY = {"vn1_1st": "competition", "vn1_2nd": "competition", "vn1_3rd": "competition", "vn1_4th": "competition",
          "vn1_5th": "competition", "naive": "official", "ma12": "official"}


CALIB_GRID = np.round(np.arange(0.30, 0.801, 0.01), 2)


def calibrate(q_val, levels_val, actual_val, q_test, levels_test, actual_test) -> dict:
    """Choose the quantile level u used as point forecast by minimising the VN1 score on phase 1 (validation),
    then apply that single u to the phase-2 forecasts. The test curve is computed only for display."""
    from export_static import interp_levels

    pv = interp_levels(q_val, levels_val, CALIB_GRID)  # (n, 13, grid)
    pt = interp_levels(q_test, levels_test, CALIB_GRID)
    val = [vn1_score(pv[..., k], actual_val)["score"] for k in range(len(CALIB_GRID))]
    test = [vn1_score(pt[..., k], actual_test)["score"] for k in range(len(CALIB_GRID))]
    k = int(np.argmin(val))
    u = float(CALIB_GRID[k])
    vmed = vn1_score(pv[..., list(CALIB_GRID).index(0.5)], actual_val)
    return {"point": pt[..., k], "meta": {
        "u": u, "test": vn1_score(pt[..., k], actual_test), "val": vn1_score(pv[..., k], actual_val), "val_median": vmed,
        "curve": {"u": CALIB_GRID.tolist(), "val": [round(x, 5) for x in val], "test": [round(x, 5) for x in test]},
        "test_best_u": float(CALIB_GRID[int(np.argmin(test))]), "test_best": float(min(test))}}


def build_vn1(args, out: Path) -> list[str]:
    from export_static import CHUNK, DISPLAY_Q, interp_levels, load_baseline_parts, load_parts, write_chunks
    from m5bench.store import mean_from_quantiles

    data = load_vn1(Path(args.data) / "vn1")
    n, actual = data.n, data.test
    ddir = out / "data" / "vn1"
    ddir.mkdir(parents=True, exist_ok=True)

    # every point forecast that is not Timestral: competition top 5, official benchmarks, statistical baselines
    points = reference_forecasts(Path(args.data) / "vn1", data)
    parts = Path(args.parts)
    if parts.exists():
        points.update({k: v for k, v in load_baseline_parts(parts, "vn1", n).items() if k != "naive_last"})
    baselines = []
    for k, v in points.items():
        r = vn1_score(np.nan_to_num(v), actual)
        baselines.append({"method": k, "family": FAMILY.get(k, "benchmark"), **r})
    baselines.sort(key=lambda b: b["score"])
    print("VN1 references: " + ", ".join(f"{b['method']} {b['score']:.4f}" for b in baselines))

    refs = [k for k in CHART_REFS if k in points]
    base = np.concatenate([data.train, actual] + [np.nan_to_num(points[k]) for k in refs], axis=1)
    write_chunks(ddir / "base", "base", base)

    # Timestral configurations
    bench_cfg = {c["name"]: c for c in json.loads((ROOT / "bench" / "configs.json").read_text())
                 if c.get("bench") == "vn1"}
    configs = load_parts(parts, n, "vn1", HORIZON) if parts.exists() else {}
    # validation runs (history = phase 0, target = phase 1) used to choose the quantile taken as point forecast
    val_cfgs = load_parts(parts, n, "vn1val", HORIZON) if parts.exists() else {}
    val_actual = load_vn1(Path(args.data) / "vn1", "val").test if val_cfgs else None
    cfg_meta = []
    for name, c in configs.items():
        if not c["complete"]:
            print(f"  {name} incomplete, skipped")
            continue
        q, levels = c["q"], c["levels"]
        med = q[..., list(levels).index(0.5)]
        mean = mean_from_quantiles(q, levels)
        lo, hi = q[..., list(levels).index(0.1)], q[..., list(levels).index(0.9)]
        entry = {"name": name, **c["info"], "label": bench_cfg.get(name, {}).get("label", name),
                 "median": vn1_score(med, actual), "mean": vn1_score(mean, actual),
                 "coverage80": float(((actual >= lo - 1e-9) & (actual <= hi + 1e-9)).mean())}
        extra = []
        v = val_cfgs.get(name.replace("vn1_", "vn1val_", 1))
        if v is not None and v["complete"]:
            calib = calibrate(v["q"], v["levels"], val_actual, q, levels, actual)
            entry["calib"] = calib["meta"]
            extra = [calib["point"]]
            print(f"  {name}: quantile chosen on phase 1 = P{100 * calib['meta']['u']:.0f} · validation score "
                  f"{calib['meta']['val']['score']:.4f} · test score {calib['meta']['test']['score']:.4f}")
        cfg_meta.append(entry)
        disp = interp_levels(q, levels, DISPLAY_Q)
        write_chunks(ddir / name, "q", np.concatenate([disp.transpose(0, 2, 1).reshape(n, -1), mean] + extra, axis=1))
        print(f"  {name}: VN1 score median {entry['median']['score']:.4f} · mean {entry['mean']['score']:.4f} · "
              f"80% band coverage {entry['coverage80']:.1%}")
    order = {k: i for i, k in enumerate(bench_cfg)}
    cfg_meta.sort(key=lambda c: order.get(c["name"], 99))
    best = min(cfg_meta, key=lambda c: min([c["median"]["score"], c["mean"]["score"]]
                                           + ([c["calib"]["test"]["score"]] if "calib" in c else [])), default=None)

    # examples: the largest series, a typical one and a lumpy one (many zero weeks)
    tot = data.train[:, -52:].sum(1)
    zs = (data.train[:, -52:] == 0).mean(1)
    big = int(np.argmax(tot))
    mid = int(np.argsort(tot)[int(0.9 * n)])
    lumpy_pool = np.flatnonzero((zs > 0.5) & (zs < 0.7) & (actual.sum(1) > 0))
    lumpy = int(lumpy_pool[np.argmax(tot[lumpy_pool])]) if len(lumpy_pool) else 0
    ids = data.ids
    meta = {
        "generated": time.strftime("%Y-%m-%d"), "chunk": CHUNK, "hist_days": int(data.train.shape[1]),
        "horizon": HORIZON, "bench": "vn1", "freq": "W", "display_q": list(DISPLAY_Q), "refs": refs,
        "start_date": str(data.dates[0].date()), "test_start": str(data.dates[data.train.shape[1]].date()),
        "configs": cfg_meta, "default_config": best["name"] if best else None, "baselines": baselines,
        "examples": [{"key": "vx_big", "i": big}, {"key": "vx_mid", "i": mid}, {"key": "vx_lumpy", "i": lumpy}],
        "n_series": n, "actual_total": float(actual.sum()),
    }
    (ddir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    (ddir / "series.json").write_text(json.dumps(
        {"client": ids["Client"].astype(int).tolist(), "warehouse": ids["Warehouse"].astype(int).tolist(),
         "product": ids["Product"].astype(int).tolist(), "vol": np.rint(tot).astype(int).tolist()},
        separators=(",", ":")), encoding="utf-8")

    lines = ["## VN1", "", "| method | VN1 score | MAE% | Bias% |", "|---|---|---|---|"]
    rows = [(f"{c['name']} ({v})", c[v]) for c in cfg_meta for v in ("median", "mean")]
    rows += [(f"{c['name']} (P{100 * c['calib']['u']:.0f}, chosen on phase 1)", c["calib"]["test"])
             for c in cfg_meta if "calib" in c]
    rows += [(b["method"], b) for b in baselines]
    rows.sort(key=lambda r: r[1]["score"])
    lines += [f"| {k} | {r['score']:.4f} | {r['mae']:.4f} | {r['bias']:+.4f} |" for k, r in rows]
    return lines
