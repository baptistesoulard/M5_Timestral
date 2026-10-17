"""Static version of the demo: Timestral forecasts computed once, then served as a static website.

Two steps, runnable on a PC or in GitHub Actions (.github/workflows/site.yml):

    # 1. Forecast all 42,840 series for one configuration (optionally split into shards)
    python export_static.py forecast --config t28m_calendar --model Timestral-28M --covariates calendar
    python export_static.py forecast --config t209m_rich --model Timestral-209M --covariates rich --shard 0 --nshards 8

    # 2. Merge the shards, score everything (official WRMSSE / WSPL) and write the website to public/
    python export_static.py build --parts parts --out public

Website data: per-series files are split into chunks of 512 series and quantised to uint16 with a per-series
scale, so the browser only downloads the series it displays.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import platform
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from m5bench.data import HORIZON, LEVELS, load_m5, level_slices
from m5bench.metrics import M5Evaluator, interpolate_quantiles
from m5bench.store import mean_from_quantiles

ROOT = Path(__file__).parent
CHUNK = 512
HIST_DAYS = 364
DISPLAY_Q = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99)
REF_KEYS = ("snaive", "tsb", "ets", "winner")


# ============================================================================ forecast
def cmd_forecast(args) -> None:
    import torch

    from m5bench.timestral_runner import forecast_series, load_forecaster

    if args.threads:
        torch.set_num_threads(args.threads)
    data = load_m5(args.data)
    hist = data.aggregate(data.train)
    n = len(data.series)
    idx = np.arange(n)[args.shard :: args.nshards]
    if args.probe:
        # spread the probe over every level, aggregates included
        lv = data.series["level"].to_numpy()[idx]
        picks = [idx[lv == l][: max(2, args.probe // 12)] for l, _ in LEVELS]
        idx = np.concatenate(picks)[: max(args.probe, 24)]
    print(f"{args.config}: {len(idx)} series · model {args.model} · covariates {args.covariates} · "
          f"{torch.get_num_threads()} threads · {platform.processor() or platform.machine()}", flush=True)

    t_load = time.time()
    fc = load_forecaster(args.model)
    print(f"model loaded in {time.time() - t_load:.0f} s", flush=True)

    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        now = time.time()
        if now - last[0] > 30 or done == total:
            last[0] = now
            rate = (now - t0) / done
            print(f"  {done}/{total} series · {rate * 1000:.0f} ms/series · ~{rate * (total - done) / 60:.0f} min left",
                  flush=True)

    q, levels = forecast_series(fc, data, idx, hist, None, args.covariates, batch_size=args.batch_size,
                                progress=progress)
    seconds = time.time() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"config": args.config, "model": args.model, "covariates": args.covariates, "shard": args.shard,
            "nshards": args.nshards, "probe": args.probe, "seconds": seconds, "series": int(len(idx)),
            "ms_per_series": 1000 * seconds / len(idx), "threads": torch.get_num_threads(),
            "cpu": platform.processor() or platform.machine(), "cpu_count": os.cpu_count()}
    np.savez_compressed(out / f"{args.config}__{args.shard:02d}of{args.nshards:02d}.npz", idx=idx,
                        q=q.astype(np.float32), levels=np.asarray(levels), meta=np.array(json.dumps(meta)))
    print(json.dumps(meta), flush=True)


# ============================================================================ build helpers
def quantize(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(n, k) non-negative floats → per-row scales (n,) float32 and uint16 codes (n, k)."""
    block = np.nan_to_num(np.maximum(block, 0), nan=0.0).astype(np.float64)
    scale = block.max(axis=1)
    scale[scale <= 0] = 1.0
    codes = np.rint(block / scale[:, None] * 65535).astype(np.uint16)
    return scale.astype(np.float32), codes


def write_chunks(folder: Path, prefix: str, values: np.ndarray) -> None:
    """values (N, k) → folder/prefix_{c}.bin = float32 scales (n) + uint16 codes (n*k), little-endian."""
    folder.mkdir(parents=True, exist_ok=True)
    for c, s in enumerate(range(0, len(values), CHUNK)):
        scale, codes = quantize(values[s : s + CHUNK])
        with open(folder / f"{prefix}_{c}.bin", "wb") as f:
            f.write(scale.astype("<f4").tobytes())
            f.write(codes.astype("<u2").tobytes())


def load_parts(parts_dir: Path, n_series: int) -> dict:
    """Group shard files by configuration; keep complete configurations only."""
    groups: dict[str, list] = {}
    for p in sorted(parts_dir.rglob("*.npz")):
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            groups.setdefault(meta["config"], []).append((z["idx"], z["q"], tuple(z["levels"]), meta))
    configs = {}
    for name, items in groups.items():
        covered = np.zeros(n_series, dtype=bool)
        levels = items[0][2]
        q_all = np.full((n_series, HORIZON, len(levels)), np.nan, dtype=np.float32)
        for idx, q, _, _ in items:
            q_all[idx] = q
            covered[idx] = True
        metas = [m for *_, m in items]
        info = {"seconds": sum(m["seconds"] for m in metas), "series": int(covered.sum()),
                "model": metas[0]["model"], "covariates": metas[0]["covariates"], "shards": len(metas),
                "threads": metas[0]["threads"], "cpu": metas[0]["cpu"], "probe": any(m["probe"] for m in metas)}
        info["ms_per_series"] = 1000 * info["seconds"] / max(info["series"], 1)
        configs[name] = {"q": q_all, "levels": levels, "complete": bool(covered.all()), "info": info}
        print(f"  {name}: {covered.sum()}/{n_series} series, {len(metas)} shard(s), "
              f"{info['ms_per_series']:.0f} ms/series")
    return configs


def interp_levels(q: np.ndarray, src_levels, dst_levels) -> np.ndarray:
    """(..., Q_src) → (..., Q_dst) by linear interpolation inside the source grid."""
    src = np.asarray(src_levels, dtype=np.float64)
    out = np.empty(q.shape[:-1] + (len(dst_levels),), dtype=np.float32)
    for j, u in enumerate(dst_levels):
        k = int(np.clip(np.searchsorted(src, u) - 1, 0, len(src) - 2))
        w = (u - src[k]) / (src[k + 1] - src[k])
        out[..., j] = q[..., k] + w * (q[..., k + 1] - q[..., k])
    return out


def pick_examples(data, ev, hist, ref) -> list[dict]:
    slices = level_slices(data.series)
    key = {(r.level, r.agg1, r.agg2): i for i, r in enumerate(data.series.itertuples())}
    l12 = slices["Level12"]
    w = ev.weights[l12]

    def distinct(i):
        vals = [ref[k][i] for k in ref]
        return all(not np.allclose(a, b) for n, a in enumerate(vals) for b in vals[n + 1:])

    order = l12[np.argsort(-w)]
    top = next(i for i in order if distinct(i))
    ranked = order[:5000]
    zs = np.array([(hist[i, -182:] == 0).mean() for i in ranked])
    gap = np.array([np.diff(np.flatnonzero(np.r_[1, hist[i, -182:], 1] > 0)).max() for i in ranked])
    inter = next((i for i in ranked[(zs > 0.6) & (zs < 0.8) & (gap < 15)] if distinct(i)), l12[0])
    return [{"key": "ex_total", "i": int(key[("Level1", "Total", "X")])},
            {"key": "ex_store", "i": int(key[("Level8", "CA_3", "FOODS")])},
            {"key": "ex_fast", "i": int(top)}, {"key": "ex_inter", "i": int(inter)}]


def learn_html(lang: str, slices) -> str:
    import markdown

    from app_texts import LEARN, LEVEL_LABELS, PIPELINE

    md = lambda s: markdown.markdown(s, extensions=["sane_lists"])  # noqa: E731
    fmt = (lambda v: f"{v:,}".replace(",", " ")) if lang == "fr" else (lambda v: f"{v:,}")
    head = ("<th>#</th><th>Niveau</th><th class='num'>Séries</th>" if lang == "fr"
            else "<th>#</th><th>Level</th><th class='num'>Series</th>")
    rows = "".join(f"<tr><td>{i + 1}</td><td>{html.escape(LEVEL_LABELS[lang][l])}</td>"
                   f"<td class='num'>{fmt(len(slices[l]))}</td></tr>" for i, (l, _) in enumerate(LEVELS))
    levels = (f"<h3>{LEARN[f'{lang}_levels_title']}</h3><div class='table-wrap'><table class='levels'>"
              f"<thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>")
    cap = "Comment Timestral produit une prévision" if lang == "fr" else "How Timestral produces a forecast"
    steps = "".join(f"<li><span class='step-n'>{n}</span><strong>{html.escape(t)}</strong><span>{html.escape(d)}</span></li>"
                    for n, t, d in PIPELINE[lang])
    pipe = f"<figure class='pipeline'><figcaption>{cap}</figcaption><ol>{steps}</ol></figure>"
    after = LEARN[f"{lang}_after_pipeline"]
    return md(LEARN[lang]) + levels + md(LEARN[f"{lang}_after_levels"]) + pipe + md(after)


# ============================================================================ build
def cmd_build(args) -> None:
    t_start = time.time()
    data = load_m5(args.data)
    ev = M5Evaluator(data)
    hist = data.aggregate(data.train)
    n = len(data.series)
    slices = level_slices(data.series)
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)
    ddir = out / "data"
    ddir.mkdir(exist_ok=True)

    # reference forecasts (bottom-up from item × store) shipped with the repo
    ref = {}
    with np.load(ROOT / "app_assets" / "reference_forecasts.npz") as z:
        for k in REF_KEYS:
            if k in z.files:
                ref[k] = data.aggregate(z[k].astype(np.float32))
    print(f"references: {list(ref)}")

    # base chunks: history (last HIST_DAYS days), actuals (28), reference forecasts (4 × 28)
    base = np.concatenate([hist[:, -HIST_DAYS:], ev.actual] + [ref[k] for k in REF_KEYS if k in ref], axis=1)
    write_chunks(ddir / "base", "base", base)
    np.asarray(ev.scale_sq, dtype="<f4").tofile(ddir / "scale_sq.bin")

    # Timestral configurations
    bench_cfg = {c["name"]: c for c in json.loads((ROOT / "bench" / "configs.json").read_text())}
    configs = load_parts(Path(args.parts), n) if Path(args.parts).exists() else {}
    scores, by_level, cfg_meta = [], [], []
    for name, c in configs.items():
        if not c["complete"]:
            print(f"  {name} incomplete, skipped")
            continue
        q, levels = c["q"], c["levels"]
        med = q[..., list(levels).index(0.5)]
        mean = mean_from_quantiles(q, levels)
        res = {}
        for variant, point in (("median", med), ("mean", mean)):
            for mode, arr in (("bottom-up", point[slices["Level12"]]), ("direct", point)):
                t = ev.wrmsse(arr)
                res[(variant, mode)] = t
                by_level += [{"method": name, "variant": variant, "mode": mode, "metric": "wrmsse",
                              "level": r.level, "score": float(r.wrmsse)} for r in t.itertuples()]
        wspl = ev.wspl(interpolate_quantiles(q, levels))
        by_level += [{"method": name, "variant": "", "mode": "direct", "metric": "wspl", "level": r.level,
                      "score": float(r.wspl)} for r in wspl.itertuples()]
        total = lambda t, col: float(t.loc[t["level"] == "Total", col].iloc[0])  # noqa: E731
        entry = {"name": name, **c["info"], "label": bench_cfg.get(name, {}).get("label", name),
                 "wrmsse": {f"{v}|{m}": total(t, "wrmsse") for (v, m), t in res.items()},
                 "wspl": total(wspl, "wspl")}
        cfg_meta.append(entry)
        disp = interp_levels(q, levels, DISPLAY_Q)  # (N, 28, 9)
        block = np.concatenate([disp.transpose(0, 2, 1).reshape(n, -1), mean], axis=1)  # 9×28 quantiles + mean
        write_chunks(ddir / name, "q", block)
        print(f"  {name}: WRMSSE bottom-up median {entry['wrmsse']['median|bottom-up']:.4f} · "
              f"mean {entry['wrmsse']['mean|bottom-up']:.4f} · WSPL {entry['wspl']:.4f}")

    order = {c["name"]: k for k, c in enumerate(json.loads((ROOT / "bench" / "configs.json").read_text()))}
    cfg_meta.sort(key=lambda c: order.get(c["name"], 99))
    best = min(cfg_meta, key=lambda c: min(c["wrmsse"]["median|bottom-up"], c["wrmsse"]["mean|bottom-up"]),
               default=None)

    # baselines and competition entries (computed by evaluate.py, shipped in app_assets)
    base_scores = pd.read_csv(ROOT / "app_assets" / "scores.csv")
    base_levels = pd.read_csv(ROOT / "app_assets" / "by_level.csv")
    for r in base_levels.itertuples():
        by_level.append({"method": r.méthode, "variant": "" if pd.isna(r.variante) else r.variante,
                         "mode": r.agrégation, "metric": r.métrique, "level": r.level, "score": float(r.score)})
    baselines = [{"method": r.méthode, "family": r.famille, "mode": r.agrégation, "metric": r.métrique,
                  "score": float(r.score)} for r in base_scores.itertuples()]

    ids = data.bottom_ids
    depts = sorted(ids["dept_id"].unique())
    cal = data.calendar
    t_end = data.train.shape[1]
    window = slice(t_end - HIST_DAYS, t_end + HORIZON)
    events = [[int(d - window.start), str(e)] for d, e in enumerate(cal["event_name_1"].fillna(""))
              if window.start <= d < window.stop and e]
    meta = {
        "generated": time.strftime("%Y-%m-%d"), "chunk": CHUNK, "hist_days": HIST_DAYS, "horizon": HORIZON,
        "display_q": DISPLAY_Q, "refs": [k for k in REF_KEYS if k in ref],
        "start_date": str(cal["date"].iloc[window.start]), "test_start": str(cal["date"].iloc[t_end]),
        "events": events, "configs": cfg_meta, "default_config": best["name"] if best else None,
        "baselines": baselines, "by_level": by_level,
        "examples": pick_examples(data, ev, hist, ref),
        "states": sorted(ids["state_id"].unique()), "stores": sorted(ids["store_id"].unique()),
        "cats": sorted(ids["cat_id"].unique()), "depts": depts,
        "items_by_dept": {d: sorted(ids.loc[ids["dept_id"] == d, "item_id"].unique()) for d in depts},
        "level_counts": {l: int(len(slices[l])) for l, _ in LEVELS},
    }
    (ddir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    level_num = data.series["level"].str.replace("Level", "").astype(int).tolist()
    (ddir / "series.json").write_text(json.dumps(
        {"level": level_num, "a1": data.series["agg1"].tolist(), "a2": data.series["agg2"].tolist()},
        separators=(",", ":")), encoding="utf-8")
    (ddir / "learn.json").write_text(json.dumps({lg: learn_html(lg, slices) for lg in ("en", "fr")},
                                                ensure_ascii=False), encoding="utf-8")
    (out / ".nojekyll").write_text("")

    # short summary for the repository (results branch)
    lines = ["| configuration | WRMSSE median (bottom-up) | WRMSSE mean (bottom-up) | WSPL | ms/series |",
             "|---|---|---|---|---|"]
    lines += [f"| {c['name']} | {c['wrmsse']['median|bottom-up']:.4f} | {c['wrmsse']['mean|bottom-up']:.4f} | "
              f"{c['wspl']:.4f} | {c['ms_per_series']:.0f} |" for c in cfg_meta]
    (out / "data" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print("\n".join(lines))
    print(f"site written to {out} · {size / 1e6:.0f} MB · {time.time() - t_start:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="forecast all series (or one shard) for one configuration")
    f.add_argument("--config", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--covariates", default="calendar", choices=["none", "calendar", "full", "rich"])
    f.add_argument("--shard", type=int, default=0)
    f.add_argument("--nshards", type=int, default=1)
    f.add_argument("--probe", type=int, default=0, help="forecast only N series spread over the levels")
    f.add_argument("--batch-size", type=int, default=512, help="upper bound; capped by the number of covariate rows")
    f.add_argument("--threads", type=int, default=None)
    f.add_argument("--data", default="data")
    f.add_argument("--out", default="parts")
    b = sub.add_parser("build", help="merge shards, score and write the static website")
    b.add_argument("--parts", default="parts")
    b.add_argument("--data", default="data")
    b.add_argument("--out", default="public")
    args = ap.parse_args()
    cmd_forecast(args) if args.cmd == "forecast" else cmd_build(args)


if __name__ == "__main__":
    main()
