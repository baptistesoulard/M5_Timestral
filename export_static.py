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
# reference version of Timestral shown across the website (the other sizes stay in the size / covariates table):
# as accurate as the 209M model on both benchmarks and about 7 times faster
REFERENCE_MODEL = "Timestral-28M"
# Published results of the M5 Uncertainty track (Makridakis et al., International Journal of Forecasting, 2022, table 5)
UNCERTAINTY_PUBLISHED = [
    {"key": "unc_winner", "wspl": 0.154,
     "levels": [0.074, 0.096, 0.113, 0.091, 0.112, 0.116, 0.136, 0.135, 0.159, 0.284, 0.270, 0.265]},
    {"key": "unc_arima", "wspl": 0.205,
     "levels": [0.158, 0.148, 0.163, 0.147, 0.167, 0.170, 0.202, 0.178, 0.201, 0.322, 0.298, 0.302]},
]
UNCERTAINTY_SOURCE = "https://www.researchgate.net/publication/356943392_The_M5_uncertainty_competition_Results_findings_and_conclusions"


# ============================================================================ forecast
def forecast_vn1(args, fc, progress):
    """VN1: weekly sales, 13-week horizon; covariate "price" = relative price as a past-only covariate."""
    from vn1bench.data import HORIZON as H_VN1
    from vn1bench.data import load_vn1, mask_leading_zeros, relative_price

    phase = "val" if args.bench == "vn1val" else args.bench[3:] if args.bench.startswith("vn1bt") else "test"
    data = load_vn1(Path(args.data) / "vn1", phase)
    idx = np.arange(data.n)[args.shard :: args.nshards]
    if args.probe:
        idx = idx[:: max(1, len(idx) // args.probe)][: args.probe]
    ctx_all = mask_leading_zeros(data.train)
    # products not launched yet at the forecast origin (backtests): no history, forecast 0
    empty = np.isnan(ctx_all).all(axis=1)
    idx_all, idx = idx, idx[~empty[idx]]
    from vn1bench.seasonal import calendar_covariates, seasonal_index, week_of_year

    price = relative_price(data) if args.covariates == "price" else None
    t_hist = data.train.shape[1]
    cal_num = cal_cat = None
    if args.covariates == "calendar":  # week of year + e-commerce peak weeks, known in the past and the future
        cal_num, cal_cat = calendar_covariates(data.dates[: t_hist + H_VN1])
    sidx = None
    if args.covariates == "sidx":  # seasonal index per client, measured on the history only
        sidx = seasonal_index(data.train, data.dates, data.ids["Client"].to_numpy())
        woy = week_of_year(data.dates[: t_hist + H_VN1])
        rows_n = np.arange(data.n)[:, None]
        ctx_all = ctx_all / sidx[rows_n, woy[None, :t_hist]]
        fut = sidx[rows_n, woy[None, t_hist:]]  # (N, 13)
    rows = 2 if price is not None else 6 if cal_num is not None else 1
    bs = max(4, min(args.batch_size, 512 // rows))
    out, levels = None, None
    if args.model.startswith("t0"):  # t0 (The Forecasting Company), sales only
        from m5bench.t0_runner import t0_forecast

        fut_fn = None
        if cal_num is not None:  # e-commerce peak weeks + week of year (sine / cosine), known in the future
            from m5bench.t0_runner import weekday_rows

            rows_t0 = np.concatenate([cal_num, weekday_rows(cal_cat[0], 52)]).astype(np.float32)
            fut_fn = lambda a, b: np.broadcast_to(rows_t0, (b - a,) + rows_t0.shape)  # noqa: E731
        out, levels = t0_forecast(fc, ctx_all[idx], H_VN1, batch_size=128, progress=progress, future=fut_fn)
        if sidx is not None:
            out = out * fut[idx][:, :, None]
        idx_iter = []
    else:
        idx_iter = range(0, len(idx), bs)
    for s in idx_iter:
        b = idx[s : s + bs]
        kw = {"past_covariates": price[b][:, None, :]} if price is not None else {}
        if cal_num is not None:
            kw = {"known_covariates": np.broadcast_to(cal_num, (len(b),) + cal_num.shape).copy(),
                  "categorical_known_covariates": np.broadcast_to(cal_cat, (len(b),) + cal_cat.shape).copy()}
        res = fc.forecast(ctx_all[b][:, None, :], prediction_length=H_VN1, **kw)
        q = res.quantiles[:, 0].cpu().numpy()
        if sidx is not None:
            q = q * fut[b][:, :, None]
        if out is None:
            out = np.empty((len(idx),) + q.shape[1:], dtype=np.float32)
            levels = res.quantile_levels
        out[s : s + len(b)] = q
        progress(s + len(b), len(idx))
    if len(idx) < len(idx_all):
        full = np.zeros((len(idx_all),) + out.shape[1:], dtype=np.float32)
        full[np.searchsorted(idx_all, idx)] = out
        out, idx = full, idx_all
    return idx, out, levels


def cmd_forecast(args) -> None:
    import torch

    from m5bench.timestral_runner import forecast_series, load_forecaster

    if args.threads:
        torch.set_num_threads(args.threads)
    if args.model.startswith("t0"):
        from m5bench.t0_runner import load_t0

        fc_loader = lambda: load_t0(args.model)  # noqa: E731
    else:
        fc_loader = lambda: load_forecaster(args.model)  # noqa: E731
    if args.bench.startswith("vn1"):
        fc = fc_loader()
        t0, last = time.time(), [0.0]

        def progress(done, total):
            now = time.time()
            if now - last[0] > 30 or done == total:
                last[0] = now
                print(f"  {done}/{total} series · {(now - t0) / done * 1000:.0f} ms/series", flush=True)

        idx, q, levels = forecast_vn1(args, fc, progress)
        seconds = time.time() - t0
        meta = {"bench": args.bench, "config": args.config, "model": args.model, "covariates": args.covariates,
                "shard": args.shard, "nshards": args.nshards, "probe": args.probe, "seconds": seconds,
                "series": int(len(idx)), "ms_per_series": 1000 * seconds / len(idx),
                "threads": torch.get_num_threads(), "cpu": platform.processor() or platform.machine()}
        Path(args.out).mkdir(parents=True, exist_ok=True)
        np.savez_compressed(Path(args.out) / f"{args.bench}__{args.config}__{args.shard:02d}of{args.nshards:02d}.npz",
                            idx=idx, q=q, levels=np.asarray(levels), meta=np.array(json.dumps(meta)))
        print(json.dumps(meta), flush=True)
        return
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
    fc = fc_loader()
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

    if args.model.startswith("t0"):  # t0 (The Forecasting Company), sales only, leading zeros masked
        from m5bench.data import mask_leading_zeros
        from m5bench.t0_runner import t0_forecast

        fut_fn = None
        if args.covariates == "calendar":  # SNAP, event day, day of week (sine / cosine), known in the future
            from m5bench.t0_runner import weekday_rows
            from m5bench.timestral_runner import covariate_bank

            bank = covariate_bank(data)
            t_all = hist.shape[1] + HORIZON
            dow = weekday_rows(bank.cal["wday"][:t_all], 7)
            event = (np.asarray(bank.cal["event"][:t_all]) > 0).astype(np.float32)[None, :]

            def fut_fn(a, b):
                snap = bank.build(idx[a:b], "calendar", 0, t_all)["known_covariates"][:, :1, :]
                rest = np.broadcast_to(np.concatenate([event, dow]), (b - a, 3, t_all))
                return np.concatenate([snap, rest], axis=1)
        q, levels = t0_forecast(fc, mask_leading_zeros(hist[idx]), HORIZON, batch_size=min(args.batch_size, 64),
                                progress=progress, future=fut_fn)
    else:
        q, levels = forecast_series(fc, data, idx, hist, None, args.covariates, batch_size=args.batch_size,
                                    progress=progress)
    seconds = time.time() - t0
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"bench": "m5", "config": args.config, "model": args.model, "covariates": args.covariates, "shard": args.shard,
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


def load_parts(parts_dir: Path, n_series: int, bench: str = "m5", horizon: int = HORIZON) -> dict:
    """Group Timestral shard files of one benchmark by configuration."""
    groups: dict[str, list] = {}
    for p in sorted(parts_dir.rglob("*.npz")):
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("kind", "timestral") != "timestral" or meta.get("bench", "m5") != bench:
                continue
            groups.setdefault(meta["config"], []).append((z["idx"], z["q"], tuple(z["levels"]), meta))
    configs = {}
    for name, items in groups.items():
        covered = np.zeros(n_series, dtype=bool)
        levels = items[0][2]
        q_all = np.full((n_series, horizon, len(levels)), np.nan, dtype=np.float32)
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


def load_baseline_parts(parts_dir: Path, bench: str, n_series: int) -> dict[str, np.ndarray]:
    """Statistical baselines computed by bench_baselines.py: {method: (n_series, horizon)}, complete ones only."""
    acc: dict[str, list] = {}
    for p in sorted(Path(parts_dir).rglob("baseline__*.npz")):
        with np.load(p) as z:
            meta = json.loads(str(z["meta"]))
            if meta.get("bench") != bench:
                continue
            for k in z.files:
                if k not in ("idx", "meta"):
                    acc.setdefault(k, []).append((z["idx"], z[k]))
    out = {}
    for k, items in acc.items():
        h = items[0][1].shape[1]
        arr = np.full((n_series, h), np.nan, dtype=np.float32)
        for idx, v in items:
            arr[idx] = v
        if np.isnan(arr).any():
            print(f"  baseline {bench}/{k} incomplete, skipped")
            continue
        out[k] = arr
    return out


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

    md = lambda s: markdown.markdown(s, extensions=["sane_lists", "tables"])  # noqa: E731
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
    m5 = md(LEARN[lang]) + levels + md(LEARN[f"{lang}_after_levels"]) + pipe + md(after)
    # VN1 page: VN1 presentation + the generic sections (foundation models, Timestral) + what is tested on VN1
    from site_texts import VN1_LEARN

    fm = "## Foundation models" if lang == "en" else "## Les modèles de fondation"
    tests = "## What this demo tests" if lang == "en" else "## Ce que teste cette démo"
    generic = md(fm + LEARN[f"{lang}_after_levels"].split(fm, 1)[1]) + pipe + md(after.split(tests, 1)[0])
    from site_texts import VN1_ANALYSIS

    vn1 = md(VN1_LEARN[lang]) + generic + md(VN1_LEARN[f"{lang}_tests"]) + md(VN1_ANALYSIS[lang])
    return {"m5": m5, "vn1": vn1}


def glossary_html(lang: str) -> str:
    import markdown

    from site_texts import GLOSSARY, GLOSSARY_INTRO

    k = 0 if lang == "en" else 1
    out = [f"<p class='lede'>{html.escape(GLOSSARY_INTRO[lang])}</p>"]
    for titles, entries in GLOSSARY:
        out.append(f"<h2>{html.escape(titles[k])}</h2><dl class='glossary'>")
        for anchor, t_en, t_fr, d_en, d_fr in entries:
            term, d = (t_en, d_en) if lang == "en" else (t_fr, d_fr)
            body = markdown.markdown(d)[3:-4]  # strip <p></p>
            out.append(f"<div id='g-{anchor}'><dt>{html.escape(term)}</dt><dd>{body}</dd></div>")
        out.append("</dl>")
    return "".join(out)


# ============================================================================ build
def build_m5(args, out: Path) -> dict:
    data = load_m5(args.data)
    ev = M5Evaluator(data)
    hist = data.aggregate(data.train)
    n = len(data.series)
    slices = level_slices(data.series)
    ddir = out / "data" / "m5"
    ddir.mkdir(parents=True, exist_ok=True)

    # reference forecasts (bottom-up from item × store): shipped with the repo, plus AutoARIMA from GitHub Actions
    bottom_ref = {}
    with np.load(ROOT / "app_assets" / "reference_forecasts.npz") as z:
        for k in REF_KEYS:
            if k in z.files:
                bottom_ref[k] = z[k].astype(np.float32)
    extra = load_baseline_parts(Path(args.parts), "m5", data.train.shape[0]) if Path(args.parts).exists() else {}
    if "arima" in extra:
        bottom_ref["arima"] = extra["arima"]
    ref_keys = [k for k in ("snaive", "tsb", "ets", "arima", "winner") if k in bottom_ref]
    ref = {k: data.aggregate(bottom_ref[k]) for k in ref_keys}
    print(f"M5 references: {ref_keys}")

    # base chunks: history (last HIST_DAYS days), actuals (28), reference forecasts (4 × 28)
    base = np.concatenate([hist[:, -HIST_DAYS:], ev.actual] + [ref[k] for k in ref_keys], axis=1)
    write_chunks(ddir / "base", "base", base)
    np.asarray(ev.scale_sq, dtype="<f4").tofile(ddir / "scale_sq.bin")

    # Timestral configurations
    bench_cfg = {c["name"]: c for c in json.loads((ROOT / "bench" / "configs.json").read_text())}
    configs = load_parts(Path(args.parts), n, "m5") if Path(args.parts).exists() else {}
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

    order = {c["name"]: k for k, c in enumerate(json.loads((ROOT / "bench" / "configs.json").read_text()))
             if c.get("bench", "m5") == "m5"}
    cfg_meta.sort(key=lambda c: order.get(c["name"], 99))
    ref_cfgs = [c for c in cfg_meta if c["model"] == REFERENCE_MODEL] or cfg_meta
    # reference configuration: the reference model with calendar covariates when available
    best = next((c for c in ref_cfgs if c["covariates"] == "calendar"),
                min(ref_cfgs, key=lambda c: min(c["wrmsse"]["median|bottom-up"], c["wrmsse"]["mean|bottom-up"]),
                    default=None))

    # baselines and competition entries (computed by evaluate.py, shipped in app_assets)
    base_scores = pd.read_csv(ROOT / "app_assets" / "scores.csv")
    base_levels = pd.read_csv(ROOT / "app_assets" / "by_level.csv")
    for r in base_levels.itertuples():
        by_level.append({"method": r.méthode, "variant": "" if pd.isna(r.variante) else r.variante,
                         "mode": r.agrégation, "metric": r.métrique, "level": r.level, "score": float(r.score)})
    baselines = [{"method": r.méthode, "family": r.famille, "mode": r.agrégation, "metric": r.métrique,
                  "score": float(r.score)} for r in base_scores.itertuples()]
    for k, v in extra.items():  # baselines computed in GitHub Actions (AutoARIMA)
        t = ev.wrmsse(v)
        baselines.append({"method": k, "family": "benchmark", "mode": "bottom-up", "metric": "wrmsse",
                          "score": float(t.loc[t["level"] == "Total", "wrmsse"].iloc[0])})
        by_level += [{"method": k, "variant": "", "mode": "bottom-up", "metric": "wrmsse", "level": r.level,
                      "score": float(r.wrmsse)} for r in t.itertuples()]
        print(f"  {k}: WRMSSE {baselines[-1]['score']:.4f}")

    ids = data.bottom_ids
    depts = sorted(ids["dept_id"].unique())
    cal = data.calendar
    t_end = data.train.shape[1]
    window = slice(t_end - HIST_DAYS, t_end + HORIZON)
    events = [[int(d - window.start), str(e)] for d, e in enumerate(cal["event_name_1"].fillna(""))
              if window.start <= d < window.stop and e]
    meta = {
        "generated": time.strftime("%Y-%m-%d"), "chunk": CHUNK, "hist_days": HIST_DAYS, "horizon": HORIZON,
        "bench": "m5", "freq": "D", "display_q": DISPLAY_Q, "refs": ref_keys, "reference_model": REFERENCE_MODEL,
        "start_date": str(cal["date"].iloc[window.start]), "test_start": str(cal["date"].iloc[t_end]),
        "events": events, "configs": cfg_meta, "default_config": best["name"] if best else None,
        "baselines": baselines, "by_level": by_level,
        "uncertainty_published": UNCERTAINTY_PUBLISHED, "uncertainty_source": UNCERTAINTY_SOURCE,
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
    lines = ["## M5", "", "| configuration | WRMSSE median (bottom-up) | WRMSSE mean (bottom-up) | WSPL | ms/series |",
             "|---|---|---|---|---|"]
    lines += [f"| {c['name']} | {c['wrmsse']['median|bottom-up']:.4f} | {c['wrmsse']['mean|bottom-up']:.4f} | "
              f"{c['wspl']:.4f} | {c['ms_per_series']:.0f} |" for c in cfg_meta]
    lines += [f"| {b['method']} | | {b['score']:.4f} | | |" for b in baselines
              if b["method"] in extra and b["metric"] == "wrmsse"]
    return {"slices": slices, "summary": lines}


def cmd_build(args) -> None:
    t_start = time.time()
    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(ROOT / "site", out)
    (out / "data").mkdir(exist_ok=True)
    benches, summary = [], []
    m5 = build_m5(args, out)
    benches.append("m5")
    summary += m5["summary"]
    vn1_dir = Path(args.data) / "vn1"
    if (vn1_dir / "Phase 2 - Sales.csv").exists():
        from vn1_static import build_vn1

        summary += [""] + build_vn1(args, out)
        benches.append("vn1")
    (out / "data" / "index.json").write_text(json.dumps({"benches": benches, "generated": time.strftime("%Y-%m-%d")}),
                                             encoding="utf-8")
    (out / "data" / "learn.json").write_text(json.dumps({lg: learn_html(lg, m5["slices"]) for lg in ("en", "fr")},
                                                        ensure_ascii=False), encoding="utf-8")
    (out / "data" / "glossary.json").write_text(json.dumps({lg: glossary_html(lg) for lg in ("en", "fr")},
                                                           ensure_ascii=False), encoding="utf-8")
    (out / ".nojekyll").write_text("")
    (out / "data" / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    print("\n".join(summary))
    print(f"site written to {out} · {size / 1e6:.0f} MB · {time.time() - t_start:.0f} s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("forecast", help="forecast all series (or one shard) for one configuration")
    f.add_argument("--config", required=True)
    f.add_argument("--model", required=True)
    f.add_argument("--bench", default="m5", choices=["m5", "vn1", "vn1val", "vn1bt131", "vn1bt144", "vn1bt157"])
    f.add_argument("--covariates", default="calendar", choices=["none", "calendar", "full", "rich", "price", "sidx"])
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
