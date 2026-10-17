"""Interactive demo, in English and French: Timestral (Mistral) on the M5 Walmart benchmark.

Run locally:               python app.py            → http://localhost:7860
Test mode (no weights):    TIMESTRAL_TEST_MODEL=tiny-random python app.py
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import threading
import time
from functools import lru_cache
from pathlib import Path

import gradio as gr
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch

from app_texts import FREQ_NAMES, LANGS, LEADERBOARD_NAMES, LEARN, LEVEL_LABELS, METHOD_LABELS, PIPELINE
from app_texts import T as TEXTS
from download_data import ensure_data
from m5bench.data import HORIZON, LEVELS, calendar_covariates, level_slices, load_m5
from m5bench.metrics import M5Evaluator, interpolate_quantiles
from m5bench.store import mean_from_quantiles
from m5bench.timestral_runner import MODELS, forecast_series, load_forecaster
from run_baselines import empirical_weekday_quantiles

ROOT = Path(__file__).parent
DATA_DIR = Path(os.environ.get("M5_DATA", ROOT / "data"))
ASSETS = ROOT / "app_assets"
TEST_MODEL = os.environ.get("TIMESTRAL_TEST_MODEL")  # "tiny-random" to test the UI without downloading weights
DEFAULT_MODEL = "Timestral-28M"
torch.set_num_threads(max(1, os.cpu_count() or 1))

# Colours: Timestral in orange (Mistral), actuals in ink, comparison methods in cool tones
C_TIMESTRAL, C_ACTUAL, C_HISTORY = "#E8590C", "#1F2328", "#8C959F"
C_METHODS = {"snaive": "#4C78A8", "tsb": "#72B7B2", "ets": "#54A24B", "winner": "#B279A2"}


def tr(lang: str, key: str, **kw) -> str:
    s = TEXTS.get(lang, TEXTS["en"]).get(key, TEXTS["en"][key])
    return s.format(**kw) if kw else s


def fnum(x: float, lang: str, d: int = 3) -> str:
    s = f"{x:.{d}f}"
    return s.replace(".", ",") if lang == "fr" else s


def fint(x: float, lang: str) -> str:
    s = f"{x:,.0f}"
    return s.replace(",", " ") if lang == "fr" else s


def fmt_pct(x: float, lang: str, signed: bool = False) -> str:
    if x is None or np.isnan(x):
        return "–"
    v = (f"{x * 100:+.0f}" if signed else f"{x * 100:.0f}").replace("-", "−")
    return v + (" %" if lang == "fr" else "%")


# ----------------------------------------------------------------------------- data
print("Loading M5 data…", flush=True)
ensure_data(DATA_DIR, top=1)
DATA = load_m5(DATA_DIR)
HIST = DATA.aggregate(DATA.train)  # (42840, 1941)
T = HIST.shape[1]
CAL = calendar_covariates(DATA.calendar)
EV = M5Evaluator(DATA)
SLICES = level_slices(DATA.series)
DATES = pd.to_datetime(DATA.calendar["date"]).to_numpy()
EVENTS = DATA.calendar["event_name_1"].fillna("").to_numpy()
SERIES_KEY = {(r.level, r.agg1, r.agg2): i for i, r in enumerate(DATA.series.itertuples())}

REF = {}
_ref_path = ASSETS / "reference_forecasts.npz"
if _ref_path.exists():
    with np.load(_ref_path) as z:
        for k in json.loads(str(z["labels"])):
            REF[k] = DATA.aggregate(z[k].astype(np.float32))  # bottom-up: (42840, 28)
else:  # fallback: seasonal naive computed on the fly
    REF["snaive"] = DATA.aggregate(np.tile(DATA.train[:, -7:], HORIZON // 7))
EMP_Q = empirical_weekday_quantiles(HIST)  # (42840, 28, 9) reference for the quantile track
SCORES = pd.read_csv(ASSETS / "scores.csv") if (ASSETS / "scores.csv").exists() else None

ids = DATA.bottom_ids
STATES = sorted(ids["state_id"].unique())
STORES = sorted(ids["store_id"].unique())
CATS = sorted(ids["cat_id"].unique())
DEPTS = sorted(ids["dept_id"].unique())
ITEMS_BY_DEPT = {d: sorted(ids.loc[ids["dept_id"] == d, "item_id"].unique()) for d in DEPTS}
DEPT_OF_ITEM = dict(zip(ids["item_id"], ids["dept_id"]))
LIEU = {"Level2": "state", "Level3": "store", "Level6": "state", "Level7": "state", "Level8": "store",
        "Level9": "store", "Level11": "state", "Level12": "store"}
PRODUIT = {"Level4": "cat", "Level5": "dept", "Level6": "cat", "Level7": "dept", "Level8": "cat", "Level9": "dept"}
ITEM_LEVELS = {"Level10", "Level11", "Level12"}


def mlabel(lang: str, k: str) -> str:
    return METHOD_LABELS[lang].get(k, k)


def level_choices(lang):
    return [(f"{i + 1}. {LEVEL_LABELS[lang][l]}", l) for i, (l, _) in enumerate(LEVELS)]


def cov_choices(lang):
    return [(tr(lang, "cov_none"), "none"), (tr(lang, "cov_calendar"), "calendar"), (tr(lang, "cov_full"), "full")]


def compare_choices(lang):
    return [(mlabel(lang, k), k) for k in REF]


def series_name(i: int, lang: str) -> str:
    r = DATA.series.iloc[i]
    parts = [p for p in (r.agg1, r.agg2) if p not in ("X", "Total")]
    return f"{LEVEL_LABELS[lang][r.level]} · {' · '.join(parts)}" if parts else tr(lang, "total_series")


def zero_share(i: int, days: int = 365) -> float:
    """Share of zero-sales days over the last `days` days of history."""
    return float((HIST[i, -days:] == 0).mean())


def pick_examples() -> list[tuple[str, int]]:
    l12 = SLICES["Level12"]
    w = EV.weights[l12]

    def distinct(i):  # skip the rare series where two comparison methods give the same forecast
        vals = [REF[k][i] for k in REF]
        return all(not np.allclose(a, b) for n, a in enumerate(vals) for b in vals[n + 1:])

    top = next(i for i in l12[np.argsort(-w)] if distinct(i))
    ranked = l12[np.argsort(-w)[:5000]]
    zs = np.array([zero_share(i, 182) for i in ranked])
    longest_gap = np.array([np.diff(np.flatnonzero(np.r_[1, HIST[i, -182:], 1] > 0)).max() for i in ranked])
    cand = ranked[(zs > 0.6) & (zs < 0.8) & (longest_gap < 15)]  # genuine intermittency, not a long stock-out
    inter = next((i for i in cand if distinct(i)), l12[0])
    return [("ex_total", SERIES_KEY[("Level1", "Total", "X")]), ("ex_store", SERIES_KEY[("Level8", "CA_3", "FOODS")]),
            ("ex_fast", int(top)), ("ex_inter", int(inter))]


EXAMPLES = pick_examples()


# ----------------------------------------------------------------------------- model
_model_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(name: str):
    return load_forecaster(TEST_MODEL or name)


def get_model(name: str, lang: str):
    with _model_lock:
        try:
            return _load(name)
        except Exception as e:  # network or Hugging Face unavailable
            raise gr.Error(tr(lang, "e_load", name=name, err=e)) from e


threading.Thread(target=lambda: _load(DEFAULT_MODEL), daemon=True).start()  # warm-up


# ----------------------------------------------------------------------------- series selection
def resolve(level, lieu, produit, article, lang="en") -> int:
    if level == "Level1":
        key = ("Level1", "Total", "X")
    elif level in ("Level2", "Level3"):
        key = (level, lieu, "X")
    elif level in ("Level4", "Level5"):
        key = (level, produit, "X")
    elif level in PRODUIT:
        key = (level, lieu, produit)
    elif level == "Level10":
        key = (level, article, "X")
    elif level == "Level11":
        key = (level, lieu, article)
    else:
        key = (level, article, lieu)
    if key not in SERIES_KEY:
        raise gr.Error(tr(lang, "e_select"))
    return SERIES_KEY[key]


def selector_updates(level, lang, lieu=None, produit=None, rayon=None, article=None):
    """Update the dropdowns for the selected aggregation level."""
    kind = LIEU.get(level)
    lieu_choices = STATES if kind == "state" else STORES
    lieu_u = gr.update(visible=kind is not None, choices=lieu_choices,
                       value=lieu if lieu in lieu_choices else lieu_choices[0],
                       label=tr(lang, "state" if kind == "state" else "store"))
    pk = PRODUIT.get(level)
    prod_choices = CATS if pk == "cat" else DEPTS
    prod_u = gr.update(visible=pk is not None, choices=prod_choices,
                       value=produit if produit in prod_choices else prod_choices[0],
                       label=tr(lang, "category" if pk == "cat" else "department"))
    is_item = level in ITEM_LEVELS
    rayon = rayon if rayon in DEPTS else "FOODS_3"
    items = ITEMS_BY_DEPT[rayon]
    rayon_u = gr.update(visible=is_item, value=rayon, label=tr(lang, "department"))
    art_u = gr.update(visible=is_item, choices=items, value=article if article in items else items[0],
                      label=tr(lang, "item"))
    return lieu_u, prod_u, rayon_u, art_u


def example_updates(i: int, lang: str):
    r = DATA.series.iloc[i]
    level = r.level
    lieu = produit = article = None
    if level in ("Level2", "Level3"):
        lieu = r.agg1
    elif level in ("Level4", "Level5"):
        produit = r.agg1
    elif level in PRODUIT:
        lieu, produit = r.agg1, r.agg2
    elif level == "Level10":
        article = r.agg1
    elif level == "Level11":
        lieu, article = r.agg1, r.agg2
    elif level == "Level12":
        article, lieu = r.agg1, r.agg2
    return (gr.update(value=level),) + selector_updates(level, lang, lieu, produit, DEPT_OF_ITEM.get(article), article)


# ----------------------------------------------------------------------------- per-series computations
def quantile_at(q: np.ndarray, levels, u: float) -> np.ndarray:
    """Quantile u (interpolated) from a (28, Q) grid."""
    lv = np.asarray(levels)
    return np.array([np.interp(u, lv, row) for row in q])


def series_metrics(i: int, f: np.ndarray) -> dict:
    y = EV.actual[i]
    tot = y.sum()
    return {"rmsse": float(EV.rmsse(f[None], np.array([i]))[0]),
            "wape": float(np.abs(f - y).sum() / tot) if tot > 0 else np.nan,
            "bias": float((f.sum() - tot) / tot) if tot > 0 else np.nan,
            "total": float(f.sum())}


def kpi_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>' \
           f'<div class="kpi-sub">{sub}</div></div>'


def make_forecast_figure(i, q, levels, compare, show_days, lang):
    h0, h1 = T - show_days, T
    hx, fx = DATES[h0:h1], DATES[T : T + HORIZON]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hx, y=HIST[i, h0:h1], name=tr(lang, "history"), mode="lines",
                             line=dict(color=C_HISTORY, width=1.4)))
    for lo_u, hi_u, key, alpha in [(0.01, 0.99, "band98", 0.12), (0.1, 0.9, "band80", 0.25)]:
        lo, hi = quantile_at(q, levels, lo_u), quantile_at(q, levels, hi_u)
        fig.add_trace(go.Scatter(x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([hi, lo[::-1]]),
                                 fill="toself", fillcolor=f"rgba(232,89,12,{alpha})", line=dict(width=0),
                                 name=tr(lang, key), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=quantile_at(q, levels, 0.5), name=tr(lang, "median"), mode="lines",
                             line=dict(color=C_TIMESTRAL, width=2.6)))
    for k in compare or []:
        if k in REF:
            fig.add_trace(go.Scatter(x=fx, y=REF[k][i], name=mlabel(lang, k), mode="lines",
                                     line=dict(color=C_METHODS.get(k, "#666"), width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=fx, y=EV.actual[i], name=tr(lang, "actual_test"), mode="lines+markers",
                             line=dict(color=C_ACTUAL, width=2), marker=dict(size=4)))
    fig.add_vline(x=DATES[T], line=dict(color="#57606A", width=1, dash="dash"))
    fig.add_annotation(x=DATES[T], y=1.02, yref="paper", text=tr(lang, "forecast_start"), showarrow=False,
                       font=dict(size=11, color="#57606A"), xanchor="left")
    for d in range(T, T + HORIZON):  # events in the horizon (known in advance)
        if EVENTS[d]:
            fig.add_annotation(x=DATES[d], y=0.98, yref="paper", text=EVENTS[d], showarrow=False, textangle=-90,
                               font=dict(size=9, color="#57606A"), yanchor="top", xshift=-6)
    fig.update_layout(title=dict(text=series_name(i, lang), font=dict(size=16)), template="plotly_white",
                      height=470, hovermode="x unified", margin=dict(l=50, r=20, t=60, b=40),
                      legend=dict(orientation="h", y=-0.12, x=0), yaxis_title=tr(lang, "units_day"),
                      xaxis=dict(tickformat="%d/%m" if lang == "fr" else "%b %d"))
    return fig


def run_forecast(level, lieu, produit, article, model_name, cov, ctx_days, compare, show_days, lang):
    i = resolve(level, lieu, produit, article, lang)
    fc = get_model(model_name, lang)
    t0 = time.time()
    q, levels = forecast_series(fc, DATA, [i], HIST, CAL, cov, None if ctx_days >= T else int(ctx_days))
    state = {"i": i, "q": q[0].tolist(), "levels": list(levels), "elapsed": time.time() - t0,
             "model": TEST_MODEL or model_name}
    return (state,) + render_forecast(state, compare, show_days, 90, lang)


def render_forecast(state, compare, show_days, target, lang):
    if not state:
        return None, "", None, "", None, ""
    i, q, levels = state["i"], np.array(state["q"]), state["levels"]
    med, mean = quantile_at(q, levels, 0.5), mean_from_quantiles(q, levels)
    y = EV.actual[i]
    rows = {"median": series_metrics(i, med), "mean": series_metrics(i, mean)}
    for k in REF:
        rows[k] = series_metrics(i, REF[k][i])

    names = {"median": tr(lang, "median"), "mean": tr(lang, "mean"), **{k: mlabel(lang, k) for k in REF}}
    table = pd.DataFrame([{tr(lang, "col_method"): names[k], "RMSSE": fnum(v["rmsse"], lang),
                           "WAPE": fmt_pct(v["wape"], lang), tr(lang, "col_bias"): fmt_pct(v["bias"], lang, True),
                           tr(lang, "col_total"): fint(v["total"], lang)} for k, v in rows.items()]
                         + [{tr(lang, "col_method"): tr(lang, "actual"), "RMSSE": "–", "WAPE": "–",
                             tr(lang, "col_bias"): "–", tr(lang, "col_total"): fint(y.sum(), lang)}])

    best_t = min(rows["median"]["rmsse"], rows["mean"]["rmsse"])
    stat_keys = [k for k in REF if k != "winner"]
    best_stat = min(stat_keys, key=lambda k: rows[k]["rmsse"]) if stat_keys else None
    in80 = int(((y >= quantile_at(q, levels, 0.1)) & (y <= quantile_at(q, levels, 0.9))).sum())

    cards = [kpi_card(tr(lang, "k_rmsse"), fnum(best_t, lang), tr(lang, "k_lower"))]
    if best_stat:
        rs = rows[best_stat]["rmsse"]
        gain = 1 - best_t / rs
        cards.append(kpi_card(f"vs {mlabel(lang, best_stat)}", fmt_pct(gain, lang, True),
                              tr(lang, "k_best_stat", v=fnum(rs, lang)), "good" if gain > 0 else "bad"))
    if "winner" in REF:
        rw = rows["winner"]["rmsse"]
        cards.append(kpi_card(tr(lang, "k_vs_winner"), fmt_pct(1 - best_t / rw, lang, True),
                              f"RMSSE {fnum(rw, lang)}", "good" if best_t < rw else "bad"))
    cards.append(kpi_card(tr(lang, "k_band"), tr(lang, "k_band_v", n=in80), tr(lang, "k_band_sub"),
                          "good" if 18 <= in80 <= 26 else "bad"))
    cards.append(kpi_card(tr(lang, "k_bias"), fmt_pct(rows["median"]["bias"], lang, True), tr(lang, "k_bias_sub")))
    cards.append(kpi_card(tr(lang, "k_time"), f"{state['elapsed'] * 1000:.0f} ms", f"{state['model']} · CPU"))
    kpis = '<div class="kpis">' + "".join(cards) + "</div>"

    zs = zero_share(i)
    mean_wins = rows["mean"]["rmsse"] < rows["median"]["rmsse"]
    reading = tr(lang, "reading", zs=fmt_pct(zs, lang),
                 verdict=tr(lang, "beats" if best_stat and best_t < rows[best_stat]["rmsse"] else "trails"),
                 n=in80, which=tr(lang, "w_mean" if mean_wins else "w_median"),
                 extra=tr(lang, "reading_extra") if zs > 0.4 and mean_wins else ".")
    fig = make_forecast_figure(i, q, levels, compare, int(show_days), lang)
    return (fig, kpis, table, reading) + service_level(state, target, lang)


def service_level(state, target, lang):
    if not state:
        return None, ""
    i, q, levels = state["i"], np.array(state["q"]), state["levels"]
    y = EV.actual[i]
    u = target / 100
    level_q = quantile_at(q, levels, u)
    med = quantile_at(q, levels, 0.5)
    cov_q, cov_m = (y <= level_q).mean(), (y <= med).mean()
    short = np.maximum(y - level_q, 0).sum()
    surplus = np.maximum(level_q - y, 0).sum()
    fill = 1 - short / y.sum() if y.sum() > 0 else np.nan
    fx = DATES[T : T + HORIZON]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fx, y=y, name=tr(lang, "actual_demand"),
                         marker_color=["#CF222E" if a > b else "#8C959F" for a, b in zip(y, level_q)]))
    fig.add_trace(go.Scatter(x=fx, y=level_q, name=tr(lang, "stock_at", target=target), mode="lines",
                             line=dict(color=C_TIMESTRAL, width=2.4, shape="hv")))
    fig.add_trace(go.Scatter(x=fx, y=med, name=tr(lang, "median_short"), mode="lines",
                             line=dict(color=C_TIMESTRAL, width=1.2, dash="dot", shape="hv")))
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=50, r=20, t=20, b=40),
                      legend=dict(orientation="h", y=-0.2), yaxis_title=tr(lang, "units"), bargap=0.25,
                      xaxis=dict(tickformat="%d/%m" if lang == "fr" else "%b %d"))
    html = '<div class="kpis">' + "".join([
        kpi_card(tr(lang, "s_covered"), fmt_pct(cov_q, lang), tr(lang, "s_covered_sub", t=target, n=int(round(cov_q * 28))),
                 "good" if abs(cov_q - u) <= 0.1 else "bad"),
        kpi_card(tr(lang, "s_fill"), fmt_pct(fill, lang), tr(lang, "s_fill_sub", v=fint(short, lang))),
        kpi_card(tr(lang, "s_surplus"), fint(surplus, lang), tr(lang, "s_surplus_sub")),
        kpi_card(tr(lang, "s_median"), fmt_pct(cov_m, lang), tr(lang, "s_median_sub")),
    ]) + "</div>"
    return fig, html


# ----------------------------------------------------------------------------- mini-benchmark
def mini_benchmark(model_name, cov, n_items, seed, lang, progress=gr.Progress()):
    rng = np.random.default_rng(int(seed))
    agg_idx = np.concatenate([SLICES[l] for l, _ in LEVELS[:9]])
    sample = {l: rng.choice(SLICES[l], size=min(int(n_items), len(SLICES[l])), replace=False)
              for l in ("Level10", "Level11", "Level12")}
    idx = np.concatenate([agg_idx] + list(sample.values()))
    progress(0, desc=tr(lang, "b_loading"))
    fc = get_model(model_name, lang)
    t0 = time.time()
    q, levels = forecast_series(fc, DATA, idx, HIST, CAL, cov, batch_size=128,
                                progress=lambda d, n: progress(d / n, desc=tr(lang, "b_progress", d=d, n=n)))
    elapsed = time.time() - t0
    methods = {"median": q[..., list(levels).index(0.5)], "mean": mean_from_quantiles(q, levels)}
    methods.update({k: REF[k][idx] for k in REF})
    r = {m: EV.rmsse(f, idx) for m, f in methods.items()}
    spl = {"wspl_t": EV.spl(interpolate_quantiles(q, levels), idx=idx), "wspl_e": EV.spl(EMP_Q[idx], idx=idx)}
    w = EV.weights[idx]
    lv = DATA.series["level"].to_numpy()[idx]
    rows = []
    for l, _ in LEVELS:
        m = lv == l
        row = {"level": l, "n": int(m.sum())}
        for name, vals in {**r, **spl}.items():
            row[name] = float((w[m] * vals[m]).sum() / w[m].sum())
        rows.append(row)
    bstate = {"rows": rows, "n": len(idx), "elapsed": elapsed, "methods": list(r)}
    return (bstate,) + render_benchmark(bstate, lang)


def render_benchmark(bstate, lang):
    if not bstate:
        return "", None, None
    df = pd.DataFrame(bstate["rows"])
    cols = bstate["methods"] + ["wspl_t", "wspl_e"]
    total = df[cols].mean()
    names = {"median": tr(lang, "median"), "mean": tr(lang, "mean"), "wspl_t": tr(lang, "b_wspl_t"),
             "wspl_e": tr(lang, "b_wspl_e"), **{k: mlabel(lang, k) for k in REF}}
    level_names = [f"{l.replace('Level', '')}. {LEVEL_LABELS[lang][l]}" for l in df["level"]]

    fig = go.Figure()
    palette = {"median": "#F4A261", "mean": C_TIMESTRAL, **{k: C_METHODS.get(k, "#999") for k in REF}}
    for m in bstate["methods"]:
        fig.add_trace(go.Bar(x=level_names, y=df[m], name=names[m], marker_color=palette[m]))
    fig.update_layout(template="plotly_white", barmode="group", height=420, yaxis_title=tr(lang, "b_yaxis"),
                      margin=dict(l=50, r=20, t=30, b=120), legend=dict(orientation="h", y=1.1))

    best_t = min(total["median"], total["mean"])
    stat = {k: total[k] for k in REF if k != "winner"}
    best_stat = min(stat, key=stat.get) if stat else None
    parts = [tr(lang, "b_summary", n=fint(bstate["n"], lang), s=f"{bstate['elapsed']:.0f}", t=fnum(best_t, lang))]
    if best_stat:
        parts.append(tr(lang, "b_best_stat", name=mlabel(lang, best_stat), v=fnum(stat[best_stat], lang)))
    if "winner" in REF:
        parts.append(tr(lang, "b_winner", v=fnum(total["winner"], lang)))
    summary = " · ".join(parts) + "\n\n" + tr(lang, "b_quant", t=fnum(total["wspl_t"], lang),
                                               e=fnum(total["wspl_e"], lang)) + "\n\n" + tr(lang, "b_note")

    head = {tr(lang, "b_level"): tr(lang, "b_overall"), tr(lang, "b_series"): fint(bstate["n"], lang),
            **{names[c]: fnum(total[c], lang) for c in cols}}
    body = [{tr(lang, "b_level"): level_names[j], tr(lang, "b_series"): fint(row["n"], lang),
             **{names[c]: fnum(row[c], lang) for c in cols}} for j, row in enumerate(bstate["rows"])]
    return summary, fig, pd.DataFrame([head] + body)


def leaderboard(lang) -> pd.DataFrame | None:
    if SCORES is None:
        return None
    n = LEADERBOARD_NAMES[lang]
    s = SCORES[SCORES["métrique"] == "wrmsse"].copy()
    s = s[~((s["famille"] == "timestral") & (s["agrégation"] == "direct"))]
    method = s["méthode"].map(lambda m: n.get(m, m)) + s["variante"].fillna("").map(
        lambda v: f" · {n.get(v, v)}" if v else "")
    return pd.DataFrame({tr(lang, "col_method"): method.to_numpy(),
                         tr(lang, "lb_type"): s["famille"].map(lambda f: n.get(f, f)).to_numpy(),
                         "WRMSSE": s["score"].map(lambda v: fnum(v, lang, 4)).to_numpy()})


# ----------------------------------------------------------------------------- your data
SEASON = {"D": (7, 28), "W": (52, 13), "M": (12, 12), "Q": (4, 4)}


def parse_dates(col: pd.Series) -> pd.Series:
    """ISO dates (2024-01-31) first, then day-first dates (31/01/2024)."""
    iso = pd.to_datetime(col, errors="coerce", format="ISO8601")
    if iso.notna().mean() > 0.9:
        return iso
    return pd.to_datetime(col, errors="coerce", dayfirst=True)


def read_user_csv(path: str, lang: str) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(path, sep=None, engine="python")
    date_col = next((c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])
                     and parse_dates(raw[c].astype(str)).notna().mean() > 0.9), None)
    if date_col is None:
        raise gr.Error(tr(lang, "e_nodate"))
    raw[date_col] = parse_dates(raw[date_col].astype(str))
    others = [c for c in raw.columns if c != date_col]
    text_cols = [c for c in others if not pd.api.types.is_numeric_dtype(raw[c])]
    if text_cols:  # long format: series, date, value
        val = [c for c in others if c not in text_cols][0]
        wide = raw.pivot_table(index=date_col, columns=text_cols[0], values=val, aggfunc="sum")
    else:
        wide = raw.set_index(date_col)[others]
    wide = wide.sort_index()
    step = pd.Series(wide.index).diff().dt.days.median()
    freq = "D" if step <= 1.5 else "W" if step <= 8 else "M" if step <= 32 else "Q"
    return wide.iloc[:, :200].astype(float), freq


def example_csv() -> str:
    """Monthly sales of the 7 M5 departments (Feb 2011 – Apr 2016): close to a monthly demand-planning case."""
    l5 = SLICES["Level5"]
    daily = pd.DataFrame(np.concatenate([HIST[l5], EV.actual[l5]], axis=1).T, index=DATES[: T + HORIZON],
                         columns=DATA.series["agg1"].to_numpy()[l5])
    monthly = daily.resample("MS").sum().iloc[1:-1]
    path = Path(tempfile.gettempdir()) / "m5_monthly_sales_example.csv"
    monthly.rename_axis("date").reset_index().to_csv(path, index=False, date_format="%Y-%m-%d")
    return str(path)


EXAMPLE_CSV = example_csv()


def run_user(path, horizon, backtest, model_name, lang):
    if not path:
        raise gr.Error(tr(lang, "e_nofile"))
    wide, freq = read_user_csv(path, lang)
    season, default_h = SEASON[freq]
    h = int(horizon) if horizon else default_h
    values = wide.to_numpy().T  # (S, T)
    ctx = values[:, :-h] if backtest else values
    if ctx.shape[1] < 3:
        raise gr.Error(tr(lang, "e_short"))
    fc = get_model(model_name, lang)
    res = fc.forecast(ctx[:, None, :].astype(np.float32), prediction_length=h)
    q = res.quantiles[:, 0].cpu().numpy()
    levels = list(res.quantile_levels)
    off = {"D": pd.DateOffset(days=1), "W": pd.DateOffset(weeks=1), "M": pd.DateOffset(months=1),
           "Q": pd.DateOffset(months=3)}[freq]
    last = wide.index[-h - 1] if backtest else wide.index[-1]
    fdates = pd.DatetimeIndex([last + off * (k + 1) for k in range(h)])
    out, metrics = [], []
    for s, name in enumerate(wide.columns):
        med = q[s, :, levels.index(0.5)]
        out.append(pd.DataFrame({"series": name, "date": fdates.strftime("%Y-%m-%d"),
                                 "p10": q[s][:, levels.index(0.1)], "p50": med, "p90": q[s][:, levels.index(0.9)],
                                 "mean": mean_from_quantiles(q[s], levels)}))
        if backtest:
            y = values[s, -h:]
            hist = ctx[s][~np.isnan(ctx[s])]
            snaive = np.resize(hist[-season:], h) if len(hist) >= season else np.repeat(hist[-1], h)
            tot = np.nansum(y)
            for key, f in (("timestral", med), ("snaive", snaive), ("naive", np.repeat(hist[-1], h))):
                metrics.append({"series": str(name), "method": key,
                                "wape": float(np.nansum(np.abs(f - y)) / tot) if tot else np.nan,
                                "bias": float((np.nansum(f) - tot) / tot) if tot else np.nan})
    fc_df = pd.concat(out, ignore_index=True)
    csv_path = Path(tempfile.gettempdir()) / "timestral_forecasts.csv"
    fc_df.round(3).to_csv(csv_path, index=False)
    ustate = {"wide": wide.reset_index().to_json(date_format="iso"), "fc": fc_df.to_json(), "h": h,
              "backtest": bool(backtest), "freq": freq, "metrics": metrics, "names": [str(c) for c in wide.columns],
              "points": int(wide.shape[0])}
    info, series_u, plot, mdf = render_user(ustate, lang, ustate["names"][0])
    return (ustate, info, series_u, plot, mdf, str(csv_path))


def render_user(ustate, lang, name=None):
    if not ustate:
        return "", gr.update(), None, None
    name = name if name in ustate["names"] else ustate["names"][0]
    info = tr(lang, "u_info", n=len(ustate["names"]), f=FREQ_NAMES[lang][ustate["freq"]], p=ustate["points"],
              h=ustate["h"], mode=tr(lang, "u_bt_mode" if ustate["backtest"] else "u_fc_mode"))
    mdf = None
    if ustate["metrics"]:
        m = pd.DataFrame(ustate["metrics"])
        avg = m.groupby("method", sort=False)[["wape", "bias"]].mean().reset_index().assign(series=tr(lang, "u_avg"))
        m = pd.concat([avg, m], ignore_index=True)
        labels = {"timestral": "Timestral", "snaive": tr(lang, "u_snaive"), "naive": tr(lang, "u_naive")}
        mdf = pd.DataFrame({tr(lang, "col_series"): m["series"], tr(lang, "col_method"): m["method"].map(labels),
                            "WAPE": m["wape"].map(lambda v: fmt_pct(v, lang)),
                            tr(lang, "col_bias"): m["bias"].map(lambda v: fmt_pct(v, lang, True))})
    return (info, gr.update(choices=ustate["names"], value=name, label=tr(lang, "u_series")),
            user_plot(name, ustate, lang), mdf)


def user_plot(name, ustate, lang):
    if not ustate or name is None:
        return None
    wide = pd.read_json(io.StringIO(ustate["wide"]))
    wide = wide.set_index(wide.columns[0])
    wide.index = pd.to_datetime(wide.index)
    fc = pd.read_json(io.StringIO(ustate["fc"]))
    fc["series"] = fc["series"].astype(str)
    f = fc[fc["series"] == str(name)]
    col = [c for c in wide.columns if str(c) == str(name)][0]
    fx = pd.to_datetime(f["date"])
    y = wide[col]
    cut = len(y) - ustate["h"] if ustate["backtest"] else len(y)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=wide.index[:cut], y=y.iloc[:cut], name=tr(lang, "u_hist"), mode="lines+markers",
                             line=dict(color=C_HISTORY, width=1.6), marker=dict(size=4)))
    if ustate["backtest"]:
        fig.add_trace(go.Scatter(x=wide.index[cut - 1:], y=y.iloc[cut - 1:], name=tr(lang, "u_hidden"),
                                 mode="lines+markers", line=dict(color=C_ACTUAL, width=2), marker=dict(size=5)))
        fig.add_vline(x=wide.index[cut - 1], line=dict(color="#57606A", width=1, dash="dash"))
    fig.add_trace(go.Scatter(x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([f["p90"], f["p10"][::-1]]),
                             fill="toself", fillcolor="rgba(232,89,12,0.22)", line=dict(width=0),
                             name=tr(lang, "u_band"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=f["p50"], name=tr(lang, "median"), mode="lines+markers",
                             line=dict(color=C_TIMESTRAL, width=2.4), marker=dict(size=5)))
    fig.update_layout(template="plotly_white", height=430, hovermode="x unified", title=str(name),
                      margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.15))
    return fig


# ----------------------------------------------------------------------------- learn tab
def levels_table(lang) -> str:
    rows = "".join(f"<tr><td>{i + 1}</td><td>{LEVEL_LABELS[lang][l]}</td>"
                   f"<td class='num'>{fint(len(SLICES[l]), lang)}</td></tr>" for i, (l, _) in enumerate(LEVELS))
    head = ("<th>#</th><th>Niveau</th><th class='num'>Séries</th>" if lang == "fr"
            else "<th>#</th><th>Level</th><th class='num'>Series</th>")
    title = LEARN[f"{lang}_levels_title"]
    return (f"<h3>{title}</h3><div class='tbl'><table class='levels'><thead><tr>{head}</tr></thead>"
            f"<tbody>{rows}</tbody></table></div>")


def pipeline_html(lang) -> str:
    steps = "".join(f"<div class='step'><div class='step-n'>{n}</div><div class='step-t'>{t}</div>"
                    f"<div class='step-d'>{d}</div></div>" for n, t, d in PIPELINE[lang])
    cap = "Comment Timestral produit une prévision" if lang == "fr" else "How Timestral produces a forecast"
    return f"<div class='pipe-cap'>{cap}</div><div class='pipeline'>{steps}</div>"


def learn_parts(lang):
    return (LEARN[lang], levels_table(lang), LEARN[f"{lang}_after_levels"], pipeline_html(lang),
            LEARN[f"{lang}_after_pipeline"])


# ----------------------------------------------------------------------------- interface
CSS = """
.kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:10px; margin:4px 0 8px}
.kpi {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px;
      background:var(--block-background-fill)}
.kpi-label {font-size:12px; color:var(--body-text-color-subdued)}
.kpi-value {font-size:22px; font-weight:650; margin-top:2px; font-variant-numeric:tabular-nums}
.kpi-sub {font-size:11.5px; color:var(--body-text-color-subdued); margin-top:2px}
.kpi.good .kpi-value {color:#1A7F37} .kpi.bad .kpi-value {color:#CF222E}
.hero h1 {margin-bottom:0} .hero p {margin-top:4px; color:var(--body-text-color-subdued); max-width:900px}
.test-banner {background:#FFF4E5; border:1px solid #F5B971; border-radius:8px; padding:8px 12px; color:#7A3E00}
.lang-row {align-items:flex-start; flex-wrap:wrap}
.learn {max-width:860px}
.tbl {overflow-x:auto}
table.levels {border-collapse:collapse; font-size:14px; min-width:280px}
table.levels th, table.levels td {padding:5px 14px 5px 0; border-bottom:1px solid var(--border-color-primary);
  text-align:left}
table.levels .num {text-align:right; font-variant-numeric:tabular-nums}
.pipe-cap {font-weight:600; margin:6px 0 8px}
.pipeline {display:grid; grid-template-columns:repeat(auto-fit,minmax(170px,1fr)); gap:10px; max-width:860px}
.step {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px;
  background:var(--block-background-fill)}
.step-n {width:22px; height:22px; border-radius:50%; background:#E8590C; color:#fff; font-size:12px; font-weight:700;
  display:flex; align-items:center; justify-content:center}
.step-t {font-weight:650; margin-top:6px} .step-d {font-size:13px; color:var(--body-text-color-subdued); margin-top:2px}
footer {display:none !important}
@media (max-width: 640px) {
  .tab-wrapper button, button[role="tab"] {padding:6px 6px !important; font-size:13px !important; min-width:0 !important}
}
"""

MODEL_CHOICES = list(MODELS)
COMPARE_DEFAULT = [k for k in ("snaive", "ets", "winner") if k in REF]
L0 = "en"  # initial language; switched to French on load for French-speaking browsers


def detect_lang(request: gr.Request):
    accept = (request.headers.get("accept-language", "") if request else "").lower()
    return "fr" if accept.startswith("fr") else "en"


with gr.Blocks(title="Timestral × M5") as demo:
    with gr.Row(elem_classes="lang-row"):
        with gr.Column(scale=4, min_width=340):
            intro = gr.HTML(tr(L0, "intro"))
        with gr.Column(scale=0, min_width=210):
            lang = gr.Radio(LANGS, value=L0, label=tr(L0, "lang_label"))
    banner = gr.HTML(f'<div class="test-banner">{tr(L0, "test_banner", model=TEST_MODEL)}</div>',
                     visible=bool(TEST_MODEL))

    with gr.Tab(tr(L0, "tab_explore")) as tab_explore:
        with gr.Accordion(tr(L0, "how_to_read"), open=False) as how_acc:
            how_md = gr.Markdown(tr(L0, "how_to_read_body"))
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                ex_title = gr.Markdown(tr(L0, "examples"))
                with gr.Row():
                    ex_btns = [gr.Button(tr(L0, key), size="sm") for key, _ in EXAMPLES]
                level = gr.Dropdown(level_choices(L0), value="Level12", label=tr(L0, "level"))
                first = DATA.series.iloc[EXAMPLES[2][1]]  # fast-moving item × store, shown at start-up
                lieu = gr.Dropdown(label=tr(L0, "store"), choices=STORES, value=first.agg2, allow_custom_value=True)
                produit = gr.Dropdown(label=tr(L0, "category"), choices=CATS, value=CATS[0], visible=False,
                                      allow_custom_value=True)
                rayon = gr.Dropdown(label=tr(L0, "department"), choices=DEPTS, value=DEPT_OF_ITEM[first.agg1])
                article = gr.Dropdown(label=tr(L0, "item"), choices=ITEMS_BY_DEPT[DEPT_OF_ITEM[first.agg1]],
                                      value=first.agg1, allow_custom_value=True)
                with gr.Accordion(tr(L0, "settings"), open=False) as settings_acc:
                    model_dd = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label=tr(L0, "model_size"),
                                           info=tr(L0, "model_info"))
                    cov = gr.Radio(cov_choices(L0), value="calendar", label=tr(L0, "covariates"))
                    ctx = gr.Slider(56, T, value=T, step=7, label=tr(L0, "ctx"), info=tr(L0, "ctx_info"))
                compare = gr.CheckboxGroup(compare_choices(L0), value=COMPARE_DEFAULT, label=tr(L0, "compare"))
                show_days = gr.Slider(28, 365, value=112, step=7, label=tr(L0, "show_days"))
                go_btn = gr.Button(tr(L0, "go"), variant="primary", size="lg")
            with gr.Column(scale=3):
                kpis = gr.HTML()
                plot = gr.Plot(label=tr(L0, "plot_label"))
                reading = gr.Markdown()
                table = gr.Dataframe(label=tr(L0, "table_label"), interactive=False, wrap=True)
                with gr.Accordion(tr(L0, "svc_title"), open=True) as svc_acc:
                    svc_md = gr.Markdown(tr(L0, "svc_body"))
                    target = gr.Slider(50, 99, value=90, step=1, label=tr(L0, "svc_target"))
                    svc_kpis = gr.HTML()
                    svc_plot = gr.Plot(show_label=False)
        fstate = gr.State()

    with gr.Tab(tr(L0, "tab_bench")) as tab_bench:
        bench_md = gr.Markdown(tr(L0, "bench_intro"))
        with gr.Row():
            b_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label=tr(L0, "model_size"))
            b_cov = gr.Radio(cov_choices(L0), value="calendar", label=tr(L0, "b_cov"))
            b_n = gr.Slider(50, 2000, value=300, step=50, label=tr(L0, "b_n"))
            b_seed = gr.Number(value=42, label=tr(L0, "b_seed"), precision=0)
        b_btn = gr.Button(tr(L0, "b_run"), variant="primary")
        b_summary = gr.Markdown()
        b_plot = gr.Plot(show_label=False)
        b_table = gr.Dataframe(interactive=False, wrap=True, max_height=620)
        bstate = gr.State()
        lb_md = gr.Markdown(tr(L0, "lb_title"))
        lb_table = gr.Dataframe(leaderboard(L0), interactive=False, visible=SCORES is not None)
        lb_note = gr.Markdown(tr(L0, "lb_note"), visible=SCORES is not None)

    with gr.Tab(tr(L0, "tab_data")) as tab_data:
        data_md = gr.Markdown(tr(L0, "data_intro"))
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                u_file = gr.File(label=tr(L0, "u_file"), file_types=[".csv"], type="filepath")
                u_example = gr.Button(tr(L0, "u_example"), size="sm")
                u_h = gr.Number(value=0, label=tr(L0, "u_h"), precision=0, info=tr(L0, "u_h_info"))
                u_bt = gr.Checkbox(value=True, label=tr(L0, "u_bt"))
                u_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label=tr(L0, "model_size"))
                u_btn = gr.Button(tr(L0, "u_go"), variant="primary")
            with gr.Column(scale=3):
                u_info = gr.Markdown()
                u_series = gr.Dropdown(label=tr(L0, "u_series"), choices=[], allow_custom_value=True)
                u_plot = gr.Plot(show_label=False)
                u_metrics = gr.Dataframe(label=tr(L0, "u_metrics"), interactive=False, max_height=420)
                u_dl = gr.File(label=tr(L0, "u_dl"))
        ustate = gr.State()

    with gr.Tab(tr(L0, "tab_learn")) as tab_learn:
        with gr.Column(elem_classes="learn"):
            lp = learn_parts(L0)
            learn_1 = gr.Markdown(lp[0])
            learn_levels = gr.HTML(lp[1])
            learn_2 = gr.Markdown(lp[2])
            learn_pipe = gr.HTML(lp[3])
            learn_3 = gr.Markdown(lp[4])

    # ---- events
    sel_inputs = [level, lieu, produit, article]
    run_inputs = sel_inputs + [model_dd, cov, ctx, compare, show_days, lang]
    view_outputs = [plot, kpis, table, reading, svc_plot, svc_kpis]
    level.input(lambda l, lg: selector_updates(l, lg), [level, lang], [lieu, produit, rayon, article])
    rayon.input(lambda r: gr.update(choices=ITEMS_BY_DEPT[r], value=ITEMS_BY_DEPT[r][0]), rayon, article)
    go_btn.click(run_forecast, run_inputs, [fstate] + view_outputs)
    for btn, (_, i) in zip(ex_btns, EXAMPLES):
        btn.click(lambda lg, i=i: example_updates(i, lg), lang, [level, lieu, produit, rayon, article]).then(
            run_forecast, run_inputs, [fstate] + view_outputs)
    target.change(service_level, [fstate, target, lang], [svc_plot, svc_kpis])
    for comp in (compare, show_days):
        comp.input(render_forecast, [fstate, compare, show_days, target, lang], view_outputs)
    b_btn.click(mini_benchmark, [b_model, b_cov, b_n, b_seed, lang], [bstate, b_summary, b_plot, b_table])
    u_example.click(lambda: EXAMPLE_CSV, None, u_file)
    u_btn.click(run_user, [u_file, u_h, u_bt, u_model, lang], [ustate, u_info, u_series, u_plot, u_metrics, u_dl])
    u_series.input(user_plot, [u_series, ustate, lang], u_plot)

    def relabel(lg, level_v, fstate_v, compare_v, show_v, target_v, bstate_v, ustate_v, series_v):
        lieu_u, prod_u, _, _ = selector_updates(level_v, lg)
        static = [
            gr.update(value=tr(lg, "intro")), gr.update(label=tr(lg, "lang_label")),
            gr.update(value=f'<div class="test-banner">{tr(lg, "test_banner", model=TEST_MODEL)}</div>'),
            gr.update(label=tr(lg, "tab_explore")), gr.update(label=tr(lg, "tab_bench")),
            gr.update(label=tr(lg, "tab_data")), gr.update(label=tr(lg, "tab_learn")),
            gr.update(label=tr(lg, "how_to_read")), gr.update(value=tr(lg, "how_to_read_body")),
            gr.update(value=tr(lg, "examples")),
            *[gr.update(value=tr(lg, key)) for key, _ in EXAMPLES],
            gr.update(choices=level_choices(lg), label=tr(lg, "level")),
            gr.update(label=lieu_u["label"]), gr.update(label=prod_u["label"]),  # keep selections, relabel only
            gr.update(label=tr(lg, "department")), gr.update(label=tr(lg, "item")),
            gr.update(label=tr(lg, "settings")),
            gr.update(label=tr(lg, "model_size"), info=tr(lg, "model_info")),
            gr.update(choices=cov_choices(lg), label=tr(lg, "covariates")),
            gr.update(label=tr(lg, "ctx"), info=tr(lg, "ctx_info")),
            gr.update(choices=compare_choices(lg), label=tr(lg, "compare")),
            gr.update(label=tr(lg, "show_days")), gr.update(value=tr(lg, "go")),
            gr.update(label=tr(lg, "svc_title")), gr.update(value=tr(lg, "svc_body")),
            gr.update(label=tr(lg, "svc_target")),
            gr.update(value=tr(lg, "bench_intro")), gr.update(label=tr(lg, "model_size")),
            gr.update(choices=cov_choices(lg), label=tr(lg, "b_cov")), gr.update(label=tr(lg, "b_n")),
            gr.update(label=tr(lg, "b_seed")), gr.update(value=tr(lg, "b_run")),
            gr.update(value=tr(lg, "lb_title")), gr.update(value=leaderboard(lg)), gr.update(value=tr(lg, "lb_note")),
            gr.update(value=tr(lg, "data_intro")), gr.update(label=tr(lg, "u_file")),
            gr.update(value=tr(lg, "u_example")), gr.update(label=tr(lg, "u_h"), info=tr(lg, "u_h_info")),
            gr.update(label=tr(lg, "u_bt")), gr.update(label=tr(lg, "model_size")), gr.update(value=tr(lg, "u_go")),
            gr.update(label=tr(lg, "u_dl")),
            *[gr.update(value=v) for v in learn_parts(lg)],
        ]
        explore = list(render_forecast(fstate_v, compare_v, show_v, target_v, lg)) if fstate_v else [None] * 6
        keep = [gr.update(value=v) if fstate_v else gr.update() for v in explore]
        keep[0] = gr.update(label=tr(lg, "plot_label"), **({"value": explore[0]} if fstate_v else {}))
        keep[2] = gr.update(label=tr(lg, "table_label"), **({"value": explore[2]} if fstate_v else {}))
        explore = keep
        bench = list(render_benchmark(bstate_v, lg)) if bstate_v else [gr.update()] * 3
        if ustate_v:
            user = list(render_user(ustate_v, lg, series_v))
            user[3] = gr.update(value=user[3], label=tr(lg, "u_metrics"))
        else:
            user = [gr.update(), gr.update(label=tr(lg, "u_series")), gr.update(),
                    gr.update(label=tr(lg, "u_metrics"))]
        return static + explore + bench + user

    relabel_outputs = [
        intro, lang, banner, tab_explore, tab_bench, tab_data, tab_learn, how_acc, how_md, ex_title, *ex_btns,
        level, lieu, produit, rayon, article, settings_acc, model_dd, cov, ctx, compare, show_days, go_btn,
        svc_acc, svc_md, target,
        bench_md, b_model, b_cov, b_n, b_seed, b_btn, lb_md, lb_table, lb_note,
        data_md, u_file, u_example, u_h, u_bt, u_model, u_btn, u_dl,
        learn_1, learn_levels, learn_2, learn_pipe, learn_3,
        *view_outputs, b_summary, b_plot, b_table, u_info, u_series, u_plot, u_metrics,
    ]
    lang.change(relabel, [lang, level, fstate, compare, show_days, target, bstate, ustate, u_series], relabel_outputs)
    demo.load(detect_lang, None, lang).then(run_forecast, run_inputs, [fstate] + view_outputs)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), theme=gr.themes.Soft(primary_hue="orange"),
        css=CSS)
