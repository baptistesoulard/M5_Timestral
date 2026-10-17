"""Interactive demo: Timestral (Mistral) on the M5 Walmart benchmark.

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

from download_data import ensure_data
from m5bench.data import HORIZON, LEVELS, calendar_covariates, level_slices, load_m5
from m5bench.metrics import M5_QUANTILES, M5Evaluator, interpolate_quantiles
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

METHOD_LABELS = {"snaive": "Seasonal naive", "tsb": "TSB (intermittent)", "ets": "Seasonal AutoETS",
                 "winner": "M5 winner (YJ_STU)"}
REF, REF_LABELS = {}, {}
_ref_path = ASSETS / "reference_forecasts.npz"
if _ref_path.exists():
    with np.load(_ref_path) as z:
        for k in json.loads(str(z["labels"])):
            REF[k] = DATA.aggregate(z[k].astype(np.float32))  # bottom-up: (42840, 28)
            REF_LABELS[k] = METHOD_LABELS.get(k, k)
else:  # fallback: seasonal naive computed on the fly
    REF_LABELS = {"snaive": METHOD_LABELS["snaive"]}
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
LEVEL_LABELS = {"Level1": "Total", "Level2": "State", "Level3": "Store", "Level4": "Category",
                "Level5": "Department", "Level6": "State × category", "Level7": "State × department",
                "Level8": "Store × category", "Level9": "Store × department", "Level10": "Item",
                "Level11": "Item × state", "Level12": "Item × store"}
LEVEL_CHOICES = [(f"{i + 1}. {LEVEL_LABELS[l]}", l) for i, (l, _) in enumerate(LEVELS)]
COV_CHOICES = [("Sales only", "none"), ("+ calendar (events, SNAP, day of week)", "calendar"),
               ("+ calendar and price", "full")]


def series_name(i: int) -> str:
    r = DATA.series.iloc[i]
    parts = [p for p in (r.agg1, r.agg2) if p not in ("X", "Total")]
    return f"{LEVEL_LABELS[r.level]} · {' · '.join(parts)}" if parts else "Walmart total (10 stores)"


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
    return [("Walmart total", SERIES_KEY[("Level1", "Total", "X")]),
            ("Store CA_3 · Foods", SERIES_KEY[("Level8", "CA_3", "FOODS")]),
            ("Fast-moving item", int(top)),
            ("Intermittent item", int(inter))]


EXAMPLES = pick_examples()


# ----------------------------------------------------------------------------- model
_model_lock = threading.Lock()


@lru_cache(maxsize=2)
def _load(name: str):
    return load_forecaster(TEST_MODEL or name)


def get_model(name: str):
    with _model_lock:
        try:
            return _load(name)
        except Exception as e:  # network or Hugging Face unavailable
            raise gr.Error(f"Could not load {name}: {e}") from e


threading.Thread(target=lambda: _load(DEFAULT_MODEL), daemon=True).start()  # warm-up


# ----------------------------------------------------------------------------- series selection
def resolve(level, lieu, produit, article) -> int:
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
        raise gr.Error("Incomplete selection: pick every element of the series.")
    return SERIES_KEY[key]


def selector_updates(level, lieu=None, produit=None, rayon=None, article=None):
    """Update the dropdowns for the selected aggregation level."""
    kind = LIEU.get(level)
    lieu_choices = STATES if kind == "state" else STORES
    lieu_u = gr.update(visible=kind is not None, choices=lieu_choices,
                       value=lieu if lieu in lieu_choices else lieu_choices[0],
                       label="State" if kind == "state" else "Store")
    pk = PRODUIT.get(level)
    prod_choices = CATS if pk == "cat" else DEPTS
    prod_u = gr.update(visible=pk is not None, choices=prod_choices,
                       value=produit if produit in prod_choices else prod_choices[0],
                       label="Category" if pk == "cat" else "Department")
    is_item = level in ITEM_LEVELS
    rayon = rayon if rayon in DEPTS else "FOODS_3"
    items = ITEMS_BY_DEPT[rayon]
    rayon_u = gr.update(visible=is_item, value=rayon)
    art_u = gr.update(visible=is_item, choices=items, value=article if article in items else items[0])
    return lieu_u, prod_u, rayon_u, art_u


def example_updates(i: int):
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
    rayon = DEPT_OF_ITEM.get(article)
    return (gr.update(value=level),) + selector_updates(level, lieu, produit, rayon, article)


# ----------------------------------------------------------------------------- per-series computations
def quantile_at(q: np.ndarray, levels, u: float) -> np.ndarray:
    """Quantile u (interpolated) from a (28, Q) grid."""
    lv = np.asarray(levels)
    return np.array([np.interp(u, lv, row) for row in q])


def series_metrics(i: int, f: np.ndarray) -> dict:
    y = EV.actual[i]
    tot = y.sum()
    return {"RMSSE": float(EV.rmsse(f[None], np.array([i]))[0]),
            "WAPE": float(np.abs(f - y).sum() / tot) if tot > 0 else np.nan,
            "Bias": float((f.sum() - tot) / tot) if tot > 0 else np.nan,
            "28-day total": float(f.sum())}


def fnum(x: float, d: int = 3) -> str:
    return f"{x:.{d}f}"


def fmt_pct(x: float, signed: bool = False) -> str:
    if x is None or np.isnan(x):
        return "–"
    s = f"{x * 100:+.0f}%" if signed else f"{x * 100:.0f}%"
    return s.replace("-", "−")


def make_forecast_figure(i, q, levels, compare, show_days, title):
    h0, h1 = T - show_days, T
    hx, fx = DATES[h0:h1], DATES[T : T + HORIZON]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hx, y=HIST[i, h0:h1], name="History", mode="lines",
                             line=dict(color=C_HISTORY, width=1.4)))
    lo99, hi99 = quantile_at(q, levels, 0.01), quantile_at(q, levels, 0.99)
    lo80, hi80 = quantile_at(q, levels, 0.1), quantile_at(q, levels, 0.9)
    for lo, hi, name, alpha in [(lo99, hi99, "Timestral · 98% interval", 0.12),
                                (lo80, hi80, "Timestral · 80% interval", 0.25)]:
        fig.add_trace(go.Scatter(x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([hi, lo[::-1]]),
                                 fill="toself", fillcolor=f"rgba(232,89,12,{alpha})", line=dict(width=0),
                                 name=name, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=quantile_at(q, levels, 0.5), name="Timestral · median", mode="lines",
                             line=dict(color=C_TIMESTRAL, width=2.6)))
    for k in compare:
        fig.add_trace(go.Scatter(x=fx, y=REF[k][i], name=REF_LABELS[k], mode="lines",
                                 line=dict(color=C_METHODS.get(k, "#666"), width=1.6, dash="dot")))
    fig.add_trace(go.Scatter(x=fx, y=EV.actual[i], name="Actual (test)", mode="lines+markers",
                             line=dict(color=C_ACTUAL, width=2), marker=dict(size=4)))
    fig.add_vline(x=DATES[T], line=dict(color="#57606A", width=1, dash="dash"))
    fig.add_annotation(x=DATES[T], y=1.02, yref="paper", text="forecast start", showarrow=False,
                       font=dict(size=11, color="#57606A"), xanchor="left")
    for d in range(T, T + HORIZON):  # events in the horizon (known in advance)
        if EVENTS[d]:
            fig.add_annotation(x=DATES[d], y=0.98, yref="paper", text=EVENTS[d], showarrow=False, textangle=-90,
                               font=dict(size=9, color="#57606A"), yanchor="top", xshift=-6)
    fig.update_layout(title=dict(text=title, font=dict(size=16)), template="plotly_white", height=470,
                      hovermode="x unified", margin=dict(l=50, r=20, t=60, b=40),
                      legend=dict(orientation="h", y=-0.12, x=0), yaxis_title="Units sold / day",
                      xaxis=dict(tickformat="%b %d"))
    return fig


def kpi_card(label: str, value: str, sub: str = "", tone: str = "") -> str:
    return f'<div class="kpi {tone}"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div>' \
           f'<div class="kpi-sub">{sub}</div></div>'


def run_forecast(level, lieu, produit, article, model_name, cov, ctx_days, compare, show_days):
    i = resolve(level, lieu, produit, article)
    fc = get_model(model_name)
    t0 = time.time()
    q, levels = forecast_series(fc, DATA, [i], HIST, CAL, cov, None if ctx_days >= T else int(ctx_days))
    elapsed = time.time() - t0
    q = q[0]
    med, mean = quantile_at(q, levels, 0.5), mean_from_quantiles(q, levels)
    y = EV.actual[i]

    rows = {"Timestral · median": series_metrics(i, med), "Timestral · mean": series_metrics(i, mean)}
    for k in REF:
        rows[REF_LABELS[k]] = series_metrics(i, REF[k][i])
    table = pd.DataFrame(rows).T
    table.loc["Actual"] = {"RMSSE": np.nan, "WAPE": np.nan, "Bias": np.nan, "28-day total": float(y.sum())}
    shown = table.copy()
    shown["RMSSE"] = shown["RMSSE"].map(lambda v: "–" if np.isnan(v) else fnum(v))
    shown["WAPE"] = shown["WAPE"].map(fmt_pct)
    shown["Bias"] = shown["Bias"].map(lambda v: fmt_pct(v, True))
    shown["28-day total"] = shown["28-day total"].map(lambda v: f"{v:,.0f}")
    shown = shown.reset_index(names="Method")

    best_t = min(rows["Timestral · median"]["RMSSE"], rows["Timestral · mean"]["RMSSE"])
    stat_keys = [k for k in REF if k != "winner"]
    best_stat = min(stat_keys, key=lambda k: rows[REF_LABELS[k]]["RMSSE"]) if stat_keys else None
    in80 = int(((y >= quantile_at(q, levels, 0.1)) & (y <= quantile_at(q, levels, 0.9))).sum())
    bias = rows["Timestral · median"]["Bias"]

    cards = [kpi_card("Timestral RMSSE", fnum(best_t), "lower is better")]
    if best_stat:
        rs = rows[REF_LABELS[best_stat]]["RMSSE"]
        gain = 1 - best_t / rs
        cards.append(kpi_card(f"vs {REF_LABELS[best_stat]}", fmt_pct(gain, True),
                              f"best statistical method (RMSSE {fnum(rs)})", "good" if gain > 0 else "bad"))
    if "winner" in REF:
        rw = rows[REF_LABELS["winner"]]["RMSSE"]
        cards.append(kpi_card("vs M5 winner", fmt_pct(1 - best_t / rw, True), f"RMSSE {fnum(rw)}",
                              "good" if best_t < rw else "bad"))
    cards.append(kpi_card("Actual inside 80% band", f"{in80}/28 days", "≈ 22 days expected",
                          "good" if 18 <= in80 <= 26 else "bad"))
    cards.append(kpi_card("Bias over 28 days", fmt_pct(bias, True), "median vs actual"))
    cards.append(kpi_card("Compute time", f"{elapsed * 1000:.0f} ms", f"{TEST_MODEL or model_name} · CPU"))
    kpis = '<div class="kpis">' + "".join(cards) + "</div>"

    zs = zero_share(i)
    verdict = "beats" if best_stat and best_t < rows[REF_LABELS[best_stat]]["RMSSE"] else "trails"
    which = "mean" if rows["Timestral · mean"]["RMSSE"] < rows["Timestral · median"]["RMSSE"] else "median"
    reading = (f"**Reading.** On this series ({fmt_pct(zs)} zero-sales days over the last year), "
               f"Timestral **{verdict}** the best statistical method. "
               f"Actual demand stayed inside the 80% band on **{in80} of 28 days**. "
               f"On RMSSE, Timestral's {which} does better"
               + ("; on an intermittent series the median is often 0 and underestimates demand."
                  if zs > 0.4 and which == "mean" else "."))
    fig = make_forecast_figure(i, q, levels, compare, int(show_days), series_name(i))
    state = {"i": i, "q": q.tolist(), "levels": list(levels)}
    return fig, kpis, shown, reading, state, *service_level(state, 90)


def service_level(state, target):
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
    fig.add_trace(go.Bar(x=fx, y=y, name="Actual demand", marker_color=["#CF222E" if a > b else "#8C959F"
                                                                        for a, b in zip(y, level_q)]))
    fig.add_trace(go.Scatter(x=fx, y=level_q, name=f"Stock set at the {target}% quantile", mode="lines",
                             line=dict(color=C_TIMESTRAL, width=2.4, shape="hv")))
    fig.add_trace(go.Scatter(x=fx, y=med, name="Median", mode="lines",
                             line=dict(color=C_TIMESTRAL, width=1.2, dash="dot", shape="hv")))
    fig.update_layout(template="plotly_white", height=330, margin=dict(l=50, r=20, t=20, b=40),
                      legend=dict(orientation="h", y=-0.2), yaxis_title="Units / day", bargap=0.25,
                      xaxis=dict(tickformat="%b %d"))
    html = '<div class="kpis">' + "".join([
        kpi_card("Days covered", fmt_pct(cov_q), f"target {target}% · {int(round(cov_q * 28))}/28 days",
                 "good" if abs(cov_q - u) <= 0.1 else "bad"),
        kpi_card("Fill rate (volume)", fmt_pct(fill), f"{short:,.0f} units not covered"),
        kpi_card("Cumulative surplus", f"{surplus:,.0f}", "units above demand"),
        kpi_card("With the median only", fmt_pct(cov_m), "days covered, for comparison"),
    ]) + "</div>"
    return fig, html


# ----------------------------------------------------------------------------- mini-benchmark
def mini_benchmark(model_name, cov, n_items, seed, progress=gr.Progress()):
    rng = np.random.default_rng(int(seed))
    agg_idx = np.concatenate([SLICES[l] for l, _ in LEVELS[:9]])
    sample = {l: rng.choice(SLICES[l], size=min(int(n_items), len(SLICES[l])), replace=False)
              for l in ("Level10", "Level11", "Level12")}
    idx = np.concatenate([agg_idx] + list(sample.values()))
    fc = get_model(model_name)
    progress(0, desc="Loading the model")
    t0 = time.time()
    q, levels = forecast_series(fc, DATA, idx, HIST, CAL, cov, batch_size=128,
                                progress=lambda d, n: progress(d / n, desc=f"Timestral: {d}/{n} series"))
    elapsed = time.time() - t0
    med = q[..., list(levels).index(0.5)]
    mean = mean_from_quantiles(q, levels)
    methods = {"Timestral · median": med, "Timestral · mean": mean}
    methods.update({REF_LABELS[k]: REF[k][idx] for k in REF})
    r = {m: EV.rmsse(f, idx) for m, f in methods.items()}
    spl_t = EV.spl(interpolate_quantiles(q, levels), idx=idx)
    spl_e = EV.spl(EMP_Q[idx], idx=idx)
    w = EV.weights[idx]
    lv = DATA.series["level"].to_numpy()[idx]
    rows = []
    for l, _ in LEVELS:
        m = lv == l
        row = {"Level": f"{l.replace('Level', '')}. {LEVEL_LABELS[l]}", "Series": int(m.sum())}
        for name, vals in r.items():
            row[name] = float((w[m] * vals[m]).sum() / w[m].sum())
        row["WSPL Timestral"] = float((w[m] * spl_t[m]).sum() / w[m].sum())
        row["WSPL empirical quantiles"] = float((w[m] * spl_e[m]).sum() / w[m].sum())
        rows.append(row)
    df = pd.DataFrame(rows)
    total = df.drop(columns=["Level", "Series"]).mean()

    fig = go.Figure()
    palette = {"Timestral · median": "#F4A261", "Timestral · mean": C_TIMESTRAL}
    palette.update({REF_LABELS[k]: C_METHODS.get(k, "#999") for k in REF})
    for name in r:
        fig.add_trace(go.Bar(x=df["Level"], y=df[name], name=name, marker_color=palette[name]))
    fig.update_layout(template="plotly_white", barmode="group", height=420, yaxis_title="Weighted RMSSE",
                      margin=dict(l=50, r=20, t=30, b=120), legend=dict(orientation="h", y=1.08))
    best_t = min(total["Timestral · median"], total["Timestral · mean"])
    stat = {REF_LABELS[k]: total[REF_LABELS[k]] for k in REF if k != "winner"}
    best_stat_name = min(stat, key=stat.get) if stat else None
    parts = [f"**{len(idx)} series forecast in {elapsed:.0f} s.** Timestral overall score: **{fnum(best_t)}**"]
    if best_stat_name:
        parts.append(f"best statistical method ({best_stat_name}): **{fnum(stat[best_stat_name])}**")
    if "winner" in REF:
        parts.append(f"competition winner: **{fnum(total[REF_LABELS['winner']])}**")
    summary = " · ".join(parts) + (
        f"\n\nQuantiles: Timestral WSPL **{fnum(total['WSPL Timestral'])}** vs "
        f"**{fnum(total['WSPL empirical quantiles'])}** for empirical day-of-week quantiles."
        "\n\n*Levels 1 to 9 are complete; levels 10 to 12 are estimated on the drawn sample. "
        "Timestral forecasts each series directly; the other methods are aggregated from the item × store "
        "level (competition rule).*")
    shown = pd.concat([pd.DataFrame([{"Level": "Overall score (mean of 12)", "Series": len(idx),
                                      **total.to_dict()}]), df], ignore_index=True)
    for c in shown.columns[2:]:
        shown[c] = shown[c].map(lambda v: fnum(v))
    return summary, fig, shown


def leaderboard() -> pd.DataFrame | None:
    if SCORES is None:
        return None
    s = SCORES[SCORES["métrique"] == "wrmsse"].copy()
    s = s[~((s["famille"] == "timestral") & (s["agrégation"] == "direct"))]
    names = {"snaive": "Seasonal naive (7 days)", "naive": "Naive (last value)", "ma28": "28-day moving average",
             "ses": "Simple exponential smoothing", "sba": "Croston SBA", "tsb": "TSB", "adida": "ADIDA",
             "imapa": "iMAPA", "ets": "Seasonal AutoETS"}
    variants = {"median": "median", "mean": "mean"}
    s["méthode"] = s["méthode"].map(lambda m: names.get(m, m))
    s["Method"] = s["méthode"] + s["variante"].fillna("").map(lambda v: f" · {variants.get(v, v)}" if v else "")
    s["Type"] = s["famille"].map({"benchmark": "Statistical", "concours M5": "Competition (LightGBM…)",
                                  "timestral": "Timestral zero-shot"}).fillna(s["famille"])
    s["WRMSSE"] = s["score"].map(lambda v: fnum(v, 4))
    return s[["Method", "Type", "WRMSSE"]].reset_index(drop=True)


# ----------------------------------------------------------------------------- your data
FREQS = {"D": ("daily", 7, 28), "W": ("weekly", 52, 13), "M": ("monthly", 12, 12), "Q": ("quarterly", 4, 4)}


def parse_dates(col: pd.Series) -> pd.Series:
    """ISO dates (2024-01-31) first, then day-first dates (31/01/2024)."""
    iso = pd.to_datetime(col, errors="coerce", format="ISO8601")
    if iso.notna().mean() > 0.9:
        return iso
    return pd.to_datetime(col, errors="coerce", dayfirst=True)


def read_user_csv(path: str) -> tuple[pd.DataFrame, str]:
    raw = pd.read_csv(path, sep=None, engine="python")
    date_col = next((c for c in raw.columns if not pd.api.types.is_numeric_dtype(raw[c])
                     and parse_dates(raw[c].astype(str)).notna().mean() > 0.9), None)
    if date_col is None:
        raise gr.Error("No date column found. Expected format: a date column, then one column per series "
                       "(or the columns series, date, value).")
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
    path = Path(tempfile.gettempdir()) / "example_m5_monthly_sales.csv"
    monthly.rename_axis("date").reset_index().to_csv(path, index=False, date_format="%Y-%m-%d")
    return str(path)


EXAMPLE_CSV = example_csv()


def run_user(path, horizon, backtest, model_name):
    if not path:
        raise gr.Error("Upload a CSV file first (or load the example).")
    wide, freq = read_user_csv(path)
    fname, season, default_h = FREQS[freq]
    h = int(horizon) if horizon else default_h
    values = wide.to_numpy().T  # (S, T)
    ctx = values[:, :-h] if backtest else values
    if ctx.shape[1] < 3:
        raise gr.Error("History too short for this horizon.")
    fc = get_model(model_name)
    res = fc.forecast(ctx[:, None, :].astype(np.float32), prediction_length=h)
    q = res.quantiles[:, 0].cpu().numpy()
    levels = res.quantile_levels
    off = {"D": pd.DateOffset(days=1), "W": pd.DateOffset(weeks=1), "M": pd.DateOffset(months=1),
           "Q": pd.DateOffset(months=3)}[freq]
    last = wide.index[-h - 1] if backtest else wide.index[-1]
    fdates = pd.DatetimeIndex([last + off * (k + 1) for k in range(h)])
    out = []
    metrics = []
    for s, name in enumerate(wide.columns):
        med = q[s, :, list(levels).index(0.5)]
        out.append(pd.DataFrame({"series": name, "date": fdates.strftime("%Y-%m-%d"),
                                 "p10": q[s][:, list(levels).index(0.1)], "p50": med,
                                 "p90": q[s][:, list(levels).index(0.9)],
                                 "mean": mean_from_quantiles(q[s], levels)}))
        if backtest:
            y = values[s, -h:]
            hist = ctx[s][~np.isnan(ctx[s])]
            snaive = np.resize(hist[-season:], h) if len(hist) >= season else np.repeat(hist[-1], h)
            for mname, f in (("Timestral", med), ("Seasonal naive", snaive), ("Naive", np.repeat(hist[-1], h))):
                tot = np.nansum(y)
                metrics.append({"Series": name, "Method": mname,
                                "WAPE": np.nansum(np.abs(f - y)) / tot if tot else np.nan,
                                "Bias": (np.nansum(f) - tot) / tot if tot else np.nan})
    fc_df = pd.concat(out, ignore_index=True)
    csv_path = Path(tempfile.gettempdir()) / "timestral_forecasts.csv"
    fc_df.round(3).to_csv(csv_path, index=False)
    mdf = None
    if metrics:
        mdf = pd.DataFrame(metrics)
        summary = mdf.groupby("Method")[["WAPE", "Bias"]].mean().reset_index().assign(Series="Average")
        mdf = pd.concat([summary, mdf], ignore_index=True)[["Series", "Method", "WAPE", "Bias"]]
        mdf["WAPE"] = mdf["WAPE"].map(fmt_pct)
        mdf["Bias"] = mdf["Bias"].map(lambda v: fmt_pct(v, True))
    state = {"wide": wide.reset_index().to_json(date_format="iso"), "fc": fc_df.to_json(), "h": h,
             "backtest": bool(backtest)}
    names = [str(c) for c in wide.columns]
    info = (f"**{len(names)} series** · **{fname}** frequency · {wide.shape[0]} points · horizon **{h}** · "
            + ("backtest on the last periods" if backtest else "forecast of the future"))
    return info, gr.update(choices=names, value=names[0]), user_plot(names[0], state), mdf, str(csv_path), state


def user_plot(name, state):
    if not state or name is None:
        return None
    wide = pd.read_json(io.StringIO(state["wide"]))
    wide = wide.set_index(wide.columns[0])
    wide.index = pd.to_datetime(wide.index)
    fc = pd.read_json(io.StringIO(state["fc"]))
    fc["series"] = fc["series"].astype(str)
    f = fc[fc["series"] == str(name)]
    col = [c for c in wide.columns if str(c) == str(name)][0]
    fx = pd.to_datetime(f["date"])
    fig = go.Figure()
    y = wide[col]
    cut = len(y) - state["h"] if state["backtest"] else len(y)
    fig.add_trace(go.Scatter(x=wide.index[:cut], y=y.iloc[:cut], name="History seen by the model",
                             mode="lines+markers", line=dict(color=C_HISTORY, width=1.6), marker=dict(size=4)))
    if state["backtest"]:
        fig.add_trace(go.Scatter(x=wide.index[cut - 1:], y=y.iloc[cut - 1:], name="Actual (hidden from the model)",
                                 mode="lines+markers", line=dict(color=C_ACTUAL, width=2), marker=dict(size=5)))
        fig.add_vline(x=wide.index[cut - 1], line=dict(color="#57606A", width=1, dash="dash"))
    fig.add_trace(go.Scatter(x=np.concatenate([fx, fx[::-1]]), y=np.concatenate([f["p90"], f["p10"][::-1]]),
                             fill="toself", fillcolor="rgba(232,89,12,0.22)", line=dict(width=0),
                             name="80% interval", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fx, y=f["p50"], name="Timestral · median", mode="lines+markers",
                             line=dict(color=C_TIMESTRAL, width=2.4), marker=dict(size=5)))
    fig.update_layout(template="plotly_white", height=430, hovermode="x unified", title=str(name),
                      margin=dict(l=50, r=20, t=50, b=40), legend=dict(orientation="h", y=-0.15))
    return fig


# ----------------------------------------------------------------------------- interface
CSS = """
.kpis {display:grid; grid-template-columns:repeat(auto-fit,minmax(160px,1fr)); gap:10px; margin:4px 0 8px}
.kpi {border:1px solid var(--border-color-primary); border-radius:10px; padding:10px 12px;
      background:var(--block-background-fill)}
.kpi-label {font-size:12px; color:var(--body-text-color-subdued)}
.kpi-value {font-size:22px; font-weight:650; margin-top:2px; font-variant-numeric:tabular-nums}
.kpi-sub {font-size:11.5px; color:var(--body-text-color-subdued); margin-top:2px}
.kpi.good .kpi-value {color:#1A7F37} .kpi.bad .kpi-value {color:#CF222E}
.hero h1 {margin-bottom:0} .hero p {margin-top:4px; color:var(--body-text-color-subdued)}
.test-banner {background:#FFF4E5; border:1px solid #F5B971; border-radius:8px; padding:8px 12px; color:#7A3E00}
footer {display:none !important}
"""

INTRO = """
<div class="hero">
<h1>Timestral × M5 Walmart</h1>
<p>Mistral's time-series foundation model, used without any training (zero-shot), against statistical methods
and the winner of the M5 competition: 42,840 daily sales series, 28-day horizon, official test period
May 23 – June 19, 2016.</p>
</div>
"""

ABOUT = """
### The benchmark
The **M5 competition** (2020) asked participants to forecast daily sales of 3,049 items in 10 Walmart stores
(California, Texas, Wisconsin) over 28 days. The 30,490 item × store series roll up into **12 levels**
(total, state, store, category, department, their crossings, item). More than half of the bottom-level
series are **intermittent** (many zero-sales days).

**Official metric: WRMSSE.** For each series, the squared error is scaled by that of a one-step naive
forecast on the history (RMSSE), then weighted by dollar sales over the last 28 days. The final score is the
mean over the 12 levels. Below 1 means better than naive. The winner scored **0.520**, the seasonal naive **0.847**.

**Quantiles: WSPL.** Same idea with the pinball loss on 9 quantiles (0.5% to 99.5%).

### Timestral
A family of models from Mistral (5M to 209M parameters, October 2026), pre-trained mostly on synthetic series.
They output 21 quantiles per time step, accept covariates known in advance and run on CPU. Here, **no tuning
on M5**: the model sees a series' history and forecasts.

### How to read the app
- **Explore**: a live forecast, compared with actuals and other methods.
- **Supply chain view**: if daily stock is set at a Timestral quantile, what share of days is actually covered?
- **Benchmark**: the score on a sample of series, level by level, in one to two minutes.
- **Your data**: your own series (CSV), as a forecast or a backtest.

### Limitations
- M5 appears in public benchmarks on which Timestral was evaluated: a good score does not prove generalisation.
- M5 measures point-of-sale sales (demand censored by stock-outs), not B2B orders.
- The 0.5% and 99.5% levels are extrapolated from Timestral's 1% – 99% grid.

Source code and full benchmark: [github.com/baptistesoulard/M5_Timestral](https://github.com/baptistesoulard/M5_Timestral) ·
[Timestral (Mistral)](https://mistral.ai/news/timestral/) · [M5 data and submissions (Nixtla)](https://github.com/Nixtla/m5-forecasts)
"""

MODEL_CHOICES = list(MODELS)
COMPARE_DEFAULT = [k for k in ("snaive", "ets", "winner") if k in REF]

with gr.Blocks(title="Timestral × M5") as demo:
    gr.HTML(INTRO)
    if TEST_MODEL:
        gr.HTML(f'<div class="test-banner">Test mode: "{TEST_MODEL}" model with random weights. '
                "Forecasts are meaningless; only the interface is being tested.</div>")

    with gr.Tab("Explore"):
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                gr.Markdown("**Examples**")
                with gr.Row():
                    ex_btns = [gr.Button(label, size="sm") for label, _ in EXAMPLES]
                level = gr.Dropdown(LEVEL_CHOICES, value="Level12", label="Aggregation level")
                first = DATA.series.iloc[EXAMPLES[2][1]]  # fast-moving item × store, shown at start-up
                lieu = gr.Dropdown(label="Store", choices=STORES, value=first.agg2, allow_custom_value=True)
                produit = gr.Dropdown(label="Category", choices=CATS, value=CATS[0], visible=False,
                                      allow_custom_value=True)
                rayon = gr.Dropdown(label="Department", choices=DEPTS, value=DEPT_OF_ITEM[first.agg1])
                article = gr.Dropdown(label="Item", choices=ITEMS_BY_DEPT[DEPT_OF_ITEM[first.agg1]], value=first.agg1,
                                      allow_custom_value=True)
                with gr.Accordion("Model settings", open=False):
                    model_dd = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model size",
                                           info="5M and 28M: fast. 209M: most accurate, a few seconds.")
                    cov = gr.Radio(COV_CHOICES, value="calendar", label="Covariates known in advance")
                    ctx = gr.Slider(56, T, value=T, step=7, label="History given to the model (days)",
                                    info="Reduce to simulate a short history")
                compare = gr.CheckboxGroup([(v, k) for k, v in REF_LABELS.items()], value=COMPARE_DEFAULT,
                                           label="Compare with")
                show_days = gr.Slider(28, 365, value=112, step=7, label="History shown (days)")
                go_btn = gr.Button("Forecast with Timestral", variant="primary", size="lg")
            with gr.Column(scale=3):
                kpis = gr.HTML()
                plot = gr.Plot(label="28-day forecast")
                reading = gr.Markdown()
                table = gr.Dataframe(label="Errors over the test period", interactive=False, wrap=True)
                with gr.Accordion("Supply chain view: what service level do the quantiles deliver?", open=True):
                    gr.Markdown("Each day's stock is set at the chosen quantile of the Timestral forecast. "
                                "Red bars are days when demand exceeded that stock.")
                    target = gr.Slider(50, 99, value=90, step=1, label="Target service level (%)")
                    svc_kpis = gr.HTML()
                    svc_plot = gr.Plot(show_label=False)
        fstate = gr.State()

        sel_inputs = [level, lieu, produit, article]
        run_inputs = sel_inputs + [model_dd, cov, ctx, compare, show_days]
        run_outputs = [plot, kpis, table, reading, fstate, svc_plot, svc_kpis]
        level.input(lambda l: selector_updates(l), level, [lieu, produit, rayon, article])
        rayon.input(lambda r: gr.update(choices=ITEMS_BY_DEPT[r], value=ITEMS_BY_DEPT[r][0]), rayon, article)
        go_btn.click(run_forecast, run_inputs, run_outputs)
        for btn, (_, i) in zip(ex_btns, EXAMPLES):
            btn.click(lambda i=i: example_updates(i), None, [level, lieu, produit, rayon, article]).then(
                run_forecast, run_inputs, run_outputs)
        target.change(service_level, [fstate, target], [svc_plot, svc_kpis])
        demo.load(run_forecast, run_inputs, run_outputs)

    with gr.Tab("Benchmark"):
        gr.Markdown("Timestral forecasts live **all 154 aggregate series** (levels 1 to 9) and a **sample** of the "
                    "item levels. Scores are computed as in the competition, on the same series for every method.")
        with gr.Row():
            b_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model size")
            b_cov = gr.Radio(COV_CHOICES, value="calendar", label="Covariates")
            b_n = gr.Slider(50, 2000, value=300, step=50, label="Series drawn per item level (10, 11, 12)")
            b_seed = gr.Number(value=42, label="Random seed", precision=0)
        b_btn = gr.Button("Run the mini-benchmark", variant="primary")
        b_summary = gr.Markdown()
        b_plot = gr.Plot(show_label=False)
        b_table = gr.Dataframe(interactive=False, wrap=True, max_height=620)
        b_btn.click(mini_benchmark, [b_model, b_cov, b_n, b_seed], [b_summary, b_plot, b_table])
        gr.Markdown("### Full leaderboard (42,840 series, official WRMSSE)")
        lb = leaderboard()
        if lb is not None:
            gr.Dataframe(lb, interactive=False)
            gr.Markdown("*Full Timestral runs appear here after `run_timestral.py`, `evaluate.py` "
                        "and `build_app_assets.py`.*")

    with gr.Tab("Your data"):
        gr.Markdown("Forecast your own series with Timestral. **Format**: a `date` column then one column per series, "
                    "or three columns `series, date, value`. Frequency is detected automatically "
                    "(day, week, month, quarter).\n\n"
                    "⚠️ This is a public app: do not upload confidential data. Run the app locally instead "
                    "(`python app.py`) so your data stays on your machine.")
        with gr.Row():
            with gr.Column(scale=1, min_width=300):
                u_file = gr.File(label="CSV file", file_types=[".csv"], type="filepath")
                u_example = gr.Button("Load the example: monthly sales of the 7 M5 departments", size="sm")
                u_h = gr.Number(value=0, label="Horizon (0 = automatic)", precision=0,
                                info="Automatic: 28 days, 13 weeks, 12 months or 4 quarters")
                u_bt = gr.Checkbox(value=True, label="Backtest: hide the last periods and measure the error")
                u_model = gr.Dropdown(MODEL_CHOICES, value=DEFAULT_MODEL, label="Model size")
                u_btn = gr.Button("Forecast", variant="primary")
            with gr.Column(scale=3):
                u_info = gr.Markdown()
                u_series = gr.Dropdown(label="Series shown", choices=[], allow_custom_value=True)
                u_plot = gr.Plot(show_label=False)
                u_metrics = gr.Dataframe(label="Backtest error (WAPE = absolute error / actual volume)",
                                         interactive=False, max_height=420)
                u_dl = gr.File(label="Forecasts (CSV: p10, p50, p90, mean)")
        ustate = gr.State()
        u_example.click(lambda: EXAMPLE_CSV, None, u_file)
        u_btn.click(run_user, [u_file, u_h, u_bt, u_model], [u_info, u_series, u_plot, u_metrics, u_dl, ustate])
        u_series.input(user_plot, [u_series, ustate], u_plot)

    with gr.Tab("Method"):
        gr.Markdown(ABOUT)

if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"), theme=gr.themes.Soft(primary_hue="orange"),
        css=CSS)
