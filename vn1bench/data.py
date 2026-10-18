"""VN1 forecasting competition (Nicolas Vandeput / SupChains, 2024): data, official score and reference forecasts.

15,053 product × warehouse series of weekly sales (Flieber e-commerce data), 13-week horizon.
History = phase 0 + phase 1 (183 weeks, July 2020 – January 2024); test = phase 2 (8 January – 1 April 2024).
Score = MAE% + |Bias%| = (Σ|forecast − sales| + |Σ(forecast − sales)|) / Σ sales.
Data and scoring code: https://github.com/SupChains/VN1 (data under CC-BY 4.0).
"""

from __future__ import annotations

import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

BASE = "https://github.com/SupChains/VN1/raw/main/"
HORIZON = 13
INDEX = ["Client", "Warehouse", "Product"]
FILES = ["Phase 0 - Sales.csv", "Phase 0 - Price.csv", "Phase 1 - Sales.csv", "Phase 1 - Price.csv",
         "Phase 2 - Sales.csv"]
TOP5 = [("vn1_1st", "Top 5 forecasts/1st - Jakub Figura and Philip Stubbs.csv"),
        ("vn1_2nd", "Top 5 forecasts/2nd - Justin Furlotte.csv"),
        ("vn1_3rd", "Top 5 forecasts/3rd - Arsa Nikzad.csv"),
        ("vn1_4th", "Top 5 forecasts/4th - Antoine Schwartz.csv"),
        ("vn1_5th", "Top 5 forecasts/5th - An Hoang.csv")]
OFFICIAL_BENCH = [("naive", "Benchmarks/Benchmark Phase 2 - Naive.csv"),
                  ("ma12", "Benchmarks/Benchmark Phase 2 - MA12.csv")]


def ensure_vn1(data_dir: str | Path) -> Path:
    data_dir = Path(data_dir)
    for f in FILES + [p for _, p in TOP5] + [p for _, p in OFFICIAL_BENCH]:
        dest = data_dir / f
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  download {f}", flush=True)
            urllib.request.urlretrieve(BASE + urllib.parse.quote(f), dest)
    return data_dir


def _read(path: Path) -> pd.DataFrame:
    t = pd.read_csv(path).set_index(INDEX)
    t.columns = pd.to_datetime(t.columns)
    return t


@dataclass
class VN1Data:
    ids: pd.DataFrame  # Client, Warehouse, Product (15,053 rows)
    train: np.ndarray  # (N, 183) weekly sales
    test: np.ndarray  # (N, 13)
    price: np.ndarray  # (N, 183) price from transactions, NaN when nothing sold
    dates: pd.DatetimeIndex  # 183 + 13 Mondays

    @property
    def n(self) -> int:
        return len(self.ids)


def load_vn1(data_dir: str | Path, phase: str = "test") -> VN1Data:
    """phase="test": history = phases 0 + 1, target = phase 2 (the official test).
    phase="val": history = phase 0, target = phase 1 (validation, as the competition's public leaderboard)."""
    d = Path(data_dir)
    s0, s1, s2 = (_read(d / f"Phase {k} - Sales.csv") for k in (0, 1, 2))
    p0, p1 = (_read(d / f"Phase {k} - Price.csv") for k in (0, 1))
    if phase == "val":
        sales, price, test = s0, p0.loc[s0.index, s0.columns], s1.loc[s0.index]
        return VN1Data(ids=sales.index.to_frame(index=False), train=sales.to_numpy(float),
                       test=test.to_numpy(float), price=price.to_numpy(float),
                       dates=sales.columns.append(test.columns))
    sales = pd.concat([s0, s1], axis=1)
    price = pd.concat([p0, p1], axis=1).loc[sales.index, sales.columns]
    test = s2.loc[sales.index]
    return VN1Data(ids=sales.index.to_frame(index=False), train=sales.to_numpy(float),
                   test=test.to_numpy(float), price=price.to_numpy(float),
                   dates=sales.columns.append(test.columns))


def reference_forecasts(data_dir: str | Path, data: VN1Data) -> dict[str, np.ndarray]:
    """Top 5 competition forecasts and the two official benchmarks, aligned on the data order."""
    d = Path(data_dir)
    key = pd.MultiIndex.from_frame(data.ids)
    out = {}
    for name, f in TOP5 + OFFICIAL_BENCH:
        t = _read(d / f)
        out[name] = t.loc[key].to_numpy(float)
    return out


def vn1_score(forecast: np.ndarray, actual: np.ndarray) -> dict:
    """Official VN1 score on a set of series: score, MAE% and Bias% (fractions)."""
    err = forecast - actual
    tot = actual.sum()
    mae, bias = np.abs(err).sum() / tot, err.sum() / tot
    return {"score": float(mae + abs(bias)), "mae": float(mae), "bias": float(bias)}


def mask_leading_zeros(x: np.ndarray) -> np.ndarray:
    out = x.astype(np.float32, copy=True)
    out[~(np.cumsum(x > 0, axis=1) > 0)] = np.nan
    return out


def relative_price(data: VN1Data) -> np.ndarray:
    """Price relative to the series' median price (NaN when nothing sold): a past-only covariate."""
    with np.errstate(invalid="ignore", divide="ignore"):
        med = np.nanmedian(data.price, axis=1, keepdims=True)
        rel = data.price / med
    return np.where(np.isfinite(rel), rel, np.nan).astype(np.float32)
