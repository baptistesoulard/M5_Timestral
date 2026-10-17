"""Covariates known in advance for every M5 series (42,840), over history + horizon (1,969 days).

Modes passed to Timestral:
- "none"     : sales only.
- "calendar" : SNAP days (numeric) + event name and day of week (categorical).
- "full"     : calendar + relative selling price (item levels only).
- "rich"     : calendar signals and assortment / price signals for every level:
    numeric     SNAP share, price index, discount depth, share of items on sale (listed), day of month,
                days to the next event, days since the last event, Christmas closure
    categorical event name, event type, day of week
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m5bench.data import M5Data, calendar_covariates, series_snap

COVARIATE_MODES = ("none", "calendar", "full", "rich")

RICH_NUMERIC = ("snap", "price_index", "discount", "listed_share", "day_of_month", "days_to_event",
                "days_since_event", "christmas")
RICH_CATEGORICAL = ("event_name", "event_type", "weekday")


def _days_to_next(flag: np.ndarray, cap: int) -> np.ndarray:
    out = np.full(len(flag), cap, dtype=np.float32)
    nxt = None
    for t in range(len(flag) - 1, -1, -1):
        if flag[t]:
            nxt = t
        if nxt is not None:
            out[t] = min(nxt - t, cap)
    return out


def _days_since_last(flag: np.ndarray, cap: int) -> np.ndarray:
    out = np.full(len(flag), cap, dtype=np.float32)
    last = None
    for t in range(len(flag)):
        if flag[t]:
            last = t
        if last is not None:
            out[t] = min(t - last, cap)
    return out


class CovariateBank:
    """Builds covariate rows for any batch of series. Price-based signals are computed once, lazily."""

    def __init__(self, data: M5Data):
        self.data = data
        cal = data.calendar
        self.cal = calendar_covariates(cal)
        self.length = len(cal)
        has_event = cal["event_name_1"].notna().to_numpy() | cal["event_name_2"].notna().to_numpy()
        dates = pd.to_datetime(cal["date"])
        self.event_type = pd.factorize(cal["event_type_1"].fillna("none"))[0].astype(np.float64)
        self.day_of_month = (dates.dt.day.to_numpy(dtype=np.float32) - 1) / 30
        self.days_to_event = _days_to_next(has_event, 14) / 14
        self.days_since_event = _days_since_last(has_event, 7) / 7
        self.christmas = ((dates.dt.month == 12) & (dates.dt.day == 25)).to_numpy(dtype=np.float32)
        self._bottom_price = None

    # ---------------------------------------------------------------- price / assortment signals
    def _price_bottom(self):
        """Bottom-level (30490, L) arrays: relative price, discount vs 52-week max, listed indicator."""
        if self._bottom_price is None:
            prices = self.data.prices  # (30490, L) float32, NaN when the item is not listed
            n, length = prices.shape
            rel = np.zeros((n, length), dtype=np.float32)
            disc = np.zeros((n, length), dtype=np.float32)
            listed = np.zeros((n, length), dtype=np.float32)
            # prices are weekly: 52-week trailing max computed on week starts, then broadcast to days
            weeks = pd.factorize(self.data.calendar["wm_yr_wk"])[0]
            first_day = np.r_[0, np.flatnonzero(np.diff(weeks)) + 1]
            for s in range(0, n, 2048):  # in blocks, to keep memory low
                p = prices[s : s + 2048]
                on = ~np.isnan(p)
                with np.errstate(invalid="ignore", divide="ignore", all="ignore"):
                    med = np.nanmedian(p, axis=1, keepdims=True).astype(np.float32)
                    rel[s : s + 2048] = np.where(on, p / med, 0.0)
                    weekly = np.where(on[:, first_day], p[:, first_day], -np.inf).astype(np.float32)
                    padded = np.concatenate([np.full((len(p), 51), -np.inf, dtype=np.float32), weekly], axis=1)
                    wmax = np.lib.stride_tricks.sliding_window_view(padded, 52, axis=1).max(axis=2)[:, weeks]
                    d = np.where(on & np.isfinite(wmax) & (wmax > 0), 1.0 - p / wmax, 0.0)
                disc[s : s + 2048] = np.clip(np.nan_to_num(d), 0, 1)
                listed[s : s + 2048] = on
            self._bottom_price = (rel, disc, listed)
        return self._bottom_price

    def price_signals(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(price index, discount depth, listed share) for series `idx`, each (len(idx), L)."""
        rel, disc, listed = self._price_bottom()
        sub = self.data.agg[idx]
        n_listed = np.asarray(sub @ listed)
        n_items = np.asarray(sub.sum(axis=1))
        with np.errstate(invalid="ignore", divide="ignore"):
            price_index = np.where(n_listed > 0, np.asarray(sub @ rel) / n_listed, np.nan)
            discount = np.where(n_listed > 0, np.asarray(sub @ disc) / n_listed, np.nan)
        return price_index.astype(np.float32), discount.astype(np.float32), (n_listed / n_items).astype(np.float32)

    def relative_price_items(self, idx: np.ndarray) -> np.ndarray | None:
        """'full' mode: mean relative price for item levels (10, 11, 12) only, else None."""
        levels = self.data.series["level"].to_numpy()[idx]
        if not np.isin(levels, ["Level10", "Level11", "Level12"]).all():
            return None
        price_index, _, _ = self.price_signals(idx)
        return price_index

    # ---------------------------------------------------------------- rows for Timestral
    def build(self, idx: np.ndarray, mode: str, start: int, end: int) -> dict:
        """kwargs for TimestralForecaster.forecast: covariates over days [start, end)."""
        if mode == "none":
            return {}
        idx = np.asarray(idx)
        span = slice(start, end)
        n = len(idx)
        snap = series_snap(self.data, idx, self.cal["snap"])[:, span]
        numeric = [snap]
        cats = [self.cal["event"][span], self.cal["wday"][span]]
        if mode == "full":
            price = self.relative_price_items(idx)
            if price is not None:
                numeric.append(price[:, span])
        elif mode == "rich":
            price_index, discount, listed_share = self.price_signals(idx)
            numeric += [price_index[:, span], discount[:, span], listed_share[:, span]]
            for row in (self.day_of_month, self.days_to_event, self.days_since_event, self.christmas):
                numeric.append(np.broadcast_to(row[span], (n, end - start)))
            cats = [self.cal["event"][span], self.event_type[span], self.cal["wday"][span]]
        elif mode != "calendar":
            raise ValueError(f"unknown covariate mode {mode!r}, expected one of {COVARIATE_MODES}")
        cat = np.stack(cats)
        return {"known_covariates": np.stack(numeric, axis=1).astype(np.float32),
                "categorical_known_covariates": np.broadcast_to(cat, (n,) + cat.shape).copy()}
