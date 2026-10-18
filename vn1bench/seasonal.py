"""Two ways of giving Timestral the yearly seasonality of VN1, without any training.

1. calendar_covariates: covariates known in the past and the future, computed from the dates only — week of the
   year (categorical) and e-commerce peak weeks (Black Friday / Cyber Monday, the two weeks before Christmas,
   Christmas - New Year, Amazon Prime Day).
2. seasonal_index: a multiplicative week-of-year index per client, measured on the history only (ratio to a
   centred 52-week moving average, averaged over the years, blended with the all-client index). The series is
   divided by the index before forecasting and the forecast multiplied back, as in seasonal-index methods.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

# Amazon Prime Day (first day of each event; 2022 and 2023 also had an October "Prime Early Access" event)
PRIME_DAYS = ["2020-10-13", "2021-06-21", "2022-07-12", "2022-10-11", "2023-07-11", "2023-10-10", "2024-07-16"]


def _thanksgiving(year: int) -> dt.date:
    nov1 = dt.date(year, 11, 1)
    first_thu = nov1 + dt.timedelta(days=(3 - nov1.weekday()) % 7)
    return first_thu + dt.timedelta(weeks=3)


def week_of_year(dates: pd.DatetimeIndex) -> np.ndarray:
    """ISO week 1..52 (week 53 folded into 52)."""
    return np.minimum(dates.isocalendar().week.to_numpy().astype(int), 52)


def calendar_covariates(dates: pd.DatetimeIndex) -> tuple[np.ndarray, np.ndarray]:
    """Weekly dates (Mondays) → numeric (4, T) and categorical (1, T) covariates."""
    t = len(dates)
    bf, xmas_pre, xmas, prime = (np.zeros(t, np.float32) for _ in range(4))
    primes = [pd.Timestamp(d) for d in PRIME_DAYS]
    for k, monday in enumerate(dates):
        week = [monday + pd.Timedelta(days=i) for i in range(7)]
        tg = pd.Timestamp(_thanksgiving(monday.year))
        if any(d in (tg, tg + pd.Timedelta(days=1), tg + pd.Timedelta(days=4)) for d in week):
            bf[k] = 1.0  # Thanksgiving / Black Friday week and Cyber Monday week
        if monday.month == 12 and 11 <= monday.day <= 24 and not any(d.month == 12 and d.day == 25 for d in week):
            xmas_pre[k] = 1.0
        if any((d.month == 12 and d.day == 25) or (d.month == 1 and d.day == 1) for d in week):
            xmas[k] = 1.0
        if any(d in primes for d in week):
            prime[k] = 1.0
    return np.stack([bf, xmas_pre, xmas, prime]), week_of_year(dates)[None, :].astype(np.int64)


def seasonal_index(train: np.ndarray, dates: pd.DatetimeIndex, groups: np.ndarray, blend: float = 0.5,
                   clip: tuple[float, float] = (0.3, 3.0)) -> np.ndarray:
    """Week-of-year index (G, 53) for each group (client), from the history only. Column 0 is unused."""
    t = train.shape[1]
    woy = week_of_year(dates[:t])
    ug, inv = np.unique(groups, return_inverse=True)
    tot = np.zeros((len(ug), t))
    np.add.at(tot, inv, train)

    def index_of(series: np.ndarray) -> np.ndarray:
        # centred 2x52 moving average: the level, without the season
        k = np.r_[0.5, np.ones(51), 0.5] / 52
        level = np.convolve(series, k, mode="same")
        ok = np.zeros(t, bool)
        ok[26 : t - 26] = True
        ok &= level > 0
        idx = np.ones(53)
        for w in range(1, 53):
            m = ok & (woy == w)
            if m.any():
                idx[w] = np.mean(series[m] / level[m])
        idx[1:] /= idx[1:].mean()
        return idx

    glob = index_of(tot.sum(0))
    out = np.empty((len(ug), 53))
    for g in range(len(ug)):
        own = index_of(tot[g]) if tot[g].sum() > 0 else glob
        out[g] = np.clip(blend * own + (1 - blend) * glob, *clip)
    return out[inv]
