"""Chargement du jeu M5 (Walmart) et construction de la hiérarchie à 12 niveaux.

Configuration officielle « évaluation » du concours M5 :
- historique : d_1 … d_1941 (sales_train_evaluation.csv)
- horizon    : 28 jours, d_1942 … d_1969 (sales_test_evaluation.csv, publié après le concours)
- 30 490 séries article × magasin, agrégées en 42 840 séries sur 12 niveaux.

Accepte les fichiers Kaggle d'origine (colonne `id`) comme la copie Nixtla (sans `id`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HORIZON = 28
TRAIN_END = 1941  # dernier jour d'historique (d_1941)
ID_COLS = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]

# (nom du niveau, colonnes d'agrégation) — ordre et libellés identiques à weights_evaluation.csv
LEVELS: list[tuple[str, list[str]]] = [
    ("Level1", []),
    ("Level2", ["state_id"]),
    ("Level3", ["store_id"]),
    ("Level4", ["cat_id"]),
    ("Level5", ["dept_id"]),
    ("Level6", ["state_id", "cat_id"]),
    ("Level7", ["state_id", "dept_id"]),
    ("Level8", ["store_id", "cat_id"]),
    ("Level9", ["store_id", "dept_id"]),
    ("Level10", ["item_id"]),
    ("Level11", ["state_id", "item_id"]),
    ("Level12", ["item_id", "store_id"]),
]
LEVEL_LABELS = {
    "Level1": "Total",
    "Level2": "État",
    "Level3": "Magasin",
    "Level4": "Catégorie",
    "Level5": "Rayon",
    "Level6": "État × catégorie",
    "Level7": "État × rayon",
    "Level8": "Magasin × catégorie",
    "Level9": "Magasin × rayon",
    "Level10": "Article",
    "Level11": "Article × État",
    "Level12": "Article × magasin",
}


@dataclass
class M5Data:
    bottom_ids: pd.DataFrame  # 30 490 lignes, colonnes ID_COLS
    train: np.ndarray  # (30490, 1941) ventes quotidiennes
    test: np.ndarray | None  # (30490, 28) ventes réelles de l'horizon, si disponibles
    calendar: pd.DataFrame  # 1969 lignes (d_1 … d_1969)
    prices: np.ndarray  # (30490, 1969) prix de vente quotidien, NaN si article non référencé
    series: pd.DataFrame  # 42 840 lignes : level, agg1, agg2 (libellés officiels)
    agg: sp.csr_matrix  # (42840, 30490) matrice d'agrégation 0/1

    def aggregate(self, bottom: np.ndarray) -> np.ndarray:
        """Agrège un tableau (30490, …) en (42840, …) par sommation."""
        flat = bottom.reshape(bottom.shape[0], -1)
        return np.asarray(self.agg @ flat).reshape((self.agg.shape[0],) + bottom.shape[1:])


def _bottom_key(df: pd.DataFrame) -> pd.Series:
    return df["item_id"] + "_" + df["store_id"]


def build_hierarchy(bottom_ids: pd.DataFrame) -> tuple[pd.DataFrame, sp.csr_matrix]:
    rows, cols, frames = [], [], []
    offset = 0
    n = len(bottom_ids)
    for level, keys in LEVELS:
        if keys:
            codes, _ = pd.factorize(bottom_ids[keys].astype(str).agg("|".join, axis=1))
            _, first = np.unique(codes, return_index=True)
            labels = bottom_ids[keys].iloc[first].reset_index(drop=True)
            agg1 = labels[keys[0]].astype(str).to_numpy()
            agg2 = labels[keys[1]].astype(str).to_numpy() if len(keys) > 1 else np.full(len(labels), "X")
        else:
            codes = np.zeros(n, dtype=np.int64)
            agg1, agg2 = np.array(["Total"]), np.array(["X"])
        frames.append(pd.DataFrame({"level": level, "agg1": agg1, "agg2": agg2}))
        rows.append(codes + offset)
        cols.append(np.arange(n))
        offset += len(agg1)
    series = pd.concat(frames, ignore_index=True)
    data = np.ones(sum(len(r) for r in rows), dtype=np.float32)
    agg = sp.csr_matrix((data, (np.concatenate(rows), np.concatenate(cols))), shape=(offset, n))
    return series, agg


def load_m5(data_dir: str | Path) -> M5Data:
    data_dir = Path(data_dir)
    sales = pd.read_csv(data_dir / "sales_train_evaluation.csv")
    bottom_ids = sales[ID_COLS].reset_index(drop=True)
    day_cols = [f"d_{i}" for i in range(1, TRAIN_END + 1)]
    train = sales[day_cols].to_numpy(dtype=np.float32)

    test = None
    test_path = data_dir / "sales_test_evaluation.csv"
    if test_path.exists():
        t = pd.read_csv(test_path)
        t = t.set_index(_bottom_key(t)).loc[_bottom_key(bottom_ids)]
        test = t[[f"d_{i}" for i in range(TRAIN_END + 1, TRAIN_END + HORIZON + 1)]].to_numpy(dtype=np.float32)

    calendar = pd.read_csv(data_dir / "calendar.csv").iloc[: TRAIN_END + HORIZON].reset_index(drop=True)

    prices_long = pd.read_csv(data_dir / "sell_prices.csv")
    weeks = calendar["wm_yr_wk"].to_numpy()
    week_pos = {w: i for i, w in enumerate(pd.unique(weeks))}
    weekly = prices_long.pivot_table(index=["item_id", "store_id"], columns="wm_yr_wk", values="sell_price")
    weekly = weekly.reindex(columns=list(week_pos)).reindex(pd.MultiIndex.from_frame(bottom_ids[["item_id", "store_id"]]))
    day_week = np.array([week_pos[w] for w in weeks])
    prices = weekly.to_numpy(dtype=np.float32)[:, day_week]

    series, agg = build_hierarchy(bottom_ids)
    return M5Data(bottom_ids, train, test, calendar, prices, series, agg)


def level_slices(series: pd.DataFrame) -> dict[str, np.ndarray]:
    return {lvl: np.flatnonzero(series["level"].to_numpy() == lvl) for lvl, _ in LEVELS}


def mask_leading_zeros(x: np.ndarray) -> np.ndarray:
    """Remplace par NaN les zéros précédant la première vente (article pas encore lancé)."""
    out = x.astype(np.float32, copy=True)
    started = np.cumsum(x > 0, axis=1) > 0
    out[~started] = np.nan
    return out


def calendar_covariates(cal: pd.DataFrame) -> dict[str, np.ndarray]:
    """Covariables calendaires connues sur tout l'historique + horizon (longueur 1969)."""
    event = pd.factorize(cal["event_name_1"].fillna("none"))[0].astype(np.float64)
    wday = cal["wday"].to_numpy(dtype=np.float64)
    month = cal["month"].to_numpy(dtype=np.float64)
    snap = cal[["snap_CA", "snap_TX", "snap_WI"]].to_numpy(dtype=np.float32).T  # (3, T)
    return {"event": event, "wday": wday, "month": month, "snap": snap}


def series_snap(data: M5Data, idx: np.ndarray, snap: np.ndarray) -> np.ndarray:
    """SNAP par série : indicateur 0/1 de l'État pour les séries d'un seul État, sinon part des États
    (pondérée par le nombre de séries de base de chaque État)."""
    states = ["CA", "TX", "WI"]
    state_of_bottom = data.bottom_ids["state_id"].map({s: i for i, s in enumerate(states)}).to_numpy()
    onehot = np.zeros((len(state_of_bottom), 3), dtype=np.float32)
    onehot[np.arange(len(state_of_bottom)), state_of_bottom] = 1
    share = np.asarray(data.agg[idx] @ onehot)
    share /= share.sum(axis=1, keepdims=True)
    return share @ snap  # (len(idx), T)


def series_relative_price(data: M5Data, idx: np.ndarray) -> np.ndarray | None:
    """Prix relatif (prix / prix médian de la série) pour les niveaux article (10, 11, 12), sinon None."""
    levels = data.series["level"].to_numpy()[idx]
    if not np.isin(levels, ["Level10", "Level11", "Level12"]).all():
        return None
    p = np.nan_to_num(data.prices, nan=0.0)
    listed = (~np.isnan(data.prices)).astype(np.float32)
    sub = data.agg[idx]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_price = np.asarray(sub @ p) / np.asarray(sub @ listed)
        rel = mean_price / np.nanmedian(mean_price, axis=1, keepdims=True)
    return rel.astype(np.float32)
