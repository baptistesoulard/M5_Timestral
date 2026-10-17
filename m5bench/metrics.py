"""Métriques officielles du M5 : WRMSSE (piste précision) et WSPL (piste incertitude).

- RMSSE : RMSE sur l'horizon, normalisée par la RMSE du naïf 1 pas calculée sur l'historique
  à partir de la première vente non nulle de la série.
- SPL   : pinball loss moyenne sur l'horizon, normalisée par la MAE du naïf 1 pas (même période).
- Poids : ventes en valeur ($) des 28 derniers jours d'historique, normalisés à 1 par niveau ;
  le score total est la moyenne des 12 niveaux.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from m5bench.data import HORIZON, LEVEL_LABELS, LEVELS, M5Data, level_slices

M5_QUANTILES = (0.005, 0.025, 0.165, 0.25, 0.5, 0.75, 0.835, 0.975, 0.995)


def _naive_scales(history: np.ndarray, chunk: int = 4096) -> tuple[np.ndarray, np.ndarray]:
    sq, ab = np.empty(len(history)), np.empty(len(history))
    for s in range(0, len(history), chunk):
        x = history[s : s + chunk].astype(np.float64)
        started = np.cumsum(x > 0, axis=1) > 0
        d = np.diff(x, axis=1)
        valid = started[:, :-1]  # différence comptée si le point de départ est après le lancement
        n = valid.sum(axis=1)
        sq[s : s + chunk] = (d**2 * valid).sum(axis=1) / np.maximum(n, 1)
        ab[s : s + chunk] = (np.abs(d) * valid).sum(axis=1) / np.maximum(n, 1)
    return sq, ab


class M5Evaluator:
    def __init__(self, data: M5Data):
        if data.test is None:
            raise ValueError("sales_test_evaluation.csv est nécessaire pour évaluer")
        self.data = data
        history = data.aggregate(data.train)
        self.scale_sq, self.scale_abs = _naive_scales(history)
        self.actual = data.aggregate(data.test).astype(np.float64)  # (42840, 28)
        last = slice(data.train.shape[1] - HORIZON, data.train.shape[1])
        dollars_bottom = np.nansum(data.train[:, last] * data.prices[:, last], axis=1)
        self.dollars = data.aggregate(dollars_bottom[:, None])[:, 0]
        self.slices = level_slices(data.series)
        self.weights = np.empty(len(self.dollars))
        for lvl, idx in self.slices.items():
            self.weights[idx] = self.dollars[idx] / self.dollars[idx].sum()

    def rmsse(self, point_all: np.ndarray) -> np.ndarray:
        mse = ((point_all.astype(np.float64) - self.actual) ** 2).mean(axis=1)
        return np.sqrt(mse / self.scale_sq)

    def spl(self, quantiles_all: np.ndarray, levels=M5_QUANTILES) -> np.ndarray:
        """quantiles_all : (42840, 28, 9) aux niveaux M5_QUANTILES. Renvoie (42840,) moyenne sur les 9 quantiles."""
        u = np.asarray(levels)[None, None, :]
        y = self.actual[:, :, None]
        q = quantiles_all.astype(np.float64)
        loss = np.where(q <= y, (y - q) * u, (q - y) * (1 - u)).mean(axis=1)  # (42840, 9)
        return loss.mean(axis=1) / self.scale_abs

    def _by_level(self, per_series: np.ndarray, name: str) -> pd.DataFrame:
        rows = []
        for lvl, _ in LEVELS:
            idx = self.slices[lvl]
            rows.append({"level": lvl, "niveau": LEVEL_LABELS[lvl], "n_series": len(idx),
                         name: float((self.weights[idx] * per_series[idx]).sum())})
        df = pd.DataFrame(rows)
        total = pd.DataFrame([{"level": "Total", "niveau": "Moyenne 12 niveaux", "n_series": len(per_series),
                               name: df[name].mean()}])
        return pd.concat([df, total], ignore_index=True)

    def wrmsse(self, point: np.ndarray) -> pd.DataFrame:
        """point : (30490, 28) prévision article × magasin (agrégée en bottom-up, règle officielle)
        ou (42840, 28) prévision directe de chaque niveau."""
        point_all = self.data.aggregate(point) if len(point) == self.data.train.shape[0] else point
        return self._by_level(self.rmsse(point_all), "wrmsse")

    def wspl(self, quantiles_all: np.ndarray) -> pd.DataFrame:
        return self._by_level(self.spl(quantiles_all), "wspl")


def interpolate_quantiles(q: np.ndarray, src_levels, dst_levels=M5_QUANTILES, nonnegative: bool = True) -> np.ndarray:
    """Interpole linéairement une grille de quantiles (…, Q_src) vers dst_levels, avec extrapolation
    linéaire aux bords (0,005 et 0,995 sont hors de la grille 0,01–0,99 de Timestral)."""
    src = np.asarray(src_levels, dtype=np.float64)
    out = np.empty(q.shape[:-1] + (len(dst_levels),), dtype=np.float32)
    for j, u in enumerate(dst_levels):
        k = int(np.clip(np.searchsorted(src, u) - 1, 0, len(src) - 2))
        w = (u - src[k]) / (src[k + 1] - src[k])
        out[..., j] = q[..., k] + w * (q[..., k + 1] - q[..., k])
    if nonnegative:
        np.maximum(out, 0, out=out)
    return np.sort(out, axis=-1)
