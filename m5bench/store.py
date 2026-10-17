"""Stockage des prévisions : un fichier forecasts/<méthode>.npz par méthode.

Clés possibles :
- point_bottom[_<variante>] : (30490, 28) prévision article × magasin, agrégée en bottom-up (règle officielle)
- point_all[_<variante>]    : (42840, 28) prévision directe de chaque série de chaque niveau
- quantiles_all             : (42840, 28, 9) quantiles aux niveaux M5 (piste incertitude)
- meta                      : description JSON (paramètres, durée)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def save_forecast(path: str | Path, meta: dict, **arrays: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, meta=np.array(json.dumps(meta, ensure_ascii=False)),
                        **{k: v.astype(np.float32) for k, v in arrays.items()})
    return path


def load_forecast(path: str | Path) -> tuple[dict, dict[str, np.ndarray]]:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"])) if "meta" in z else {}
        arrays = {k: z[k] for k in z.files if k != "meta"}
    return meta, arrays


def mean_from_quantiles(q: np.ndarray, levels) -> np.ndarray:
    """Espérance approchée à partir d'une grille de quantiles (…, Q) : intégrale de la fonction quantile
    (trapèzes entre les niveaux, queues prolongées à plat jusqu'à 0 et 1)."""
    u = np.asarray(levels, dtype=np.float64)
    q = q.astype(np.float64)
    inner = ((q[..., 1:] + q[..., :-1]) / 2 * np.diff(u)).sum(axis=-1)
    return (inner + q[..., 0] * u[0] + q[..., -1] * (1 - u[-1])).astype(np.float32)
