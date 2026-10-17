"""Chargement de Timestral et prévision d'un lot de séries M5 (utilisé par run_timestral.py et l'appli)."""

from __future__ import annotations

import numpy as np
import torch

from m5bench.data import HORIZON, mask_leading_zeros, series_relative_price, series_snap

MODELS = {
    "Timestral-5M": "mistralai/Timestral-5M",
    "Timestral-28M": "mistralai/Timestral-28M",
    "Timestral-123M": "mistralai/Timestral-123M",
    "Timestral-209M": "mistralai/Timestral-209M",
}


def load_forecaster(model: str, device: str = "cpu"):
    from timestral import TimestralForecaster

    if model == "tiny-random":  # mini-modèle aléatoire : sert uniquement à tester le code, pas à prévoir
        from timestral import TimestralConfig, TimestralModel

        knots = (0.01, 0.05) + tuple(round(0.05 * i, 2) for i in range(2, 19)) + (0.95, 0.99)
        cfg = TimestralConfig(patch_size=32, d_model=32, num_heads=4, mlp_hidden_dim=64,
                              block_schedule=("time", "time_variate"), quantile_knots=knots,
                              block_size=4, context_length=4096)
        torch.manual_seed(0)
        m = TimestralModel(cfg)
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.02)
        return TimestralForecaster(m.eval())
    return TimestralForecaster.from_pretrained(MODELS.get(model, model), device=device)


def build_inputs(data, idx, history_all, cal, covariates: str, max_context: int | None):
    """Contexte (B, 1, L) avec zéros de lancement masqués, et covariables connues sur l'horizon."""
    t = history_all.shape[1]
    start = 0 if max_context is None else max(0, t - max_context)
    context = mask_leading_zeros(history_all[idx, start:])[:, None, :]
    kwargs = {}
    if covariates in ("calendar", "full"):
        span = slice(start, t + HORIZON)
        known = [series_snap(data, idx, cal["snap"])[:, span]]
        if covariates == "full":
            price = series_relative_price(data, idx)
            if price is not None:
                known.append(price[:, span])
        kwargs["known_covariates"] = np.stack(known, axis=1).astype(np.float32)
        cats = np.stack([cal["event"][span], cal["wday"][span]])  # (2, L + H)
        kwargs["categorical_known_covariates"] = np.broadcast_to(cats, (len(idx),) + cats.shape).copy()
    return context, kwargs


def forecast_series(fc, data, idx, history_all, cal, covariates="calendar", max_context=None,
                    symmetric=True, batch_size=256, progress=None):
    """Quantiles (len(idx), 28, Q) et niveaux de quantiles. `progress(done, total)` est appelé à chaque lot."""
    idx = np.asarray(idx)
    out, levels = None, None
    for s in range(0, len(idx), batch_size):
        b = idx[s : s + batch_size]
        context, kwargs = build_inputs(data, b, history_all, cal, covariates, max_context)
        res = fc.forecast(context, prediction_length=HORIZON, symmetric_averaging=symmetric, **kwargs)
        q = res.quantiles[:, 0].cpu().numpy()
        if out is None:
            out = np.empty((len(idx),) + q.shape[1:], dtype=np.float32)
            levels = res.quantile_levels
        out[s : s + len(b)] = q
        if progress is not None:
            progress(s + len(b), len(idx))
    return out, levels
