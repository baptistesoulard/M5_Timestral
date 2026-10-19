"""t0 (The Forecasting Company) as a second foundation model, run with the same protocol as Timestral.

Open weights on Hugging Face (theforecastingcompany/t0-gamma), code `tfc-t0` under Apache 2.0.
Univariate forecasts with the settings that produce the published scores (trim leading missing values,
symmetric averaging, clip at zero, snap to integers), and the same 21 quantile levels as Timestral.
"""

from __future__ import annotations

import numpy as np

LEVELS = (0.01,) + tuple(round(float(x), 2) for x in np.arange(0.05, 0.951, 0.05)) + (0.99,)


def load_t0(name: str = "t0-gamma"):
    import torch
    from t0 import T0Forecaster

    if name == "t0-tiny-random":  # local smoke test without downloading weights
        torch.manual_seed(0)
        return T0Forecaster(embed_dim=64, num_layers=2, num_heads=2, mlp_hidden_dim=128, patch_size=32,
                            group_every_n=2, dropout=0.0, quantile_levels=LEVELS).eval()
    return T0Forecaster.from_pretrained(f"theforecastingcompany/{name}").eval()


def t0_forecast(model, context: np.ndarray, horizon: int, batch_size: int = 128, progress=None):
    """context (N, T) with NaN before each series starts → quantiles (N, horizon, 21), levels."""
    import torch

    ctx = np.asarray(context, dtype=np.float32).copy()
    empty = np.isnan(ctx).all(axis=1)
    ctx[empty, -1] = 0.0  # a series without history gets a flat zero forecast
    out = np.empty((len(ctx), horizon, len(LEVELS)), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(ctx), batch_size):
            res = model.predict(ctx[s : s + batch_size], horizon=horizon, quantile_levels=LEVELS,
                                trim_leading_missing=True, symmetric_averaging=True,
                                clip_nonnegative=True, snap_integers=True)
            q = res.quantiles
            out[s : s + len(q)] = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
            if progress is not None:
                progress(min(s + batch_size, len(ctx)), len(ctx))
    out[empty] = 0.0
    return out, LEVELS
