"""Loading Timestral and forecasting a batch of M5 series (used by run_timestral.py, export_static.py and the app)."""

from __future__ import annotations

import os

import numpy as np
import torch

from m5bench.covariates import CovariateBank
from m5bench.data import HORIZON, mask_leading_zeros

MODELS = {
    "Timestral-5M": "mistralai/Timestral-5M",
    "Timestral-28M": "mistralai/Timestral-28M",
    "Timestral-123M": "mistralai/Timestral-123M",
    "Timestral-209M": "mistralai/Timestral-209M",
}

_BANKS: dict[int, CovariateBank] = {}

# Rows sent to the model per series (target + covariate rows). Timestral attends across the rows of one series
# with a mask built over all rows of the batch, so memory and time grow with (batch × rows)²: the batch size is
# capped so that batch × rows stays under ROW_BUDGET.
ROWS_PER_SERIES = {"none": 1, "calendar": 4, "full": 5, "rich": 12}
ROW_BUDGET = int(os.environ.get("TIMESTRAL_ROW_BUDGET", 512))


def load_forecaster(model: str, device: str = "cpu"):
    from timestral import TimestralForecaster

    if model == "tiny-random":  # small model with random weights: tests the code, not the forecasts
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


def covariate_bank(data, cal=None) -> CovariateBank:
    """One bank per dataset; `cal` may already be a bank (older call sites pass the calendar dict)."""
    if isinstance(cal, CovariateBank):
        return cal
    key = id(data)
    if key not in _BANKS:
        _BANKS[key] = CovariateBank(data)
    return _BANKS[key]


def build_inputs(data, idx, history_all, cal, covariates: str, max_context: int | None):
    """Context (B, 1, L) with launch zeros masked, and covariates known over history and horizon."""
    t = history_all.shape[1]
    start = 0 if max_context is None else max(0, t - max_context)
    context = mask_leading_zeros(history_all[idx, start:])[:, None, :]
    kwargs = covariate_bank(data, cal).build(np.asarray(idx), covariates, start, t + HORIZON)
    return context, kwargs


def forecast_series(fc, data, idx, history_all, cal, covariates="calendar", max_context=None,
                    symmetric=True, batch_size=256, progress=None):
    """Quantiles (len(idx), 28, Q) and quantile levels. `progress(done, total)` is called after each batch."""
    idx = np.asarray(idx)
    batch_size = max(4, min(batch_size, ROW_BUDGET // ROWS_PER_SERIES.get(covariates, 1)))
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
