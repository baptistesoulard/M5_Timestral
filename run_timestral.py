"""Prévision M5 avec Timestral (Mistral), en zero-shot, sur CPU.

Prévoit directement chacune des 42 840 séries (12 niveaux), ou seulement les 30 490 séries
article × magasin avec --levels 12. Enregistre la médiane et l'espérance (moyenne de la
grille de quantiles), ainsi que les 9 quantiles M5 pour la piste incertitude.

Exemples :
    python run_timestral.py --dry-run 500                       # estime la durée d'un run complet
    python run_timestral.py --model mistralai/Timestral-28M     # run complet, covariables calendrier
    python run_timestral.py --model mistralai/Timestral-209M --covariates full --name timestral209_full
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from m5bench.data import HORIZON, LEVELS, calendar_covariates, level_slices, load_m5, mask_leading_zeros, \
    series_relative_price, series_snap
from m5bench.metrics import interpolate_quantiles
from m5bench.store import mean_from_quantiles, save_forecast


def load_forecaster(model: str, device: str):
    from timestral import TimestralForecaster

    if model == "tiny-random":  # mini-modèle aléatoire : sert uniquement à tester le code, pas à prévoir
        from timestral import TimestralConfig, TimestralModel

        knots = (0.01, 0.05) + tuple(round(0.1 * i, 1) for i in range(1, 10)) + (0.95, 0.99)
        knots = tuple(sorted(set(knots) | {0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85}))
        cfg = TimestralConfig(patch_size=32, d_model=32, num_heads=4, mlp_hidden_dim=64,
                              block_schedule=("time", "time_variate"), quantile_knots=knots,
                              block_size=4, context_length=4096)
        torch.manual_seed(0)
        m = TimestralModel(cfg)
        for p in m.parameters():
            torch.nn.init.normal_(p, std=0.02)
        return TimestralForecaster(m.eval())
    return TimestralForecaster.from_pretrained(model, device=device)


def build_inputs(data, idx, history_all, cal, covariates: str, max_context: int | None):
    t = history_all.shape[1]
    start = 0 if max_context is None else max(0, t - max_context)
    context = mask_leading_zeros(history_all[idx, start:])[:, None, :]  # (B, 1, L)
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


def forecast_level(fc, data, idx, history_all, cal, args) -> np.ndarray:
    out = None
    t0 = time.time()
    for s in range(0, len(idx), args.batch_size):
        b = idx[s : s + args.batch_size]
        context, kwargs = build_inputs(data, b, history_all, cal, args.covariates, args.max_context)
        res = fc.forecast(context, prediction_length=HORIZON, symmetric_averaging=not args.no_symmetric, **kwargs)
        q = res.quantiles[:, 0].cpu().numpy()  # (B, H, Q)
        if out is None:
            out = np.empty((len(idx),) + q.shape[1:], dtype=np.float32)
            levels = res.quantile_levels
        out[s : s + len(b)] = q
        done = s + len(b)
        if len(idx) > args.batch_size:
            rate = (time.time() - t0) / done
            print(f"\r    {done:>6}/{len(idx)} séries · reste ~{rate * (len(idx) - done) / 60:.1f} min", end="", flush=True)
    if len(idx) > args.batch_size:
        print()
    return out, levels


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="forecasts")
    ap.add_argument("--model", default="mistralai/Timestral-28M",
                    help="dépôt Hugging Face (Timestral-5M/28M/123M/209M) ou dossier local")
    ap.add_argument("--covariates", choices=["none", "calendar", "full"], default="calendar",
                    help="none : ventes seules · calendar : SNAP, événements, jour de semaine · full : + prix relatif")
    ap.add_argument("--max-context", type=int, default=None, help="nb de jours d'historique (défaut : tout, 1941)")
    ap.add_argument("--levels", default="all", help="'all' (42 840 séries) ou '12' (article × magasin seulement)")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--no-symmetric", action="store_true", help="désactive le double passage (2× plus rapide)")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--name", default=None, help="nom du fichier de sortie (défaut : déduit des options)")
    ap.add_argument("--dry-run", type=int, default=0, metavar="N", help="chronomètre N séries et estime la durée totale")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)

    data = load_m5(args.data)
    history_all = data.aggregate(data.train)
    cal = calendar_covariates(data.calendar)
    slices = level_slices(data.series)
    fc = load_forecaster(args.model, args.device)
    levels = [lvl for lvl, _ in LEVELS] if args.levels == "all" else ["Level12"]
    n_total = sum(len(slices[l]) for l in levels)

    if args.dry_run:
        idx = slices["Level12"][: args.dry_run]
        forecast_level(fc, data, idx[: min(32, len(idx))], history_all, cal, args)  # préchauffage
        t0 = time.time()
        forecast_level(fc, data, idx, history_all, cal, args)
        per = (time.time() - t0) / len(idx)
        print(f"{per * 1000:.1f} ms/série → ~{per * n_total / 60:.0f} min pour {n_total} séries ({args.levels})")
        return

    t0 = time.time()
    q_all = None
    for lvl in levels:
        idx = slices[lvl]
        print(f"  {lvl} ({len(idx)} séries)")
        q, qlevels = forecast_level(fc, data, idx, history_all, cal, args)
        if q_all is None:
            q_all = np.full((len(data.series),) + q.shape[1:], np.nan, dtype=np.float32)
        q_all[idx] = q
    elapsed = time.time() - t0

    median_all = q_all[..., list(qlevels).index(0.5)]
    mean_all = mean_from_quantiles(q_all, qlevels)
    bottom = slices["Level12"]
    arrays = {"point_bottom_median": median_all[bottom], "point_bottom_mean": mean_all[bottom]}
    if args.levels == "all":
        arrays.update(point_all_median=median_all, point_all_mean=mean_all,
                      quantiles_all=interpolate_quantiles(q_all, qlevels))
    model_tag = args.model.rstrip("/").split("/")[-1].lower().replace("timestral-", "timestral")
    name = args.name or f"{model_tag}_{args.covariates}" + (f"_ctx{args.max_context}" if args.max_context else "")
    meta = {"method": name, "family": "timestral", "model": args.model, "covariates": args.covariates,
            "max_context": args.max_context, "symmetric": not args.no_symmetric, "levels": args.levels,
            "seconds": round(elapsed), "threads": torch.get_num_threads()}
    path = save_forecast(f"{args.out}/{name}.npz", meta, **arrays)
    print(f"{path} enregistré · {elapsed / 60:.1f} min")


if __name__ == "__main__":
    main()
