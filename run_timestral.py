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

from m5bench.data import LEVELS, calendar_covariates, level_slices, load_m5
from m5bench.metrics import interpolate_quantiles
from m5bench.store import mean_from_quantiles, save_forecast
from m5bench.timestral_runner import forecast_series, load_forecaster


def forecast_level(fc, data, idx, history_all, cal, args):
    t0 = time.time()

    def progress(done, total):
        if total > args.batch_size:
            rate = (time.time() - t0) / done
            end = "\n" if done == total else ""
            print(f"\r    {done:>6}/{total} séries · reste ~{rate * (total - done) / 60:.1f} min", end=end, flush=True)

    return forecast_series(fc, data, idx, history_all, cal, args.covariates, args.max_context,
                           symmetric=not args.no_symmetric, batch_size=args.batch_size, progress=progress)


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
