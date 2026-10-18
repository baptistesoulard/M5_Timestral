"""Builds the GitHub Actions job matrix from bench/configs.json and bench/run.json."""
import json

run = json.load(open("bench/run.json"))
configs = json.load(open("bench/configs.json"))
mode = run.get("mode", "full")
probe = int(run.get("probe_series", 240)) if mode == "probe" else 0
only = set(run.get("only", []))
include = []
for c in configs:
    if only and c["name"] not in only:
        continue
    n = 1 if mode == "probe" else int(c.get("shards", 1))
    big = any(s in c.get("model", "") for s in ("123M", "209M"))
    for s in range(n):
        include.append({"config": c["name"], "kind": c.get("kind", "timestral"), "bench": c.get("bench", "m5"),
                        "method": c.get("method", ""), "model": c.get("model", "none"),
                        "covariates": c.get("covariates", "none"), "shard": s, "nshards": n, "probe": probe,
                        "batch": 256 if big else 512})
# longest jobs first, so they start while the 20-job concurrency limit is not reached
cost = {c["name"]: c.get("probe_ms", 0) / max(1, int(c.get("shards", 1))) for c in configs}
include.sort(key=lambda j: -cost.get(j["config"], 0))
print("matrix=" + json.dumps({"include": include}))
print("mode=" + mode)
print("count=" + str(len(include)))
print("target=" + ("live" if run.get("target") == "live" else "staging"))
# earlier runs whose shards are reused: "reuse_run_id" and/or the list "reuse_run_ids" (earlier in the list wins)
reuse = [str(r) for r in ([run["reuse_run_id"]] if run.get("reuse_run_id") else []) + run.get("reuse_run_ids", [])]
print("reuse_runs=" + " ".join(reuse))
