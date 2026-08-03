"""Full 2x2 (model x scaffold) analysis. Emits JSON for the figure scripts.

Run:  python analysis/run_all.py --limit 150
Output: data/results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loopengg.costmodel import (  # noqa: E402
    Pricing,
    score_cache_aware,
    score_design_a,
    score_design_b,
    score_flat,
)
from loopengg.ingest import Tokenizer, load_shard, row_to_trajectory  # noqa: E402

REPO = "nvidia/Open-SWE-Traces"

CELLS = {
    "qwen35 / OpenHands": "data/qwen35_openhands_trajectories/train-00022-of-00023.parquet",
    "minimax / SWE-agent": "data/minimax_m25_sweagent_trajectories/train-00015-of-00023.parquet",
    "minimax / OpenHands": "data/minimax_m25_openhands_trajectories/train-00006-of-00020.parquet",
    "qwen35 / SWE-agent": "data/qwen35_sweagent_trajectories/train-00017-of-00018.parquet",
}

STRONG = Pricing("strong", input_per_mtok=1.0, output_per_mtok=5.0)
WEAK = Pricing("weak", input_per_mtok=0.05, output_per_mtok=0.20)
THRESHOLDS = [200, 500, 1000, 2000, 5000]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else 0


def analyse_cell(label: str, shard: str, limit: int, tok: Tokenizer) -> dict | None:
    t0 = time.time()
    print(f"[{label}] downloading/loading {shard} ...", flush=True)
    try:
        rows = load_shard(REPO, shard, limit=limit)
    except Exception as e:
        print(f"[{label}] FAILED: {type(e).__name__}: {e}", flush=True)
        return None
    print(f"[{label}] {len(rows)} rows in {time.time() - t0:.0f}s", flush=True)

    # Parse once with a permissive threshold; we re-flag per threshold below.
    parsed = []
    for row in rows:
        traj, _ = row_to_trajectory(row, tok, min_delegatable_obs=1)
        if traj:
            parsed.append((traj, row.get("resolved", 0) == 1))
    if not parsed:
        return None
    trajs = [p[0] for p in parsed]
    resolved = [p[1] for p in parsed]

    out: dict = {"label": label, "n_trajectories": len(trajs)}

    # --- shape ---
    n_turns = [t.n_turns for t in trajs]
    final_ctx = [t.turns[-1].context_tokens for t in trajs]
    all_obs = [x.observation_tokens for t in trajs for x in t.turns if x.observation_tokens]
    out["turns"] = {"median": q(n_turns, 0.5), "p25": q(n_turns, 0.25),
                    "p75": q(n_turns, 0.75), "max": max(n_turns)}
    out["final_context"] = {"median": q(final_ctx, 0.5), "p25": q(final_ctx, 0.25),
                            "p75": q(final_ctx, 0.75)}
    out["observation_tokens"] = {
        "n": len(all_obs),
        "mean": sum(all_obs) / len(all_obs) if all_obs else 0,
        **{f"p{int(p * 100)}": q(all_obs, p) for p in (0.5, 0.75, 0.9, 0.95, 0.99, 1.0)},
    }
    out["observation_sample"] = sorted(all_obs)[:: max(1, len(all_obs) // 2000)]

    # --- turn types ---
    tc = Counter()
    to = Counter()
    for t in trajs:
        for turn in t.turns:
            tc[turn.turn_type] += 1
            to[turn.turn_type] += turn.observation_tokens
    out["turn_types"] = {k: {"n": v, "obs_tokens": to[k]} for k, v in tc.most_common()}

    # --- growth decomposition ---
    tot_obs = sum(x.observation_tokens for t in trajs for x in t.turns)
    tot_out = sum(x.output_tokens for t in trajs for x in t.turns)
    out["growth"] = {
        "observations": tot_obs,
        "assistant_output": tot_out,
        "obs_share": tot_obs / (tot_obs + tot_out) if (tot_obs + tot_out) else 0,
    }

    # --- headline: flat vs cache-aware ---
    flat = [score_flat(t, STRONG).dollars for t in trajs]
    cached = [score_cache_aware(t, STRONG).dollars for t in trajs]
    infl = [f / c for f, c in zip(flat, cached) if c > 0]
    hits = sum(score_cache_aware(t, STRONG).cached_read_tokens for t in trajs)
    miss = sum(score_cache_aware(t, STRONG).uncached_input_tokens for t in trajs)
    out["inflation"] = {
        "aggregate": sum(flat) / sum(cached),
        "median": q(infl, 0.5), "p25": q(infl, 0.25), "p75": q(infl, 0.75),
        "max": max(infl), "samples": infl,
        "cache_hit_rate": hits / (hits + miss),
        "total_flat": sum(flat), "total_cached": sum(cached),
    }

    # --- saving vs threshold ---
    sweep = []
    for thr in THRESHOLDS:
        re_trajs = []
        for row in rows:
            t, _ = row_to_trajectory(row, tok, min_delegatable_obs=thr)
            if t:
                re_trajs.append(t)
        base = sum(score_cache_aware(t, STRONG).dollars for t in re_trajs)
        rec = {"threshold": thr, "baseline": base}
        for name, kw in (("R1", {"compressor": "weak"}),
                         ("B2", {"compressor": "strong"}),
                         ("B3", {"compressor": "script"})):
            v = sum(score_design_b(t, STRONG, WEAK, sigma=0.05, **kw).dollars for t in re_trajs)
            rec[name] = v
            rec[f"{name}_saving_pct"] = 100 * (base - v) / base if base else 0
        a = sum(score_design_a(t, STRONG, WEAK).dollars for t in re_trajs)
        rec["A"] = a
        rec["A_saving_pct"] = 100 * (base - a) / base if base else 0
        n_del = sum(1 for t in re_trajs for x in t.turns if x.delegatable)
        del_tok = sum(x.observation_tokens for t in re_trajs for x in t.turns if x.delegatable)
        rec["n_delegatable"] = n_del
        rec["delegatable_obs_share"] = del_tok / tot_obs if tot_obs else 0
        sweep.append(rec)
    out["threshold_sweep"] = sweep

    # --- ceiling: compress every observation, free ---
    for t in trajs:
        for turn in t.turns:
            turn.delegatable = turn.observation_tokens > 0
    base = sum(score_cache_aware(t, STRONG).dollars for t in trajs)
    ideal = sum(score_design_b(t, STRONG, WEAK, sigma=0.05, compressor="script").dollars
                for t in trajs)
    out["ceiling"] = {"baseline": base, "ideal": ideal,
                      "saving_pct": 100 * (base - ideal) / base if base else 0}

    out["resolved_rate"] = sum(resolved) / len(resolved)
    print(f"[{label}] done: inflation={out['inflation']['aggregate']:.2f}x  "
          f"p50_obs={out['observation_tokens']['p50']}  "
          f"ceiling={out['ceiling']['saving_pct']:.1f}%", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--out", default="data/results.json")
    args = ap.parse_args()

    tok = Tokenizer()
    results = {"tokenizer": tok.name, "repo": REPO, "cells": []}
    for label, shard in CELLS.items():
        r = analyse_cell(label, shard, args.limit, tok)
        if r:
            results["cells"].append(r)
            # Write incrementally so a mid-run failure still leaves usable data.
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(json.dumps(results, indent=1))

    print(f"\nwrote {args.out} with {len(results['cells'])} cells")


if __name__ == "__main__":
    main()
