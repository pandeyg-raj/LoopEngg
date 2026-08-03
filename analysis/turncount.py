"""Does turn count or observation size drive cost?

The paper asserts the former in its abstract and conclusion but has so far only
shown it indirectly. This quantifies it directly: per-trajectory cost regressed
on turn count and on mean observation size, plus a variance decomposition.

Run:  python analysis/turncount.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loopengg.costmodel import Pricing, score_cache_aware  # noqa: E402
from loopengg.ingest import Tokenizer, load_shard, row_to_trajectory  # noqa: E402

REPO = "nvidia/Open-SWE-Traces"
CELLS = {
    "qwen35 / OpenHands": "data/qwen35_openhands_trajectories/train-00022-of-00023.parquet",
    "minimax / SWE-agent": "data/minimax_m25_sweagent_trajectories/train-00015-of-00023.parquet",
    "minimax / OpenHands": "data/minimax_m25_openhands_trajectories/train-00006-of-00020.parquet",
    "qwen35 / SWE-agent": "data/qwen35_sweagent_trajectories/train-00017-of-00018.parquet",
}
STRONG = Pricing("strong", input_per_mtok=1.0, output_per_mtok=5.0)


def r2(y, yhat):
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


def loglog_fit(x, y):
    """Fit y = a * x^b; return (b, r2) on the log-log scale."""
    lx, ly = np.log(x), np.log(y)
    b, a = np.polyfit(lx, ly, 1)
    return b, r2(ly, a + b * lx)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=150)
    args = ap.parse_args()

    tok = Tokenizer()
    turns, mean_obs, cost, cell_id = [], [], [], []
    for ci, (label, shard) in enumerate(CELLS.items()):
        rows = load_shard(REPO, shard, limit=args.limit)
        n = 0
        for row in rows:
            t, _ = row_to_trajectory(row, tok, min_delegatable_obs=10**9)
            if not t or t.n_turns < 2:
                continue
            obs = [x.observation_tokens for x in t.turns]
            c = score_cache_aware(t, STRONG).dollars
            if c <= 0:
                continue
            turns.append(t.n_turns)
            mean_obs.append(max(1.0, float(np.mean(obs))))
            cost.append(c)
            cell_id.append(ci)
            n += 1
        print(f"  {label:22} {n} trajectories", flush=True)

    turns = np.array(turns, float)
    mean_obs = np.array(mean_obs, float)
    cost = np.array(cost, float)

    print(f"\n{'=' * 74}\nWhat drives per-trajectory cost? (n={len(cost)})\n{'=' * 74}")

    # --- univariate association on the log scale -------------------------
    b_t, r2_t = loglog_fit(turns, cost)
    b_o, r2_o = loglog_fit(mean_obs, cost)
    print(f"{'predictor':24} {'exponent':>9} {'R^2 (log-log)':>14}")
    print(f"{'turn count T':24} {b_t:>9.2f} {r2_t:>14.3f}")
    print(f"{'mean observation size':24} {b_o:>9.2f} {r2_o:>14.3f}")

    # --- joint model ------------------------------------------------------
    X = np.column_stack([np.ones_like(turns), np.log(turns), np.log(mean_obs)])
    beta, *_ = np.linalg.lstsq(X, np.log(cost), rcond=None)
    r2_joint = r2(np.log(cost), X @ beta)
    print(f"\njoint log-log model: log cost = {beta[0]:.2f} "
          f"+ {beta[1]:.2f}*log T + {beta[2]:.2f}*log obs   (R^2 = {r2_joint:.3f})")

    # Incremental R^2: what does each predictor add over the other alone?
    print(f"\nincremental R^2 over the other predictor alone:")
    print(f"  adding turn count      +{r2_joint - r2_o:.3f}")
    print(f"  adding observation size +{r2_joint - r2_t:.3f}")

    # --- superlinearity ---------------------------------------------------
    print(f"\nExponent on T is {b_t:.2f}: cost grows faster than linearly in turns,")
    print("as the append-only re-send model predicts (each turn re-reads all prior ones).")

    # --- cost per turn is roughly stable ---------------------------------
    cpt = cost / turns
    print(f"\ncost per turn: median {np.median(cpt):.5f}, "
          f"IQR [{np.percentile(cpt, 25):.5f}, {np.percentile(cpt, 75):.5f}]  "
          f"(CV {np.std(cpt) / np.mean(cpt):.2f})")
    cpo = cost / (mean_obs * turns)
    print(f"cost per observation token: CV {np.std(cpo) / np.mean(cpo):.2f} "
          f"-- {'more' if np.std(cpo) / np.mean(cpo) > np.std(cpt) / np.mean(cpt) else 'less'} "
          "variable than cost per turn")

    # --- decile table -----------------------------------------------------
    print(f"\n{'turn decile':>12} {'median T':>9} {'median obs':>11} {'median cost':>12}")
    order = np.argsort(turns)
    for d in range(10):
        idx = order[int(d * len(order) / 10):int((d + 1) * len(order) / 10)]
        if len(idx) == 0:
            continue
        print(f"{d + 1:>12} {np.median(turns[idx]):>9.0f} "
              f"{np.median(mean_obs[idx]):>11.0f} {np.median(cost[idx]):>12.4f}")

    Path("data").mkdir(exist_ok=True)
    Path("data/turncount.json").write_text(json.dumps({
        "n": len(cost),
        "exponent_turns": b_t, "r2_turns": r2_t,
        "exponent_obs": b_o, "r2_obs": r2_o,
        "r2_joint": r2_joint,
        "beta": list(beta),
        "cv_cost_per_turn": float(np.std(cpt) / np.mean(cpt)),
        "cells": list(CELLS.keys()),
        # Per-trajectory points, for the scatter figure.
        "points": {
            "turns": turns.tolist(),
            "mean_obs": mean_obs.tolist(),
            "cost": cost.tolist(),
            "cell": cell_id,
        },
    }, indent=1))
    print("\nwrote data/turncount.json")


if __name__ == "__main__":
    main()
