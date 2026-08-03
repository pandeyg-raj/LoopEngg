"""S1-real: cache-aware re-scoring of published agent trajectories.

This is the paper's backbone. It costs nothing to run: no inference, just
tokenizing public conversations and simulating a deterministic prefix cache.

Run:  python analysis/s1_real.py [--limit N]
"""

from __future__ import annotations

import argparse
import statistics as stats
import sys
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

# One shard per (model, scaffold) cell. Gives a 2x2 generality axis for free.
SHARDS = {
    "qwen35 / openhands": "data/qwen35_openhands_trajectories/train-00022-of-00023.parquet",
}

# Normalised: strong input = $1.00/Mtok. Only ratios matter.
STRONG = Pricing("strong", input_per_mtok=1.0, output_per_mtok=5.0)
WEAK = Pricing("weak", input_per_mtok=0.05, output_per_mtok=0.20)


def pct(xs):
    xs = sorted(xs)
    if not xs:
        return (0, 0, 0)
    return (
        xs[len(xs) // 2],
        xs[int(len(xs) * 0.25)],
        xs[int(len(xs) * 0.75)],
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=200, help="trajectories per shard")
    ap.add_argument("--min-obs", type=int, default=1000, help="delegatable obs threshold")
    ap.add_argument("--sigma", type=float, default=0.05, help="compression ratio")
    args = ap.parse_args()

    tok = Tokenizer()
    print(f"tokenizer: {tok.name}")

    for label, shard in SHARDS.items():
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        rows = load_shard(REPO, shard, limit=args.limit)
        print(f"loaded {len(rows)} rows")

        trajs, resolved_flags, skipped = [], [], 0
        for row in rows:
            traj, st = row_to_trajectory(row, tok, min_delegatable_obs=args.min_obs)
            if traj is None:
                skipped += 1
                continue
            trajs.append(traj)
            resolved_flags.append(row.get("resolved", 0) == 1)
        print(f"parsed {len(trajs)} trajectories ({skipped} skipped)")
        if not trajs:
            continue

        # ---- trajectory shape -------------------------------------------------
        n_turns = [t.n_turns for t in trajs]
        ctx_final = [t.turns[-1].context_tokens for t in trajs]
        med, q1, q3 = pct(n_turns)
        print(f"\nturns   median={med}  IQR=[{q1}, {q3}]  max={max(n_turns)}")
        med, q1, q3 = pct(ctx_final)
        print(f"final context tokens  median={med:,}  IQR=[{q1:,}, {q3:,}]")

        n_deleg = [sum(1 for x in t.turns if x.delegatable) for t in trajs]
        deleg_tok = [sum(x.observation_tokens for x in t.turns if x.delegatable) for t in trajs]
        all_tok = [sum(x.observation_tokens for x in t.turns) for t in trajs]
        share = [d / a if a else 0 for d, a in zip(deleg_tok, all_tok)]
        print(f"delegatable turns  median={pct(n_deleg)[0]}  "
              f"share of observation tokens median={pct(share)[0]:.1%}")

        # ---- the headline: flat vs cache-aware --------------------------------
        flat = [score_flat(t, STRONG).dollars for t in trajs]
        cached = [score_cache_aware(t, STRONG).dollars for t in trajs]
        infl = [f / c for f, c in zip(flat, cached) if c > 0]

        c0 = score_cache_aware(trajs[0], STRONG)
        hit_rate = c0.cached_read_tokens / (c0.cached_read_tokens + c0.uncached_input_tokens)

        print(f"\n--- flat vs cache-aware baseline ---")
        print(f"  total flat      ${sum(flat):.2f}")
        print(f"  total cached    ${sum(cached):.2f}")
        print(f"  INFLATION       {sum(flat) / sum(cached):.2f}x  (aggregate)")
        m, q1, q3 = pct(infl)
        print(f"  per-trajectory  median={m:.2f}x  IQR=[{q1:.2f}, {q3:.2f}]  max={max(infl):.2f}x")

        # cache hit rate across the corpus
        hits, misses = 0, 0
        for t in trajs:
            c = score_cache_aware(t, STRONG)
            hits += c.cached_read_tokens
            misses += c.uncached_input_tokens
        print(f"  cache hit rate  {hits / (hits + misses):.1%} of input tokens")

        # ---- design comparison ------------------------------------------------
        base = sum(cached)
        a = sum(score_design_a(t, STRONG, WEAK).dollars for t in trajs)
        r1 = sum(score_design_b(t, STRONG, WEAK, sigma=args.sigma, compressor="weak").dollars
                 for t in trajs)
        b2 = sum(score_design_b(t, STRONG, WEAK, sigma=args.sigma, compressor="strong").dollars
                 for t in trajs)
        b3 = sum(score_design_b(t, STRONG, WEAK, sigma=args.sigma, compressor="script").dollars
                 for t in trajs)

        print(f"\n--- designs (cache-aware, totals) ---")
        print(f"  B0 baseline            ${base:8.3f}   ---")
        print(f"  A* in-band swap        ${a:8.3f}   {100 * (a - base) / base:+6.1f}%")
        print(f"  B2 strong compression  ${b2:8.3f}   {100 * (b2 - base) / base:+6.1f}%")
        print(f"  B3 scripted truncation ${b3:8.3f}   {100 * (b3 - base) / base:+6.1f}%")
        print(f"  R1 edge delegation     ${r1:8.3f}   {100 * (r1 - base) / base:+6.1f}%")

        if base - r1 > 0:
            comp_share = (base - b2) / (base - r1)
            print(f"\n  compression accounts for {comp_share:.1%} of R1's saving")
        print(f"  R1 vs B3: {'R1 cheaper' if r1 < b3 else 'B3 CHEAPER -- routing must win on quality'}"
              f" (diff ${abs(r1 - b3):.3f}, {100 * abs(r1 - b3) / base:.1f}% of baseline)")

        # ---- what prior work would have reported ------------------------------
        flat_r1 = sum(score_flat(t, STRONG).dollars for t in trajs)
        r1_flat_world = sum(
            score_design_b(t, STRONG, WEAK, sigma=args.sigma,
                           cache_efficiency=0.0, compressor="weak").dollars
            for t in trajs
        )
        base_flat_world = sum(
            score_cache_aware(t, STRONG, cache_efficiency=0.0).dollars for t in trajs
        )
        saving_flat = 100 * (base_flat_world - r1_flat_world) / base_flat_world
        saving_cached = 100 * (base - r1) / base
        print(f"\n--- reported saving, two accountings ---")
        print(f"  no cache (prior work): {saving_flat:.1f}%")
        print(f"  with cache (ours):     {saving_cached:.1f}%")
        print(f"  overstatement:         {saving_flat - saving_cached:.1f} points")

        res = sum(resolved_flags)
        print(f"\nresolved: {res}/{len(resolved_flags)} ({res / len(resolved_flags):.1%})")


if __name__ == "__main__":
    main()
