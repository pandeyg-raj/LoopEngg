"""Why is the delegatable share so small? Characterise where tokens actually go.

This is the Phase-0 measurement study on real data. If the compression
opportunity is small, we need to know that now, and we need to know why.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loopengg.costmodel import Pricing, score_cache_aware, score_design_b  # noqa: E402
from loopengg.ingest import Tokenizer, load_shard, row_to_trajectory  # noqa: E402

REPO = "nvidia/Open-SWE-Traces"
SHARDS = {
    "qwen35/openhands": "data/qwen35_openhands_trajectories/train-00022-of-00023.parquet",
    "minimax/sweagent": "data/minimax_m25_sweagent_trajectories/train-00015-of-00023.parquet",
}
STRONG = Pricing("strong", input_per_mtok=1.0, output_per_mtok=5.0)
WEAK = Pricing("weak", input_per_mtok=0.05, output_per_mtok=0.20)


def quantiles(xs, qs=(0.5, 0.75, 0.9, 0.95, 0.99, 1.0)):
    xs = sorted(xs)
    if not xs:
        return {}
    return {q: xs[min(len(xs) - 1, int(len(xs) * q))] for q in qs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()
    tok = Tokenizer()

    for label, shard in SHARDS.items():
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
        try:
            rows = load_shard(REPO, shard, limit=args.limit)
        except Exception as e:
            print(f"  could not load: {type(e).__name__}: {e}")
            continue

        trajs = []
        for row in rows:
            t, _ = row_to_trajectory(row, tok, min_delegatable_obs=1)
            if t:
                trajs.append(t)
        if not trajs:
            print("  no trajectories parsed")
            continue
        print(f"parsed {len(trajs)} trajectories")

        # --- what tools are actually being called? -------------------------
        tool_counter = Counter()
        obs_by_tool = Counter()
        for t in trajs:
            for turn in t.turns:
                tool_counter[turn.turn_type] += 1
                obs_by_tool[turn.turn_type] += turn.observation_tokens
        print("\n--- turn types (count, observation tokens, mean obs) ---")
        for name, cnt in tool_counter.most_common(12):
            mean = obs_by_tool[name] / cnt if cnt else 0
            print(f"  {name:16} n={cnt:6}  obs_tok={obs_by_tool[name]:10,}  mean={mean:8.0f}")

        # --- observation size distribution ---------------------------------
        all_obs = [x.observation_tokens for t in trajs for x in t.turns if x.observation_tokens]
        q = quantiles(all_obs)
        print(f"\n--- observation size (tokens), n={len(all_obs):,} ---")
        print("  " + "  ".join(f"p{int(k * 100)}={v:,}" for k, v in q.items()))
        print(f"  mean={sum(all_obs) / len(all_obs):.0f}")

        # --- where does the CONTEXT actually come from? --------------------
        # Final context = system+user+tools + sum(outputs) + sum(observations)
        tot_obs = sum(x.observation_tokens for t in trajs for x in t.turns)
        tot_out = sum(x.output_tokens for t in trajs for x in t.turns)
        base_ctx = sum(t.turns[0].context_tokens for t in trajs)
        total_growth = tot_obs + tot_out
        print(f"\n--- what grows the transcript ---")
        print(f"  initial context (system+user+tools)  {base_ctx:12,}")
        print(f"  observations                         {tot_obs:12,}  "
              f"({100 * tot_obs / total_growth:.1f}% of growth)")
        print(f"  assistant output (carried forward)   {tot_out:12,}  "
              f"({100 * tot_out / total_growth:.1f}% of growth)")

        # --- threshold sweep: how much is delegatable? ---------------------
        print(f"\n--- delegatable share vs threshold ---")
        print(f"  {'min_obs':>8} {'turns':>8} {'%obs tok':>9} {'R1 saving':>10} {'B3 saving':>10}")
        for thr in (200, 500, 1000, 2000, 5000):
            re_trajs = []
            for row in rows:
                t, _ = row_to_trajectory(row, tok, min_delegatable_obs=thr)
                if t:
                    re_trajs.append(t)
            n_del = sum(1 for t in re_trajs for x in t.turns if x.delegatable)
            del_tok = sum(x.observation_tokens for t in re_trajs for x in t.turns if x.delegatable)
            share = 100 * del_tok / tot_obs if tot_obs else 0
            base = sum(score_cache_aware(t, STRONG).dollars for t in re_trajs)
            r1 = sum(score_design_b(t, STRONG, WEAK, compressor="weak").dollars for t in re_trajs)
            b3 = sum(score_design_b(t, STRONG, WEAK, compressor="script").dollars for t in re_trajs)
            print(f"  {thr:>8} {n_del:>8} {share:>8.1f}% "
                  f"{100 * (base - r1) / base:>9.1f}% {100 * (base - b3) / base:>9.1f}%")

        # --- upper bound: what if we could compress EVERYTHING? ------------
        for t in trajs:
            for turn in t.turns:
                turn.delegatable = turn.observation_tokens > 0
        base = sum(score_cache_aware(t, STRONG).dollars for t in trajs)
        ideal = sum(score_design_b(t, STRONG, WEAK, sigma=0.05, compressor="script").dollars
                    for t in trajs)
        print(f"\n--- CEILING: compress every observation to 5% (free, oracle) ---")
        print(f"  baseline ${base:.2f} -> ${ideal:.2f}  =  {100 * (base - ideal) / base:.1f}% saving")
        print("  This is the absolute maximum any compression/routing scheme can achieve here.")


if __name__ == "__main__":
    main()
