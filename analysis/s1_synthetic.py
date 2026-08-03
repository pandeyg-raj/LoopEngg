"""S1 (synthetic): how much does cache-aware pricing change the picture?

Produces the numbers behind the paper's headline claims, parameterised by price
RATIOS rather than absolute vendor prices, so the conclusions survive price
changes. Strong input is normalised to $1.00/Mtok throughout.

Run:  python analysis/s1_synthetic.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loopengg.costmodel import (  # noqa: E402
    Pricing,
    Trajectory,
    breakeven_remaining_turns,
    score_cache_aware,
    score_design_a,
    score_design_b,
    score_flat,
)

STRONG_IN = 1.0
STRONG_OUT = 5.0


def strong(cache_ratio: float = 0.10) -> Pricing:
    return Pricing(
        "strong",
        input_per_mtok=STRONG_IN,
        output_per_mtok=STRONG_OUT,
        cache_read_per_mtok=cache_ratio * STRONG_IN,
    )


def weak(input_ratio: float) -> Pricing:
    return Pricing(
        "weak",
        input_per_mtok=input_ratio * STRONG_IN,
        output_per_mtok=4 * input_ratio * STRONG_IN,
    )


def build(n_turns: int, big_obs: int = 6000, every: int = 2) -> Trajectory:
    """A loop where every `every`-th turn returns a large log/test output."""
    obs = [big_obs if i % every == 0 else 250 for i in range(n_turns)]
    dele = [i % every == 0 for i in range(n_turns)]
    return Trajectory.from_observations(
        system_tokens=2500, observations=obs, output_tokens=450, delegatable=dele
    )


def rule(title: str) -> None:
    print(f"\n{'=' * 74}\n{title}\n{'=' * 74}")


def table1_inflation() -> None:
    """How much does flat pricing overstate the baseline?"""
    rule("T1  Flat vs cache-aware baseline cost (the inflated reference)")
    print(f"{'turns':>6} {'flat $':>10} {'cached $':>10} {'inflation':>10} {'%cached tok':>12}")
    S = strong()
    for n in (5, 10, 20, 30, 50, 80):
        t = build(n)
        f = score_flat(t, S)
        c = score_cache_aware(t, S)
        pct = 100 * c.cached_read_tokens / (c.cached_read_tokens + c.uncached_input_tokens)
        print(f"{n:>6} {f.dollars:>10.4f} {c.dollars:>10.4f} {f.dollars / c.dollars:>9.2f}x {pct:>11.1f}%")
    print("\n  A saving reported against the 'flat' column is measured against a")
    print("  baseline nobody actually pays. This is the paper's core observation.")


def table2_design_a_phase() -> None:
    """Where does in-band swapping (Design A) win or lose?"""
    rule("T2  Design A (in-band swap): profitable only if p_in_W < p_cache_S")
    cache_ratios = [0.05, 0.10, 0.25, 0.50]
    weak_ratios = [0.02, 0.05, 0.10, 0.20, 0.30]
    header = "  p_in_W/p_in_S |" + "".join(f"  cache={cr:<6.2f}" for cr in cache_ratios)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for wr in weak_ratios:
        cells = []
        for cr in cache_ratios:
            S, W = strong(cr), weak(wr)
            t = build(30)
            base = score_cache_aware(t, S).dollars
            a = score_design_a(t, S, W).dollars
            delta = 100 * (a - base) / base
            cells.append(f"  {delta:>+8.1f}%")
        print(f"  {wr:>13.2f} |" + "".join(cells))
    print("\n  Positive = routing COSTS MORE than not routing. Under a typical")
    print("  cache ratio of 0.10, in-band swapping loses money for any weak model")
    print("  priced above 10% of the strong model's list price.")


def table3_compression_share() -> None:
    """H4: how much of the win is compression rather than cheap tokens?"""
    rule("T3  Decomposition: is the win compression, or is it routing?")
    print(f"{'p_in_W/p_in_S':>14} {'baseline $':>11} {'B2 strong $':>12} {'R1 weak $':>11} "
          f"{'compression':>12} {'routing':>9}")
    S = strong()
    t = build(30)
    base = score_cache_aware(t, S).dollars
    for wr in (0.02, 0.05, 0.10, 0.20, 0.40):
        W = weak(wr)
        b2 = score_design_b(t, S, W, compressor="strong").dollars
        r1 = score_design_b(t, S, W, compressor="weak").dollars
        total = base - r1
        comp = base - b2
        share = comp / total if total else float("nan")
        print(f"{wr:>14.2f} {base:>11.4f} {b2:>12.4f} {r1:>11.4f} "
              f"{share:>11.1%} {1 - share:>8.1%}")
    b3 = score_design_b(t, S, weak(0.05), compressor="script").dollars
    print(f"\n  B3 (scripted truncation, no model call at all): ${b3:.4f}")
    print("  B3 is the floor. If R1 barely beats B3, the LLM delegation is not")
    print("  earning its keep and the honest finding is 'just truncate'.")


def table4_breakeven() -> None:
    rule("T4  Break-even: remaining turns needed for delegation to pay")
    print(f"{'p_in_W/p_in_S':>14} {'cache=0.05':>12} {'cache=0.10':>12} {'cache=0.25':>12}")
    for wr in (0.02, 0.05, 0.10, 0.20, 0.40):
        row = [breakeven_remaining_turns(strong(cr), weak(wr), sigma=0.05)
               for cr in (0.05, 0.10, 0.25)]
        print(f"{wr:>14.2f} " + "".join(f"{v:>12.1f}" for v in row))
    print("\n  Values below ~1 mean delegation pays immediately. Values above the")
    print("  typical trajectory length (~20-30 turns) mean it never pays.")


def table5_sensitivity() -> None:
    rule("T5  Sensitivity to imperfect cache hits")
    S, W = strong(), weak(0.05)
    t = build(30)
    print(f"{'cache_eff':>10} {'baseline $':>12} {'R1 $':>10} {'saving':>10}")
    for eff in (1.0, 0.9, 0.75, 0.5, 0.0):
        base = score_cache_aware(t, S, cache_efficiency=eff).dollars
        r1 = score_design_b(t, S, W, cache_efficiency=eff, compressor="weak").dollars
        print(f"{eff:>10.2f} {base:>12.4f} {r1:>10.4f} {100 * (base - r1) / base:>9.1f}%")
    print("\n  cache_eff=0.0 is the flat-pricing world prior work assumes.")
    print("  Reported savings shrink as the real cache gets better.")


if __name__ == "__main__":
    print("LoopEngg S1 -- synthetic cache-aware analysis")
    print("Strong input normalised to $1.00/Mtok; only ratios matter.")
    print("ILLUSTRATIVE price ratios -- substitute the verified vendor table before publication.")
    table1_inflation()
    table2_design_a_phase()
    table3_compression_share()
    table4_breakeven()
    table5_sensitivity()
    print("\nDone.")
