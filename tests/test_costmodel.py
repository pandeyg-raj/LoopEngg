"""Sanity tests for the cost model.

These are guardrails for the paper's central numbers. If any of these break,
a headline claim is wrong.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loopengg.costmodel import (  # noqa: E402
    Pricing,
    Trajectory,
    breakeven_remaining_turns,
    design_a_is_profitable,
    score_cache_aware,
    score_design_a,
    score_design_b,
    score_flat,
)

# Normalised: strong input = $1.00/Mtok. Only ratios matter for the conclusions.
STRONG = Pricing("strong", input_per_mtok=1.0, output_per_mtok=5.0)
WEAK = Pricing("weak", input_per_mtok=0.05, output_per_mtok=0.20)


def make_traj(n_turns=20, obs=4000, system=2000, out=400, every=2):
    """Loop with a large observation every `every` turns marked delegatable."""
    observations = [obs if i % every == 0 else 200 for i in range(n_turns)]
    delegatable = [i % every == 0 for i in range(n_turns)]
    return Trajectory.from_observations(
        system_tokens=system,
        observations=observations,
        output_tokens=out,
        delegatable=delegatable,
    )


def test_append_only_growth():
    """Context must grow by exactly output + observation each turn."""
    t = make_traj(n_turns=5, obs=1000, system=500, out=100, every=1)
    for i in range(1, len(t.turns)):
        prev, cur = t.turns[i - 1], t.turns[i]
        expected = prev.context_tokens + prev.output_tokens + prev.observation_tokens
        assert cur.context_tokens == expected, f"turn {i}: {cur.context_tokens} != {expected}"


def test_cache_default_ratios():
    assert abs(STRONG.cache_read_ratio - 0.10) < 1e-9
    assert abs(STRONG.cache_write_per_mtok - 1.25) < 1e-9


def test_flat_overstates_cost():
    """The core claim: pricing without the cache inflates the baseline."""
    t = make_traj()
    flat = score_flat(t, STRONG)
    cached = score_cache_aware(t, STRONG)
    assert flat.dollars > cached.dollars
    ratio = flat.dollars / cached.dollars
    # With a long transcript the inflation should be substantial, not marginal.
    assert ratio > 1.5, f"expected meaningful inflation, got {ratio:.2f}x"


def test_inflation_grows_with_trajectory_length():
    """Longer loops re-send more context, so the flat/cached gap must widen."""
    short = make_traj(n_turns=5)
    long = make_traj(n_turns=40)
    r_short = score_flat(short, STRONG).dollars / score_cache_aware(short, STRONG).dollars
    r_long = score_flat(long, STRONG).dollars / score_cache_aware(long, STRONG).dollars
    assert r_long > r_short


def test_cache_accounting_conserves_tokens():
    """Every input token is either a hit or a miss -- nothing invented or lost."""
    t = make_traj()
    c = score_cache_aware(t, STRONG)
    assert c.cached_read_tokens + c.uncached_input_tokens == sum(
        x.context_tokens for x in t.turns
    )


def test_design_b_beats_baseline():
    """H2: out-of-band delegation with slicing should reduce cost."""
    t = make_traj()
    base = score_cache_aware(t, STRONG).dollars
    b = score_design_b(t, STRONG, WEAK, sigma=0.05).dollars
    assert b < base, f"design B {b:.4f} should undercut baseline {base:.4f}"


def test_design_a_loses_when_weak_costs_more_than_cached_reads():
    """H1: in-band swapping trades a cached read for an uncached one."""
    # weak input (0.05) is BELOW strong cached read (0.10) -> should help
    assert design_a_is_profitable(STRONG, WEAK)

    # A weak model that is cheap vs list price but dear vs cached reads.
    pricey_weak = Pricing("weak2", input_per_mtok=0.30, output_per_mtok=1.0)
    assert not design_a_is_profitable(STRONG, pricey_weak)

    t = make_traj()
    base = score_cache_aware(t, STRONG).dollars
    a = score_design_a(t, STRONG, pricey_weak).dollars
    assert a > base, "in-band swap should lose money against cached reads here"


def test_design_b_dominates_design_a():
    """The paper's wedge: same weak model, opposite outcomes by implementation."""
    t = make_traj()
    a = score_design_a(t, STRONG, WEAK).dollars
    b = score_design_b(t, STRONG, WEAK, sigma=0.05).dollars
    assert b < a, f"design B {b:.4f} must beat design A {a:.4f}"


def test_compression_explains_most_of_the_win():
    """H4: strong-model compression (B2) should capture most of R1's benefit.

    If this fails, routing genuinely adds a lot and the paper's framing changes.
    """
    t = make_traj()
    base = score_cache_aware(t, STRONG).dollars
    b2 = score_design_b(t, STRONG, WEAK, compressor="strong").dollars
    r1 = score_design_b(t, STRONG, WEAK, compressor="weak").dollars

    total_saving = base - r1
    compression_saving = base - b2
    share = compression_saving / total_saving
    assert 0.0 < share <= 1.0
    print(f"\n  compression accounts for {share:.1%} of the total saving")


def test_breakeven_matches_simple_ratio():
    """As sigma -> 0 the exact form collapses to p_in_W / p_cache_S."""
    approx = WEAK.input_per_mtok / STRONG.cache_read_per_mtok
    exact = breakeven_remaining_turns(STRONG, WEAK, sigma=1e-6)
    assert abs(exact - approx) < 1e-3


def test_breakeven_worse_with_bad_compressor():
    """A compressor that barely compresses pushes break-even further out."""
    good = breakeven_remaining_turns(STRONG, WEAK, sigma=0.05)
    bad = breakeven_remaining_turns(STRONG, WEAK, sigma=0.60)
    assert bad > good


def test_cache_efficiency_monotonic():
    """Lower cache efficiency must cost more -- never less."""
    t = make_traj()
    costs = [score_cache_aware(t, STRONG, cache_efficiency=e).dollars for e in (0.0, 0.5, 1.0)]
    assert costs[0] > costs[1] > costs[2]


if __name__ == "__main__":
    import traceback

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
