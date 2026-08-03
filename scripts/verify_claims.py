"""Check the paper's headline numbers against data/results.json + expansion.json.

Cheap insurance against prose drifting from the data after edits.
"""

import json
from pathlib import Path

res = json.loads(Path("data/results.json").read_text())
cells = res["cells"]


def rng(vals, fmt="{:.1f}"):
    return f"{fmt.format(min(vals))}--{fmt.format(max(vals))}"


hits = [c["inflation"]["cache_hit_rate"] * 100 for c in cells]
infl = [c["inflation"]["aggregate"] for c in cells]
p50 = [c["observation_tokens"]["p50"] for c in cells]
p90 = [c["observation_tokens"]["p90"] for c in cells]
p99 = [c["observation_tokens"]["p99"] for c in cells]
mx = [c["observation_tokens"]["p100"] for c in cells]
turns = [c["turns"]["median"] for c in cells]
ctx = [c["final_context"]["median"] for c in cells]
obs_share = [c["growth"]["obs_share"] * 100 for c in cells]
ceil = [c["ceiling"]["saving_pct"] for c in cells]

CLAIMS = [
    ("cache hit rate range",   rng(hits) + "%",            "98.1--99.0%"),
    ("cache hit mean",         f"{sum(hits) / 4:.1f}%",    "98.6%"),
    ("inflation range",        rng(infl, "{:.2f}") + "x",  "6.5--7.8x"),
    ("inflation mean",         f"{sum(infl) / 4:.2f}x",    "7.14x"),
    ("median turns range",     rng(turns, "{:.0f}"),       "60--134"),
    ("final context range",    rng(ctx, "{:,.0f}"),        "45k--73k"),
    ("observation p50 range",  rng(p50, "{:.0f}"),         "118--286"),
    ("observation p90 range",  rng(p90, "{:.0f}"),         "749--1,258"),
    ("observation p99 range",  rng(p99, "{:.0f}"),         "~4.3k--6.2k"),
    ("observation max range",  rng(mx, "{:.0f}"),          "12.9k--33.5k"),
    ("obs share of growth",    rng(obs_share) + "%",       "66--74%"),
    ("ceiling range",          rng(ceil) + "%",            "47.8--60.3%"),
]

print(f"{'claim':26} {'computed':>22}   {'in paper':<16}")
print("-" * 70)
for name, computed, paper in CLAIMS:
    print(f"{name:26} {computed:>22}   {paper:<16}")

exp = json.loads(Path("data/expansion.json").read_text())
tt = exp["turn_types"]
total = sum(t["n"] for t in tt)
print(f"\npooled turns: {total:,}   (paper says 56,164)")
for t in tt:
    print(f"  {t['turn_type']:12} n={t['n']:>7,}  mean obs={t['obs_tokens'] / t['n']:>7.0f}")

tc = json.loads(Path("data/turncount.json").read_text())
print(f"\nR^2 turns={tc['r2_turns']:.3f}  R^2 obs={tc['r2_obs']:.3f}  "
      f"joint={tc['r2_joint']:.3f}")
print(f"exponents: turns={tc['exponent_turns']:+.2f}  obs={tc['exponent_obs']:+.2f}")
