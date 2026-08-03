"""Print the complete per-cell numbers the paper's tables need."""

import json
from pathlib import Path

res = json.loads(Path("data/results.json").read_text())

print("=== Table: corpus shape + inflation ===")
print(f"{'cell':22} {'medT':>5} {'IQR':>12} {'maxT':>5} {'medCtx':>8} "
      f"{'hit%':>6} {'infl':>6}")
for c in res["cells"]:
    t, f, i = c["turns"], c["final_context"], c["inflation"]
    print(f"{c['label']:22} {t['median']:>5} "
          f"{f'{t[chr(112)+chr(50)+chr(53)]}-{t[chr(112)+chr(55)+chr(53)]}':>12} "
          f"{t['max']:>5} {f['median']:>8,} "
          f"{100 * i['cache_hit_rate']:>5.1f}% {i['aggregate']:>5.2f}x")

print("\n=== Table: observation size distribution ===")
print(f"{'cell':22} {'p50':>6} {'p75':>6} {'p90':>7} {'p95':>7} {'p99':>7} {'max':>8}")
for c in res["cells"]:
    o = c["observation_tokens"]
    print(f"{c['label']:22} {o['p50']:>6,} {o['p75']:>6,} {o['p90']:>7,} "
          f"{o['p95']:>7,} {o['p99']:>7,} {o['p100']:>8,}")

print("\n=== Table: growth decomposition + ceiling ===")
print(f"{'cell':22} {'obs%':>7} {'out%':>7} {'ceiling':>8}")
for c in res["cells"]:
    g, ce = c["growth"], c["ceiling"]
    print(f"{c['label']:22} {100 * g['obs_share']:>6.1f}% "
          f"{100 * (1 - g['obs_share']):>6.1f}% {ce['saving_pct']:>7.1f}%")
