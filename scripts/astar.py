"""Extract the A* (in-band model swap) results the paper defines but never reports."""

import json
from pathlib import Path

res = json.loads(Path("data/results.json").read_text())

print(f"{'cell':22} " + "".join(f"{r['threshold']:>9}" for r in res["cells"][0]["threshold_sweep"]))
print("-" * 70)
for c in res["cells"]:
    row = "".join(f"{r['A_saving_pct']:>+9.1f}" for r in c["threshold_sweep"])
    print(f"{c['label']:22} {row}")

print("\n(positive = A* is CHEAPER than baseline; negative = it costs more)")

# Aggregate at the mid threshold
for thr_i, thr in enumerate([r["threshold"] for r in res["cells"][0]["threshold_sweep"]]):
    vals = [c["threshold_sweep"][thr_i]["A_saving_pct"] for c in res["cells"]]
    r1 = [c["threshold_sweep"][thr_i]["R1_saving_pct"] for c in res["cells"]]
    print(f"threshold {thr:>5}: A* mean {sum(vals) / len(vals):>+6.1f}%   "
          f"R1 mean {sum(r1) / len(r1):>+6.1f}%   "
          f"gap {sum(r1) / len(r1) - sum(vals) / len(vals):>+5.1f} pts")
