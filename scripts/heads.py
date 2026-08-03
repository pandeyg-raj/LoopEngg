"""Print the title/author block of each extracted paper for citation purposes."""

import sys
from pathlib import Path

WANT = sys.argv[1:] or ["swerouter", "mtrouter", "opendev", "openswetraces",
                        "trace_router", "explainable_routing"]

for f in sorted((Path("papers") / "txt").glob("*.txt")):
    if not any(w in f.stem.lower() for w in WANT):
        continue
    lines = [l.rstrip() for l in f.read_text(encoding="utf-8").splitlines()]
    lines = [l for l in lines if l.strip()][:26]
    print(f"\n{'=' * 78}\n{f.stem}\n{'=' * 78}")
    print("\n".join(lines))
