"""ACI recovery behaviour: what does the agent do after the harness truncates?

Reproduces the numbers in Section 5.9 of the paper. Scans every observation in the
four-cell sample for the two scaffold truncation markers, then classifies the agent's
next tool call.

    python analysis/aci_recovery.py            # -> data/aci_recovery.json

Truncation is interesting here for two reasons. It is the mechanism the scaffolds
already use to bound observation size, and it is the only place in an *observed*
corpus where we can see how an agent responds to having its input cut -- which is the
closest available evidence on whether a deletion-based reduction policy is safe.

Caveat, and it matters: these are observed trajectories, not an intervention. We can
see what the agent did next, not whether it recovered the right content.
"""

from __future__ import annotations

import collections
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from loopengg.ingest import load_shard  # noqa: E402

REPO = "nvidia/Open-SWE-Traces"
CELLS = {
    "qwen35 / OpenHands": "data/qwen35_openhands_trajectories/train-00022-of-00023.parquet",
    "minimax / SWE-agent": "data/minimax_m25_sweagent_trajectories/train-00015-of-00023.parquet",
    "minimax / OpenHands": "data/minimax_m25_openhands_trajectories/train-00006-of-00020.parquet",
    "qwen35 / SWE-agent": "data/qwen35_sweagent_trajectories/train-00017-of-00018.parquet",
}

# The two harness truncation markers. CLIP is SWE-agent's editor/viewer clip, which the
# scaffold follows with a <NOTE> telling the agent how to re-query; BARE is OpenHands'
# length cap, which says only that something was removed.
CLIP = "<response clipped>"
BARE = "[... Observation truncated due to length ...]"
NOTE = re.compile(r"<NOTE>(.*?)</NOTE>", re.S)

# A follow-up counts as "narrowed" when it restricts the next read: an explicit line
# range, a search instead of a full read, or an incremental listing.
NARROW = re.compile(
    r"view_range|grep\s+-[a-z]*n|\bgrep\b|ls\s+-la|\|\s*head|\|\s*tail|"
    r"\bhead\s+-|\btail\s+-|sed\s+-n|wc\s+-l|--max-count|-m\s+\d",
    re.I,
)
TERMINAL = {"finish", "submit", "stop"}


def tool_calls(msg: dict) -> list[tuple[str, str]]:
    out = []
    for call in msg.get("tool_calls") or []:
        if isinstance(call, dict):
            fn = call.get("function") or {}
            out.append((str(fn.get("name") or ""), str(fn.get("arguments") or "")))
    return out


def classify(messages: list[dict], i: int) -> str:
    """Label the agent's next action after the truncated observation at index i."""
    prev = None
    for j in range(i - 1, -1, -1):
        if messages[j].get("role") == "assistant" and tool_calls(messages[j]):
            prev = tool_calls(messages[j])[0]
            break
    nxt = None
    for j in range(i + 1, len(messages)):
        if messages[j].get("role") == "assistant":
            calls = tool_calls(messages[j])
            nxt = calls[0] if calls else ("<no_tool>", "")
            break
    if nxt is None:
        return "end_of_trajectory"
    name, args = nxt
    if name == "<no_tool>":
        return "no_tool_call"
    if name.split("__")[-1].lower() in TERMINAL:
        return "terminate"
    if NARROW.search(args):
        return "narrowed_retry_same_tool" if prev and prev[0] == name else "narrowed_other_tool"
    if prev and prev[0] == name and args.strip() == prev[1].strip():
        return "identical_retry"
    return "other_action"


def main() -> None:
    kinds = collections.Counter()
    actions = collections.Counter()
    per_cell = {}
    n_obs_total = 0

    for label, shard in CELLS.items():
        rows = load_shard(REPO, shard, limit=150)
        n_obs = n_trunc = n_traj = 0
        for row in rows:
            traj = row.get("trajectory")
            if isinstance(traj, str):
                traj = json.loads(traj)
            traj = traj or []
            touched = False
            for i, msg in enumerate(traj):
                if msg.get("role") not in ("tool", "user"):
                    continue
                content = msg.get("content") or ""
                if not isinstance(content, str):
                    continue
                n_obs += 1
                has_clip, has_bare = CLIP in content, BARE in content
                if not (has_clip or has_bare):
                    continue
                touched = True
                n_trunc += 1
                kind = "guided" if has_clip and "<NOTE>" in content else "bare"
                kinds[kind] += 1
                actions[(kind, classify(traj, i))] += 1
            n_traj += touched
        n_obs_total += n_obs
        per_cell[label] = {"observations": n_obs, "truncated": n_trunc,
                           "rate_pct": round(n_trunc / n_obs * 100, 2),
                           "trajectories_touched": n_traj}
        print(f"{label:22s} obs={n_obs:6d} truncated={n_trunc:4d} "
              f"({n_trunc / n_obs * 100:.2f}%)  trajectories={n_traj}/150", flush=True)

    total = sum(kinds.values())
    print(f"\n{total} truncation events over {n_obs_total} observations "
          f"({total / n_obs_total * 100:.2f}%); "
          f"{sum(c['trajectories_touched'] for c in per_cell.values())}/600 trajectories")

    summary = {}
    for kind in ("guided", "bare"):
        sub = {a: n for (k, a), n in actions.items() if k == kind}
        denom = sum(sub.values())
        if not denom:
            continue
        narrowed = sub.get("narrowed_retry_same_tool", 0) + sub.get("narrowed_other_tool", 0)
        summary[kind] = {"n": denom, "narrowed_pct": round(narrowed / denom * 100, 1),
                         "breakdown": sub}
        print(f"\n-- {kind} (n={denom}): {narrowed / denom * 100:.1f}% narrowed re-query")
        for action, n in sorted(sub.items(), key=lambda x: -x[1]):
            print(f"   {n:5d}  {n / denom * 100:5.1f}%  {action}")

    out = Path("data/aci_recovery.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"per_cell": per_cell, "kinds": dict(kinds), "summary": summary,
         "observations_total": n_obs_total}, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    os.chdir(Path(__file__).resolve().parent.parent)
    main()
