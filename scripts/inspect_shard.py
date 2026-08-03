"""Download one small shard and dump its schema + a sample trajectory shape."""

import json

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "nvidia/Open-SWE-Traces"
FILE = "data/qwen35_openhands_trajectories/train-00022-of-00023.parquet"

path = hf_hub_download(REPO, FILE, repo_type="dataset", cache_dir="data/hf_cache")
print(f"downloaded -> {path}\n")

pf = pq.ParquetFile(path)
print("=== schema ===")
print(pf.schema_arrow)
print(f"\nrows: {pf.metadata.num_rows}")

batch = next(pf.iter_batches(batch_size=2))
rows = batch.to_pylist()
row = rows[0]

print("\n=== field types / sizes for row 0 ===")
for k, v in row.items():
    if isinstance(v, str):
        print(f"  {k:20} str    len={len(v)}")
    elif isinstance(v, list):
        print(f"  {k:20} list   n={len(v)}")
    else:
        print(f"  {k:20} {type(v).__name__:6} {str(v)[:60]}")

# The trajectory is the interesting part -- find out whether it is a JSON
# string or a native list, and what a message looks like.
traj = row.get("trajectory")
if isinstance(traj, str):
    print("\ntrajectory is a JSON string; parsing...")
    traj = json.loads(traj)
print(f"\n=== trajectory: {len(traj)} messages ===")
for i, msg in enumerate(traj[:6]):
    if isinstance(msg, dict):
        keys = list(msg.keys())
        role = msg.get("role", msg.get("from", "?"))
        content = msg.get("content", "")
        clen = len(content) if isinstance(content, str) else f"[{type(content).__name__}]"
        print(f"  [{i}] keys={keys} role={role!r} content_len={clen}")
    else:
        print(f"  [{i}] {type(msg).__name__}: {str(msg)[:100]}")

print("\n=== role sequence (first 24) ===")
roles = [m.get("role", m.get("from", "?")) if isinstance(m, dict) else "?" for m in traj]
print("  " + " -> ".join(roles[:24]))
from collections import Counter  # noqa: E402

print(f"\n  role counts: {dict(Counter(roles))}")

print("\n=== sample message content (truncated) ===")
for i, msg in enumerate(traj[:4]):
    if isinstance(msg, dict) and isinstance(msg.get("content"), str):
        print(f"\n--- [{i}] {msg.get('role')} ---")
        print(msg["content"][:400])
