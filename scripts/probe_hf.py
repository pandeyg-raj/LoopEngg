"""List files in candidate trajectory datasets so we can pull one small shard
rather than the whole thing (Open-SWE-Traces is 18.3 GB)."""

from huggingface_hub import HfApi

CANDIDATES = [
    "nebius/SWE-agent-trajectories",
    "nvidia/Open-SWE-Traces",
]

api = HfApi()
for repo in CANDIDATES:
    print(f"\n{'=' * 70}\n{repo}\n{'=' * 70}")
    try:
        info = api.repo_info(repo, repo_type="dataset", files_metadata=True)
        files = sorted(
            ((s.rfilename, s.size or 0) for s in info.siblings),
            key=lambda x: x[1],
        )
        print(f"{len(files)} files")
        for name, size in files[:30]:
            mb = size / 1e6
            print(f"  {mb:>10.1f} MB  {name}")
        if len(files) > 30:
            print(f"  ... and {len(files) - 30} more")
    except Exception as e:
        print(f"ERR: {type(e).__name__}: {e}")
