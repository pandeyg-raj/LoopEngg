# TurnCost — Artifact

Replication package for "TurnCost: Turns, Not Observations: Cache-Aware Loop
Engineering for Coding Agents", POVC '26 (1st International Workshop on PromptOps
and Vibe Coding, co-located with ASE 2026), doi:10.1145/3843779.3844635. Every number, table and figure regenerates from
public data. **No API keys, no paid inference, no GPU.** A full run takes a few minutes
plus one ~600 MB download.

## What this does

Published agent-trajectory corpora ship full conversations but no token accounting.
That is not a limitation: prefix-cache behaviour is deterministic on an append-only
transcript, so instead of trusting provider-reported numbers we reconstruct per-turn
context from the message sequence, tokenize it, and simulate the cache exactly
(`h_t = C_t − r_{t−1}`). The result is a re-scoring tool that recomputes the cost of
*any* published trajectory corpus under a stated cache model.

## Setup

```bash
pip install huggingface_hub pyarrow tiktoken pandas numpy matplotlib
```

Python 3.10+. `tiktoken` is optional — without it the tokenizer falls back to a
characters/4 heuristic, which changes absolute counts but not the ratios the paper
reports.

## Reproducing the paper

```bash
python tests/test_costmodel.py      # 12 sanity checks on the cost model
python analysis/run_all.py          # main 2x2 analysis  -> data/results.json
python analysis/expansion.py        # sensitivity + turn types -> data/expansion.json
python analysis/make_tables.py      # LaTeX tables -> figures/tables.tex
python analysis/figures.py          # Figures 1-4 -> figures/*.pdf
python analysis/diagnose.py         # per-cell characterisation detail
python analysis/aci_recovery.py     # Sec 5.9 truncation recovery -> data/aci_recovery.json
```

`run_all.py` downloads four parquet shards on first use and caches them under
`data/hf_cache/`.

## Exact data provenance

Corpus: [`nvidia/Open-SWE-Traces`](https://huggingface.co/datasets/nvidia/Open-SWE-Traces),
CC BY 4.0, DOI [10.48550/arXiv.2606.16038](https://doi.org/10.48550/arXiv.2606.16038).

We use **one shard per model × scaffold cell** and the **first 150 rows** of each,
giving 600 trajectories:

| Cell | Shard |
|---|---|
| Qwen3.5 / OpenHands | `data/qwen35_openhands_trajectories/train-00022-of-00023.parquet` |
| Minimax-M2.5 / SWE-agent | `data/minimax_m25_sweagent_trajectories/train-00015-of-00023.parquet` |
| Minimax-M2.5 / OpenHands | `data/minimax_m25_openhands_trajectories/train-00006-of-00020.parquet` |
| Qwen3.5 / SWE-agent | `data/qwen35_sweagent_trajectories/train-00017-of-00018.parquet` |

Sampling is deterministic — the first *n* rows in file order, no shuffling — so the
selection is reproducible byte-for-byte.

**Known selection bias.** The corpus excludes trajectories where the agent hit its
maximum iteration limit or was terminated by the harness. This truncates the right tail
of the turn-count distribution. It makes our flat-pricing figure *conservative*
(inflation grows with length) but may bias the observation-size distribution *small*.
The paper states this as its main threat to validity.

## Layout

```
loopengg/costmodel.py   cost model, prefix-cache simulator, break-even condition
loopengg/ingest.py      trajectory parsing, tokenization, turn classification
analysis/               the analyses behind each table and figure
tests/                  sanity checks — these guard the paper's headline numbers
scripts/                utilities (environment probe, PDF text extraction, checks)
```

## Cost model

Prices are normalised to **strong input = $1.00/Mtok** and reported as ratios, so the
conclusions do not depend on a vendor price list and do not expire when prices change.
Defaults: cached reads at `0.1 × input`, cache writes at `1.25 × input`, weak input at
`0.05 × strong input`, compression ratio `σ = 0.05`.

To re-score under real prices, construct a `Pricing` with your own numbers:

```python
from loopengg.costmodel import Pricing, score_cache_aware, score_flat
strong = Pricing("gpt-x", input_per_mtok=2.50, output_per_mtok=10.0,
                 cache_read_per_mtok=0.25)
```

`score_flat` reproduces the accounting used by several routing papers;
`score_cache_aware` prices the prefix cache. The gap between them is the paper's
first result.

## Conditions

| ID | Meaning |
|---|---|
| B0 | baseline, no intervention |
| A\* | in-band model swap (weak model receives the full transcript, cold) |
| R1 | out-of-band delegation (weak model receives only the observation) |
| B2 | same as R1 but the strong model compresses — isolates compression from routing |
| B3 | deterministic truncation, no model call |

B2 and B3 are load-bearing. Without them any saving could be attributed to compression
rather than to routing.

## License

Code released under MIT. The corpus is CC BY 4.0 and is not redistributed here — it is
downloaded from Hugging Face at run time.
