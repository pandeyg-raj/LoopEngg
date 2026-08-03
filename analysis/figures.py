"""Paper figures from data/results.json.

IEEE double-column: single-column figures are 3.4in wide, full-width 7.0in.
Every multi-series figure carries a marker/linestyle in addition to hue so it
survives greyscale printing -- identity is never colour-alone.

Run:  python analysis/figures.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loopengg.costmodel import Pricing, breakeven_remaining_turns  # noqa: E402

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

# Validated categorical slots 1-4, with a redundant marker/linestyle channel.
SERIES = [
    {"color": "#2a78d6", "marker": "o", "ls": "-"},
    {"color": "#eb6834", "marker": "s", "ls": "--"},
    {"color": "#1baf7a", "marker": "^", "ls": "-."},
    {"color": "#eda100", "marker": "D", "ls": ":"},
]
SURFACE = "#fcfcfb"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED = "#898781"
INK = "#0b0b0b"
INK2 = "#52514e"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9,
    "legend.fontsize": 7.5,
    "font.size": 8,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "legend.frameon": False,
    "figure.dpi": 200,
})


def tidy(ax, grid_axis="y"):
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def load():
    p = Path("data/results.json")
    if not p.exists():
        sys.exit("data/results.json not found -- run analysis/run_all.py first")
    return json.loads(p.read_text())


# ---------------------------------------------------------------- Fig 1
def fig1_inflation(res):
    cells = res["cells"]
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    labels = [c["label"] for c in cells]
    x = np.arange(len(cells))
    w = 0.36

    flat = [c["inflation"]["total_flat"] for c in cells]
    cached = [c["inflation"]["total_cached"] for c in cells]

    ax.bar(x - w / 2 - 0.01, flat, w, label="Flat pricing (prior work)",
           color="#eb6834", zorder=3)
    ax.bar(x + w / 2 + 0.01, cached, w, label="Cache-aware (ours)",
           color="#2a78d6", zorder=3)

    for i, c in enumerate(cells):
        ax.annotate(f"{c['inflation']['aggregate']:.1f}×",
                    xy=(i - w / 2, flat[i]), xytext=(0, 3), textcoords="offset points",
                    ha="center", fontsize=7.5, color=INK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([l.replace(" / ", "\n") for l in labels], fontsize=7)
    ax.set_ylabel("Cost, 150 trajectories (norm.)")
    # Headroom so the ×-labels clear the legend.
    ax.set_ylim(0, max(flat) * 1.30)
    ax.legend(loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.16),
              columnspacing=1.0, handlelength=1.4)
    tidy(ax)
    fig.tight_layout()
    fig.savefig(OUT / "fig1_inflation.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig1_inflation.png", bbox_inches="tight")
    plt.close(fig)
    print("fig1_inflation")


# ---------------------------------------------------------------- Fig 2
def fig2_observation_cdf(res):
    """The figure that kills the motivating premise."""
    fig, ax = plt.subplots(figsize=(3.4, 2.6))

    for i, c in enumerate(res["cells"]):
        obs = np.array([v for v in c["observation_sample"] if v > 0])
        if obs.size == 0:
            continue
        obs.sort()
        y = np.arange(1, obs.size + 1) / obs.size * 100
        s = SERIES[i % 4]
        ax.plot(obs, y, color=s["color"], ls=s["ls"], lw=1.6,
                label=c["label"], zorder=3)

    ax.set_xscale("log")
    ax.set_xlabel("Observation size (tokens, log scale)")
    ax.set_ylabel("Cumulative % of observations")
    ax.set_ylim(0, 100)

    # Mark the region the routing literature actually motivates with.
    ax.axvspan(4000, 40000, color="#eb6834", alpha=0.10, zorder=1)
    # Sits high in the band: the curves are flat at ~100% out here, and the
    # legend occupies the lower right.
    ax.annotate("the “huge log”\nthat motivates\nrouting (top ~1%)",
                xy=(4600, 56), fontsize=6.3, color=INK2, ha="left", va="bottom")
    ax.axhline(50, color=MUTED, lw=0.7, ls=(0, (2, 2)), zorder=2)
    ax.annotate("median\n< 200 tokens", xy=(12, 54), fontsize=6.5,
                color=INK, fontweight="bold", ha="left", va="bottom")

    ax.xaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{int(v):,}" if v >= 1 else ""))
    ax.legend(loc="lower right")
    tidy(ax, grid_axis="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig2_observation_cdf.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_observation_cdf.png", bbox_inches="tight")
    plt.close(fig)
    print("fig2_observation_cdf")


# ---------------------------------------------------------------- Fig 3
def fig3_saving_vs_threshold(res):
    """Small multiples: 3 designs per cell. B3 dominates; delegation goes negative."""
    cells = res["cells"]
    n = len(cells)
    ncol = 2
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.0, 2.5 * nrow), sharey=True)
    axes = np.atleast_1d(axes).ravel()

    # B3 is drawn thick underneath and R1 thin on top: the two coincide almost
    # exactly, and that coincidence is the result, so it must stay legible.
    designs = [("B3", "scripted truncation", 2, 3.2, 0),
               ("B2", "strong compression", 1, 1.6, 4.5),
               ("R1", "edge delegation", 0, 1.4, 4.0)]

    for k, c in enumerate(cells):
        ax = axes[k]
        thr = [r["threshold"] for r in c["threshold_sweep"]]
        for key, name, si, lw, ms in designs:
            vals = [r[f"{key}_saving_pct"] for r in c["threshold_sweep"]]
            s = SERIES[si]
            ax.plot(thr, vals, color=s["color"], ls=s["ls"], lw=lw,
                    marker=s["marker"] if ms else None, ms=ms,
                    label=f"{key} — {name}", zorder=3,
                    alpha=0.85 if key == "B3" else 1.0)
        ax.axhline(0, color=INK, lw=0.9, zorder=2)
        ax.set_xscale("log")
        ax.set_title(c["label"], color=INK, pad=4)
        ax.set_xlabel("Delegation threshold (min observation tokens)")
        if k % ncol == 0:
            ax.set_ylabel("Cost saving vs. baseline (%)")
        tidy(ax, grid_axis="both")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    axes[0].legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "fig3_saving_vs_threshold.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig3_saving_vs_threshold.png", bbox_inches="tight")
    plt.close(fig)
    print("fig3_saving_vs_threshold")


# ---------------------------------------------------------------- Fig 4
def fig4_breakeven_phase(res):
    """Where delegation pays, over the price-ratio space. Sequential single hue."""
    weak_ratios = np.logspace(np.log10(0.01), np.log10(0.6), 160)
    cache_ratios = np.logspace(np.log10(0.02), np.log10(1.0), 160)
    W, C = np.meshgrid(weak_ratios, cache_ratios)

    Z = np.zeros_like(W)
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            s = Pricing("s", input_per_mtok=1.0, output_per_mtok=5.0,
                        cache_read_per_mtok=C[i, j])
            w = Pricing("w", input_per_mtok=W[i, j], output_per_mtok=4 * W[i, j])
            Z[i, j] = breakeven_remaining_turns(s, w, sigma=0.05)

    fig, ax = plt.subplots(figsize=(3.4, 2.8))
    lv = [0.5, 1, 2, 5, 10, 25, 50, 100]
    cf = ax.contourf(W, C, np.clip(Z, 0.3, 200), levels=lv,
                     cmap="Blues", norm=matplotlib.colors.LogNorm(), zorder=1)
    cs = ax.contour(W, C, Z, levels=lv, colors="#0d366b", linewidths=0.5, zorder=2)
    ax.clabel(cs, inline=True, fontsize=6, fmt=lambda v: f"{v:g}")

    # Where real deployments actually sit: small models run 2-10% of a frontier
    # model's input price, and cached reads are ~10% of list. Marking this shows
    # the point of the figure -- price ratios are NOT what makes delegation fail.
    ax.add_patch(plt.Rectangle((0.02, 0.08), 0.08, 0.07, fill=False,
                               edgecolor="#e34948", lw=1.6, ls="--", zorder=5))
    ax.annotate("typical deployment:\nbreak-even < 2 turns", xy=(0.045, 0.115),
                xytext=(0.016, 0.35), fontsize=6.3, color="#e34948",
                fontweight="bold", ha="left", zorder=6,
                arrowprops=dict(arrowstyle="->", color="#e34948", lw=1.0))

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(weak_ratios[0], weak_ratios[-1])
    ax.set_ylim(cache_ratios[0], cache_ratios[-1])
    ax.set_xlabel("weak input price / strong input price")
    ax.set_ylabel("cached-read / input price (strong)")
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label("Remaining turns needed to break even", fontsize=7.5)
    cb.ax.tick_params(labelsize=6.5)
    ax.set_title("Price ratios are not the binding constraint", fontsize=8.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig4_breakeven_phase.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig4_breakeven_phase.png", bbox_inches="tight")
    plt.close(fig)
    print("fig4_breakeven_phase")


# ---------------------------------------------------------------- Fig 5
def fig5_cost_drivers():
    """Two panels: the axis that explains cost, and the one that does not.

    Same y-axis and the same points in both, so the contrast is the message
    rather than an artefact of scaling.
    """
    p = Path("data/turncount.json")
    if not p.exists():
        print("fig5 skipped -- run analysis/turncount.py first")
        return
    d = json.loads(p.read_text())
    pts = d["points"]
    turns = np.array(pts["turns"], float)
    obs = np.array(pts["mean_obs"], float)
    cost = np.array(pts["cost"], float)
    cell = np.array(pts["cell"], int)
    labels = d["cells"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)

    for ax, x, xlabel, expo, r2v in (
        (axes[0], turns, "Turns per trajectory", d["exponent_turns"], d["r2_turns"]),
        (axes[1], obs, "Mean observation size (tokens)", d["exponent_obs"], d["r2_obs"]),
    ):
        for i, lab in enumerate(labels):
            m = cell == i
            s = SERIES[i % 4]
            ax.scatter(x[m], cost[m], s=7, alpha=0.55, color=s["color"],
                       marker=s["marker"], linewidths=0, label=lab, zorder=3)
        # Fitted power law, drawn over the full x-range.
        lx = np.log(x)
        b, a = np.polyfit(lx, np.log(cost), 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, np.exp(a) * xs ** b, color=INK, lw=1.6, ls="--", zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.annotate(f"exponent {expo:+.2f}\n$R^2$ = {r2v:.3f}",
                    xy=(0.04, 0.93), xycoords="axes fraction", va="top",
                    fontsize=7.5, color=INK, fontweight="bold")
        tidy(ax, grid_axis="both")

    axes[0].set_ylabel("Cost per trajectory (norm. \\$)")
    axes[0].legend(loc="lower right", markerscale=1.8, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig5_cost_drivers.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig5_cost_drivers.png", bbox_inches="tight")
    plt.close(fig)
    print("fig5_cost_drivers")


if __name__ == "__main__":
    res = load()
    print(f"{len(res['cells'])} cells loaded")
    fig1_inflation(res)
    fig2_observation_cdf(res)
    fig3_saving_vs_threshold(res)
    fig4_breakeven_phase(res)
    fig5_cost_drivers()
    print(f"\nwrote PDFs + PNGs to {OUT.resolve()}")
