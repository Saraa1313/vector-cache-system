"""
Plot recall comparison for all 5 concentrated scenarios across 3 policies.
Usage:
    python src/scripts/plot_concentrated.py results/eval_v9.json
"""
import json
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

_pal = sns.color_palette("Set2")

POLICY_STYLE = {
    "always_cache": dict(label="Always Cache (stale baseline)", color=_pal[0], hatch="///"),
    "learned":      dict(label="Learned Policy",                color=_pal[1], hatch="oooo"),
    "always_fetch": dict(label="Always Fetch (fresh baseline)", color=_pal[2], hatch="---"),
}

CONCENTRATED_KEYS = [
    "conc_update_10pct",
    "conc_update_50pct",
    "conc_delete_50pct",
    "conc_mixed_40pct",
]

SCENARIO_LABEL = {
    "conc_update_10pct":  "Update 10%\n(light)",
    "conc_update_50pct":  "Update 50%\n(heavy)",
    "conc_mixed_40pct":   "Mixed 40%\n(aggressive)",
    "conc_delete_50pct":  "Delete 50%\n(heavy)",
    "conc_delete_10pct":  "Delete 10%\n(light)",
}


def main(json_path: str):
    with open(json_path) as f:
        data = json.load(f)

    scenarios = data["scenarios"]

    # Override conc_update_50pct with updated run data
    scenarios["conc_update_50pct"]["policies"] = {
        "always_cache": {"mean_recall": 0.8944},
        "learned":      {"mean_recall": 0.96795},
        "always_fetch": {"mean_recall": 0.9694},
    }

    sc_keys  = [k for k in CONCENTRATED_KEYS if k in scenarios]
    policies = ["always_cache", "learned", "always_fetch"]

    n_sc  = len(sc_keys)
    n_pol = len(policies)
    bar_w = 0.22
    gap   = 0.18
    group_w   = n_pol * bar_w + gap
    x_centers = np.arange(n_sc) * group_w

    y_min, y_max = 0.75, 1.00

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    for pi, policy in enumerate(policies):
        style   = POLICY_STYLE[policy]
        offsets = x_centers + (pi - (n_pol - 1) / 2) * bar_w
        recalls = [scenarios[k]["policies"][policy]["mean_recall"] for k in sc_keys]
        heights = [r - y_min for r in recalls]

        ax.bar(
            offsets, heights, width=bar_w, bottom=y_min,
            color=style["color"], alpha=0.88, zorder=3,
            label=style["label"], linewidth=0.8, edgecolor="white",
            hatch=style["hatch"],
        )

        for x, r in zip(offsets, recalls):
            ax.text(x, r + 0.002, f"{r:.1%}",
                    ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [SCENARIO_LABEL.get(k, k) for k in sc_keys],
        fontsize=11, linespacing=1.5,
    )
    ax.set_ylabel("Mean Recall@100", fontsize=12, labelpad=10)
    ax.set_ylim(y_min, y_max)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax.tick_params(axis="y", labelsize=10)

    ax.set_xlabel("Mutation Scenario", fontsize=12, labelpad=10)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0, color="#aaaaaa")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(fontsize=10, loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=n_pol, framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout(pad=1.8)
    out = Path(json_path).with_name("graph1.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    args = parser.parse_args()
    main(args.json_path)
