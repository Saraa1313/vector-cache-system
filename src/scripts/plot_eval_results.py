"""
Plot grouped bar chart from evaluate_live_policy.py JSON output.
Usage:
    python src/scripts/plot_eval_results.py results/eval_<timestamp>.json [--n 6]
"""
import argparse
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from pathlib import Path

import seaborn as sns

rcParams["font.family"] = "DejaVu Sans"

_pal = sns.color_palette("Set2")
POLICY_STYLE = {
    "always_cache":  dict(label="Always Cache  (stale baseline)", color=_pal[0]),
    "learned":       dict(label="Learned Policy",                 color=_pal[1]),
    "always_fetch":  dict(label="Always Fetch  (fresh baseline)", color=_pal[2]),
}

SCENARIO_SHORT = {
    "conc_update_10pct":        "Concentrated\nUpdate 10%\n(light)",
    "conc_update_50pct":        "Concentrated\nUpdate 50%\n(heavy)",
    "conc_mixed_40pct":         "Concentrated\nMixed 40%+40%\n(aggressive)",
    "conc_delete_50pct":        "Concentrated\nDelete 50%\n(heavy)",
    "conc_delete_10pct":        "Concentrated\nDelete 10%\n(light)",
    "rank_close_update_50pct":  "Rank Close\nUpdate 50%",
    "rank_mid_update_50pct":    "Rank Mid\nUpdate 50%",
    "rank_late_update_50pct":   "Rank Late\nUpdate 50%",
    "rank_close_delete_50pct":  "Rank Close\nDelete 50%",
    "rank_mid_delete_50pct":    "Rank Mid\nDelete 50%",
    "rank_late_delete_50pct":   "Rank Late\nDelete 50%",
}


def main(json_path: str, n_scenarios: int | None = None):
    with open(json_path) as f:
        data = json.load(f)

    all_scenarios = data["scenarios"]
    policies      = data["config"]["policies"]

    sc_keys   = list(all_scenarios.keys())[:n_scenarios]  # None → all
    scenarios = {k: all_scenarios[k] for k in sc_keys}

    n_sc       = len(scenarios)
    n_pol      = len(policies)
    bar_w      = 0.24
    group_gap  = 0.18
    group_w    = n_pol * bar_w + group_gap
    x_centers  = np.arange(n_sc) * group_w

    fig, ax = plt.subplots(figsize=(max(11, 2.2 * n_sc), 6.5))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Derive y_min from data so bars below 0.75 aren't silently clipped
    all_recalls = [
        sc["policies"][pol]["mean_recall"]
        for sc in scenarios.values()
        for pol in policies
    ]
    y_min = max(0.0, np.floor((min(all_recalls) - 0.02) / 0.05) * 0.05)
    y_max = 1.00

    for pi, policy in enumerate(policies):
        style   = POLICY_STYLE[policy]
        offsets = x_centers + (pi - (n_pol - 1) / 2) * bar_w

        recalls = []
        for sc in scenarios.values():
            pol_data = sc["policies"][policy]
            recalls.append(pol_data["mean_recall"])

        bar_heights = [r - y_min for r in recalls]
        bars = ax.bar(
            offsets, bar_heights, width=bar_w, bottom=y_min,
            color=style["color"], alpha=0.88, zorder=3,
            label=style["label"],
            linewidth=0.8, edgecolor="white",
        )


    # Axes formatting
    ax.set_xticks(x_centers)
    ax.set_xticklabels(
        [SCENARIO_SHORT.get(k, k) for k in scenarios],
        fontsize=10, linespacing=1.4,
    )
    ax.set_ylabel("Mean Recall@100", fontsize=12, labelpad=10)
    ax.set_ylim(y_min, y_max + 0.035)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))
    ax.tick_params(axis="y", labelsize=10)

    ax.set_title(
        "Cache Policy Recall Comparison by Mutation Scenario",
        fontsize=14, fontweight="bold", pad=48,
    )
    ax.set_xlabel("Mutation Scenario", fontsize=12, labelpad=10)

    # Recovery rate brackets — drawn per scenario
    for si, (sc_key, sc) in enumerate(scenarios.items()):
        ac  = sc["policies"]["always_cache"]["mean_recall"]
        lp  = sc["policies"]["learned"]["mean_recall"]
        af  = sc["policies"]["always_fetch"]["mean_recall"]
        total_gap   = af - ac
        recovered   = lp - ac
        if total_gap < 0.005:
            continue
        recovery_pct = recovered / total_gap * 100

        af_idx    = policies.index("always_fetch")
        x_af      = x_centers[si] + (af_idx - (n_pol - 1) / 2) * bar_w
        bracket_x = x_af + bar_w * 0.62

        # Full gap bracket (always_cache → always_fetch), dim gray
        ax.annotate("", xy=(bracket_x, af), xytext=(bracket_x, ac),
                    arrowprops=dict(arrowstyle="<->", color="#bbbbbb", lw=1.4),
                    zorder=5)

        # Recovered portion bracket (always_cache → learned), bold red
        ax.annotate("", xy=(bracket_x - 0.045, lp), xytext=(bracket_x - 0.045, ac),
                    arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=2.5),
                    zorder=6)

        # Recovery % label in red with white background box
        ax.text(bracket_x + 0.01, (ac + lp) / 2,
                f"{recovery_pct:.0f}% of gap\nrecovered",
                ha="left", va="center", fontsize=9, fontweight="bold",
                color="#c0392b",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b",
                          alpha=0.92, linewidth=1.2))

    # Subtle grid
    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0, color="#aaaaaa")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Legend above the plot
    ax.legend(
        fontsize=10, loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=n_pol,
        framealpha=0.92, edgecolor="#cccccc",
        handlelength=1.6, handleheight=1.2,
    )

    plt.tight_layout(pad=1.8)
    out_path = Path(json_path).with_suffix(".png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot policy recall comparison chart")
    parser.add_argument("json_path", help="Path to eval results JSON")
    parser.add_argument("--n", type=int, default=None,
                        help="Number of scenarios to plot (default: all)")
    args = parser.parse_args()
    main(args.json_path, n_scenarios=args.n)
