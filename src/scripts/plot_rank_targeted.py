"""
Plot recall for rank-targeted scenarios — close/mid/late × update/delete.
Groups: Near (1-4), Mid (9-16), Far (25-32)
Each group has two sub-groups: Update 50% and Delete 50%, side by side.
Usage:
    python src/scripts/plot_rank_targeted.py results/eval_v9.json
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
POLICIES = ["always_cache", "learned", "always_fetch"]

# 3 rank groups, each with (update_key, delete_key)
GROUPS = [
    ("Near (Ranks 1–4)",  "rank_close_update_50pct", None),
    ("Mid (Ranks 9–16)",  "rank_mid_update_50pct",   None),
    ("Far (Ranks 25–32)", "rank_late_update_50pct",  None),
]


def main(json_path: str):
    with open(json_path) as f:
        data = json.load(f)
    scenarios = data["scenarios"]

    bar_w      = 0.18   # width of each policy bar
    sub_gap    = 0.08   # gap between Update and Delete sub-groups within a rank group
    group_gap  = 0.12   # gap between Near / Mid / Far groups
    n_pol      = len(POLICIES)
    sub_w      = n_pol * bar_w  # width of one mutation-type sub-group

    # x positions for the center of each sub-group (Update=0, Delete=1 within group)
    group_centers = []
    x = 0.0
    for _ in GROUPS:
        update_center = x + sub_w / 2
        delete_center = update_center + sub_w + sub_gap
        group_centers.append((update_center, delete_center))
        x = delete_center + sub_w / 2 + group_gap

    total_width = x - group_gap + sub_w / 2

    # y range
    all_recalls = [
        scenarios[key]["policies"][pol]["mean_recall"]
        for _, upd_key, del_key in GROUPS
        for key in (upd_key, del_key)
        for pol in POLICIES
        if key in scenarios
    ]
    y_min = max(0.0, np.floor((min(all_recalls) - 0.02) / 0.05) * 0.05)
    y_max = 1.00

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Draw bars
    for gi, (rank_label, upd_key, del_key) in enumerate(GROUPS):
        for si, sc_key in enumerate((upd_key, del_key)):
            if sc_key not in scenarios:
                continue
            sub_center = group_centers[gi][si]
            for pi, policy in enumerate(POLICIES):
                style  = POLICY_STYLE[policy]
                x_bar  = sub_center - sub_w / 2 + pi * bar_w + bar_w / 2
                recall = scenarios[sc_key]["policies"][policy]["mean_recall"]
                height = recall - y_min
                ax.bar(x_bar, height, width=bar_w, bottom=y_min,
                       color=style["color"], alpha=0.88, zorder=3,
                       hatch=style["hatch"], linewidth=0.8, edgecolor="white",
                       label=style["label"] if (gi == 0 and si == 0) else "_nolegend_")
                ax.text(x_bar, recall + 0.003, f"{recall:.1%}",
                        ha="center", va="bottom", fontsize=8, fontweight="bold", rotation=0)

    # Sub-group labels (only for non-None keys)
    mut_labels = ("Update 50%", "Delete 50%")
    for gi, (rank_label, upd_key, del_key) in enumerate(GROUPS):
        for si, (sc_key, mut_label) in enumerate(zip((upd_key, del_key), mut_labels)):
            if sc_key is None:
                continue
            cx = group_centers[gi][si]
            ax.text(cx, y_min - 0.022, mut_label,
                    ha="center", va="top", fontsize=10, fontweight="bold")

    # Rank group labels below sub-group labels
    for gi, (rank_label, _, _) in enumerate(GROUPS):
        cx = (group_centers[gi][0] + group_centers[gi][1]) / 2
        ax.text(cx, y_min - 0.048, rank_label,
                ha="center", va="top", fontsize=11, color="#333333")

    # Vertical dividers between rank groups
    for gi in range(len(GROUPS) - 1):
        divider_x = (group_centers[gi][1] + sub_w / 2 + group_centers[gi + 1][0] - sub_w / 2) / 2
        ax.axvline(divider_x, color="#cccccc", linewidth=1.2, linestyle="--", zorder=1)

    ax.set_xlim(-bar_w, total_width + bar_w)
    ax.set_ylim(y_min, y_max)
    ax.set_xticks([])
    ax.set_ylabel("Mean Recall@100", fontsize=12, labelpad=10)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.02))
    ax.tick_params(axis="y", labelsize=10)

    ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5, zorder=0, color="#aaaaaa")
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(fontsize=11, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              ncol=n_pol, framealpha=0.92, edgecolor="#cccccc")

    plt.tight_layout(pad=2.0)
    out = Path(json_path).with_name("graph3.png")
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_path")
    args = parser.parse_args()
    main(args.json_path)
