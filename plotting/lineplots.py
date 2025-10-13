import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.25, 1.10, 0.05)

sequences = [
    {
        "data": [
            0.05,
            0.05,
            0.05,
            0.03,
            0.07,
            0.16,
            0.21,
            0.36,
            0.64,
            0.89,
            0.95,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "label": "Math (Locked) Refusal",
        "color": "#FFCC99",
        "linestyle": "--",
        "marker": "o",
        "show_in_legend": True,
    },
    {
        "data": [
            0.06,
            0.65,
            0.98,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "label": "SQL (Locked) Refusal",
        "color": "#A6B3B7",
        "linestyle": "--",
        "marker": "s",
        "show_in_legend": True,
        "markersize": 10,
    },
    {
        "data": [
            0.02,
            0.01,
            0.05,
            0.48,
            0.97,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
            1.0,
        ],
        "label": "Summarize (Locked) Refusal",
        "color": "#8EC6E8",
        "linestyle": "--",
        "marker": "v",
        "show_in_legend": True,
        "markersize": 13,
    },
    {
        "data": [
            0.53,
            0.52,
            0.52,
            0.54,
            0.52,
            0.53,
            0.52,
            0.55,
            0.54,
            0.54,
            0.52,
            0.53,
            0.48,
            # 0.45,
            # 0.39,
            # 0.4,
            # 0.38,
            0.3431,
            0.19,
            0.0799,
            0.0188,
        ],
        "label": "MMLU (Unlocked) Utility",
        "color": "#8B5E3C",
        "linestyle": "-",
        "marker": "d",
        "show_in_legend": True,
    },
    # {
    #     "data": [
    #         0.01,
    #         0.01,
    #         0.01,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.0,
    #         0.01,
    #         0.01,
    #         0.03,
    #         0.08,
    #         0.27,
    #         0.55,
    #         0.83,
    #         0.96,
    #         0.99,
    #     ],
    #     "label": "MMLU (Unlocked) Refusal",
    #     "color": "#8B5E3C",
    #     "linestyle": "--",
    #     "marker": "p",
    #     "show_in_legend": True,
    #     "markersize": 14,
    # },
]

baselines = [
    {
        "y": 0.53,
        "color": "#8B5E3C",
        "linestyle": "--",
        "label": "Baseline",
        "show_in_legend": False,
    },
]

fig, ax = plt.subplots(figsize=(9, 6))

# ax.axvspan(0.7, 0.8, alpha=0.2, color="gray")
ax.axvline(x=0.8, linestyle="--", linewidth=6, color="#009587", zorder=100)

for seq in sequences:
    show_label = seq.get("show_in_legend", True)
    ax.plot(
        x,
        seq["data"],
        label=seq["label"] if show_label else None,
        color=seq["color"],
        linestyle=seq["linestyle"],
        marker=seq["marker"],
        markersize=seq["markersize"] if "markersize" in seq else 12,
        linewidth=3,
    )

for baseline in baselines:
    show_label = baseline.get("show_in_legend", True)
    ax.axhline(
        y=baseline["y"],
        color=baseline["color"],
        linestyle=baseline["linestyle"],
        label=baseline["label"] if show_label else None,
        linewidth=2,
    )

ax.set_xlabel("Tau (τ)", fontsize=24)
ax.set_ylabel("Utility / Refusal Rate (%)", fontsize=24)
ax.legend(fontsize=20, loc="lower left", bbox_to_anchor=(0.0, 0.05))
ax.tick_params(axis="both", labelsize=18)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("lineplot.png", dpi=600)
