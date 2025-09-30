import matplotlib.pyplot as plt
import numpy as np

from locket.config import PROJECT_DIR

SERIF_FONT = "serif"
BOLD_FONT_WEIGHT = "bold"
AXIS_FONT_SIZE = 16
TICKS_FONT_SIZE = 12
LABEL_FONT_SIZE = 10
LEGEND_FONT_SIZE = 12
FIG_SIZE = (10, 6)
BAR_WIDTH = 0.1
Y_MIN = 0
Y_MAX = 200

# Custom color palette
COLOR_LINEAR = "#386795"
COLOR_CAT = "#73bcd5"
COLOR_TIES = "#abdce0"
COLOR_DARE = "#ffd06e"
COLOR_MAGNITUDE = "#fee6b5"
COLOR_OURS = "#e86254"

x_categories = ["DeepSeek-7B-Math", "Llama-3-8B"]
y_m = [39, 42]
y_c = [43, 30]
y_s = [32, 44]
y_u = [40, 29]
y_mcsu = [148, 143]
y_m_c_s_u = [153, 145]

plt.rcParams["font.family"] = SERIF_FONT

x = np.arange(len(x_categories))

fig, ax = plt.subplots(figsize=FIG_SIZE)

rects1 = ax.bar(
    x - 2.5 * BAR_WIDTH, y_m, BAR_WIDTH, label="Math (M)", color=COLOR_LINEAR
)
rects2 = ax.bar(x - 1.5 * BAR_WIDTH, y_c, BAR_WIDTH, label="Code (C)", color=COLOR_CAT)
rects3 = ax.bar(
    x - 0.5 * BAR_WIDTH, y_s, BAR_WIDTH, label="Summarize (S)", color=COLOR_TIES
)
rects4 = ax.bar(x + 0.5 * BAR_WIDTH, y_u, BAR_WIDTH, label="MMLU (U)", color=COLOR_DARE)
rects5 = ax.bar(
    x + 1.5 * BAR_WIDTH,
    y_mcsu,
    BAR_WIDTH,
    label="M, C, S, U",
    color=COLOR_MAGNITUDE,
)
rects6 = ax.bar(
    x + 2.5 * BAR_WIDTH,
    y_m_c_s_u,
    BAR_WIDTH,
    label="M + C + S + U",
    color=COLOR_OURS,
)

ax.set_ylabel(
    "# of Successful Attacks (out of 1000 samples)",
    fontsize=AXIS_FONT_SIZE,
    family=SERIF_FONT,
)
ax.set_xticks(x)
ax.set_xticklabels(x_categories, fontsize=AXIS_FONT_SIZE, family=SERIF_FONT)
plt.yticks(fontsize=TICKS_FONT_SIZE, family=SERIF_FONT)
ax.set_ylim(Y_MIN, Y_MAX)


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(
            f"{height}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            rotation=60,
            fontsize=LABEL_FONT_SIZE,
            family=SERIF_FONT,
            fontweight="normal",
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6)


handles, labels = ax.get_legend_handles_labels()
custom_handles = []
custom_labels = []
for h, label in zip(handles, labels):
    if label == "Ours":
        custom_handles.append(h)
        custom_labels.append(label)
    else:
        custom_handles.append(h)
        custom_labels.append(label)

legend = ax.legend(
    custom_handles,
    custom_labels,
    loc="upper left",
    fontsize=LEGEND_FONT_SIZE,
    frameon=True,
    shadow=True,
    prop={"family": SERIF_FONT},
)

fig.tight_layout()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.7)

plt.savefig(f"{PROJECT_DIR}/figs/bar_multi_robustness.png", dpi=600)
