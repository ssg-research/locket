import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerBase
from matplotlib.patches import Rectangle

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
Y_MAX = 1

# Custom color palette
COLOR_LINEAR = "#386795"
COLOR_CAT = "#73bcd5"
COLOR_TIES = "#abdce0"
COLOR_DARE = "#ffd06e"
COLOR_MAGNITUDE = "#fee6b5"
COLOR_OURS = "#e86254"

x_categories = ["Math", "Code", "Summarize", "MMLU"]
y_linear = [0.08, 0.00, 0.24, 0.03]
y_cat = [0.00, 0.00, 0.04, 0.00]
y_ties = [0.42, 0.93, 0.23, 0.49]
y_dare_linear = [0.42, 0.93, 0.24, 0.52]
y_magnitude_prune = [0.08, 0.00, 0.24, 0.06]
y_ours = [0.45, 0.94, 0.24, 0.53]

plt.rcParams["font.family"] = SERIF_FONT

x = np.arange(len(x_categories))

fig, ax = plt.subplots(figsize=FIG_SIZE)

rects1 = ax.bar(
    x - 2.5 * BAR_WIDTH, y_linear, BAR_WIDTH, label="Linear", color=COLOR_LINEAR
)
rects2 = ax.bar(x - 1.5 * BAR_WIDTH, y_cat, BAR_WIDTH, label="Cat", color=COLOR_CAT)
rects3 = ax.bar(x - 0.5 * BAR_WIDTH, y_ties, BAR_WIDTH, label="TIES", color=COLOR_TIES)
rects4 = ax.bar(
    x + 0.5 * BAR_WIDTH, y_dare_linear, BAR_WIDTH, label="DARE", color=COLOR_DARE
)
rects5 = ax.bar(
    x + 1.5 * BAR_WIDTH,
    y_magnitude_prune,
    BAR_WIDTH,
    label="Magnitude",
    color=COLOR_MAGNITUDE,
)
rects6 = ax.bar(
    x + 2.5 * BAR_WIDTH,
    y_ours,
    BAR_WIDTH,
    label="Ours",
    color=COLOR_OURS,
    linewidth=2,
    edgecolor="black",
)

ax.set_ylabel(
    "Accuracy",
    fontsize=AXIS_FONT_SIZE,
    family=SERIF_FONT,
)
ax.set_xlabel("Unlocked Feature", fontsize=AXIS_FONT_SIZE, family=SERIF_FONT)
ax.set_xticks(x)
ax.set_xticklabels(x_categories, fontsize=TICKS_FONT_SIZE, family=SERIF_FONT)
plt.yticks(fontsize=TICKS_FONT_SIZE, family=SERIF_FONT)
ax.set_ylim(Y_MIN, Y_MAX)


def autolabel(rects, bold=False):
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
            fontweight=BOLD_FONT_WEIGHT if bold else "normal",
        )


autolabel(rects1)
autolabel(rects2)
autolabel(rects3)
autolabel(rects4)
autolabel(rects5)
autolabel(rects6, bold=True)


class BoldLegendHandler(HandlerBase):
    def create_artists(
        self, legend, orig_handle, x0, y0, width, height, fontsize, trans
    ):
        patch = Rectangle(
            [x0, y0],
            width,
            height,
            facecolor=orig_handle.get_facecolor()[0],
            edgecolor="black",
            lw=2,
        )
        return [patch]


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
    handler_map={"Ours": BoldLegendHandler()},
)

# Bold the "Ours" label in the legend
for text, label in zip(legend.get_texts(), custom_labels):
    if label == "Ours":
        text.set_fontweight(BOLD_FONT_WEIGHT)

fig.tight_layout()
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["bottom"].set_visible(True)
ax.spines["left"].set_visible(True)
ax.yaxis.grid(True, linestyle="--", which="major", color="grey", alpha=0.7)

plt.savefig(f"{PROJECT_DIR}/figs/bar_merging_methods.png", dpi=600)
