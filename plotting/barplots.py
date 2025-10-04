import matplotlib.pyplot as plt
import numpy as np

# Data for the plot (you can replace these with your actual data)
bins = ["Math", "SQL", "Summarize", "MMLU"]
linear = [1.0, 1.00, 1.00, 1.00]
cat = [1.00, 1.00, 1.00, 1.00]
ties = [0.75, 0.30, 0.04, 0.94]
ours = [0.03, 0.01, 0.01, 0.08]

# Create a bar plot
x = np.arange(len(bins))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots(figsize=(9, 6))

# Colors for the trait types (match the style of the image)
color_linear = "#FFCC99"  # light peach
color_cat = "#A6B3B7"  # light gray
color_ties = "#8EC6E8"  # light blue
color_ours = "#8B5E3C"  # brownish

# Plotting each trait type with black border around the bars
rects1 = ax.bar(
    x - 1.5 * width,
    linear,
    width,
    label="Linear",
    color=color_linear,
    edgecolor="black",
)
rects2 = ax.bar(
    x - 0.5 * width,
    cat,
    width,
    label="CAT",
    color=color_cat,
    edgecolor="black",
)
rects3 = ax.bar(
    x + 0.5 * width,
    ties,
    width,
    label="TIES",
    color=color_ties,
    edgecolor="black",
)
rects4 = ax.bar(
    x + 1.5 * width,
    ours,
    width,
    label="LOCKET",
    color=color_ours,
    edgecolor="black",
)

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_xlabel("Unlocked Feature", fontsize=24)  # Increase font size of x-axis label
ax.set_ylabel(
    "Over-Refusal Rate (%)", fontsize=24
)  # Increase font size of y-axis label
# ax.set_title('Trait Type Distribution by Confidence Score Bin (7 sessions)', fontsize=22)  # Increase font size of title
ax.set_xticks(x)
ax.set_xticklabels(bins, fontsize=18)  # Increase font size of x-axis tick labels
ax.legend(
    fontsize=12, loc="upper right", bbox_to_anchor=(1.0, 0.854)
)  # Increase font size of legend

# Increase font size of ticks on both axes
ax.tick_params(axis="both", labelsize=18)  # Increase font size of ticks

# Display the plot
plt.tight_layout()
plt.savefig("barplot.png", dpi=600)
