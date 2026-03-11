import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# -----------------------------
# Data (from your table)
# -----------------------------
data = {
    "Method": [
        "YOLO11n", "YOLO11n", "YOLO11n",
        "YOLOv8n", "YOLOv8n", "YOLOv8n",
        "Xiong et al. (2021)", "Xiong et al. (2021)", "Xiong et al. (2021)",
        "Our Method", "Our Method", "Our Method"
    ],
    "Metric": [
        "Precision", "Recall", "F1",
        "Precision", "Recall", "F1",
        "Precision", "Recall", "F1",
        "Precision", "Recall", "F1"
    ],
    "Score": [
        0.896, 0.959, 0.927,     # YOLO11n
        0.919, 0.958, 0.938,     # YOLOv8n
        0.916, 0.891, 0.904,     # Xiong et al.
        0.955, 0.938, 0.946      # Our Method
    ]
}

df = pd.DataFrame(data)

# -----------------------------
# Plot styling (AAAI-safe)
# -----------------------------
sns.set_theme(style="whitegrid", font_scale=1.4)

palette = {
    "YOLO11n": "#9ca3af",
    "YOLOv8n": "#6b7280",
    "Xiong et al. (2021)": "#374151",
    "Our Method": "#1d4ed8"   # highlighted
}

plt.figure(figsize=(10, 6))

ax = sns.barplot(
    data=df,
    x="Metric",
    y="Score",
    hue="Method",
    palette=palette
)

# -----------------------------
# Value labels (bold for Our Method)
# -----------------------------
for container, method in zip(ax.containers, df["Method"].unique()):
    for bar in container:
        height = bar.get_height()
        ax.annotate(
            f"{height:.3f}",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold" if method == "Our Method" else "normal"
        )

# -----------------------------
# Axis & legend formatting
# -----------------------------
ax.set_ylabel("Score")
ax.set_xlabel("")
ax.set_ylim(0.85, 1.0)

ax.legend(
    title="Method",
    frameon=False,
    loc="upper left"
)

sns.despine()
plt.tight_layout()

plt.show()
