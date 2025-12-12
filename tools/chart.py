import matplotlib.pyplot as plt
import seaborn as sns

# ======================================================
# Accuracy Results (put your final values here)
# ======================================================
methods = ["DeepHunter V1 (Ensemble)", "DeepHunter V2 (ProtoNet Few-Shot)"]
accuracies = [68.42, 94.74]  # update automatically if needed

# ======================================================
# Modern Plot Style
# ======================================================
sns.set_theme(style="whitegrid", font_scale=1.4)

plt.figure(figsize=(10, 6))
bar_palette = sns.color_palette("viridis", len(methods))

ax = sns.barplot(
    x=methods,
    y=accuracies,
    palette=bar_palette,
    width=0.6
)

# ======================================================
# Labels & Aesthetics
# ======================================================
plt.title("DeepHunter Experiment Accuracy Comparison", fontsize=20, weight="bold")
plt.ylabel("Accuracy (%)", fontsize=16)
plt.ylim(0, 110)

# Add value labels on bars
for i, acc in enumerate(accuracies):
    ax.text(
        i,
        acc + 2,
        f"{acc:.2f}%",
        ha="center",
        fontsize=16,
        weight="bold"
    )

# Adjust label rotation for readability
plt.xticks(rotation=10)

# Tight layout
plt.tight_layout()

# Save the figure (optional)
plt.savefig("deephunter_v1_v2_accuracy.png", dpi=300)

# Show plot
plt.show()
