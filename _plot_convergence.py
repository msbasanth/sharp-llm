import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Load V2 metrics
with open("outputs/icmlde2026/convergence/seed_50/codet5-small/v2/outputs/epoch_metrics.json") as f:
    v2_metrics = json.load(f)

# Extract data
epochs = [x["epoch"] for x in v2_metrics]
train_loss = [x["train_loss"] for x in v2_metrics]
val_loss = [x["val_loss"] for x in v2_metrics]
train_f1 = [x["train_f1"] for x in v2_metrics]
test_f1 = [x["test_f1"] for x in v2_metrics]
test_mcc = [x["test_mcc"] for x in v2_metrics]
test_acc = [x["test_accuracy"] for x in v2_metrics]

# Create figure with 2x2 subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle("CodeT5-Small Convergence Analysis (8 Epochs, Seed 50)", 
             fontsize=14, fontweight="bold", y=0.995)

# --- TOP LEFT: Loss ---
ax = axes[0, 0]
ax.plot(epochs, train_loss, "o-", color="steelblue", linewidth=2.5, markersize=7, label="Train Loss")
ax.plot(epochs, val_loss, "s--", color="darkorange", linewidth=2.5, markersize=6, label="Val Loss")
ax.set_title("Training & Validation Loss", fontweight="bold", fontsize=11)
ax.set_xlabel("Epoch")
ax.set_ylabel("Cross-Entropy Loss")
ax.set_xticks(epochs)
ax.legend(loc="upper right", fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
# Add annotations
for e, tl, vl in zip(epochs, train_loss, val_loss):
    ax.annotate(f"{tl:.4f}", (e, tl), textcoords="offset points", xytext=(-15, 8), fontsize=7, color="steelblue")
    ax.annotate(f"{vl:.4f}", (e, vl), textcoords="offset points", xytext=(3, -12), fontsize=7, color="darkorange")

# --- TOP RIGHT: F1 Scores ---
ax = axes[0, 1]
ax.plot(epochs, train_f1, "o-", color="steelblue", linewidth=2.5, markersize=7, label="Train F1")
ax.plot(epochs, test_f1, "s--", color="crimson", linewidth=2.5, markersize=6, label="Test F1")
ax.set_title("F1 Score Evolution", fontweight="bold", fontsize=11)
ax.set_xlabel("Epoch")
ax.set_ylabel("Macro F1")
ax.set_xticks(epochs)
ax.set_ylim(0.70, 1.005)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3, linestyle="--")
# Add annotations for key epochs
for i in [0, 3, 7]:  # Epochs 1, 4, 8
    ax.annotate(f"{train_f1[i]:.4f}", (epochs[i], train_f1[i]), textcoords="offset points", 
                xytext=(-15, 8), fontsize=7, color="steelblue")
    ax.annotate(f"{test_f1[i]:.4f}", (epochs[i], test_f1[i]), textcoords="offset points", 
                xytext=(3, -12), fontsize=7, color="crimson")

# --- BOTTOM LEFT: Test Metrics (Accuracy & MCC) ---
ax = axes[1, 0]
ax.plot(epochs, test_acc, "^-", color="green", linewidth=2.5, markersize=7, label="Test Accuracy")
ax.plot(epochs, test_mcc, "D--", color="purple", linewidth=2.5, markersize=6, label="Test MCC")
ax.set_title("Test Set Metrics", fontweight="bold", fontsize=11)
ax.set_xlabel("Epoch")
ax.set_ylabel("Score")
ax.set_xticks(epochs)
ax.set_ylim(0.9970, 1.0002)
ax.legend(loc="lower right", fontsize=10)
ax.grid(alpha=0.3, linestyle="--")

# --- BOTTOM RIGHT: Convergence Table ---
ax = axes[1, 1]
ax.axis("off")

# Create summary table
summary_data = [
    ["Metric", "Epoch 1", "Epoch 4", "Epoch 8", "Improvement"],
    ["Test F1", f"{test_f1[0]:.4f}", f"{test_f1[3]:.4f}", f"{test_f1[7]:.4f}", 
     f"+{(test_f1[7]-test_f1[0])*100:.2f}%"],
    ["Test Acc", f"{test_acc[0]:.4f}", f"{test_acc[3]:.4f}", f"{test_acc[7]:.4f}", 
     f"+{(test_acc[7]-test_acc[0])*100:.2f}%"],
    ["Test MCC", f"{test_mcc[0]:.4f}", f"{test_mcc[3]:.4f}", f"{test_mcc[7]:.4f}", 
     f"+{(test_mcc[7]-test_mcc[0])*100:.2f}%"],
    ["Train F1", f"{train_f1[0]:.4f}", f"{train_f1[3]:.4f}", f"{train_f1[7]:.4f}", 
     f"+{(train_f1[7]-train_f1[0])*100:.2f}%"],
    ["Train Loss", f"{train_loss[0]:.4f}", f"{train_loss[3]:.4f}", f"{train_loss[7]:.4f}", 
     f"{(train_loss[7]-train_loss[0])*100:.2f}%"],
]

table = ax.table(cellText=summary_data, cellLoc="center", loc="center",
                colWidths=[0.20, 0.16, 0.16, 0.16, 0.20])
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 2.2)

# Style header row
for i in range(5):
    table[(0, i)].set_facecolor("#40466e")
    table[(0, i)].set_text_props(weight="bold", color="white")

# Color alternate rows
for i in range(1, len(summary_data)):
    for j in range(5):
        if i % 2 == 0:
            table[(i, j)].set_facecolor("#f0f0f0")
        else:
            table[(i, j)].set_facecolor("white")

ax.text(0.5, 1.08, "Convergence Summary", ha="center", fontsize=11, fontweight="bold",
        transform=ax.transAxes)

plt.tight_layout()
out = "outputs/icmlde2026/convergence/seed_50/codet5-small/convergence_detailed.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
