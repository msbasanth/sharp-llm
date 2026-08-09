import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Load both versions
with open("outputs/icmlde2026/convergence/seed_50/codet5-small/epoch_metrics.json") as f:
    v1_metrics = json.load(f)
    
with open("outputs/icmlde2026/convergence/seed_50/codet5-small/v2/outputs/epoch_metrics.json") as f:
    v2_metrics = json.load(f)

# Extract data
v1_epochs = [x["epoch"] for x in v1_metrics]
v1_train_f1 = [x["train_f1"] for x in v1_metrics]
v1_test_f1 = [x["test_f1"] for x in v1_metrics]
v1_test_loss = [x["val_loss"] for x in v1_metrics]
v1_test_mcc = [x["test_mcc"] for x in v1_metrics]

v2_epochs = [x["epoch"] for x in v2_metrics]
v2_train_f1 = [x["train_f1"] for x in v2_metrics]
v2_test_f1 = [x["test_f1"] for x in v2_metrics]
v2_test_loss = [x["val_loss"] for x in v2_metrics]
v2_test_mcc = [x["test_mcc"] for x in v2_metrics]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
fig.suptitle("CodeT5-Small Convergence Study: V1 (4 Epochs) vs V2 (8 Epochs) | Seed 50", 
             fontsize=13, fontweight="bold", y=1.02)

# --- SUBPLOT 1: Loss ---
ax = axes[0]
ax.plot(v1_epochs, v1_test_loss, "o-", color="steelblue", linewidth=2.5, markersize=6, label="V1 (4 ep)")
ax.plot(v2_epochs, v2_test_loss, "s--", color="darkorange", linewidth=2.5, markersize=5, label="V2 (8 ep)")
ax.set_title("Validation Loss", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss (CE)")
ax.set_xticks(range(1, 9))
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")

# --- SUBPLOT 2: Test F1 ---
ax = axes[1]
ax.plot(v1_epochs, v1_test_f1, "o-", color="steelblue", linewidth=2.5, markersize=6, label="V1 (4 ep) - Test F1")
ax.plot(v2_epochs, v2_test_f1, "s--", color="darkorange", linewidth=2.5, markersize=5, label="V2 (8 ep) - Test F1")
ax.axhline(y=0.9983, color="steelblue", linestyle=":", alpha=0.5, linewidth=1.5, label="V1 Final (0.9983)")
ax.axhline(y=0.9990, color="darkorange", linestyle=":", alpha=0.5, linewidth=1.5, label="V2 Final (0.9990)")
ax.set_title("Test F1 Score", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("Macro F1")
ax.set_ylim(0.90, 1.005)
ax.set_xticks(range(1, 9))
ax.legend(fontsize=9)
ax.grid(alpha=0.3, linestyle="--")

# --- SUBPLOT 3: MCC ---
ax = axes[2]
ax.plot(v1_epochs, v1_test_mcc, "^-", color="seagreen", linewidth=2.5, markersize=6, label="V1 (4 ep)")
ax.plot(v2_epochs, v2_test_mcc, "D--", color="crimson", linewidth=2.5, markersize=5, label="V2 (8 ep)")
ax.set_title("Matthews Correlation Coefficient", fontweight="bold")
ax.set_xlabel("Epoch")
ax.set_ylabel("MCC")
ax.set_ylim(0.9970, 1.0002)
ax.set_xticks(range(1, 9))
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle="--")

plt.tight_layout()
out = "outputs/icmlde2026/convergence/seed_50/codet5-small/convergence_v1_vs_v2.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved: {out}")
