import json

# Load v2 metrics
with open("outputs/icmlde2026/convergence/seed_50/codet5-small/v2/outputs/epoch_metrics.json") as f:
    v2_metrics = json.load(f)

print("=== V2 EPOCH METRICS ===")
for ep in v2_metrics:
    print(f"Epoch {ep[\"epoch\"]}: Train F1={ep[\"train_f1\"]:.4f}, Val F1={ep[\"val_f1\"]:.4f}, Test F1={ep[\"test_f1\"]:.4f}, MCC={ep[\"test_mcc\"]:.4f}")

print("\n=== V2 SUMMARY ===")
# Get final performance
final = v2_metrics[-1]
print(f"Final Test F1: {final[\"test_f1\"]:.4f}")
print(f"Final Test Accuracy: {final[\"test_accuracy\"]:.4f}")
print(f"Final Test MCC: {final[\"test_mcc\"]:.4f}")
