import json
import pandas as pd

# Load both versions
with open("outputs/icmlde2026/convergence/seed_50/codet5-small/epoch_metrics.json") as f:
    v1_metrics = json.load(f)
    
with open("outputs/icmlde2026/convergence/seed_50/codet5-small/v2/outputs/epoch_metrics.json") as f:
    v2_metrics = json.load(f)

print("=" * 80)
print("KAGGLE RUN V2 CONVERGENCE REPORT (8 Epochs)")
print("=" * 80)
print()
print(f"V1: {len(v1_metrics)} epochs | V2: {len(v2_metrics)} epochs")
print()

print("V2 FULL CONVERGENCE TRAJECTORY:")
print("-" * 80)
print(f"{'Epoch':<6} {'Train F1':<12} {'Test F1':<12} {'Train Loss':<14} {'Val Loss':<14} {'MCC':<10}")
print("-" * 80)
for ep in v2_metrics:
    print(f"{ep['epoch']:<6.0f} {ep['train_f1']:<12.4f} {ep['test_f1']:<12.4f} {ep['train_loss']:<14.4f} {ep['val_loss']:<14.4f} {ep['test_mcc']:<10.4f}")
print("-" * 80)
print()

# Compare key milestones
print("PERFORMANCE COMPARISON:")
print("-" * 80)
print(f"{'Metric':<25} {'Epoch 1':<15} {'Epoch 4 (V1)':<15} {'Epoch 8 (V2)':<15}")
print("-" * 80)

v1_ep4 = v1_metrics[3]  # 4th epoch in v1
v2_ep8 = v2_metrics[7]  # 8th epoch in v2
v2_ep1 = v2_metrics[0]  # 1st epoch in v2 (same as v1 ep1)

print(f"{'Test F1':<25} {v2_ep1['test_f1']:<15.4f} {v1_ep4['test_f1']:<15.4f} {v2_ep8['test_f1']:<15.4f}")
print(f"{'Test Accuracy':<25} {v2_ep1['test_accuracy']:<15.4f} {v1_ep4['test_accuracy']:<15.4f} {v2_ep8['test_accuracy']:<15.4f}")
print(f"{'Test MCC':<25} {v2_ep1['test_mcc']:<15.4f} {v1_ep4['test_mcc']:<15.4f} {v2_ep8['test_mcc']:<15.4f}")
print(f"{'Train F1':<25} {v2_ep1['train_f1']:<15.4f} {v1_ep4['train_f1']:<15.4f} {v2_ep8['train_f1']:<15.4f}")
print("-" * 80)
print()

# Compute improvements
print("CONVERGENCE IMPROVEMENTS (V1 Ep4 -> V2 Ep8):")
print("-" * 80)
f1_gain = (v2_ep8['test_f1'] - v1_ep4['test_f1']) * 100
acc_gain = (v2_ep8['test_accuracy'] - v1_ep4['test_accuracy']) * 100
print(f"Test F1 Gain: {f1_gain:+.2f}% ({v1_ep4['test_f1']:.4f} -> {v2_ep8['test_f1']:.4f})")
print(f"Test Accuracy Gain: {acc_gain:+.2f}% ({v1_ep4['test_accuracy']:.4f} -> {v2_ep8['test_accuracy']:.4f})")
print()

# Check if converged (plateau detection)
print("CONVERGENCE PLATEAU ANALYSIS:")
print("-" * 80)
recent_f1 = [v2_metrics[i]['test_f1'] for i in range(max(0, len(v2_metrics)-3), len(v2_metrics))]
f1_variance = max(recent_f1) - min(recent_f1)
print(f"Last 3 epochs F1 values: {[f'{x:.4f}' for x in recent_f1]}")
print(f"F1 Variance (last 3 epochs): {f1_variance:.6f}")
if f1_variance < 0.001:
    print("Status: CONVERGED (plateau detected)")
else:
    print("Status: STILL IMPROVING (consider more epochs if needed)")
print()
