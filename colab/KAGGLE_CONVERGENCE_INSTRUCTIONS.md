# Running CodeT5-Small Convergence Experiment on Kaggle

## Quick Start

### Option 1: Use Kaggle Web UI (Easiest) ⭐ RECOMMENDED

1. **Navigate to Kaggle Kernels**
   - Go to https://www.kaggle.com/code
   - Click "New Notebook" → "Create Notebook"
   - Name: `sharp-llm-icmlde-convergence`

2. **Configure GPU FIRST**
   - Click "⚙️ Settings" (top-right)
   - Select "GPU" as accelerator
   - Save

3. **Copy and Paste This Code** (handles directory cleanup safely)
   - In a notebook cell, paste this complete code block:

```python
# ============================================================================
# CodeT5-Small Convergence Experiment on Kaggle (FIXED)
# ============================================================================
import os
import shutil
import subprocess
import sys

print("=" * 75)
print("KAGGLE CONVERGENCE EXPERIMENT - FIXED VERSION")
print("=" * 75)

# STEP 1: Move to safe directory FIRST (before any deletions)
os.chdir('/kaggle/working')
print(f"\n[1] Safe directory: {os.getcwd()}")

# STEP 2: Remove old repo (now safe, we're not inside it)
print("\n[2] Cleaning up old repo...")
repo_path = '/kaggle/working/repo'
if os.path.exists(repo_path):
    try:
        shutil.rmtree(repo_path)
        print(f"    ✓ Removed stale repo")
    except Exception as e:
        print(f"    ! Warning: {e}")

# STEP 3: Install dependencies
print("\n[3] Installing dependencies...")
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 
                'torch', 'transformers', 'scikit-learn', 'tqdm'])
print("    ✓ Done")

# STEP 4: Clone fresh repository
print("\n[4] Cloning repository...")
result = subprocess.run(['git', 'clone', '--depth', '1',
                        'https://github.com/msbasanth/sharp-llm.git', repo_path],
                       capture_output=True, text=True)
if result.returncode != 0:
    print(f"    ✗ Clone failed: {result.stderr[:200]}")
    raise RuntimeError("Git clone failed")
print(f"    ✓ Cloned")

# STEP 5: Verify files exist
print("\n[5] Verifying repository...")
script = os.path.join(repo_path, 'scripts/convergence_experiment_codet5_small.py')
config = os.path.join(repo_path, 'config.yaml')
if not os.path.exists(script) or not os.path.exists(config):
    raise FileNotFoundError(f"Missing script or config in {repo_path}")
print("    ✓ Files verified")

# STEP 6: Change to repo and run
print("\n[6] Running convergence experiment...")
os.chdir(repo_path)
print("=" * 75)

cmd = [sys.executable, 'scripts/convergence_experiment_codet5_small.py',
       '--config', 'config.yaml', '--model', 'Salesforce/codet5-small',
       '--epochs', '4', '--seed', '50',
       '--output-dir', '/kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small']
result = subprocess.run(cmd)

print("=" * 75)
if result.returncode == 0:
    print("\n✅ CONVERGENCE EXPERIMENT COMPLETED SUCCESSFULLY")
else:
    print(f"\n❌ Failed with exit code {result.returncode}")
```

4. **Run**
   - Click "Run All" (▶️)
   - Wait ~2-3 hours for completion

### Option 2: Use Kaggle CLI (For Automation)

```bash
# 1. Download kernel metadata
kaggle kernels pull your-username/sharp-llm-icmlde-convergence -p ./kernel

# 2. Edit kernel-metadata.json
# Set:
# "accelerator": "gpu",
# "isPrivate": false,
# "enable_gpu": true

# 3. Push to Kaggle
kaggle kernels push -p ./kernel

# 4. Monitor execution
kaggle kernels status your-username/sharp-llm-icmlde-convergence
```

### Option 3: Direct Python Wrapper (If You Have Kaggle API)

```bash
cd d:\Repositories\sharp-llm
python colab/kaggle_convergence_runner.py
```

---

## Expected Output

After ~2-3 hours, you'll find results in:

```
/kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small/
├── train.log                 # Detailed training log
├── epoch_metrics.json        # Per-epoch metrics (JSON)
├── convergence_summary.txt   # Human-readable summary
└── checkpoints/
    └── final.pt             # Model checkpoint
```

### Download Results

In Kaggle notebook, after execution:

```python
import json

# Load and display epoch metrics
with open('/kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small/epoch_metrics.json') as f:
    metrics = json.load(f)
    for m in metrics:
        print(f"Epoch {m['epoch']}: Test F1={m['test_f1']:.4f}, Train Loss={m['train_loss']:.4f}")

# Download files (Kaggle notebook environment)
# Files are automatically available in Output tab
```

---

## Monitoring Progress

### During Execution

Check logs in real-time (in Kaggle notebook):

```python
!tail -20 /kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small/train.log
```

### Expected Training Time

| Phase | Time | Status |
|-------|------|--------|
| Setup + Load data | ~5 min | One-time |
| Epoch 1 (training + val + test) | ~30 min | First epoch slowest |
| Epoch 2 | ~28 min | Slightly faster |
| Epoch 3 | ~27 min | Cache warming |
| Epoch 4 | ~25 min | Optimal speed |
| **Total** | **~110-130 min** | ~2 hours |

---

## Expected Results

### convergence_summary.txt (Preview)

```
===========================================================================
CodeT5-Small Convergence Analysis (Seed 50, 4 Epochs)
===========================================================================

Epoch | Train Loss | Train F1 | Val Loss | Val F1 | Test F1 | Test Acc
------+----------+----------+----------+--------+---------+----------
    1 |   3.2145 |   0.4521 |   3.1892 |  0.4687|  0.4654 |   0.5798
    2 |   1.8432 |   0.7123 |   1.7901 |  0.7245|  0.7189 |   0.8234
    3 |   0.9234 |   0.8765 |   0.9012 |  0.8834|  0.8756 |   0.9123
    4 |   0.5678 |   0.9234 |   0.5812 |  0.9167|  0.9145 |   0.9456

===========================================================================
Convergence Observations:
---
Initial Test F1 (Epoch 1): 0.4654
Final Test F1 (Epoch 4):   0.9145
Improvement: +0.4491
Training Loss Trend: 3.2145 → 0.5678
```

### epoch_metrics.json (Preview)

```json
[
  {
    "epoch": 1,
    "train_loss": 3.2145,
    "train_f1": 0.4521,
    "train_accuracy": 0.5632,
    "val_loss": 3.1892,
    "val_f1": 0.4687,
    "val_accuracy": 0.5801,
    "test_f1": 0.4654,
    "test_accuracy": 0.5798,
    "test_mcc": 0.3421,
    "test_precision": 0.4812,
    "test_recall": 0.4567
  },
  { ... epoch 2, 3, 4 ... }
]
```

---

## Analysis After Completion

### 1. Plot Learning Curves (Kaggle Notebook)

```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('/kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small/epoch_metrics.json') as f:
    metrics = json.load(f)

epochs = [m['epoch'] for m in metrics]
test_f1 = [m['test_f1'] for m in metrics]
train_loss = [m['train_loss'] for m in metrics]
val_f1 = [m['val_f1'] for m in metrics]

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(epochs, test_f1, 'o-', label='Test F1', linewidth=2, markersize=8)
axes[0].plot(epochs, val_f1, 's--', label='Val F1', linewidth=2, markersize=6)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Macro F1')
axes[0].set_title('F1 Score Convergence')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, train_loss, 'o-', color='orange', linewidth=2, markersize=8)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Training Loss')
axes[1].set_title('Loss Decay')
axes[1].grid(True, alpha=0.3)

# Epoch-to-epoch improvement
improvements = [test_f1[i] - test_f1[i-1] for i in range(1, len(test_f1))]
axes[2].bar(epochs[1:], improvements, color='green', alpha=0.7)
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('ΔF1 Improvement')
axes[2].set_title('Per-Epoch Improvement')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/kaggle/working/convergence_plot.png', dpi=150)
plt.show()

print("Plot saved to output")
```

### 2. Compare Against Main Study

```python
import pandas as pd

# Main study results (ICMLDE)
main_study = {
    'Model': 'CodeT5-Small',
    'Final F1': 0.9539,
    'Std Dev': 0.0149,
    'Epochs': 2,  # default config
}

# This convergence study
convergence = {
    'Model': 'CodeT5-Small (Seed 50)',
    'Final F1': metrics[-1]['test_f1'],  # Epoch 4
    'Std Dev': 'N/A (single seed)',
    'Epochs': 4,
}

df = pd.DataFrame([main_study, convergence])
print(df.to_string(index=False))
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Out of Memory (OOM)** | Reduce `--batch-size` to 4 |
| **Kernel timeout (9h limit)** | Should not happen (~2-3h expected) |
| **Low initial F1** | Normal for transformer training (random initialization) |
| **GPU not detected** | Restart kernel, ensure GPU is selected in settings |
| **Import errors** | Ensure all dependencies installed: `pip install torch transformers scikit-learn` |

---

## Next Steps

1. **Run experiment** on Kaggle (2-3 hours)
2. **Download results**:
   - `epoch_metrics.json` (metrics data)
   - `convergence_summary.txt` (human-readable)
   - `train.log` (debug log)
3. **Analyze convergence** using provided plotting code
4. **Add to manuscript**:
   - Figure: Learning curves
   - Table: Per-epoch metrics (Appendix)
   - Discussion: Convergence behavior vs main study

---

## Files

- `colab/kaggle_convergence_runner.py` — Standalone Kaggle setup script
- `scripts/convergence_experiment_codet5_small.py` — Main experiment
- `scripts/CONVERGENCE_EXPERIMENT_README.md` — Detailed documentation

## Questions?

See `scripts/CONVERGENCE_EXPERIMENT_README.md` for full documentation.
