# CodeT5-Small Convergence Experiment for ICMLDE 2026

## Overview

This experiment trains CodeT5-Small on the Juliet 118-CWE dataset for **4 epochs** with a **new random seed (50)** to understand training convergence behavior and how model performance improves across iterations.

## Motivation

The main ICMLDE study used 5 seeds (42-46) with typical configurations. This convergence study:
- Uses a fresh seed (50) to avoid bias from previous runs
- Tracks **per-epoch metrics** to visualize learning curves
- Enables understanding of early stopping behavior
- Demonstrates model training stability and convergence speed

## Configuration

| Parameter | Value | Notes |
|-----------|-------|-------|
| **Model** | Salesforce/codet5-small | 60M parameters, encoder-decoder |
| **Seed** | 50 | NEW seed, different from main study (42-46) |
| **Epochs** | 4 | Number of training iterations |
| **Batch Size** | 8 | Standard, limited by T4 VRAM |
| **Learning Rate** | 5e-5 | Standard for fine-tuning transformers |
| **Dataset** | Juliet 118-CWE | ~2,800 test samples for evaluation |

## Running Locally

```bash
cd d:\Repositories\sharp-llm

# Option 1: Use convenience script
python scripts/run_convergence_experiment.py

# Option 2: Run directly
python scripts/convergence_experiment_codet5_small.py \
  --config config.yaml \
  --model Salesforce/codet5-small \
  --epochs 4 \
  --seed 50 \
  --output-dir outputs/icmlde2026/convergence/seed_50/codet5-small
```

## Expected Runtime

- **~2 hours** on NVIDIA T4 GPU (Kaggle free tier)
- **~30-45 minutes** per epoch on T4
- Can be run locally on consumer GPU (~1 hour per epoch)

## Output Files

```
outputs/icmlde2026/convergence/seed_50/codet5-small/
├── train.log                    # Detailed training log
├── epoch_metrics.json          # Per-epoch F1, accuracy, loss (machine-readable)
├── convergence_summary.txt     # Human-readable convergence table
└── checkpoints/
    └── final.pt                # Final model weights (PyTorch checkpoint)
```

### epoch_metrics.json Format

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
    "test_mcc": 0.4210,
    "test_precision": 0.4821,
    "test_recall": 0.4512
  },
  { "epoch": 2, ... },
  { "epoch": 3, ... },
  { "epoch": 4, ... }
]
```

## Analysis

After running, analyze convergence:

```python
import json
import matplotlib.pyplot as plt

# Load metrics
with open('outputs/icmlde2026/convergence/seed_50/codet5-small/epoch_metrics.json') as f:
    metrics = json.load(f)

epochs = [m['epoch'] for m in metrics]
test_f1 = [m['test_f1'] for m in metrics]
train_loss = [m['train_loss'] for m in metrics]

# Plot convergence
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(epochs, test_f1, marker='o', label='Test F1')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Macro-F1')
ax1.set_title('CodeT5-Small Convergence (Seed 50)')
ax1.grid(True)

ax2.plot(epochs, train_loss, marker='s', color='orange', label='Train Loss')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Loss')
ax2.set_title('Training Loss Decay')
ax2.grid(True)

plt.tight_layout()
plt.savefig('convergence_plot.png', dpi=150)
plt.show()
```

## Key Metrics to Track

1. **Test F1 Improvement**: From Epoch 1 → Epoch 4
   - Shows how quickly model converges to reasonable performance
   - Compare against baseline ICMLDE runs

2. **Train Loss Decay**: Should decrease monotonically
   - Steep initial drop (Epoch 1-2) indicates fast learning
   - Flatter tail (Epoch 3-4) shows asymptotic behavior

3. **Val/Test Gap**: Difference between validation and test F1
   - Small gap indicates good generalization
   - Large gap might suggest overfitting

4. **Epoch-to-Epoch Improvement**: ΔF1 per epoch
   - Diminishing returns indicate convergence
   - Can inform optimal stopping point

## Comparison with Main Study

After convergence experiment completes:

```
ICMLDE Main Study (Seed 42-46, default epochs):
  CodeT5-Small: F1 = 0.9539 ± 0.0149

This Convergence Study (Seed 50, 4 epochs):
  Epoch 1: F1 = ?
  Epoch 2: F1 = ?
  Epoch 3: F1 = ?
  Epoch 4: F1 = ?
```

## Integration with Manuscript

Include convergence analysis in paper as:
- **Figure:** Learning curves (F1 vs epoch, loss vs epoch)
- **Table:** Per-epoch metrics (Appendix)
- **Discussion:** Comment on convergence speed and early stopping efficacy

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Out of memory (OOM) | Reduce batch_size to 4 |
| Slow training | Verify GPU usage with `nvidia-smi` |
| Low initial F1 | This is normal; transformer training has slow start |

## Files

- `scripts/convergence_experiment_codet5_small.py` — Main experiment
- `scripts/run_convergence_experiment.py` — Quick-start wrapper
- `scripts/convergence_experiment_codet5_small.md` — This file

## Next Steps

1. Run the experiment (local or Kaggle)
2. Analyze `epoch_metrics.json` and `convergence_summary.txt`
3. Generate convergence plots
4. Add findings to manuscript discussion/appendix
5. Compare against main study results
