# Fix for Kaggle Convergence Experiment

## Error Message
```
fatal: destination path '/kaggle/working/repo' already exists and is not an empty directory.
python3: can't open file '/kaggle/working/repo/scripts/convergence_experiment_codet5_small.py': [Errno 2] No such file or directory
```

## Solution

### Option 1: Use Fresh Clone (Recommended)

Replace the existing code in your Kaggle notebook with:

```python
# Clean up old repo if it exists
import shutil
import os

if os.path.exists('/kaggle/working/repo'):
    shutil.rmtree('/kaggle/working/repo')
    print("✓ Removed old repo")

# Install dependencies
!pip install -q torch transformers scikit-learn tqdm

# Clone fresh repository
!git clone https://github.com/msbasanth/sharp-llm.git /kaggle/working/repo
%cd /kaggle/working/repo

# Verify script exists
!ls -la scripts/convergence_experiment_codet5_small.py

# Run convergence experiment
!python scripts/convergence_experiment_codet5_small.py \
  --config config.yaml \
  --model Salesforce/codet5-small \
  --epochs 4 --seed 50 \
  --output-dir /kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small
```

### Option 2: Update Existing Clone (Faster)

If you want to keep the existing repo and just update it:

```python
import os

# Update existing repo
os.chdir('/kaggle/working/repo')
!git pull origin master

# Verify script exists
!ls -la scripts/convergence_experiment_codet5_small.py

# Run convergence experiment
!python scripts/convergence_experiment_codet5_small.py \
  --config config.yaml \
  --model Salesforce/codet5-small \
  --epochs 4 --seed 50 \
  --output-dir /kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small
```

### Option 3: Use Kaggle Datasets (If Repo in Dataset)

If sharp-llm is uploaded as a Kaggle dataset:

```python
import os
import shutil

# Copy dataset to working directory
os.chdir('/kaggle/working')

# Assuming dataset is mounted at /kaggle/input/sharp-llm
if os.path.exists('/kaggle/input/sharp-llm'):
    if os.path.exists('/kaggle/working/repo'):
        shutil.rmtree('/kaggle/working/repo')
    shutil.copytree('/kaggle/input/sharp-llm', '/kaggle/working/repo')
    print("✓ Copied dataset to working directory")

os.chdir('/kaggle/working/repo')

# Install dependencies
!pip install -q torch transformers scikit-learn tqdm

# Verify script exists
!ls -la scripts/convergence_experiment_codet5_small.py

# Run convergence experiment
!python scripts/convergence_experiment_codet5_small.py \
  --config config.yaml \
  --model Salesforce/codet5-small \
  --epochs 4 --seed 50 \
  --output-dir /kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small
```

---

## What Each Option Does

| Option | When to Use | Speed | Fresh Code |
|--------|-----------|-------|-----------|
| **Option 1** | Kaggle kernel is stale/needs latest code | Slow (full clone) | ✅ Always fresh |
| **Option 2** | Repo exists and just needs update | Fast (incremental) | ✅ If you `git pull` |
| **Option 3** | Using Kaggle dataset upload | Very fast (copy) | ✅ If dataset updated |

---

## Debug Checklist

After running the fixed code:

1. **Verify script exists:**
   ```python
   import os
   path = '/kaggle/working/repo/scripts/convergence_experiment_codet5_small.py'
   print(f"Script exists: {os.path.exists(path)}")
   ```

2. **Check repository structure:**
   ```python
   !ls -la /kaggle/working/repo/scripts/
   ```

3. **Verify dependencies:**
   ```python
   import torch
   import transformers
   print(f"PyTorch: {torch.__version__}")
   print(f"Transformers: {transformers.__version__}")
   ```

---

## Recommended: Start Fresh with Option 1

Copy this complete cell and paste into your Kaggle notebook:

```python
# ============================================================================
# CodeT5-Small Convergence Experiment (ICMLDE 2026)
# ============================================================================

import shutil
import os
import subprocess

# 1. Clean up old repo
if os.path.exists('/kaggle/working/repo'):
    shutil.rmtree('/kaggle/working/repo')
    print("✓ Removed stale repo")

# 2. Install dependencies
print("\n[1] Installing dependencies...")
subprocess.run(['pip', 'install', '-q', 'torch', 'transformers', 'scikit-learn', 'tqdm'])
print("✓ Dependencies installed")

# 3. Clone fresh repository
print("\n[2] Cloning repository...")
os.chdir('/kaggle/working')
subprocess.run(['git', 'clone', 'https://github.com/msbasanth/sharp-llm.git', 'repo'])
os.chdir('/kaggle/working/repo')
print("✓ Repository cloned")

# 4. Verify script
print("\n[3] Verifying convergence script...")
script_path = 'scripts/convergence_experiment_codet5_small.py'
if os.path.exists(script_path):
    print(f"✓ Found: {script_path}")
else:
    print(f"✗ NOT FOUND: {script_path}")
    print("Available scripts:", os.listdir('scripts/'))

# 5. Run convergence experiment
print("\n[4] Starting convergence experiment...")
print("=" * 75)
cmd = [
    'python', 'scripts/convergence_experiment_codet5_small.py',
    '--config', 'config.yaml',
    '--model', 'Salesforce/codet5-small',
    '--epochs', '4',
    '--seed', '50',
    '--output-dir', '/kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small'
]
result = subprocess.run(cmd)
print("=" * 75)

if result.returncode == 0:
    print("\n✅ CONVERGENCE EXPERIMENT COMPLETED SUCCESSFULLY")
else:
    print(f"\n❌ EXPERIMENT FAILED (Exit code: {result.returncode})")
```

---

## After Experiment Completes

Download results from Kaggle Output tab:
- `outputs/icmlde2026/convergence/seed_50/codet5-small/epoch_metrics.json`
- `outputs/icmlde2026/convergence/seed_50/codet5-small/convergence_summary.txt`
- `outputs/icmlde2026/convergence/seed_50/codet5-small/train.log`
