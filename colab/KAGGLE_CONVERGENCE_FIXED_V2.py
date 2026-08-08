# ============================================================================
# FIXED: CodeT5-Small Convergence Experiment on Kaggle
# ============================================================================
# Issue: Shell was inside /kaggle/working/repo when we tried to delete it
# Solution: Change to safe directory FIRST, then delete, then clone

import os
import shutil
import subprocess
import sys

print("=" * 75)
print("KAGGLE CONVERGENCE EXPERIMENT - FIXED VERSION")
print("=" * 75)

# STEP 1: Change to safe directory FIRST (before any deletions)
print("\n[1] Moving to safe directory...")
os.chdir('/kaggle/working')
print(f"    Current dir: {os.getcwd()}")

# STEP 2: Remove old repo (now safe, we're not inside it)
print("\n[2] Cleaning up old repo...")
repo_path = '/kaggle/working/repo'
if os.path.exists(repo_path):
    try:
        shutil.rmtree(repo_path)
        print(f"    ✓ Removed {repo_path}")
    except Exception as e:
        print(f"    ! Warning: Could not remove repo ({e}), continuing anyway...")
else:
    print(f"    (repo not found, skipping)")

# STEP 3: Install dependencies
print("\n[3] Installing dependencies...")
result = subprocess.run(
    [sys.executable, '-m', 'pip', 'install', '-q', 
     'torch', 'transformers', 'scikit-learn', 'tqdm'],
    capture_output=True,
    text=True
)
if result.returncode == 0:
    print("    ✓ Dependencies installed")
else:
    print(f"    ✗ pip install failed: {result.stderr[:200]}")
    raise RuntimeError("Dependency installation failed")

# STEP 4: Clone repository
print("\n[4] Cloning repository...")
clone_result = subprocess.run(
    ['git', 'clone', '--depth', '1', 
     'https://github.com/msbasanth/sharp-llm.git', repo_path],
    capture_output=True,
    text=True
)
if clone_result.returncode == 0:
    print(f"    ✓ Cloned to {repo_path}")
else:
    print(f"    ✗ Clone failed: {clone_result.stderr[:300]}")
    raise RuntimeError("Git clone failed")

# STEP 5: Verify directory structure
print("\n[5] Verifying repository structure...")
if not os.path.isdir(repo_path):
    raise RuntimeError(f"Clone failed: {repo_path} does not exist")

script_path = os.path.join(repo_path, 'scripts', 'convergence_experiment_codet5_small.py')
if not os.path.exists(script_path):
    print(f"    ✗ Script not found: {script_path}")
    print(f"    Available scripts: {os.listdir(os.path.join(repo_path, 'scripts'))}")
    raise FileNotFoundError(f"Convergence script missing: {script_path}")
else:
    print(f"    ✓ Found convergence script")

config_path = os.path.join(repo_path, 'config.yaml')
if not os.path.exists(config_path):
    print(f"    ✗ Config not found: {config_path}")
    raise FileNotFoundError(f"Config missing: {config_path}")
else:
    print(f"    ✓ Found config.yaml")

# STEP 6: Change to repo directory (now safe, repo definitely exists)
print("\n[6] Entering repository...")
os.chdir(repo_path)
print(f"    Current dir: {os.getcwd()}")

# STEP 7: Run convergence experiment
print("\n[7] Starting convergence experiment...")
print("=" * 75)

cmd = [
    sys.executable, 
    'scripts/convergence_experiment_codet5_small.py',
    '--config', 'config.yaml',
    '--model', 'Salesforce/codet5-small',
    '--epochs', '4',
    '--seed', '50',
    '--output-dir', '/kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small'
]

print(f"Command: {' '.join(cmd)}\n")
result = subprocess.run(cmd)

print("=" * 75)
if result.returncode == 0:
    print("\n✅ CONVERGENCE EXPERIMENT COMPLETED SUCCESSFULLY")
    print("\nOutputs saved to: /kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small/")
    print("  - epoch_metrics.json (metrics)")
    print("  - convergence_summary.txt (human-readable)")
    print("  - train.log (training details)")
    print("  - checkpoints/final.pt (model weights)")
else:
    print(f"\n❌ CONVERGENCE EXPERIMENT FAILED (Exit code: {result.returncode})")
    print("Check output above for error details")
    sys.exit(1)
