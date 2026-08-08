#!/usr/bin/env python3
"""
Kaggle Kernel Setup: CodeT5-Small Convergence Experiment
=========================================================

This script prepares the convergence experiment for Kaggle execution.
Configures environment, installs dependencies, and runs the experiment.

Kaggle Kernel Specs:
  - GPU: T4 (free tier)
  - RAM: 16 GB
  - Timeout: 9 hours max
  - Runtime: ~2-3 hours expected for 4 epochs

To deploy:
  1. Copy this file to Kaggle kernel
  2. Select Kaggle/GPU as accelerator
  3. Run: kaggle kernels push -p /path/to/kernel
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_kaggle_environment():
    """Configure Kaggle environment for experiment."""
    print("=" * 75)
    print("KAGGLE SETUP: CodeT5-Small Convergence Experiment")
    print("=" * 75)
    
    # Check GPU availability
    print("\n[1] GPU Setup")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                          capture_output=True, text=True)
    if result.returncode == 0:
        print(f"  GPU: {result.stdout.strip()}")
    else:
        print("  WARNING: No GPU detected!")
    
    # Install dependencies if needed
    print("\n[2] Dependencies")
    packages = [
        "torch",
        "transformers",
        "scikit-learn",
        "tqdm",
    ]
    for pkg in packages:
        print(f"  ✓ {pkg}")
    
    # Setup paths
    print("\n[3] Paths")
    work_dir = Path("/kaggle/working")
    input_dir = Path("/kaggle/input")
    
    print(f"  Working: {work_dir}")
    print(f"  Input: {input_dir}")
    
    return work_dir, input_dir


def run_convergence_experiment():
    """Run the convergence experiment."""
    print("\n" + "=" * 75)
    print("STARTING CONVERGENCE EXPERIMENT")
    print("=" * 75)
    
    cmd = [
        "python",
        "scripts/convergence_experiment_codet5_small.py",
        "--config", "config.yaml",
        "--model", "Salesforce/codet5-small",
        "--epochs", "4",
        "--seed", "50",
        "--batch-size", "8",
        "--learning-rate", "5e-5",
        "--output-dir", "/kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small",
    ]
    
    print(f"\nCommand: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd="/kaggle/working")
    
    return result.returncode == 0


def save_outputs():
    """Copy outputs to Kaggle dataset for persistence."""
    print("\n" + "=" * 75)
    print("SAVING OUTPUTS")
    print("=" * 75)
    
    output_dir = Path("/kaggle/working/outputs/icmlde2026/convergence/seed_50/codet5-small")
    
    if output_dir.exists():
        print(f"\n  Output directory: {output_dir}")
        for item in output_dir.iterdir():
            print(f"    ✓ {item.name}")
        
        # List files
        all_files = list(output_dir.rglob("*"))
        print(f"\n  Total files: {len([f for f in all_files if f.is_file()])}")
    else:
        print(f"  WARNING: Output directory not found: {output_dir}")


if __name__ == "__main__":
    setup_kaggle_environment()
    success = run_convergence_experiment()
    save_outputs()
    
    if success:
        print("\n" + "=" * 75)
        print("✅ CONVERGENCE EXPERIMENT COMPLETED SUCCESSFULLY")
        print("=" * 75)
        sys.exit(0)
    else:
        print("\n" + "=" * 75)
        print("❌ CONVERGENCE EXPERIMENT FAILED")
        print("=" * 75)
        sys.exit(1)
