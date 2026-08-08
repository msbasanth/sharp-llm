#!/usr/bin/env python3
"""
Quick start: Run CodeT5-Small Convergence Experiment

This runs the convergence experiment locally with 4 epochs, seed 50 (new).
Tracks how F1 score, loss, and accuracy improve over iterations.
"""

import subprocess
import sys
from pathlib import Path

def run():
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / "outputs" / "icmlde2026" / "convergence" / "seed_50" / "codet5-small"
    
    cmd = [
        sys.executable,
        str(repo_root / "scripts" / "convergence_experiment_codet5_small.py"),
        "--config", str(repo_root / "config.yaml"),
        "--model", "Salesforce/codet5-small",
        "--epochs", "4",
        "--seed", "50",
        "--batch-size", "8",
        "--learning-rate", "5e-5",
        "--output-dir", str(output_dir),
    ]
    
    print("Running CodeT5-Small Convergence Experiment")
    print(f"Output: {output_dir}")
    print()
    
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    run()
