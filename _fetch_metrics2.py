"""
Download seeds 45 & 46 codet5-base eval files immediately after kernel completion.
Seeds 42-43 already exist in _check_now (will be skipped by download_needed).
Seed 44 eval was recovered locally. Only 45 & 46 are needed.
"""
import json, pathlib, subprocess, sys

KERNEL = "msbasanth/sharp-llm-icmlde-five-seed-runner"
CHECK_NOW = pathlib.Path(r"d:\Repositories\sharp-llm\_check_now")
PYTHON = sys.executable

print("Downloading with --file-pattern (small files only, no checkpoints)...")
result = subprocess.run([
    PYTHON, "-m", "kaggle", "kernels", "output",
    KERNEL,
    "-p", str(CHECK_NOW),
    "--file-pattern", r"metrics\.json|classification_report\.txt|confusion_pairs\.csv|epoch_metrics\.json|train\.log|status\.json|run_manifest\.json",
    "--force",
], capture_output=False)
print(f"Exit code: {result.returncode}")

# Report coverage
print("\n--- codet5-base seed coverage ---")
base = CHECK_NOW / "icmlde2026" / "juliet118" / "juliet118"
for seed in range(42, 47):
    p = base / f"seed_{seed}" / "codet5-base" / "evaluation" / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        print(f"  seed_{seed}: OK  macro_f1={m.get('macro_f1', 'N/A'):.4f}")
    else:
        print(f"  seed_{seed}: MISSING")

