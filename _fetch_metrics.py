"""Download only small evaluation files (no .pt checkpoints) from Kaggle kernel output."""
import json, pathlib

import kaggle.api as kaggle_api

KERNEL = "msbasanth/sharp-llm-icmlde-five-seed-runner"
OUT_DIR = r"d:\Repositories\sharp-llm\_check_now"
PATTERN = r"metrics\.json|classification_report|confusion_pairs|epoch_metrics|train\.log|status\.json|run_manifest"

page_token = None
total_downloaded = 0

print("Downloading small eval files (skipping .pt checkpoints)...")
while True:
    files, next_token = kaggle_api.kernels_output(
        kernel=KERNEL,
        path=OUT_DIR,
        file_pattern=PATTERN,
        force=True,
        quiet=False,
        page_token=page_token,
        page_size=200,
    )
    total_downloaded += len(files)
    print(f"  Page done: {len(files)} files, next_token={next_token!r}")
    if not next_token:
        break
    page_token = next_token

print(f"\nTotal downloaded: {total_downloaded}")

# Summary check
print("\n--- Seed coverage (codet5-base) ---")
base = pathlib.Path(OUT_DIR) / "icmlde2026" / "juliet118" / "juliet118"
for seed in range(42, 47):
    p = base / f"seed_{seed}" / "codet5-base" / "evaluation" / "metrics.json"
    if p.exists():
        m = json.loads(p.read_text())
        print(f"  seed_{seed}: macro_f1={m.get('macro_f1', 'N/A'):.4f}")
    else:
        print(f"  seed_{seed}: MISSING")
