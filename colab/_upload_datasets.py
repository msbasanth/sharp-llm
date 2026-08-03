"""Create or update the two Kaggle datasets needed for the ICMLDE runner."""
import json, shutil, sys
from pathlib import Path

# Change to repo root regardless of where this is called from
ROOT = Path(__file__).parent.parent
import os; os.chdir(ROOT)

from kaggle import KaggleApi
api = KaggleApi()
api.authenticate()

# ── Dataset 1: processed data ─────────────────────────────────────────────────
d1 = Path("kaggle-data-upload")
d1.mkdir(exist_ok=True)
shutil.copy("data/processed/train.parquet",  d1)
shutil.copy("data/processed/test.parquet",   d1)
shutil.copy("data/processed/label_map.json", d1)
(d1 / "dataset-metadata.json").write_text(json.dumps({
    "title": "sharp-llm-processed-data",
    "id":    "msbasanth/sharp-llm-processed-data",
    "licenses": [{"name": "other"}]
}, indent=2))

print("Uploading sharp-llm-processed-data …")
try:
    api.dataset_create_new(str(d1), dir_mode="zip", quiet=False, convert_to_csv=False)
    print("✓ Created sharp-llm-processed-data")
except Exception as e:
    msg = str(e).lower()
    if "already exists" in msg or "409" in msg or "already been created" in msg:
        print("  Dataset exists — pushing new version …")
        api.dataset_create_version(str(d1), "Initial upload", dir_mode="zip",
                                   quiet=False, convert_to_csv=False)
        print("✓ Versioned sharp-llm-processed-data")
    else:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

# ── Dataset 2: outputs placeholder ───────────────────────────────────────────
d2 = Path("kaggle-outputs-upload")
d2.mkdir(exist_ok=True)
(d2 / "placeholder.json").write_text("{}")
(d2 / "dataset-metadata.json").write_text(json.dumps({
    "title": "sharp-llm-icmlde-outputs",
    "id":    "msbasanth/sharp-llm-icmlde-outputs",
    "licenses": [{"name": "other"}]
}, indent=2))

print("Uploading sharp-llm-icmlde-outputs placeholder …")
try:
    api.dataset_create_new(str(d2), dir_mode="zip", quiet=False, convert_to_csv=False)
    print("✓ Created sharp-llm-icmlde-outputs")
except Exception as e:
    msg = str(e).lower()
    if "already exists" in msg or "409" in msg or "already been created" in msg:
        print("✓ sharp-llm-icmlde-outputs already exists — skipping")
    else:
        print(f"✗ Error: {e}", file=sys.stderr)
        sys.exit(1)

# Cleanup
shutil.rmtree("kaggle-data-upload",    ignore_errors=True)
shutil.rmtree("kaggle-outputs-upload", ignore_errors=True)
print("\nBoth datasets ready on Kaggle.")
