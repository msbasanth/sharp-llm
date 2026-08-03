"""Patch Cell 2 of kaggle_icmlde_runner.ipynb to wrap nvidia-smi in try/except."""
import json
from pathlib import Path

path = Path(__file__).parent / "kaggle_icmlde_runner.ipynb"
nb = json.loads(path.read_text(encoding="utf-8"))

new_source = """\
# ── Cell 2 · GPU check ────────────────────────────────────────────────────────
import subprocess, torch

# nvidia-smi is informational only — torch CUDA is the authoritative check
try:
    r = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
    if r.returncode == 0:
        print(r.stdout)
    else:
        print("nvidia-smi returned non-zero (GPU may still be available via torch)")
except (FileNotFoundError, OSError) as e:
    print(f"nvidia-smi not in PATH ({e}) — checking torch CUDA directly ...")

if not torch.cuda.is_available():
    raise RuntimeError(
        "No GPU detected by torch.\\n"
        "On Kaggle: Settings (right panel) -> Accelerator -> GPU T4 x1, then Save & Run All."
    )

p = torch.cuda.get_device_properties(0)
print(f"GPU  : {p.name}")
print(f"VRAM : {p.total_memory / 1e9:.1f} GB")
print(f"CUDA : {torch.version.cuda}")
"""

# Replace first code cell
code_idx = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
nb["cells"][code_idx]["source"] = new_source

path.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")

# Verify
check = "".join(nb["cells"][code_idx]["source"])
assert "FileNotFoundError" in check, "Patch not applied!"
assert "torch.cuda.is_available" in check, "torch check missing!"
print(f"Patched {path}")
print("First 200 chars:", check[:200])
