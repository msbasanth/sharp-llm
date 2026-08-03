# setup_kaggle.ps1
# Run once from the repo root to create Kaggle datasets and push the kernel.
# Usage:  .\colab\setup_kaggle.ps1

Set-Location $PSScriptRoot\..

$python = "D:\Repositories\sharp-llm\.venv\Scripts\python.exe"
$kaggle = "D:\Repositories\sharp-llm\.venv\Scripts\kaggle.exe"
if (-not (Test-Path $kaggle)) {
    Write-Error "kaggle.exe not found — run: pip install kaggle"
    exit 1
}
Set-Alias -Name kaggle -Value $kaggle -Scope Script

# ── Step 1 + 2: Create datasets via Python API (avoids Windows CLI path bug) ─
Write-Host "`n[1/5] Creating Kaggle datasets via Python API …" -ForegroundColor Cyan

& $python "colab/_upload_datasets.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "Dataset creation failed — check error above."
    exit 1
}
Write-Host "  ✓ Both datasets ready on Kaggle." -ForegroundColor Green

# ── Step 3: Update kernel-metadata.json with dataset sources ─────────────────
Write-Host "`n[3/5] Updating kernel-metadata.json with dataset sources …" -ForegroundColor Cyan

$meta = Get-Content "colab/kernel-metadata.json" | ConvertFrom-Json
$meta.dataset_sources = @(
    "msbasanth/sharp-llm-processed-data",
    "msbasanth/sharp-llm-icmlde-outputs"
)
$meta | ConvertTo-Json -Depth 5 | Set-Content "colab/kernel-metadata.json"
Write-Host "  kernel-metadata.json updated." -ForegroundColor Green

# ── Step 4: Push kernel ───────────────────────────────────────────────────────
Write-Host "`n[4/5] Pushing kernel …" -ForegroundColor Cyan
kaggle kernels push -p colab/
if ($LASTEXITCODE -ne 0) {
    Write-Error "Kernel push failed — check error above."
    exit 1
}

# ── Step 5: Poll status ───────────────────────────────────────────────────────
Write-Host "`n[5/5] Polling kernel status (Ctrl+C to stop polling) …" -ForegroundColor Cyan
$slug = "msbasanth/sharp-llm-icmlde-five-seed-runner"

do {
    Start-Sleep -Seconds 30
    $statusOut = & $kaggle kernels status $slug 2>&1
    Write-Host "  $(Get-Date -Format 'HH:mm:ss')  $statusOut"
    $done = $statusOut -match "complete|error|cancel"
} while (-not $done)

Write-Host "`nKernel finished. Download outputs with:" -ForegroundColor Green
Write-Host "  kaggle kernels output $slug -p outputs/icmlde2026/" -ForegroundColor White

# ── Cleanup temp dirs ─────────────────────────────────────────────────────────
Remove-Item -Recurse -Force $dataDir  -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force $outDir   -ErrorAction SilentlyContinue
