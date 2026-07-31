# SHARP-LLM

A framework for source code vulnerability detection, CWE classification, and **healthcare-specific risk prioritization** using deep learning models. Supports multi-model comparison across CodeT5, CodeBERT, GraphCodeBERT, CodeGemma (QLoRA), and BiLSTM-Attention architectures, with an integrated Healthcare Software Vulnerability Scoring System (HSVSS).

## Current Status

| Component | Status |
|---|---|
| Vulnerability detection pipeline (5 models) | ✅ Complete |
| 118-CWE classification (macro-F1 0.96) | ✅ Complete |
| Template-aware anti-leakage splitting | ✅ Complete |
| Healthcare risk prioritization (`src/risk.py`) | ✅ Complete (5-weight baseline) |
| HVSS ML models (retrained for sklearn 1.8.0) | ✅ Complete |
| Streamlit web UI | ✅ Running |
| HSVSS 8-dimensional scoring engine | 🔜 Phase 2 (next) |
| CVSS vs HSVSS comparison study | 🔜 Phase 3 |
| Journal paper manuscript | 🔜 Phase 4 |

**Active development:** Transitioning from conference paper (vulnerability detection focus) to journal paper (healthcare-specific risk prioritization with HSVSS as the core contribution). See `context/journal-paper/the-plan.md` for the full roadmap.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Requirements:** Python 3.14+, NVIDIA GPU with CUDA support (8 GB VRAM sufficient).

## Configuration

All hyperparameters, model paths, and experiment settings are defined in `config.yaml`.

## Supported Models

| Model | Type | Parameters |
|---|---|---|
| CodeT5-Small | Encoder | 60M |
| CodeT5-Base | Encoder | 220M |
| CodeBERT-Base | Encoder (RoBERTa) | 125M |
| GraphCodeBERT-Base | Graph-aware Encoder | 125M |
| CodeGemma-2B (QLoRA) | Decoder + LoRA (4-bit) | 2B |
| BiLSTM-Attention | Custom PyTorch | Lightweight |

## Data Pipeline

1. **Extract** — Parse Juliet Test Suite C/C++ samples into parquet:
   ```bash
   python -m src.data.extract_dataset
   ```
2. **Preprocess** — Strip comments, headers, and normalize whitespace:
   ```bash
   python -m src.data.preprocess
   ```
3. **Split** — Template-aware 80:20 train/test split (prevents structural data leakage):
   ```bash
   python -m src.data.split
   ```

## Training

```bash
python -m src.train --config config.yaml --experiment exp_a_juliet118 --model Salesforce/codet5-small --epochs 2 --patience 1
```

QLoRA fine-tuning (CodeGemma-2B):
```bash
python -m src.train_qlora --config config.yaml
```

Pre-configured experiment scripts:
```powershell
.\run_train_exp_a.ps1   # Juliet-118 baseline
.\run_train_exp_f.ps1   # Union dataset, 6 epochs
.\run_train_exp_g.ps1   # BiLSTM-Attention
```

## Evaluation

```bash
python -m src.evaluate --config config.yaml --checkpoint outputs/checkpoints/codet5-small/best.pt --experiment exp_a_juliet118 --model Salesforce/codet5-small
```

Outputs: `metrics.json`, `classification_report.txt`, `confusion_pairs.csv`.

`metrics.json` includes overall macro/weighted metrics plus `macro_f1_across_cwe_mean`,
`macro_f1_across_cwe_std`, and `macro_f1_class_count` so Macro-F1 can be reported as
variation across CWE classes (for example, `0.84 ± 0.07`) without retraining multiple seeds.

## Prediction

```bash
# Single file
python -m src.predict path/to/code.c

# Directory scan with top-K results
python -m src.predict path/to/directory/ --top-k 3
```

## Web UI

```bash
streamlit run src/app.py
```

Interactive Streamlit interface for uploading C/C++ code, selecting models, viewing CWE predictions with confidence scores, and healthcare risk assessment.

## Experiments

| Exp | Dataset | Models | Epochs |
|---|---|---|---|
| A | Juliet (118 CWEs) | 4 encoders | 2 |
| B | Juliet (19 CWE subset) | 4 encoders | 2 |
| C | Juliet + Big-Vul (overlap) | 4 encoders | 2 |
| D | Big-Vul only | 4 encoders | 2 |
| E | Union (187 CWEs) | 4 encoders | 2 |
| F | Union (187 CWEs) | 4 encoders | 6 |
| G | Juliet | BiLSTM-Attention | 10 |

## Healthcare Risk Prioritization

### Current: 3-Stage Baseline (`src/risk.py`)

1. **CWE → LINDDUN** — Maps vulnerabilities to privacy threat categories
2. **LINDDUN → Control Domains** — Maps threats to healthcare policy domains (access control, audit, confidentiality)
3. **Risk Scoring** — Weighted composite score based on severity, exploitability, detectability, scope, and compliance

### Upcoming: HSVSS 8-Dimensional Scoring Engine

$$\text{HSVSS}(v) = \sum_{i=1}^{8} w_i \cdot D_i(v)$$

| Dimension | Source |
|---|---|
| D1: Technical Severity | CVSS base score / CWE severity lookup |
| D2: Exploitability Likelihood | HVSS `exploitability_model.pkl` (R²=0.967) |
| D3: Patient Safety Impact | HVSS `xps_model.pkl` (R²=0.882) |
| D4: PHI/PII Exposure Impact | LINDDUN + HVSS `xsd_model.pkl` (R²=0.804) |
| D5: Clinical Workflow Disruption | CWE→workflow mapping (new) |
| D6: Clinical Data Integrity | LINDDUN + integrity analysis |
| D7: Interoperability Impact | CWE→API/integration mapping (new) |
| D8: Regulatory Compliance Impact | HIPAA/GDPR/FDA mapping |

### HVSS ML Models

Five retrained models (StandardScaler + MLP(256,128)) at `src/insights/hvss-calculator-lab-main/models/`:

| Model | Features | R² Score |
|---|---|---|
| Exploitability | 4 | 0.967 |
| Patient Safety (XPS) | 5 | 0.882 |
| Sensitive Data (XSD) | 5 | 0.804 |
| Hospital Breach (XHB) | 5 | 0.958 |
| CIA Impact (XCIA) | 7 | 0.218 |

Retrain with: `python scripts/retrain_hvss_models.py`

## Project Structure

```
src/
├── app.py              # Streamlit web UI
├── model.py            # CWEClassifier, CWEBiLSTM, QLoRA loading
├── train.py            # Training loop (transformer models)
├── train_qlora.py      # QLoRA fine-tuning
├── evaluate.py         # Metrics and error analysis
├── predict.py          # CLI inference
├── risk.py             # Healthcare risk prioritization (3-stage baseline)
├── utils.py            # Seed, config, device, logging helpers
├── insights/           # HVSS calculator + retrained ML models
└── data/
    ├── extract_dataset.py   # Juliet → parquet
    ├── extract_bigvul.py    # Big-Vul CSV → parquet
    ├── preprocess.py        # Code normalization
    ├── split.py             # Template-aware splitting
    ├── dataset.py           # PyTorch Dataset / DataLoaders
    └── merge_datasets.py    # Combine datasets
scripts/
├── retrain_hvss_models.py   # Reproducible HVSS model retraining
context/
└── journal-paper/
    └── the-plan.md          # Full HSVSS implementation roadmap
```

## Roadmap

| Phase | Timeline | Deliverable |
|---|---|---|
| Phase 1: Reframing | Week 1–2 | HSVSS formal definition, architecture design |
| Phase 2: Implementation | Week 3–5 | `hsvss.py`, `hvss_adapter.py`, `compliance.py`, mappings |
| Phase 3: Evaluation | Week 6–9 | CVSS vs HSVSS comparison, ablation, case studies |
| Phase 4: Writing | Week 8–12 | Full journal manuscript (~20 pages) |
| Phase 5: Submission | Week 12–14 | Camera-ready, reproducibility package |

Target journals: *Computers & Security* (IF 5.4), *Int. J. Critical Infrastructure Protection* (IF 5.3), *Journal of Medical Systems* (IF 5.7).
