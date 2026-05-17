# SHARP-LLM

A framework for source code vulnerability detection and CWE classification using deep learning models. Supports multi-model comparison across CodeT5, CodeBERT, GraphCodeBERT, CodeGemma (QLoRA), and BiLSTM-Attention architectures.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Requirements:** Python 3.9+, NVIDIA GPU with CUDA support.

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

Interactive Streamlit interface for uploading C/C++ code, selecting models, and viewing CWE predictions with confidence scores.

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

## Healthcare Risk Pipeline

`src/risk.py` implements a 3-stage prioritization pipeline:

1. **CWE → LINDDUN** — Maps vulnerabilities to privacy threat categories
2. **LINDDUN → Control Domains** — Maps threats to healthcare policy domains (access control, audit, confidentiality)
3. **Risk Scoring** — Weighted composite score based on severity, exploitability, detectability, scope, and compliance

## Project Structure

```
src/
├── app.py              # Streamlit web UI
├── model.py            # CWEClassifier, CWEBiLSTM, QLoRA loading
├── train.py            # Training loop (transformer models)
├── train_qlora.py      # QLoRA fine-tuning
├── evaluate.py         # Metrics and error analysis
├── predict.py          # CLI inference
├── risk.py             # Healthcare risk prioritization
├── utils.py            # Seed, config, device, logging helpers
└── data/
    ├── extract_dataset.py   # Juliet → parquet
    ├── extract_bigvul.py    # Big-Vul CSV → parquet
    ├── preprocess.py        # Code normalization
    ├── split.py             # Template-aware splitting
    ├── dataset.py           # PyTorch Dataset / DataLoaders
    └── merge_datasets.py    # Combine datasets
```
