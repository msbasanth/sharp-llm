"""Compare all trained models with comprehensive metrics.

Metrics: Accuracy, Precision, Recall, F1-score, MCC, FPR (all macro-averaged).
Also computes parameter count, checkpoint size, and inference latency.
Saves results to outputs/model_comparison.json for the Streamlit UI.
"""

import json
import gc
import os
import sys
import time

import truststore
truststore.inject_into_ssl()

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from src.data.dataset import get_dataloaders
from src.gemma_predict import load_zero_shot_model, predict_code_zero_shot, _is_instruction_tuned
from src.model import CWEClassifier, CWEBiLSTM, load_qlora_for_inference
from src.utils import load_config, get_device, load_label_map, count_parameters


SAMPLE_CODE = """void bad() {
    char *ptr = (char *)malloc(10);
    free(ptr);
    printf("%s", ptr);
}"""


def predict_all(model, dataloader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
            all_preds.extend(logits.argmax(dim=-1).cpu().tolist())
            all_labels.extend(batch["label"].tolist())
    return all_preds, all_labels


def macro_fpr(y_true, y_pred, num_classes):
    """Compute macro-averaged False Positive Rate."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fprs = []
    for i in range(num_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        if fp + tn > 0:
            fprs.append(fp / (fp + tn))
        else:
            fprs.append(0.0)
    return np.mean(fprs)


MODELS = [
    ("CodeT5-Small", "Salesforce/codet5-small"),
    ("CodeT5-Base", "Salesforce/codet5-base"),
    ("CodeBERT-Base", "microsoft/codebert-base"),
    ("GraphCodeBERT-Base", "microsoft/graphcodebert-base"),
]

# Zero-shot models (inference-only, no checkpoint required).
# Evaluated on a random sample of the test set for practical speed.
# Set N_ZERO_SHOT_SAMPLES=None to evaluate on the full test set (very slow).
# Instruction-tuned models use fewer samples due to much higher latency.
ZERO_SHOT_MODELS = [
    ("CodeGemma-2B (Zero-shot)", "google/codegemma-2b"),
    ("Gemma 4 E2B IT (Zero-shot)", "google/gemma-4-E2B-it"),
]
# Parameter and model-size metadata for zero-shot models.
# Values are used in UI summaries because these models are inference-only.
ZERO_SHOT_MODEL_META = {
    "google/codegemma-2b": {"Parameters": 2510000000, "Size (MB)": 4677.0},
    "google/gemma-4-E2B-it": {"Parameters": 5130000000, "Size (MB)": 9561.0},
}
N_ZERO_SHOT_SAMPLES = 500  # samples per zero-shot model

config = load_config("config.yaml")
device = get_device()
num_classes = config["num_classes"]

# Load test data once with a common tokenizer, then reload per model
# We need to reload per model because tokenizers differ
results = []

for display_name, model_id in MODELS:
    variant = model_id.split("/")[-1]
    ckpt_path = os.path.join(config["checkpoint_dir"], variant, "best.pt")
    latest_ckpt_path = os.path.join(config["checkpoint_dir"], variant, "latest.pt")

    if not os.path.exists(ckpt_path):
        print(f"\n{display_name}: checkpoint not found at {ckpt_path}, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating: {display_name} ({model_id})")
    print(f"{'='*60}")

    # Override model name for tokenizer/dataloader
    eval_config = {**config, "model_name": model_id}

    # Load model
    model = CWEClassifier(model_name=model_id, num_classes=num_classes, dropout=0.0)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    # Parameter count and checkpoint size
    params = count_parameters(model)
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    best_epoch = ckpt.get("epoch", "N/A")
    # Read total training time from latest checkpoint (covers all epochs)
    if os.path.exists(latest_ckpt_path):
        latest_ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        training_time_s = latest_ckpt.get("training_time", None)
        del latest_ckpt
    else:
        training_time_s = ckpt.get("training_time", None)

    # Load test data with this model's tokenizer
    _, test_loader, _ = get_dataloaders(eval_config)

    # Measure inference latency (average over multiple runs)
    from transformers import AutoTokenizer as _AT
    _tok = _AT.from_pretrained(model_id)
    _enc = _tok(SAMPLE_CODE, max_length=config["max_length"],
                padding="max_length", truncation=True, return_tensors="pt")
    _ids = _enc["input_ids"].to(device)
    _mask = _enc["attention_mask"].to(device)
    # Warm up
    with torch.no_grad():
        for _ in range(5):
            model(_ids, _mask)
    # Timed runs
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            model(_ids, _mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
    avg_latency_ms = np.mean(latencies) * 1000
    del _tok, _enc, _ids, _mask

    # Predict
    y_pred, y_true = predict_all(model, test_loader, device)

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr = macro_fpr(y_true, y_pred, num_classes)

    results.append({
        "Model": display_name,
        "model_id": model_id,
        "Parameters": params["total"],
        "Size (MB)": round(ckpt_size_mb, 1),
        "Best Epoch": best_epoch,
        "Training Time (s)": round(training_time_s, 1) if training_time_s else "N/A",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "MCC": mcc,
        "FPR": fpr,
        "Latency (ms)": round(avg_latency_ms, 2),
    })

    # Free GPU memory
    del model
    torch.cuda.empty_cache()

# ── QLoRA fine-tuned models ─────────────────────────────────────────────────
QLORA_MODELS = [
    ("CodeGemma-2B (QLoRA)", "google/codegemma-1.1-2b"),
]

for display_name, model_id in QLORA_MODELS:
    variant = model_id.split("/")[-1] + "-qlora"
    adapter_path = os.path.join(config["checkpoint_dir"], variant, "best_adapter")
    state_path = os.path.join(config["checkpoint_dir"], variant, "best.pt")

    if not os.path.exists(adapter_path):
        print(f"\n{display_name}: adapter not found at {adapter_path}, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating (QLoRA): {display_name} ({model_id})")
    print(f"{'='*60}")

    qlora_model, qlora_tokenizer = load_qlora_for_inference(
        model_id=model_id,
        num_classes=num_classes,
        adapter_path=adapter_path,
    )

    # Load training state
    qlora_ckpt = torch.load(state_path, map_location="cpu", weights_only=False) if os.path.exists(state_path) else {}
    best_epoch = qlora_ckpt.get("epoch", "N/A")
    training_time_s = qlora_ckpt.get("training_time", None)

    # Parameter count (trainable via PEFT)
    params = count_parameters(qlora_model)

    # Build test dataloader with QLoRA tokenizer
    from src.data.dataset import CWEDataset
    from torch.utils.data import DataLoader
    import pandas as pd

    _test_df_qlora = pd.read_parquet(config["test_path"])
    _test_dataset = CWEDataset(
        codes=_test_df_qlora["code"].tolist(),
        labels=_test_df_qlora["label"].tolist(),
        tokenizer=qlora_tokenizer,
        max_length=config["max_length"],
    )
    qlora_test_loader = DataLoader(
        _test_dataset,
        batch_size=config.get("qlora", {}).get("per_device_batch_size", 2),
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    # Predict
    def predict_all_qlora(model, dataloader, device):
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="QLoRA eval", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                all_preds.extend(outputs.logits.argmax(dim=-1).cpu().tolist())
                all_labels.extend(batch["label"].tolist())
        return all_preds, all_labels

    y_pred, y_true = predict_all_qlora(qlora_model, qlora_test_loader, device)

    # Latency measurement
    _enc = qlora_tokenizer(SAMPLE_CODE, max_length=config["max_length"],
                           padding="max_length", truncation=True, return_tensors="pt")
    _ids = _enc["input_ids"].to(device)
    _mask = _enc["attention_mask"].to(device)
    with torch.no_grad():
        for _ in range(3):
            qlora_model(input_ids=_ids, attention_mask=_mask)
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            qlora_model(input_ids=_ids, attention_mask=_mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
    avg_latency_ms = np.mean(latencies) * 1000
    del _enc, _ids, _mask

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr = macro_fpr(y_true, y_pred, num_classes)

    # Adapter size
    adapter_size_mb = sum(
        os.path.getsize(os.path.join(adapter_path, f))
        for f in os.listdir(adapter_path)
        if os.path.isfile(os.path.join(adapter_path, f))
    ) / (1024 * 1024)

    results.append({
        "Model": display_name,
        "model_id": model_id,
        "qlora": True,
        "Parameters": params["total"],
        "Size (MB)": round(adapter_size_mb, 1),
        "Best Epoch": best_epoch,
        "Training Time (s)": round(training_time_s, 1) if training_time_s else "N/A",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "MCC": mcc,
        "FPR": fpr,
        "Latency (ms)": round(avg_latency_ms, 2),
    })

    del qlora_model, qlora_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# ── DL (non-transformer) models ─────────────────────────────────────────────
DL_MODELS = [
    ("BiLSTM-Attention", "bilstm-attention", "outputs/exp_g_bilstm/checkpoints"),
]

for display_name, model_id, dl_ckpt_dir in DL_MODELS:
    variant = model_id
    ckpt_path = os.path.join(dl_ckpt_dir, variant, "best.pt")
    latest_ckpt_path = os.path.join(dl_ckpt_dir, variant, "latest.pt")

    if not os.path.exists(ckpt_path):
        print(f"\n{display_name}: checkpoint not found at {ckpt_path}, skipping.")
        continue

    print(f"\n{'='*60}")
    print(f"Evaluating (DL): {display_name} ({model_id})")
    print(f"{'='*60}")

    dl_config = config.get("dl", {})
    tokenizer_name = dl_config.get("tokenizer_name", "Salesforce/codet5-small")
    from transformers import AutoTokenizer as _AT_DL
    dl_tokenizer = _AT_DL.from_pretrained(tokenizer_name)

    # Load model
    dl_model = CWEBiLSTM(
        vocab_size=dl_tokenizer.vocab_size,
        num_classes=num_classes,
        embedding_dim=dl_config.get("embedding_dim", 128),
        hidden_dim=dl_config.get("hidden_dim", 256),
        num_layers=dl_config.get("num_layers", 2),
        dropout=0.0,
        pad_idx=dl_tokenizer.pad_token_id or 0,
    )
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    dl_model.load_state_dict(ckpt["model_state_dict"])
    dl_model.to(device)

    params = count_parameters(dl_model)
    ckpt_size_mb = os.path.getsize(ckpt_path) / (1024 * 1024)
    best_epoch = ckpt.get("epoch", "N/A")
    if os.path.exists(latest_ckpt_path):
        latest_ckpt = torch.load(latest_ckpt_path, map_location="cpu", weights_only=False)
        training_time_s = latest_ckpt.get("training_time", None)
        del latest_ckpt
    else:
        training_time_s = ckpt.get("training_time", None)

    # Load test data with DL tokenizer
    eval_config = {**config, "model_name": model_id}
    _, dl_test_loader, _ = get_dataloaders(eval_config)

    # Latency measurement
    _enc = dl_tokenizer(SAMPLE_CODE, max_length=config["max_length"],
                        padding="max_length", truncation=True, return_tensors="pt")
    _ids = _enc["input_ids"].to(device)
    _mask = _enc["attention_mask"].to(device)
    with torch.no_grad():
        for _ in range(5):
            dl_model(_ids, _mask)
    latencies = []
    with torch.no_grad():
        for _ in range(20):
            t0 = time.perf_counter()
            dl_model(_ids, _mask)
            if device.type == "cuda":
                torch.cuda.synchronize()
            latencies.append(time.perf_counter() - t0)
    avg_latency_ms = np.mean(latencies) * 1000
    del _enc, _ids, _mask

    # Predict
    y_pred, y_true = predict_all(dl_model, dl_test_loader, device)

    # Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr = macro_fpr(y_true, y_pred, num_classes)

    results.append({
        "Model": display_name,
        "model_id": model_id,
        "dl_model": True,
        "Parameters": params["total"],
        "Size (MB)": round(ckpt_size_mb, 1),
        "Best Epoch": best_epoch,
        "Training Time (s)": round(training_time_s, 1) if training_time_s else "N/A",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "MCC": mcc,
        "FPR": fpr,
        "Latency (ms)": round(avg_latency_ms, 2),
    })

    del dl_model
    torch.cuda.empty_cache()

# ── Zero-shot models ────────────────────────────────────────────────────────
from huggingface_hub import scan_cache_dir as _scan_cache
_cached_ids = {r.repo_id for r in _scan_cache().repos}

label_map = load_label_map(config["label_map_path"])
# Inverse: class index → CWE string (for decoding parquet labels)
idx_to_cwe = {v: k for k, v in label_map.items()}

# Load test parquet for zero-shot evaluation (raw code + labels)
_test_df_full = pd.read_parquet(config["test_path"], columns=["code", "cwe_id"])
N_IT_SAMPLES = 50  # fewer samples for slow instruction-tuned models

for display_name, model_id in ZERO_SHOT_MODELS:
    if model_id not in _cached_ids:
        print(f"\n{display_name}: not cached locally, skipping. "
              f"Download with HF_TOKEN set and load_zero_shot_model('{model_id}').")
        continue

    # Use fewer samples for instruction-tuned models (much slower inference)
    n_samples = N_IT_SAMPLES if _is_instruction_tuned(model_id) else N_ZERO_SHOT_SAMPLES
    if n_samples and n_samples < len(_test_df_full):
        _test_df = _test_df_full.sample(n_samples, random_state=42).reset_index(drop=True)
    else:
        _test_df = _test_df_full

    print(f"\n{'='*60}")
    print(f"Evaluating (zero-shot): {display_name} ({model_id})")
    print(f"{'='*60}")

    try:
        zs_model, zs_tokenizer = load_zero_shot_model(model_id)
    except Exception as e:
        print(f"  Failed to load: {type(e).__name__}: {e}")
        continue

    y_pred_zs, y_true_zs = [], []
    latencies_zs = []
    # Use 2 beams for large IT models to balance speed vs quality
    _beams = 2 if _is_instruction_tuned(model_id) else None
    for _, row in tqdm(_test_df.iterrows(), total=len(_test_df), desc="Zero-shot eval"):
        t0 = time.perf_counter()
        try:
            preds = predict_code_zero_shot(
                row["code"], zs_model, zs_tokenizer, model_id, label_map,
                top_k=1, num_beams=_beams,
            )
        except Exception as e:
            print(f"  Prediction error: {e}")
            preds = [{"cwe": "CWE-0", "confidence": 0.0}]
        if device.type == "cuda":
            torch.cuda.synchronize()
        latencies_zs.append((time.perf_counter() - t0) * 1000)

        predicted_cwe = preds[0]["cwe"].replace("CWE-", "")
        true_cwe = str(row["cwe_id"])
        y_pred_zs.append(label_map.get(predicted_cwe, 0))
        y_true_zs.append(label_map.get(true_cwe, 0))

    acc = accuracy_score(y_true_zs, y_pred_zs)
    prec = precision_score(y_true_zs, y_pred_zs, average="macro", zero_division=0)
    rec = recall_score(y_true_zs, y_pred_zs, average="macro", zero_division=0)
    f1 = f1_score(y_true_zs, y_pred_zs, average="macro", zero_division=0)
    mcc = matthews_corrcoef(y_true_zs, y_pred_zs)
    fpr = macro_fpr(y_true_zs, y_pred_zs, num_classes)
    avg_latency_ms = np.mean(latencies_zs)

    n_eval = len(_test_df)
    n_total = len(_test_df_full)
    sample_note = f"sampled {n_eval}/{n_total}" if n_eval < n_total else "full test set"
    meta = ZERO_SHOT_MODEL_META.get(model_id, {})

    results.append({
        "Model": display_name,
        "model_id": model_id,
        "inference_only": True,
        "eval_samples": sample_note,
        "Parameters": meta.get("Parameters", "N/A"),
        "Size (MB)": meta.get("Size (MB)", "N/A"),
        "Best Epoch": "N/A",
        "Training Time (s)": "N/A",
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "MCC": mcc,
        "FPR": fpr,
        "Latency (ms)": round(avg_latency_ms, 2),
    })

    del zs_model, zs_tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# Print comparison table
print(f"\n\n{'='*90}")
print("MODEL COMPARISON — All metrics on test set (macro-averaged)")
print(f"{'='*90}")
header = f"{'Model':<16} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'MCC':>10} {'FPR':>10}"
print(header)
print("-" * 90)
for r in results:
    print(
        f"{r['Model']:<16} "
        f"{r['Accuracy']:>10.4f} "
        f"{r['Precision']:>10.4f} "
        f"{r['Recall']:>10.4f} "
        f"{r['F1-Score']:>10.4f} "
        f"{r['MCC']:>10.4f} "
        f"{r['FPR']:>10.6f}"
    )
print(f"{'='*90}")

# Save results to JSON for UI consumption
output_path = os.path.join("outputs", "model_comparison.json")
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to {output_path}")
