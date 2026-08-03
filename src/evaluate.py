"""Evaluation and error analysis for the CWE classifier.

Generates:
- Overall metrics (accuracy, macro precision/recall/F1)
- Per-class classification report
- Top confused CWE pairs
"""

import argparse
import json
import numbers
import os
import statistics
import sys

import truststore
truststore.inject_into_ssl()

import pandas as pd
import torch
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import get_dataloaders
from src.model import CWEClassifier, CWEBiLSTM, load_qlora_for_inference
from src.utils import load_config, setup_logging, get_device, load_label_map


def predict_all(model, dataloader, device, is_qlora=False):
    """Run inference on all samples, return predictions and labels."""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"]

            if is_qlora:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.argmax(dim=-1).cpu()
            else:
                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1).cpu()

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    return all_preds, all_labels


def get_confused_pairs(y_true, y_pred, label_names, top_k=20):
    """Find the most frequently confused CWE pairs."""
    cm = confusion_matrix(y_true, y_pred)
    pairs = []

    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j and cm[i][j] > 0:
                pairs.append({
                    "actual": label_names[i],
                    "predicted": label_names[j],
                    "count": int(cm[i][j]),
                })

    pairs.sort(key=lambda x: x["count"], reverse=True)
    return pairs[:top_k]


def compute_macro_fpr(y_true, y_pred, num_classes):
    """Compute macro-averaged false positive rate across classes."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    fprs = []
    for i in range(num_classes):
        fp = cm[:, i].sum() - cm[i, i]
        tn = cm.sum() - cm[i, :].sum() - cm[:, i].sum() + cm[i, i]
        denom = fp + tn
        fprs.append((fp / denom) if denom > 0 else 0.0)
    return float(sum(fprs) / len(fprs)) if fprs else 0.0


def build_classification_report_summary(y_true, y_pred, label_names):
    """Return the text report and class-wise F1 dispersion for present labels."""
    present_labels = sorted(set(y_true) | set(y_pred))
    present_names = [label_names[i] for i in present_labels]
    report_dict = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        labels=present_labels,
        target_names=present_names,
        zero_division=0,
    )
    class_f1_scores = [report_dict[name]["f1-score"] for name in present_names]
    macro_f1_mean = statistics.fmean(class_f1_scores) if class_f1_scores else 0.0
    macro_f1_std = statistics.pstdev(class_f1_scores) if len(class_f1_scores) > 1 else 0.0

    return {
        "report_text": report_text,
        "present_labels": present_labels,
        "present_names": present_names,
        "macro_f1_across_cwe_mean": macro_f1_mean,
        "macro_f1_across_cwe_std": macro_f1_std,
        "macro_f1_class_count": len(class_f1_scores),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate CWE classifier")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (default: best.pt)")
    parser.add_argument("--qlora", action="store_true", help="Evaluate QLoRA model instead of encoder model")
    parser.add_argument("--dl-model", action="store_true", help="Evaluate a DL model (BiLSTM) instead of encoder model")
    parser.add_argument("--experiment", default=None, help="Experiment name (e.g. exp_a_juliet118)")
    parser.add_argument("--model", default=None, help="Model name override (e.g. Salesforce/codet5-small)")
    parser.add_argument("--test-path", default=None, help="Override test parquet path")
    parser.add_argument("--label-map-path", default=None, help="Override label map JSON path")
    parser.add_argument("--output-dir", default=None, help="Override output directory")
    parser.add_argument("--seed", type=int, default=None, help="Run seed metadata for traceability")
    parser.add_argument("--experiment-tag", default=None,
                        help="Optional run tag (e.g. ICMLDE:Juliet118:CodeT5-Base)")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.qlora:
        config["qlora_eval"] = True

    # Apply CLI overrides
    if args.model:
        config["model_name"] = args.model
    if args.test_path:
        config["test_path"] = args.test_path
    if args.label_map_path:
        config["label_map_path"] = args.label_map_path

    # Determine output directory
    if args.output_dir:
        output_dir = args.output_dir
    elif args.experiment:
        model_variant = config["model_name"].split("/")[-1]
        output_dir = os.path.join("outputs", args.experiment, model_variant)
    else:
        output_dir = "outputs"

    logger = setup_logging()
    device = get_device()

    # Load label map (int_label → CWE name)
    label_map = load_label_map(config["label_map_path"])
    # Override num_classes to match the label map (important for 19-CWE experiments)
    config["num_classes"] = len(label_map)
    inv_label_map = {v: k for k, v in label_map.items()}
    label_names = [f"CWE-{inv_label_map[i]}" for i in range(len(label_map))]

    # Load model
    is_qlora = config.get("qlora_eval", False)
    if is_qlora or args.checkpoint and "qlora" in args.checkpoint:
        # QLoRA model evaluation
        qlora_model_entry = None
        for m in config.get("available_models", []):
            if m.get("qlora", False):
                qlora_model_entry = m
                break
        if qlora_model_entry is None:
            raise ValueError("No model with 'qlora: true' found in available_models")

        model_id = qlora_model_entry["model_id"]
        variant = model_id.split("/")[-1] + "-qlora"
        adapter_path = args.checkpoint or os.path.join(config["checkpoint_dir"], variant, "best_adapter")
        logger.info(f"Loading QLoRA model: {model_id} from {adapter_path}")

        model, qlora_tokenizer = load_qlora_for_inference(
            model_id=model_id,
            num_classes=config["num_classes"],
            adapter_path=adapter_path,
        )

        # Load QLoRA training state for epoch info
        state_path = os.path.join(config["checkpoint_dir"], variant, "best.pt")
        if os.path.exists(state_path):
            ckpt = torch.load(state_path, map_location="cpu", weights_only=False)
            logger.info(f"Checkpoint epoch: {ckpt.get('epoch', 'N/A')}, F1: {ckpt.get('test_f1', 'N/A')}")

        # Build dataloaders with QLoRA tokenizer
        from src.data.dataset import CWEDataset
        from torch.utils.data import DataLoader
        test_df = pd.read_parquet(config["test_path"])
        test_dataset = CWEDataset(
            codes=test_df["code"].tolist(),
            labels=test_df["label"].tolist(),
            tokenizer=qlora_tokenizer,
            max_length=config["max_length"],
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.get("qlora", {}).get("per_device_batch_size", 2),
            shuffle=False,
            num_workers=config.get("num_workers", 0),
            pin_memory=config.get("pin_memory", True),
        )
    else:
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        elif args.experiment:
            model_variant = config["model_name"].split("/")[-1]
            checkpoint_path = os.path.join("outputs", args.experiment, "checkpoints", model_variant, "best.pt")
        else:
            checkpoint_path = os.path.join(config["checkpoint_dir"], "best.pt")
        logger.info(f"Loading checkpoint: {checkpoint_path}")

        model_name = config.get("model_name", "")
        is_dl = args.dl_model or model_name.startswith("bilstm") or model_name.startswith("textcnn")

        if is_dl:
            dl_config = config.get("dl", {})
            tokenizer_name = dl_config.get("tokenizer_name", "Salesforce/codet5-small")
            from transformers import AutoTokenizer
            dl_tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            model = CWEBiLSTM(
                vocab_size=dl_tokenizer.vocab_size,
                num_classes=config["num_classes"],
                embedding_dim=dl_config.get("embedding_dim", 128),
                hidden_dim=dl_config.get("hidden_dim", 256),
                num_layers=dl_config.get("num_layers", 2),
                dropout=0.0,  # No dropout during inference
                pad_idx=dl_tokenizer.pad_token_id or 0,
            )
        else:
            model = CWEClassifier(
                model_name=config["model_name"],
                num_classes=config["num_classes"],
                dropout=config["dropout"],
            )
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        model.to(device)

        logger.info(f"Checkpoint epoch: {ckpt['epoch']}, F1: {ckpt.get('test_f1', 'N/A')}")

        # Load test data
        _, test_loader, _ = get_dataloaders(config)

    # Predict
    all_preds, all_labels = predict_all(model, test_loader, device, is_qlora=is_qlora)

    # Overall metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "macro_precision": precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_recall": recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "weighted_f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "mcc": matthews_corrcoef(all_labels, all_preds),
        "macro_fpr": compute_macro_fpr(all_labels, all_preds, config["num_classes"]),
        "run_seed": args.seed if args.seed is not None else config.get("seed"),
        "experiment_tag": args.experiment_tag,
        "model_name": config.get("model_name"),
    }

    report_summary = build_classification_report_summary(all_labels, all_preds, label_names)
    metrics["macro_f1_across_cwe_mean"] = report_summary["macro_f1_across_cwe_mean"]
    metrics["macro_f1_across_cwe_std"] = report_summary["macro_f1_across_cwe_std"]
    metrics["macro_f1_class_count"] = report_summary["macro_f1_class_count"]

    logger.info("--- Overall Metrics ---")
    for k, v in metrics.items():
        if isinstance(v, numbers.Real):
            logger.info(f"  {k}: {v:.4f}")
        else:
            logger.info(f"  {k}: {v}")
    logger.info(
        "  macro_f1_across_cwe: %.4f ± %.4f over %d classes",
        metrics["macro_f1_across_cwe_mean"],
        metrics["macro_f1_across_cwe_std"],
        metrics["macro_f1_class_count"],
    )

    report = report_summary["report_text"]

    # Confused pairs
    confused = get_confused_pairs(all_labels, all_preds, label_names)

    # Save outputs
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    pd.DataFrame(confused).to_csv(os.path.join(output_dir, "confusion_pairs.csv"), index=False)

    logger.info(f"Saved metrics to {output_dir}/metrics.json")

    # Source-stratified evaluation (for combined datasets)
    experiment = config.get("experiment", {})
    dataset_mode = experiment.get("dataset_mode", "juliet_only")
    test_path = config["test_path"]
    if dataset_mode != "juliet_only":
        test_path = experiment.get("combined_test_path", test_path)

    test_df = pd.read_parquet(test_path)
    if "source" in test_df.columns:
        logger.info("--- Source-Stratified Evaluation ---")
        sources = test_df["source"].values

        for src_name in sorted(test_df["source"].unique()):
            src_mask = [s == src_name for s in sources]
            src_preds = [p for p, m in zip(all_preds, src_mask) if m]
            src_labels = [l for l, m in zip(all_labels, src_mask) if m]

            if not src_labels:
                continue

            src_metrics = {
                "source": src_name,
                "sample_count": len(src_labels),
                "accuracy": accuracy_score(src_labels, src_preds),
                "macro_precision": precision_score(src_labels, src_preds, average="macro", zero_division=0),
                "macro_recall": recall_score(src_labels, src_preds, average="macro", zero_division=0),
                "macro_f1": f1_score(src_labels, src_preds, average="macro", zero_division=0),
                "weighted_f1": f1_score(src_labels, src_preds, average="weighted", zero_division=0),
                "mcc": matthews_corrcoef(src_labels, src_preds),
                "macro_fpr": compute_macro_fpr(src_labels, src_preds, config["num_classes"]),
                "run_seed": metrics.get("run_seed"),
                "experiment_tag": args.experiment_tag,
                "model_name": config.get("model_name"),
            }
            src_report_summary = build_classification_report_summary(src_labels, src_preds, label_names)
            src_metrics["macro_f1_across_cwe_mean"] = src_report_summary["macro_f1_across_cwe_mean"]
            src_metrics["macro_f1_across_cwe_std"] = src_report_summary["macro_f1_across_cwe_std"]
            src_metrics["macro_f1_class_count"] = src_report_summary["macro_f1_class_count"]

            logger.info(f"  {src_name}: acc={src_metrics['accuracy']:.4f}, "
                        f"f1={src_metrics['macro_f1']:.4f}, n={src_metrics['sample_count']}")

            with open(os.path.join(output_dir, f"metrics_{src_name}.json"), "w") as f:
                json.dump(src_metrics, f, indent=2)

        logger.info(f"Saved source-stratified metrics to {output_dir}/metrics_<source>.json")
    logger.info(f"Saved classification report to {output_dir}/classification_report.txt")
    logger.info(f"Saved confusion pairs to {output_dir}/confusion_pairs.csv")

    # Print summary
    print("\n--- Classification Report (abbreviated) ---")
    print(report[:2000])

    print("\n--- Top 10 Confused Pairs ---")
    for pair in confused[:10]:
        print(f"  {pair['actual']} -> {pair['predicted']}: {pair['count']}")


if __name__ == "__main__":
    main()
