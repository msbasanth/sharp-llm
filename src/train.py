"""Training loop for the CWE vulnerability classifier.

Supports CUDA acceleration, mixed precision (fp16), early stopping,
checkpoint saving/resumption, and TensorBoard logging.
"""

import argparse
import json
import os
import sys
import time

import truststore
truststore.inject_into_ssl()

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data.dataset import get_dataloaders
from src.model import CWEClassifier, CWEBiLSTM
from src.utils import load_config, setup_logging, set_seed, get_device, count_parameters


def _is_dl_model(config: dict) -> bool:
    """Check if the selected model is a traditional DL model (non-transformer)."""
    model_name = config.get("model_name", "")
    return model_name.startswith("bilstm") or model_name.startswith("textcnn")


def evaluate(model, dataloader, device):
    """Run evaluation and return loss, accuracy, macro-F1, precision, recall."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    n = len(all_labels)
    avg_loss = total_loss / n
    accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)

    return avg_loss, accuracy, macro_f1, macro_precision, macro_recall


def train(config: dict):
    logger = setup_logging(log_file=os.path.join(config["log_dir"], "train.log"))
    device = get_device()
    logger.info(f"Using device: {device}")

    set_seed(config["seed"])
    logger.info(f"Run seed: {config['seed']}")
    if config.get("run_tag"):
        logger.info(f"Run tag: {config['run_tag']}")

    # Data — support combined datasets via experiment config
    logger.info("Loading data...")
    experiment = config.get("experiment", {})
    dataset_mode = experiment.get("dataset_mode", "juliet_only")
    if dataset_mode != "juliet_only":
        # Use combined dataset paths
        train_config = dict(config)
        train_config["train_path"] = experiment.get("combined_train_path", config["train_path"])
        train_config["test_path"] = experiment.get("combined_test_path", config["test_path"])
        train_config["label_map_path"] = experiment.get("combined_label_map_path", config["label_map_path"])
        # Update num_classes from combined label map
        from src.utils import load_label_map
        label_map = load_label_map(train_config["label_map_path"])
        train_config["num_classes"] = len(label_map)
        logger.info(f"Dataset mode: {dataset_mode}, num_classes: {train_config['num_classes']}")
        train_loader, test_loader, tokenizer = get_dataloaders(train_config)
        num_classes = train_config["num_classes"]
    else:
        train_loader, test_loader, tokenizer = get_dataloaders(config)
        num_classes = config["num_classes"]
    logger.info(f"Train batches: {len(train_loader)}, Test batches: {len(test_loader)}")

    # Model
    dl_config = config.get("dl", {})
    if _is_dl_model(config):
        # DL model — use tokenizer vocab size for embedding layer
        vocab_size = tokenizer.vocab_size
        model = CWEBiLSTM(
            vocab_size=vocab_size,
            num_classes=num_classes,
            embedding_dim=dl_config.get("embedding_dim", 128),
            hidden_dim=dl_config.get("hidden_dim", 256),
            num_layers=dl_config.get("num_layers", 2),
            dropout=dl_config.get("dropout", 0.3),
            pad_idx=tokenizer.pad_token_id or 0,
        ).to(device)
        # Override hyperparameters for DL models
        effective_lr = dl_config.get("learning_rate", 1e-3)
        effective_wd = config.get("weight_decay", 0.01)
    else:
        model = CWEClassifier(
            model_name=config["model_name"],
            num_classes=num_classes,
            dropout=config["dropout"],
        ).to(device)
        effective_lr = config["learning_rate"]
        effective_wd = config["weight_decay"]

    params = count_parameters(model)
    logger.info(f"Model parameters — total: {params['total']:,}, trainable: {params['trainable']:,}")

    # Optimizer & loss
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=effective_lr,
        weight_decay=effective_wd,
    )
    criterion = nn.CrossEntropyLoss()

    # Mixed precision
    use_fp16 = config.get("fp16", False) and device.type == "cuda"
    scaler = GradScaler("cuda") if use_fp16 else None
    logger.info(f"Mixed precision (fp16): {use_fp16}")

    # TensorBoard
    writer = SummaryWriter(log_dir=config["log_dir"])

    # Checkpoint directory (model-specific or explicit model dir)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)
    model_variant = config["model_name"].split("/")[-1]
    if config.get("checkpoint_dir_is_model_dir", False):
        model_checkpoint_dir = config["checkpoint_dir"]
    else:
        model_checkpoint_dir = os.path.join(config["checkpoint_dir"], model_variant)
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # Resume from checkpoint
    start_epoch = 0
    best_f1 = 0.0
    patience_counter = 0

    checkpoint_path = os.path.join(model_checkpoint_dir, "latest.pt")
    if os.path.exists(checkpoint_path):
        logger.info(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        logger.info(f"Resumed at epoch {start_epoch}, best F1: {best_f1:.4f}")

    # Training loop
    accum_steps = config.get("gradient_accumulation_steps", 1)
    log_every = config.get("log_every_n_steps", 100)
    training_start = time.time()

    # Per-epoch metrics log for before/after comparison
    if config.get("log_dir_is_model_dir", False):
        epoch_metrics_path = os.path.join(config["log_dir"], "epoch_metrics.json")
    else:
        epoch_metrics_path = os.path.join(config["log_dir"], f"{model_variant}_epoch_metrics.json")
    epoch_metrics_log = []

    # Running best trackers (each metric tracked independently)
    best_metrics = {
        "best_f1": 0.0,
        "best_acc": 0.0,
        "best_precision": 0.0,
        "best_recall": 0.0,
        "best_f1_epoch": -1,
        "best_acc_epoch": -1,
        "best_precision_epoch": -1,
        "best_recall_epoch": -1,
    }

    for epoch in range(start_epoch, config["epochs"]):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        progress = tqdm(
            enumerate(train_loader),
            total=len(train_loader),
            desc=f"Epoch {epoch+1}/{config['epochs']}",
        )

        optimizer.zero_grad()

        for step, batch in progress:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            if use_fp16:
                with autocast("cuda"):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss = loss / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                loss = loss / accum_steps

                loss.backward()

                if (step + 1) % accum_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()

            epoch_loss += loss.item() * accum_steps
            global_step = epoch * len(train_loader) + step

            if (step + 1) % log_every == 0:
                avg = epoch_loss / (step + 1)
                writer.add_scalar("train/loss", avg, global_step)
                progress.set_postfix(loss=f"{avg:.4f}")

        epoch_time = time.time() - epoch_start
        total_training_time = time.time() - training_start
        train_avg_loss = epoch_loss / len(train_loader)

        # Evaluate
        logger.info("Evaluating on test set...")
        test_loss, test_acc, test_f1, test_prec, test_rec = evaluate(model, test_loader, device)

        logger.info(
            f"Epoch {epoch+1} | "
            f"Train Loss: {train_avg_loss:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"Test Macro-F1: {test_f1:.4f} | "
            f"Test Precision: {test_prec:.4f} | "
            f"Test Recall: {test_rec:.4f} | "
            f"Time: {epoch_time:.0f}s"
        )

        writer.add_scalar("test/loss", test_loss, epoch)
        writer.add_scalar("test/accuracy", test_acc, epoch)
        writer.add_scalar("test/macro_f1", test_f1, epoch)
        writer.add_scalar("test/macro_precision", test_prec, epoch)
        writer.add_scalar("test/macro_recall", test_rec, epoch)

        # Update running best metrics (each tracked independently)
        if test_f1 > best_metrics["best_f1"]:
            best_metrics["best_f1"] = test_f1
            best_metrics["best_f1_epoch"] = epoch + 1
        if test_acc > best_metrics["best_acc"]:
            best_metrics["best_acc"] = test_acc
            best_metrics["best_acc_epoch"] = epoch + 1
        if test_prec > best_metrics["best_precision"]:
            best_metrics["best_precision"] = test_prec
            best_metrics["best_precision_epoch"] = epoch + 1
        if test_rec > best_metrics["best_recall"]:
            best_metrics["best_recall"] = test_rec
            best_metrics["best_recall_epoch"] = epoch + 1

        # Save per-epoch metrics
        epoch_entry = {
            "epoch": epoch + 1,
            "train_loss": round(train_avg_loss, 6),
            "test_loss": round(test_loss, 6),
            "test_acc": round(test_acc, 6),
            "test_f1": round(test_f1, 6),
            "test_precision": round(test_prec, 6),
            "test_recall": round(test_rec, 6),
            "best_f1_so_far": round(best_metrics["best_f1"], 6),
            "best_acc_so_far": round(best_metrics["best_acc"], 6),
            "best_precision_so_far": round(best_metrics["best_precision"], 6),
            "best_recall_so_far": round(best_metrics["best_recall"], 6),
            "training_time_seconds": round(total_training_time, 1),
        }
        epoch_metrics_log.append(epoch_entry)

        # Persist epoch metrics to disk after each epoch
        os.makedirs(os.path.dirname(epoch_metrics_path), exist_ok=True)
        with open(epoch_metrics_path, "w") as f:
            json.dump({
                "model": config["model_name"],
                "model_variant": model_variant,
                "seed": config.get("seed"),
                "run_tag": config.get("run_tag"),
                "epoch_metrics": epoch_metrics_log,
                "best_metrics": best_metrics,
            }, f, indent=2)

        # Save checkpoint
        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_f1": max(best_f1, test_f1),
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_f1": test_f1,
            "test_precision": test_prec,
            "test_recall": test_rec,
            "best_metrics": best_metrics,
            "config": config,
            "seed": config.get("seed"),
            "run_tag": config.get("run_tag"),
            "training_time": total_training_time,
        }
        if scaler:
            ckpt_data["scaler_state_dict"] = scaler.state_dict()

        torch.save(ckpt_data, os.path.join(model_checkpoint_dir, "latest.pt"))

        # Best model
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(ckpt_data, os.path.join(model_checkpoint_dir, "best.pt"))
            logger.info(f"New best model saved (Macro-F1: {best_f1:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{config['patience']}")

        if patience_counter >= config["patience"]:
            logger.info("Early stopping triggered")
            break

    writer.close()
    logger.info(f"Training complete. Best Macro-F1: {best_f1:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train CWE classifier")
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--experiment", default=None,
                        help="Experiment name for output isolation (e.g. 'exp_b_juliet19'). "
                             "Checkpoints/logs go to outputs/<experiment>/checkpoints/ etc.")
    parser.add_argument("--model", default=None, help="Override model_name from config")
    parser.add_argument("--epochs", type=int, default=None, help="Override max epochs from config")
    parser.add_argument("--patience", type=int, default=None, help="Override early-stopping patience from config")
    parser.add_argument("--seed", type=int, default=None, help="Override run seed from config")
    parser.add_argument("--train-path", default=None, help="Override train parquet path")
    parser.add_argument("--test-path", default=None, help="Override test parquet path")
    parser.add_argument("--label-map-path", default=None, help="Override label map path")
    parser.add_argument("--checkpoint-dir", default=None, help="Override checkpoint directory")
    parser.add_argument("--log-dir", default=None, help="Override logging directory")
    parser.add_argument("--checkpoint-dir-is-model-dir", action="store_true",
                        help="Treat --checkpoint-dir as the final per-model directory (no variant suffix)")
    parser.add_argument("--log-dir-is-model-dir", action="store_true",
                        help="Treat --log-dir as the final per-model directory")
    parser.add_argument("--run-tag", default=None, help="Optional run tag for metadata (e.g. ICMLDE:Juliet118:CodeT5-Base)")
    args = parser.parse_args()

    config = load_config(args.config)

    # Override model from CLI
    if args.model:
        config["model_name"] = args.model

    # Override training hyperparameters from CLI
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.patience is not None:
        config["patience"] = args.patience
    if args.seed is not None:
        config["seed"] = args.seed

    # Experiment-specific output directories
    if args.experiment:
        config["checkpoint_dir"] = os.path.join("outputs", args.experiment, "checkpoints")
        config["log_dir"] = os.path.join("outputs", args.experiment, "logs")
        config["experiment_name"] = args.experiment

    # Explicit output overrides (useful for seed-isolated runs)
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.log_dir:
        config["log_dir"] = args.log_dir
    if args.checkpoint_dir_is_model_dir:
        config["checkpoint_dir_is_model_dir"] = True
    if args.log_dir_is_model_dir:
        config["log_dir_is_model_dir"] = True
    if args.run_tag:
        config["run_tag"] = args.run_tag

    # Override data paths from CLI
    if args.train_path:
        config["train_path"] = args.train_path
        # When using explicit paths, bypass experiment section logic
        config.setdefault("experiment", {})["dataset_mode"] = "juliet_only"
    if args.test_path:
        config["test_path"] = args.test_path
    if args.label_map_path:
        config["label_map_path"] = args.label_map_path

    # Auto-detect num_classes from label map when overriding paths
    if args.label_map_path:
        from src.utils import load_label_map
        label_map = load_label_map(args.label_map_path)
        config["num_classes"] = len(label_map)

    train(config)


if __name__ == "__main__":
    main()
