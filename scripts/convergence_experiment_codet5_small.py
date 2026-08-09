#!/usr/bin/env python3
"""
Convergence Experiment: CodeT5-Small with 4 Epochs
====================================================

Run CodeT5-Small on Juliet 118-CWE with a new seed (50) for 4 epochs
to understand training convergence behavior.

Usage:
  python scripts/convergence_experiment_codet5_small.py
    --config config.yaml
    --model Salesforce/codet5-small
    --epochs 4
    --seed 50
    --output-dir outputs/icmlde2026/convergence/seed_50/codet5-small
"""

import argparse
import json
import os
import sys
import logging
from pathlib import Path

import truststore
truststore.inject_into_ssl()

import torch
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, precision_score, recall_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils import load_config, set_seed, get_device, setup_logging
from src.model import CWEClassifier
from src.data.dataset import get_dataloaders


def main():
    parser = argparse.ArgumentParser(
        description="Convergence experiment: train CodeT5-Small for 4 epochs with new seed"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Config file path")
    parser.add_argument("--model", type=str, default="Salesforce/codet5-small", help="Model ID")
    parser.add_argument("--epochs", type=int, default=4, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=50, help="Random seed (NEW: not 42-46)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    
    # Setup
    config = load_config(args.config)
    config["model_name"] = args.model
    config["epochs"] = args.epochs
    config["seed"] = args.seed
    config["batch_size"] = args.batch_size
    config["learning_rate"] = args.learning_rate
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    set_seed(args.seed)
    device = get_device()
    logger = setup_logging(output_dir / "train.log")
    
    logger.info("=" * 75)
    logger.info("ICMLDE Convergence Experiment: CodeT5-Small")
    logger.info("=" * 75)
    logger.info(f"Model: {args.model}")
    logger.info(f"Epochs: {args.epochs}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_dir}")
    
    # Load data
    logger.info("Loading datasets...")
    train_loader, test_loader, tokenizer = get_dataloaders(config)
    val_loader = test_loader  # reuse test split for per-epoch validation
    logger.info(f"  Train samples: {len(train_loader.dataset)}")
    logger.info(f"  Val/Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    logger.info(f"Initializing {args.model}...")
    model = CWEClassifier(
        model_name=args.model,
        num_classes=config["num_classes"],
        dropout=config.get("dropout", 0.1),
    )
    model.to(device)
    
    # Training setup
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=config.get("weight_decay", 0.01)
    )
    criterion = torch.nn.CrossEntropyLoss()
    
    # Track epoch-level metrics
    epoch_metrics = []
    
    logger.info("\n" + "=" * 75)
    logger.info("Starting training...")
    logger.info("=" * 75)
    
    for epoch in range(1, args.epochs + 1):
        logger.info(f"\n--- Epoch {epoch}/{args.epochs} ---")
        
        # Training
        model.train()
        train_loss = 0.0
        train_preds = []
        train_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=-1)
            train_preds.extend(preds.cpu().tolist())
            train_labels.extend(labels.cpu().tolist())
            
            if (batch_idx + 1) % 50 == 0:
                logger.info(f"  Batch {batch_idx + 1}/{len(train_loader)}: loss={loss.item():.4f}")
        
        train_loss /= len(train_loader)
        train_f1 = f1_score(train_labels, train_preds, average="macro", zero_division=0)
        train_acc = accuracy_score(train_labels, train_preds)
        
        logger.info(f"Train: loss={train_loss:.4f} | F1={train_f1:.4f} | Acc={train_acc:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)
                
                val_loss += loss.item()
                preds = logits.argmax(dim=-1)
                val_preds.extend(preds.cpu().tolist())
                val_labels.extend(labels.cpu().tolist())
        
        val_loss /= len(val_loader)
        val_f1 = f1_score(val_labels, val_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(val_labels, val_preds)
        
        logger.info(f"Val:   loss={val_loss:.4f} | F1={val_f1:.4f} | Acc={val_acc:.4f}")
        
        # Test evaluation
        model.eval()
        test_preds = []
        test_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                
                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=-1)
                test_preds.extend(preds.cpu().tolist())
                test_labels.extend(labels.cpu().tolist())
        
        test_f1 = f1_score(test_labels, test_preds, average="macro", zero_division=0)
        test_acc = accuracy_score(test_labels, test_preds)
        test_mcc = matthews_corrcoef(test_labels, test_preds)
        test_prec = precision_score(test_labels, test_preds, average="macro", zero_division=0)
        test_rec = recall_score(test_labels, test_preds, average="macro", zero_division=0)
        
        logger.info(f"Test:  F1={test_f1:.4f} | Acc={test_acc:.4f} | MCC={test_mcc:.4f} | Prec={test_prec:.4f} | Rec={test_rec:.4f}")
        
        # Save epoch metrics
        epoch_metric = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_f1": train_f1,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_f1": val_f1,
            "val_accuracy": val_acc,
            "test_f1": test_f1,
            "test_accuracy": test_acc,
            "test_mcc": test_mcc,
            "test_precision": test_prec,
            "test_recall": test_rec,
        }
        epoch_metrics.append(epoch_metric)
    
    # Save final checkpoint
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    checkpoint_path = checkpoint_dir / "final.pt"
    torch.save(model.state_dict(), checkpoint_path)
    logger.info(f"\nCheckpoint saved to {checkpoint_path}")
    
    # Save epoch metrics
    epoch_metrics_path = output_dir / "epoch_metrics.json"
    with open(epoch_metrics_path, "w") as f:
        json.dump(epoch_metrics, f, indent=2)
    logger.info(f"Epoch metrics saved to {epoch_metrics_path}")
    
    # Save convergence summary
    summary_path = output_dir / "convergence_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 75 + "\n")
        f.write("CodeT5-Small Convergence Analysis (Seed 50, 4 Epochs)\n")
        f.write("=" * 75 + "\n\n")
        f.write("Epoch | Train Loss | Train F1 | Val Loss | Val F1 | Test F1 | Test Acc\n")
        f.write("-" * 75 + "\n")
        for m in epoch_metrics:
            f.write(
                "{:5d} | {:10.4f} | {:8.4f} | {:8.4f} | {:6.4f} | {:7.4f} | {:8.4f}\n".format(
                    m["epoch"], m["train_loss"], m["train_f1"], m["val_loss"],
                    m["val_f1"], m["test_f1"], m["test_accuracy"]
                )
            )
        f.write("\n" + "=" * 75 + "\n")
        f.write("Convergence Observations:\n")
        f.write("-" * 75 + "\n")
        f.write(f"Initial Test F1 (Epoch 1): {epoch_metrics[0]['test_f1']:.4f}\n")
        f.write(f"Final Test F1 (Epoch {args.epochs}): {epoch_metrics[-1]['test_f1']:.4f}\n")
        f.write(f"Improvement: {epoch_metrics[-1]['test_f1'] - epoch_metrics[0]['test_f1']:+.4f}\n")
        f.write(f"Training Loss Trend: {epoch_metrics[0]['train_loss']:.4f} → {epoch_metrics[-1]['train_loss']:.4f}\n")
    
    logger.info(f"Summary saved to {summary_path}")

    # Print epoch metrics as JSON to stdout so they're always captured in the
    # kernel execution log (recoverable even if /kaggle/working/ files are lost)
    logger.info("\nEPOCH_METRICS_JSON: " + json.dumps(epoch_metrics))

    logger.info("\n" + "=" * 75)
    logger.info("Convergence experiment completed!")
    logger.info("=" * 75)


if __name__ == "__main__":
    main()
