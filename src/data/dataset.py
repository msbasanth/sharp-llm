"""PyTorch Dataset and DataLoader for CWE classification."""

import os
import sys

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
from src.utils import load_config, load_label_map


class CWEDataset(Dataset):
    """Dataset for CWE vulnerability classification.

    Stores raw code strings and labels. Tokenization happens
    on-the-fly in __getitem__ to keep memory usage low.
    """

    def __init__(
        self,
        codes: list[str],
        labels: list[int],
        tokenizer,
        max_length: int = 256,
    ):
        self.codes = codes
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.codes[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, AutoTokenizer]:
    """Create train and test DataLoaders from processed parquet files.

    Returns:
        (train_loader, test_loader, tokenizer)
    """
    # For DL models (BiLSTM etc.), use a separate tokenizer specified in dl config
    model_name = config.get("model_name", "")
    dl_config = config.get("dl", {})
    if model_name.startswith("bilstm") or model_name.startswith("textcnn"):
        tokenizer_name = dl_config.get("tokenizer_name", "Salesforce/codet5-small")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        batch_size = dl_config.get("batch_size", config["batch_size"])
    else:
        # codet5-base shares the same RoBERTa BPE tokenizer as codet5-small.
        # Its tokenizer_config.json has an extra_special_tokens format that newer
        # transformers (4.40+) rejects. Load via codet5-small to bypass this.
        tokenizer_name = (
            "Salesforce/codet5-small"
            if "codet5-base" in model_name.lower()
            else model_name
        )
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=False)
        batch_size = config["batch_size"]

    train_df = pd.read_parquet(config["train_path"])
    test_df = pd.read_parquet(config["test_path"])

    train_dataset = CWEDataset(
        codes=train_df["code"].tolist(),
        labels=train_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config["max_length"],
    )

    test_dataset = CWEDataset(
        codes=test_df["code"].tolist(),
        labels=test_df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config["max_length"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )

    return train_loader, test_loader, tokenizer
