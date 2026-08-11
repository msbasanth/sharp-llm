"""PyTorch Dataset and DataLoader for CWE classification."""

import os
import sys

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
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


def _build_tokenizer_and_batch_size(config: dict) -> tuple[AutoTokenizer, int]:
    """Create tokenizer and resolve the correct batch size for the model type."""
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

    return tokenizer, batch_size


def _build_loader(
    df: pd.DataFrame,
    tokenizer,
    config: dict,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = CWEDataset(
        codes=df["code"].tolist(),
        labels=df["label"].tolist(),
        tokenizer=tokenizer,
        max_length=config["max_length"],
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.get("num_workers", 0),
        pin_memory=config.get("pin_memory", True),
    )


def _split_train_val_group_aware(
    train_df: pd.DataFrame,
    val_size: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create train/val split from train_df while respecting template groups per CWE."""
    if val_size <= 0.0 or val_size >= 1.0:
        raise ValueError("val_size must be between 0 and 1")

    train_indices: list[int] = []
    val_indices: list[int] = []
    has_template = "template_id" in train_df.columns

    for _, cwe_group in train_df.groupby("cwe_id"):
        if len(cwe_group) < 2:
            train_indices.extend(cwe_group.index.tolist())
            continue

        if has_template:
            templates = cwe_group["template_id"].astype(str)
            if templates.nunique() >= 2:
                splitter = GroupShuffleSplit(
                    n_splits=1,
                    test_size=val_size,
                    random_state=seed,
                )
                train_idx, val_idx = next(splitter.split(cwe_group, groups=templates))
                train_indices.extend(cwe_group.index[train_idx].tolist())
                val_indices.extend(cwe_group.index[val_idx].tolist())
                continue

        # Fallback: row-level split if grouping is not available for this class.
        shuffled = cwe_group.sample(frac=1.0, random_state=seed)
        val_count = max(1, int(round(len(shuffled) * val_size)))
        if val_count >= len(shuffled):
            val_count = len(shuffled) - 1

        val_indices.extend(shuffled.iloc[:val_count].index.tolist())
        train_indices.extend(shuffled.iloc[val_count:].index.tolist())

    if not val_indices:
        raise ValueError("Validation split is empty; adjust val_size or dataset")

    train_split = train_df.loc[train_indices].reset_index(drop=True)
    val_split = train_df.loc[val_indices].reset_index(drop=True)

    if has_template:
        train_templates = set(train_split["template_id"].astype(str).unique())
        val_templates = set(val_split["template_id"].astype(str).unique())
        overlap = train_templates & val_templates
        if overlap:
            raise ValueError(
                f"Template leakage detected between train/val: {len(overlap)} overlapping templates"
            )

    return train_split, val_split


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, AutoTokenizer]:
    """Create train and test DataLoaders from processed parquet files.

    Returns:
        (train_loader, test_loader, tokenizer)
    """
    tokenizer, batch_size = _build_tokenizer_and_batch_size(config)

    train_df = pd.read_parquet(config["train_path"])
    test_df = pd.read_parquet(config["test_path"])

    train_loader = _build_loader(train_df, tokenizer, config, batch_size, shuffle=True)
    test_loader = _build_loader(test_df, tokenizer, config, batch_size, shuffle=False)

    return train_loader, test_loader, tokenizer


def get_dataloaders_with_validation(
    config: dict,
    val_size: float = 0.1,
    val_seed: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, AutoTokenizer]:
    """Create train, validation, and test DataLoaders with an independent val split."""
    tokenizer, batch_size = _build_tokenizer_and_batch_size(config)
    train_df = pd.read_parquet(config["train_path"])
    test_df = pd.read_parquet(config["test_path"])

    split_seed = config.get("seed", 42) if val_seed is None else val_seed
    train_split, val_split = _split_train_val_group_aware(train_df, val_size, split_seed)

    train_loader = _build_loader(train_split, tokenizer, config, batch_size, shuffle=True)
    val_loader = _build_loader(val_split, tokenizer, config, batch_size, shuffle=False)
    test_loader = _build_loader(test_df, tokenizer, config, batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tokenizer
