"""Encoder with classification head for CWE detection.

Supports T5-family (CodeT5-small, CodeT5-base) and RoBERTa-family (CodeBERT-base)
encoders, plus QLoRA-based decoder models (CodeGemma) for sequence classification.

Architecture (encoder models):
    Encoder → mean pooling → dropout → Linear(hidden_size, num_classes)

Architecture (QLoRA models):
    AutoModelForSequenceClassification (4-bit NF4) → LoRA adapters + classification head
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig
from transformers.models.t5.modeling_t5 import T5EncoderModel

# Model families that use T5EncoderModel (encoder-only).
# All others default to AutoModel.
_T5_PREFIXES = ("Salesforce/codet5",)


def _is_t5_family(model_name: str) -> bool:
    return any(model_name.startswith(p) for p in _T5_PREFIXES)


class CWEClassifier(nn.Module):
    """Multi-class CWE classifier supporting CodeT5 and CodeBERT encoders."""

    def __init__(self, model_name: str, num_classes: int, dropout: float = 0.1):
        super().__init__()
        if _is_t5_family(model_name):
            self.encoder = T5EncoderModel.from_pretrained(model_name)
        else:
            self.encoder = AutoModel.from_pretrained(model_name)

        # T5 exposes d_model; RoBERTa exposes hidden_size
        cfg = self.encoder.config
        hidden_size = getattr(cfg, "hidden_size", None) or getattr(cfg, "d_model")

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        # Encode
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Mean pooling over non-padding tokens
        mask_expanded = attention_mask.unsqueeze(-1).float()  # (batch, seq_len, 1)
        sum_hidden = (hidden_states * mask_expanded).sum(dim=1)  # (batch, hidden_size)
        count = mask_expanded.sum(dim=1).clamp(min=1e-9)  # (batch, 1)
        pooled = sum_hidden / count  # (batch, hidden_size)

        # Classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        return logits


class SelfAttention(nn.Module):
    """Learnable self-attention pooling over sequence outputs."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.query = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor):
        """Compute attention-weighted sum.

        Args:
            lstm_output: (batch, seq_len, hidden_dim)
            mask: (batch, seq_len) — 1 for real tokens, 0 for padding

        Returns:
            context: (batch, hidden_dim)
            weights: (batch, seq_len) — attention weights (for visualization)
        """
        scores = self.query(lstm_output).squeeze(-1)  # (batch, seq_len)
        scores = scores.masked_fill(mask == 0, float("-inf"))
        weights = torch.softmax(scores, dim=-1)  # (batch, seq_len)
        context = (lstm_output * weights.unsqueeze(-1)).sum(dim=1)  # (batch, hidden_dim)
        return context, weights


class CWEBiLSTM(nn.Module):
    """BiLSTM with self-attention for CWE vulnerability classification.

    Architecture:
        Embedding → BiLSTM (2 layers) → Self-Attention → Dropout → Linear

    Uses the same forward(input_ids, attention_mask) → logits interface
    as CWEClassifier for pipeline compatibility.
    """

    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        embedding_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
        pad_idx: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.attention = SelfAttention(hidden_dim * 2)  # *2 for bidirectional
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)

        Returns:
            logits: (batch_size, num_classes)
        """
        embedded = self.embedding(input_ids)  # (batch, seq_len, embedding_dim)

        # Pack padded sequences for efficient LSTM processing
        lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=input_ids.size(1),
        )  # (batch, seq_len, hidden_dim*2)

        # Attention pooling
        context, _ = self.attention(lstm_out, attention_mask)  # (batch, hidden_dim*2)

        # Classify
        logits = self.classifier(self.dropout(context))
        return logits

    def forward_with_attention(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass that also returns attention weights for visualization."""
        embedded = self.embedding(input_ids)
        lengths = attention_mask.sum(dim=1).cpu().clamp(min=1)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False,
        )
        lstm_out, _ = self.lstm(packed)
        lstm_out, _ = nn.utils.rnn.pad_packed_sequence(
            lstm_out, batch_first=True, total_length=input_ids.size(1),
        )
        context, attn_weights = self.attention(lstm_out, attention_mask)
        logits = self.classifier(self.dropout(context))
        return logits, attn_weights


def load_qlora_classifier(model_id: str, num_classes: int, qlora_config: dict):
    """Load a causal LM as a sequence classifier with QLoRA for training.

    Args:
        model_id: HuggingFace model ID (e.g. 'google/codegemma-1.1-2b').
        num_classes: Number of output classes.
        qlora_config: Dict with keys: lora_rank, lora_alpha, lora_dropout,
            target_modules, quant_type, use_double_quant, gradient_checkpointing.

    Returns:
        (model, tokenizer) tuple. Model is PEFT-wrapped and ready for training.
    """
    import os
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    hf_token = os.environ.get("HF_TOKEN")

    # Tokenizer setup — decoder models need a pad token
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # standard for decoder classification

    # 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=qlora_config.get("quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=qlora_config.get("use_double_quant", True),
    )

    # Load as sequence classification model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        quantization_config=bnb_config,
        device_map={"": 0},
        token=hf_token,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    # Prepare for k-bit training (freeze base, cast norms to fp32)
    model = prepare_model_for_kbit_training(model)

    if qlora_config.get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # LoRA config — train adapters + the classification head (score layer)
    lora_config = LoraConfig(
        r=qlora_config.get("lora_rank", 16),
        lora_alpha=qlora_config.get("lora_alpha", 32),
        lora_dropout=qlora_config.get("lora_dropout", 0.05),
        target_modules=qlora_config.get("target_modules", ["q_proj", "v_proj"]),
        bias="none",
        task_type="SEQ_CLS",
        modules_to_save=["score"],
    )

    model = get_peft_model(model, lora_config)
    return model, tokenizer


def load_qlora_for_inference(model_id: str, num_classes: int, adapter_path: str):
    """Load a QLoRA-trained model for inference.

    Args:
        model_id: Base HuggingFace model ID.
        num_classes: Number of output classes.
        adapter_path: Path to saved PEFT adapter directory.

    Returns:
        (model, tokenizer) tuple. Model is in eval mode.
    """
    import os
    from peft import PeftModel

    # PEFT treats relative paths as HuggingFace repo IDs — resolve to absolute
    adapter_path = os.path.abspath(adapter_path)

    if not os.path.isdir(adapter_path):
        raise FileNotFoundError(
            f"No QLoRA adapter found at {adapter_path}. "
            "Train with: python -m src.train_qlora"
        )

    hf_token = os.environ.get("HF_TOKEN")

    tokenizer = AutoTokenizer.from_pretrained(model_id, token=hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_id,
        num_labels=num_classes,
        quantization_config=bnb_config,
        device_map={"":0},
        token=hf_token,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id

    model = PeftModel.from_pretrained(base_model, adapter_path)
    model.eval()
    return model, tokenizer
