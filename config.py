"""
Configuration for the ASR system re-implementation.
Based on: "Improving Contextual ASR via Multi-grained Fusion with Large Language Models"

Adapted for Vietnamese medical domain using VietMed-NER dataset.
Uses smaller models suitable for inference without heavy GPU requirements.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ASRConfig:
    """Configuration for the ASR backbone model."""
    # Using whisper-small for lighter inference (244M params)
    # Paper uses whisper-small as well
    model_name: str = "openai/whisper-small"
    language: str = "vi"
    task: str = "transcribe"
    device: str = "cuda"  # will fallback to cpu if cuda unavailable
    batch_size: int = 8
    max_length: int = 448  # max tokens for generation
    beam_size: int = 5


@dataclass
class LLMConfig:
    """Configuration for the LLM used in token-level fusion."""
    # Using Qwen2-0.5B for lighter inference (paper uses Qwen2-1.5B for Chinese)
    # For Vietnamese, Qwen2 has decent multilingual support
    model_name: str = "Qwen/Qwen2-0.5B"
    device: str = "cuda"
    max_length: int = 512


@dataclass
class KeywordEncoderConfig:
    """Configuration for the keyword LSTM encoder (phrase-level fusion)."""
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1


@dataclass
class FusionConfig:
    """Configuration for multi-grained fusion."""
    # Token-level fusion weight (uncertainty-based, so this is initial scaling)
    token_fusion_enabled: bool = True
    phrase_fusion_enabled: bool = True

    # Phrase-level fusion
    phrase_query_dim: int = 256  # dimension of the fused query vector

    # Training
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 40
    warmup_epochs: int = 4
    batch_size: int = 8  # reduced from paper's 60 for smaller GPUs

    # Keyword list
    max_keywords: int = 50  # default keyword list size


@dataclass
class DataConfig:
    """Configuration for dataset handling."""
    dataset_name: str = "leduckhai/VietMed-NER"
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    sample_rate: int = 16000

    # NER label mapping for VietMed-NER (18 entity types)
    # BIO scheme: B-X, I-X, O
    entity_types: List[str] = field(default_factory=lambda: [
        "DISEASESYMTOM",     # Disease/Symptom
        "DATETIME",          # Date/Time
        "ORGAN",             # Body organ
        "UNITCALIBRATOR",    # Unit/Calibrator
        "DRUG",              # Drug/Medicine
        "PROCEDURE",         # Medical procedure
        "LOCATION",          # Location
        "PERSON",            # Person
        "ORGANIZATION",      # Organization
        "QUANTITY",          # Quantity
        "DURATION",          # Duration
        "FREQUENCY",         # Frequency
        "AGE",               # Age
        "GENDER",            # Gender
        "OCCUPATION",        # Occupation
        "MEASUREMENT",       # Measurement
        "BODYPART",          # Body part
        "MEDICALTOOL",       # Medical tool
    ])


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics."""
    output_dir: str = "./outputs"
    # Metrics to compute (following paper)
    compute_cer: bool = True       # Character Error Rate (for Vietnamese)
    compute_wer: bool = True       # Word Error Rate
    compute_b_cer: bool = True     # Biased CER (keyword spans only)
    compute_u_cer: bool = True     # Unbiased CER (non-keyword spans only)
    compute_recall: bool = True    # Keyword Recall
    compute_ner_f1: bool = True    # NER F1 (from VietMed-NER paper)
    compute_entity_metrics: bool = True  # Per-entity-type metrics


@dataclass
class FullConfig:
    """Full configuration combining all sub-configs."""
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    keyword_encoder: KeywordEncoderConfig = field(default_factory=KeywordEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        os.makedirs(self.data.cache_dir, exist_ok=True)
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.eval.output_dir, exist_ok=True)
