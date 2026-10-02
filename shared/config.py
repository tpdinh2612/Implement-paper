"""
Shared Configuration Module.
Centralized configs used across all three paper implementations:
- Multi-grained Fusion (Multigrained)
- CopyNE
- PEEKING-FUTURE

Adapted for Vietnamese medical domain using VietMed-NER dataset.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ASRConfig:
    """Configuration for the ASR backbone model (shared across all papers)."""
    model_name: str = "openai/whisper-small"
    language: str = "vi"
    task: str = "transcribe"
    device: str = "cuda"  # will fallback to cpu if cuda unavailable
    batch_size: int = 8
    max_length: int = 448
    beam_size: int = 5


@dataclass
class LLMConfig:
    """Configuration for the LLM used in token-level fusion (Multigrained paper)."""
    model_name: str = "Qwen/Qwen2-0.5B"
    device: str = "cuda"
    max_length: int = 512


@dataclass
class KeywordEncoderConfig:
    """Configuration for the keyword LSTM encoder (CopyNE & Multigrained)."""
    embedding_dim: int = 256
    hidden_dim: int = 256
    num_layers: int = 3
    dropout: float = 0.1


@dataclass
class FusionConfig:
    """Configuration for multi-grained fusion (Multigrained paper)."""
    token_fusion_enabled: bool = True
    phrase_fusion_enabled: bool = True
    phrase_query_dim: int = 256
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    num_epochs: int = 40
    warmup_epochs: int = 4
    batch_size: int = 8
    max_keywords: int = 50


@dataclass
class CopyNEConfig:
    """Configuration specific to CopyNE paper."""
    copy_threshold: float = 0.9  # gamma threshold for copy confidence
    ne_encoder_layers: int = 3
    ne_encoder_hidden_dim: int = 256
    ne_encoder_embedding_dim: int = 256
    max_entities: int = 50
    use_dict_repr: bool = True  # whether to use Dict representation ru
    similarity_threshold: float = 0.7  # for fuzzy matching in inference


@dataclass
class PeekingFutureConfig:
    """Configuration specific to PEEKING-FUTURE paper (multi-token prediction)."""
    num_mtp_heads: int = 4  # K = 4 future tokens
    biasing_weight: float = 4.4  # lambda for entity score boosting
    confidence_threshold: float = 0.3  # gamma for pruning
    max_entities: int = 50
    mtp_head_type: str = "linear"  # "linear" or "transformer"
    entity_scorer_hidden_dim: int = 128


@dataclass
class DataConfig:
    """Configuration for dataset handling (shared)."""
    dataset_name: str = "leduckhai/VietMed-NER"
    cache_dir: str = "./cache"
    output_dir: str = "./outputs"
    sample_rate: int = 16000
    entity_types: List[str] = field(default_factory=lambda: [
        "DISEASESYMTOM", "DATETIME", "ORGAN", "UNITCALIBRATOR",
        "DRUG", "PROCEDURE", "LOCATION", "PERSON",
        "ORGANIZATION", "QUANTITY", "DURATION", "FREQUENCY",
        "AGE", "GENDER", "OCCUPATION", "MEASUREMENT",
        "BODYPART", "MEDICALTOOL",
    ])


@dataclass
class EvalConfig:
    """Configuration for evaluation metrics (shared)."""
    output_dir: str = "./outputs"
    compute_cer: bool = True
    compute_wer: bool = True
    compute_b_cer: bool = True
    compute_u_cer: bool = True
    compute_recall: bool = True
    compute_ner_f1: bool = True
    compute_entity_metrics: bool = True


@dataclass
class FullConfig:
    """Full configuration combining all sub-configs."""
    asr: ASRConfig = field(default_factory=ASRConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    keyword_encoder: KeywordEncoderConfig = field(default_factory=KeywordEncoderConfig)
    fusion: FusionConfig = field(default_factory=FusionConfig)
    copyne: CopyNEConfig = field(default_factory=CopyNEConfig)
    peeking_future: PeekingFutureConfig = field(default_factory=PeekingFutureConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def __post_init__(self):
        os.makedirs(self.data.cache_dir, exist_ok=True)
        os.makedirs(self.data.output_dir, exist_ok=True)
        os.makedirs(self.eval.output_dir, exist_ok=True)
