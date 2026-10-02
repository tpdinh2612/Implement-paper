"""
Shared ASR Backbone: Whisper model wrapper.
Used by all three paper implementations as the base ASR system.
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)
from tqdm import tqdm

from shared.config import ASRConfig


class WhisperASR:
    """
    Whisper-based ASR model wrapper.
    Shared backbone across all three papers.
    """

    def __init__(self, config: ASRConfig):
        self.config = config
        self.device = self._get_device(config.device)

        print(f"Loading Whisper model: {config.model_name}")
        self.processor = WhisperProcessor.from_pretrained(config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()

        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=config.language, task=config.task
        )

    def _get_device(self, device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """Transcribe a single audio sample."""
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        generated_ids = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            max_length=self.config.max_length,
        )

        transcription = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]
        return transcription.strip()

    @torch.no_grad()
    def transcribe_batch(
        self, audios: List[np.ndarray], sample_rate: int = 16000
    ) -> List[str]:
        """Transcribe a batch of audio samples."""
        inputs = self.processor(
            audios, sampling_rate=sample_rate, return_tensors="pt", padding=True
        )
        input_features = inputs.input_features.to(self.device)

        generated_ids = self.model.generate(
            input_features,
            forced_decoder_ids=self.forced_decoder_ids,
            max_length=self.config.max_length,
        )

        transcriptions = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        return [t.strip() for t in transcriptions]

    @torch.no_grad()
    def get_logits_and_hidden(
        self,
        audio: np.ndarray,
        decoder_input_ids: torch.Tensor,
        sample_rate: int = 16000,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get logit scores and hidden states from ASR decoder.
        Used for token-level and phrase-level fusion.
        """
        inputs = self.processor(
            audio, sampling_rate=sample_rate, return_tensors="pt"
        )
        input_features = inputs.input_features.to(self.device)

        outputs = self.model(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids.to(self.device),
            output_hidden_states=True,
        )

        logits = outputs.logits
        hidden = outputs.decoder_hidden_states[-1]

        return logits, hidden
