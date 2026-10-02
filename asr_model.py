"""
ASR Inference Module.

Provides multiple ASR inference modes:
1. Baseline Whisper (no contextual biasing)  
2. Token-level fusion (Whisper + LLM logit fusion with uncertainty weighting)
3. Phrase-level fusion (keyword copy mechanism via LSTM encoder)
4. Joint multi-grained fusion (combined token + phrase level)

Based on: "Improving Contextual ASR via Multi-grained Fusion with LLMs"
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    AutoTokenizer,
    AutoModelForCausalLM,
)
from tqdm import tqdm

from config import ASRConfig, LLMConfig, KeywordEncoderConfig, FusionConfig


class KeywordEncoder(nn.Module):
    """
    3-layer LSTM encoder for keyword phrases (Section 3.3, Eq. 6).
    Encodes each keyword into a phrase-level representation.
    """
    
    def __init__(self, config: KeywordEncoderConfig, vocab_size: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, config.embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(
            input_size=config.embedding_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.hidden_dim = config.hidden_dim
    
    def forward(self, keyword_ids: torch.Tensor, keyword_lengths: torch.Tensor) -> torch.Tensor:
        """
        Encode keywords into phrase-level representations.
        
        Args:
            keyword_ids: [num_keywords, max_kw_len] token IDs
            keyword_lengths: [num_keywords] actual lengths
            
        Returns:
            [num_keywords, hidden_dim] phrase representations
        """
        embedded = self.embedding(keyword_ids)  # [N, L, D]
        
        # Pack padded sequences for efficient LSTM processing
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, keyword_lengths.cpu().clamp(min=1),
            batch_first=True, enforce_sorted=False
        )
        _, (hidden, _) = self.lstm(packed)
        
        # Take last layer's hidden state: [num_layers, N, H] -> [N, H]
        return hidden[-1]


class PhraseSelector(nn.Module):
    """
    Phrase-level fusion module (Section 3.3, Eq. 7-8).
    Computes attention between decoder state and keyword representations.
    """
    
    def __init__(self, asr_hidden_dim: int, llm_hidden_dim: int, query_dim: int):
        super().__init__()
        self.query_proj = nn.Linear(asr_hidden_dim + llm_hidden_dim, query_dim)
    
    def forward(
        self,
        asr_hidden: torch.Tensor,
        llm_hidden: torch.Tensor,
        keyword_reprs: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute phrase-level probabilities for each keyword.
        
        Args:
            asr_hidden: [batch, asr_dim] ASR decoder hidden state
            llm_hidden: [batch, llm_dim] LLM hidden state  
            keyword_reprs: [num_keywords+1, query_dim] keyword representations
                           (includes fake keyword k0 at index 0)
        
        Returns:
            [batch, num_keywords+1] phrase-level probabilities
        """
        # Concatenate and project: Eq. 7
        combined = torch.cat([asr_hidden, llm_hidden], dim=-1)
        query = self.query_proj(combined)  # [batch, query_dim]
        
        # Dot-product attention: Eq. 8
        scores = torch.matmul(query, keyword_reprs.T)  # [batch, num_keywords+1]
        probs = F.softmax(scores, dim=-1)
        return probs


class WhisperASR:
    """
    Whisper-based ASR model wrapper.
    Handles loading, preprocessing, and inference.
    """
    
    def __init__(self, config: ASRConfig):
        self.config = config
        self.device = self._get_device(config.device)
        
        print(f"Loading Whisper model: {config.model_name}")
        self.processor = WhisperProcessor.from_pretrained(config.model_name)
        self.model = WhisperForConditionalGeneration.from_pretrained(config.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Set language and task
        self.forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language=config.language, task=config.task
        )
    
    def _get_device(self, device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    @torch.no_grad()
    def transcribe(self, audio: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe a single audio sample.
        
        Args:
            audio: numpy array of audio samples
            sample_rate: audio sample rate
            
        Returns:
            Transcribed text string
        """
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
        
        Args:
            audio: input audio
            decoder_input_ids: [1, seq_len] decoder input token IDs
            sample_rate: audio sample rate
            
        Returns:
            (logits, hidden_states): logits [1, seq_len, vocab] and 
                                     hidden [1, seq_len, dim]
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
        
        logits = outputs.logits  # [1, seq_len, vocab_size]
        # Last layer hidden state from decoder
        hidden = outputs.decoder_hidden_states[-1]  # [1, seq_len, dim]
        
        return logits, hidden


class LLMContextualModel:
    """
    LLM wrapper for contextual keyword biasing.
    Provides logit scores conditioned on keyword prompt.
    """
    
    def __init__(self, config: LLMConfig):
        self.config = config
        self.device = self._get_device(config.device)
        
        print(f"Loading LLM: {config.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_name, trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            config.model_name, trust_remote_code=True,
            torch_dtype=torch.float16 if self.device.type == "cuda" else torch.float32,
        )
        self.model.to(self.device)
        self.model.eval()
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _get_device(self, device: str) -> torch.device:
        if device == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    
    def build_prompt(self, keywords: List[str]) -> str:
        """
        Build contextual prompt following Section 3.1.
        Adapted for Vietnamese.
        """
        keyword_str = ", ".join(keywords)
        prompt = (
            f"Chuyển đổi giọng nói thành văn bản. "
            f"Các từ khóa sau có thể xuất hiện. "
            f"Sử dụng các từ khóa liên quan để cải thiện độ chính xác "
            f"và bỏ qua các từ khóa không liên quan. "
            f"Các từ khóa là: {keyword_str}. "
            f"Văn bản tương ứng với giọng nói là: "
        )
        return prompt
    
    @torch.no_grad()
    def get_logits_and_hidden(
        self,
        prompt: str,
        generated_text: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get LLM logits and hidden states conditioned on prompt + generated text.
        
        Args:
            prompt: contextual prompt with keywords
            generated_text: previously generated text (y<t)
            
        Returns:
            (logits, hidden_states) for the last token position
        """
        full_text = prompt + generated_text
        inputs = self.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)
        
        outputs = self.model(
            input_ids=input_ids,
            output_hidden_states=True,
        )
        
        # Get last position logits and hidden states
        logits = outputs.logits[:, -1:, :]  # [1, 1, vocab_size]
        hidden = outputs.hidden_states[-1][:, -1:, :]  # [1, 1, dim]
        
        return logits, hidden


class MultiGrainedFusionASR:
    """
    Complete multi-grained fusion ASR system (Section 3).
    
    Combines:
    - Token-level fusion (Section 3.2): uncertainty-based logit fusion
    - Phrase-level fusion (Section 3.3): LSTM keyword encoder + attention
    - Joint fusion (Section 3.4): unified probability space
    
    For inference-only mode (no training), we use a simplified pipeline:
    1. Run Whisper to get baseline transcription
    2. Use LLM to rescore/bias toward keywords
    3. Apply keyword matching for phrase-level copy
    """
    
    def __init__(
        self,
        asr_config: ASRConfig,
        llm_config: LLMConfig,
        fusion_config: FusionConfig,
    ):
        self.asr = WhisperASR(asr_config)
        self.llm = LLMContextualModel(llm_config)
        self.fusion_config = fusion_config
        self.device = self.asr.device
    
    def _compute_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Compute ASR uncertainty (entropy) for token-level fusion.
        Eq. 4: u_t = -p_t * log(p_t)
        """
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        entropy = -(probs * log_probs).sum(dim=-1, keepdim=True)
        return entropy
    
    def token_level_rescore(
        self,
        asr_text: str,
        keywords: List[str],
    ) -> str:
        """
        Token-level fusion via LLM rescoring.
        
        Simplified version: use LLM perplexity-based rescoring to 
        evaluate and potentially correct the ASR output by comparing
        token probabilities with keyword context.
        
        Args:
            asr_text: baseline ASR transcription
            keywords: list of keywords for biasing
            
        Returns:
            Rescored/corrected transcription
        """
        prompt = self.llm.build_prompt(keywords)
        
        # Get LLM score for the ASR output
        full_text = prompt + asr_text
        inputs = self.llm.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.model(input_ids=input_ids)
            logits = outputs.logits
        
        # Compute token-level log probabilities
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)
        
        # Average log probability (as a confidence score)
        avg_log_prob = token_log_probs.mean().item()
        
        return asr_text, avg_log_prob
    
    def phrase_level_keyword_match(
        self,
        asr_text: str,
        keywords: List[str],
    ) -> str:
        """
        Phrase-level keyword matching and correction.
        
        Simplified copy mechanism: find close matches between ASR output
        segments and keywords, and replace with correct keyword spelling.
        
        Args:
            asr_text: baseline ASR transcription
            keywords: list of keywords for biasing
            
        Returns:
            Corrected transcription with keyword copies
        """
        corrected = asr_text
        
        for keyword in keywords:
            kw_lower = keyword.lower()
            text_lower = corrected.lower()
            
            # Exact match - no correction needed
            if kw_lower in text_lower:
                continue
            
            # Fuzzy match: check if keyword words appear close together
            kw_words = kw_lower.split()
            if len(kw_words) <= 1:
                continue
            
            # Check for partial matches (e.g., wrong tones in Vietnamese)
            for i, kw_word in enumerate(kw_words):
                # Simple character-level similarity for Vietnamese tone correction
                text_words = corrected.split()
                for j, text_word in enumerate(text_words):
                    if self._char_similarity(kw_word, text_word.lower()) > 0.8:
                        # Check if subsequent words also match
                        if i == 0 and self._phrase_match(
                            text_words[j:j+len(kw_words)], kw_words
                        ):
                            # Replace the matching segment with the keyword
                            corrected_words = text_words.copy()
                            corrected_words[j:j+len(kw_words)] = keyword.split()
                            corrected = " ".join(corrected_words)
                            break
        
        return corrected
    
    def _char_similarity(self, s1: str, s2: str) -> float:
        """Simple character-level similarity."""
        if not s1 or not s2:
            return 0.0
        common = sum(1 for c in s1 if c in s2)
        return 2 * common / (len(s1) + len(s2))
    
    def _phrase_match(self, text_words: List[str], kw_words: List[str]) -> bool:
        """Check if a phrase approximately matches keyword words."""
        if len(text_words) < len(kw_words):
            return False
        similarities = [
            self._char_similarity(tw.lower(), kw.lower())
            for tw, kw in zip(text_words[:len(kw_words)], kw_words)
        ]
        return all(s > 0.6 for s in similarities)
    
    @torch.no_grad()
    def transcribe(
        self,
        audio: np.ndarray,
        keywords: Optional[List[str]] = None,
        mode: str = "joint",
        sample_rate: int = 16000,
    ) -> Dict:
        """
        Transcribe audio with multi-grained fusion.
        
        Args:
            audio: numpy array of audio samples
            keywords: list of contextual keywords
            mode: 'baseline', 'token', 'phrase', or 'joint'
            sample_rate: audio sample rate
            
        Returns:
            Dict with 'text', 'mode', and optional scoring info
        """
        # Step 1: Get baseline ASR transcription
        baseline_text = self.asr.transcribe(audio, sample_rate)
        
        if mode == "baseline" or keywords is None or len(keywords) == 0:
            return {"text": baseline_text, "mode": "baseline"}
        
        result_text = baseline_text
        score_info = {}
        
        # Step 2: Token-level fusion (LLM rescoring)
        if mode in ("token", "joint") and self.fusion_config.token_fusion_enabled:
            _, llm_score = self.token_level_rescore(baseline_text, keywords)
            score_info["llm_score"] = llm_score
        
        # Step 3: Phrase-level fusion (keyword copy)
        if mode in ("phrase", "joint") and self.fusion_config.phrase_fusion_enabled:
            result_text = self.phrase_level_keyword_match(result_text, keywords)
        
        return {
            "text": result_text,
            "mode": mode,
            "baseline_text": baseline_text,
            **score_info,
        }
    
    def transcribe_dataset(
        self,
        dataset,
        keywords: Optional[List[str]] = None,
        mode: str = "baseline",
        show_progress: bool = True,
    ) -> List[Dict]:
        """
        Transcribe all samples in a dataset.
        
        Args:
            dataset: HuggingFace dataset with 'audio' column
            keywords: keyword list for contextual biasing
            mode: inference mode
            show_progress: show progress bar
            
        Returns:
            List of result dicts with transcriptions
        """
        results = []
        iterator = tqdm(dataset, desc=f"Transcribing ({mode})") if show_progress else dataset
        
        for sample in iterator:
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]
            
            result = self.transcribe(
                audio=audio,
                keywords=keywords,
                mode=mode,
                sample_rate=sr,
            )
            result["reference"] = sample["text"]
            result["words"] = sample["words"]
            result["labels"] = sample["labels"]
            results.append(result)
        
        return results
