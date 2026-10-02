"""
PEEKING-FUTURE: Peeking into the Future for Contextual Biasing
Level-1 architectural reimplementation using Whisper backbone for VietMed-NER.

Paper: Selvakumar et al., arXiv:2512.17657, Dec 2025

This is a Level-1 architectural reimplementation of PEEKING-FUTURE's 
Multi-Token Prediction (MTP), Entity Scoring, and Unified Search decoding loop.
It uses a pretrained Whisper backbone and randomly initialized MTP/Scorer modules 
to demonstrate the tensor flow (Eq. 4-12) during inference.

Core components:
1. MTPModule (Eq 4-6): K FFN heads predict K future tokens in parallel from h_d_s.
2. EntityScorer (Eq 7-8): Learned FFN f_theta scores entity representations.
3. Unified Search (Eq 11-12): Decoder loop fuses static vocab probs and dynamic entity probs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import random

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import ASRConfig, PeekingFutureConfig
from shared.asr_backbone import WhisperASR
from shared.data_loader import extract_entities_from_sample


class MTPModule(nn.Module):
    """
    Multi-Token Prediction module (Section 3.1, Eq. 4-6).
    K FFN heads that each predict a different future token from decoder hidden state h^d_s.
    """
    def __init__(self, hidden_dim: int, lm_head: nn.Linear, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.g = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(num_heads)
        ])
        # Shared final output projection W_o from the backbone
        self.output_projection = lm_head

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: [B, D] decoder hidden state at step s
        Returns:
            logits: [B, K, V] future token logits
        """
        logits = []
        for head in self.g:
            z = head(h)
            l = self.output_projection(z)
            logits.append(l)
        return torch.stack(logits, dim=1)


class EntityScorer(nn.Module):
    """
    Entity Scoring module (Section 3.2, Eq. 7-8).
    Scores an entity based on its collected future logits.
    """
    def __init__(self, K: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, entity_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            entity_logits: [B, N, K] where N is number of entities (including null)
        Returns:
            scores: [B, N]
        """
        return self.net(entity_logits).squeeze(-1)


def collect_entity_logits(
    future_logits: torch.Tensor, 
    entity_token_ids: List[int], 
) -> torch.Tensor:
    """
    Implements Eq. 7: Gather logits for entity subwords across the K MTP heads.
    Args:
        future_logits: [K, V]
        entity_token_ids: list of token IDs for the entity
    Returns:
        p_n: [K] logit values (padded with 0 if length < K)
    """
    K = future_logits.shape[0]
    L = min(len(entity_token_ids), K)
        
    values = []
    for k in range(L):
        token_id = entity_token_ids[k]
        values.append(future_logits[k, token_id])
        
    if L < K:
        # Mask out missing positions with 0
        values.extend([future_logits.new_zeros(())] * (K - L))
        
    return torch.stack(values)


def unified_search(
    token_probs: torch.Tensor,
    entity_probs: torch.Tensor,
    lambda_: float,
    gamma: float,
    null_idx: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Unified Search Space (Section 3.4, Eq. 11-12).
    """
    N = entity_probs.shape[0]
    
    # max Pe(E) for E in entities (excluding null)
    if N > 1:
        entity_probs_except_null = entity_probs[..., :null_idx]
        best_entity_prob = entity_probs_except_null.max(dim=-1).values
        
        # Apply pruning threshold gamma
        if best_entity_prob < gamma:
            entity_probs = torch.zeros_like(entity_probs)
            entity_probs[..., null_idx] = 1.0
            
    pe_null = entity_probs[..., null_idx]
    
    # Q(i) = Pe(empty)*P1(i) for static vocab
    Q_vocab = pe_null * token_probs
    # Q(E) = lambda * Pe(E) for entities
    Q_entity = lambda_ * entity_probs
    
    # Null is not an entity candidate to emit
    Q_entity = Q_entity.clone()
    Q_entity[..., null_idx] = -float("inf")
    
    return Q_vocab, Q_entity


def build_bias_list(positive_entities: List[str], all_entities: List[str], kappa: int = 2) -> List[str]:
    """Builds bias list with positives and sampled negatives (kappa=2)."""
    pos_set = set(positive_entities)
    negatives = [e for e in all_entities if e not in pos_set]
    
    rng = random.Random(42)
    num_negs = min(len(negatives), kappa * len(positive_entities))
    sampled_negs = rng.sample(negatives, num_negs) if num_negs > 0 else []
    
    return list(pos_set) + sampled_negs


class PeekingFutureInference:
    """
    ToyPeekingFuture Implementation for educational purposes (Level 1).
    Runs manual greedy decoding over Whisper's decoder loop to demonstrate
    the MTP tensor flow and Unified Search.
    
    Note: MTP and Scorer are randomly initialized, so entity probabilities
    will be arbitrary. This class is for architectural understanding.
    """
    def __init__(self, asr_config: ASRConfig, peeking_future_config: PeekingFutureConfig):
        self.asr = WhisperASR(asr_config)
        self.config = peeking_future_config
        self.device = self.asr.device
        
        self.tokenizer = self.asr.processor.tokenizer
        
        hidden_dim = self.asr.model.config.d_model
        vocab_size = self.asr.model.config.vocab_size
        
        # Pass the backbone's lm_head directly
        lm_head = getattr(self.asr.model, "proj_out", None) or getattr(self.asr.model, "lm_head", None)
        if lm_head is None:
            # Fallback
            lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
            
        self.mtp = MTPModule(hidden_dim, lm_head, num_heads=self.config.num_mtp_heads).to(self.device)
        self.scorer = EntityScorer(K=self.config.num_mtp_heads).to(self.device)
            
        self.pad_token_id = self.tokenizer.pad_token_id or 0
        
        # Global entity list for negative sampling
        self.global_entities = []

    def set_global_entities(self, entities: List[str]):
        self.global_entities = entities

    @torch.no_grad()
    def transcribe(
        self,
        audio: np.ndarray,
        entity_list: Optional[List[str]] = None,
        sample_rate: int = 16000,
        debug: bool = False,
        return_baseline: bool = False,
    ) -> Dict:
        """
        Transcribe with PEEKING-FUTURE manual decoding loop.
        """
        baseline_text = None
        if return_baseline:
            baseline_text = self.asr.transcribe(audio, sample_rate)
        
        if not entity_list:
            if baseline_text is None:
                baseline_text = self.asr.transcribe(audio, sample_rate)
            return {"text": baseline_text, "mode": "peeking_future_baseline", "baseline_text": baseline_text}

        # Build bias list with negatives (kappa=2)
        bias_list = build_bias_list(entity_list, self.global_entities, kappa=2)
        if not bias_list:
            bias_list = entity_list
            
        # Tokenize entities
        entity_token_ids_list = []
        for e in bias_list:
            tokens = self.tokenizer.encode(e, add_special_tokens=False)
            entity_token_ids_list.append(tokens)
            
        N_entities = len(bias_list)
        null_idx = N_entities
        
        # Prepare Whisper Inputs
        inputs = self.asr.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        encoder_outputs = self.asr.model.get_encoder()(input_features)
        
        # Determine prompt tokens dynamically
        forced_decoder_ids = self.asr.model.generation_config.forced_decoder_ids or []
        prompt_ids = [token_id for _, token_id in forced_decoder_ids if token_id is not None]
        
        decoder_input_ids = [self.asr.model.config.decoder_start_token_id] + prompt_ids
        
        MAX_DECISIONS = 100
        output_ids = []
        
        lambda_ = self.config.biasing_weight
        gamma = self.config.confidence_threshold
        
        for decision_step in range(MAX_DECISIONS):
            input_tensor = torch.tensor([decoder_input_ids]).to(self.device)
            
            # Forward pass decoder (inefficient no-KV-cache mode for architectural demo)
            decoder_outputs = self.asr.model.model.decoder(
                input_ids=input_tensor,
                encoder_hidden_states=encoder_outputs.last_hidden_state
            )
            # h^d_s
            h_d_s = decoder_outputs.last_hidden_state[:, -1, :] # [1, D]
            
            # Normal LM logits
            lm_logits = self.mtp.output_projection(h_d_s) # [1, V]
            token_probs = F.softmax(lm_logits[0], dim=-1) # [V]
            
            # MTP logits
            mtp_logits = self.mtp(h_d_s) # [1, K, V]
            
            # Gather entity logits p_n
            entity_logits_list = []
            for e_tokens in entity_token_ids_list:
                p_n = collect_entity_logits(mtp_logits[0], e_tokens)
                entity_logits_list.append(p_n)
                
            # Null entity (zero representation)
            p_null = torch.zeros(self.config.num_mtp_heads, device=self.device)
            entity_logits_list.append(p_null)
            
            entity_logits_tensor = torch.stack(entity_logits_list).unsqueeze(0) # [1, N+1, K]
            
            # Score entities Pe
            entity_scores = self.scorer(entity_logits_tensor)[0] # [N+1]
            entity_probs = F.softmax(entity_scores, dim=-1) # [N+1]
            
            # Unified Search
            Q_vocab, Q_entity = unified_search(
                token_probs, entity_probs, lambda_, gamma, null_idx
            )
            
            best_token = torch.argmax(Q_vocab).item()
            best_entity = torch.argmax(Q_entity).item()
            
            if debug and decision_step < 5:
                print(f"\n--- Decision Step {decision_step} ---")
                print(f"h_d_s shape: {h_d_s.shape}")
                print(f"mtp_logits shape: {mtp_logits.shape}")
                print(f"entity_logits shape: {entity_logits_tensor.shape}")
                print(f"Pe(null): {entity_probs[null_idx].item():.4f}")
                if N_entities > 0:
                    print(f"Max Pe(entity): {entity_probs[:null_idx].max().item():.4f}")
                print(f"Q_vocab max: {Q_vocab.max().item():.4f}, Q_entity max: {Q_entity.max().item():.4f}")
            
            # Select best candidate
            if best_entity == null_idx or Q_vocab[best_token] >= Q_entity[best_entity]:
                decoder_input_ids.append(best_token)
                output_ids.append(best_token)
                if best_token == self.tokenizer.eos_token_id:
                    break
            else:
                # Emit whole entity!
                e_tokens = entity_token_ids_list[best_entity]
                decoder_input_ids.extend(e_tokens)
                output_ids.extend(e_tokens)
                if debug:
                    print(f"EMITTING ENTITY: {bias_list[best_entity]}")
                    
        biased_text = self.tokenizer.decode(output_ids, skip_special_tokens=True)
        
        return {
            "text": biased_text,
            "mode": "peeking_future_architecture",
            "baseline_text": baseline_text or ""
        }

    def transcribe_dataset(
        self,
        dataset,
        entity_list: Optional[List[str]] = None,
        show_progress: bool = True,
        debug: bool = False,
    ) -> List[Dict]:
        """Transcribe dataset with PEEKING-FUTURE."""
        # Collect all entities in dataset for negative sampling
        all_dataset_entities = []
        for sample in dataset:
            ents = extract_entities_from_sample(sample)
            all_dataset_entities.extend([e["text"] for e in ents])
        
        combined_global = list(set((entity_list or []) + all_dataset_entities))
        self.set_global_entities(combined_global)

        results = []
        iterator = tqdm(dataset, desc="PEEKING-FUTURE inference (Architecture Level 1)") if show_progress else dataset

        for i, sample in enumerate(iterator):
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            sample_entities = extract_entities_from_sample(sample)
            positives = [e["text"] for e in sample_entities]

            result = self.transcribe(
                audio=audio,
                entity_list=positives,
                sample_rate=sr,
                debug=debug and i < 3, # print debug info for first 3 samples
            )
            result["reference"] = sample["text"]
            result["words"] = sample["words"]
            result["labels"] = sample["labels"]
            results.append(result)

        return results
