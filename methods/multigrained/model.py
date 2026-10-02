"""
Multi-grained Fusion: Improving Contextual ASR via Multi-grained Fusion with LLMs
Re-implementation for inference on VietMed-NER dataset.

Paper: Zhou et al., arXiv:2507.12252, Jul 2025

Simplified inference-time prototype of Multi-Grained Fusion contextual ASR.
This implementation demonstrates:
1. Whisper ASR baseline
2. LLM contextual prompting
3. LLM-based candidate scoring
4. Heuristic phrase-level keyword correction
It does NOT reproduce the paper's full training. But it implements the structural architecture for demonstration:
- 3-layer LSTM keyword encoder
- token-level logit fusion
- ASR/LLM hidden-state fusion
- joint V ∪ K decoding space
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from shared.config import ASRConfig, LLMConfig, FusionConfig
from shared.asr_backbone import WhisperASR


class KeywordEncoder(nn.Module):
    """3-layer LSTM keyword encoder for phrase-level fusion."""
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=3, batch_first=True
        )

    def forward(self, keyword_ids):
        # keyword_ids: (batch, seq_len)
        x = self.embedding(keyword_ids)
        _, (h, _) = self.lstm(x)
        # return the hidden state of the last layer
        return h[-1]


class PhraseFusion(nn.Module):
    """Phrase-level fusion via dot-product attention."""
    def __init__(self, llm_dim, asr_dim, phrase_dim):
        super().__init__()
        self.query = nn.Linear(llm_dim + asr_dim, phrase_dim)

    def forward(self, h_llm, h_asr, phrase_repr):
        # h_llm, h_asr: (batch, dim)
        # phrase_repr: (num_keywords, phrase_dim)
        q = self.query(torch.cat([h_llm, h_asr], dim=-1))
        # scores: (batch, num_keywords)
        scores = torch.matmul(phrase_repr, q.unsqueeze(-1)).squeeze(-1)
        return F.softmax(scores, dim=-1)


class LLMContextualModel:
    """LLM wrapper for contextual keyword biasing (Section 3.1-3.2)."""

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
        """Build contextual prompt following Section 3.1. Adapted for Vietnamese."""
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
        self, prompt: str, generated_text: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get LLM logits and hidden states conditioned on prompt + generated text."""
        full_text = prompt + generated_text
        inputs = self.tokenizer(full_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.device)

        outputs = self.model(input_ids=input_ids, output_hidden_states=True)

        logits = outputs.logits[:, -1:, :]
        hidden = outputs.hidden_states[-1][:, -1:, :]

        return logits, hidden


class MultiGrainedFusionASR:
    """
    This prototype implements the STRUCTURAL components of the paper:
    - 3-layer LSTM keyword encoder
    - token-level uncertainty-weighted fusion
    - ASR/LLM hidden-state fusion
    - phrase-level attention
    - joint V ∪ K probability construction
    
    However, these components are provided as architectural demonstrations and are NOT fully integrated 
    into the actual Whisper decoding loop.
    
    It does NOT reproduce:
    - the trained fusion parameters
    - the original training procedure
    - the original beam-search decoder
    - the original experimental results
    
    The active inference pipeline uses a lightweight heuristic approximation for keyword scoring, 
    while the full architecture classes are provided for educational/debug purposes.
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
        
        # Instantiate architectural components for demonstration
        # Note: These are untrained in this prototype
        vocab_size = 50000 # Dummy vocab size for prototype
        llm_dim = self.llm.model.config.hidden_size
        asr_dim = self.asr.model.config.d_model
        phrase_dim = 256
        
        self.keyword_encoder = KeywordEncoder(vocab_size, embed_dim=256, hidden_dim=phrase_dim).to(self.device)
        self.phrase_fusion = PhraseFusion(llm_dim, asr_dim, phrase_dim).to(self.device)


    def _compute_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute ASR uncertainty aligned with token vocabulary distribution."""
        probs = F.softmax(logits, dim=-1)
        log_probs = torch.log(probs + 1e-10)
        # Expected u_t^a = -p_t^a \log p_t^a as per Eq. 4 used in fusion
        uncertainty = -probs * log_probs
        return uncertainty
        
    def token_fusion(self, asr_logits: torch.Tensor, llm_logits: torch.Tensor) -> torch.Tensor:
        """
        Token-level fusion via uncertainty weighting (Eq 4).
        s_t = s_t^a + \sigma(u_t^a)s_t^l
        
        IMPORTANT: ASR and LLM tokenizers must share/align vocabulary 
        before logits can be successfully fused in a real implementation.
        """
        uncertainty = self._compute_uncertainty(asr_logits)
        fused_logits = asr_logits + torch.sigmoid(uncertainty) * llm_logits
        return F.softmax(fused_logits, dim=-1)

    def demo_joint_decoding_step(self, asr_logits, asr_hidden, llm_logits, llm_hidden, keyword_ids):
        """
        Architectural demonstration only. This function is not used by the simplified transcription pipeline.
        
        Demonstrate a single decoding step in the unified space Z = V ∪ K.
        This is an architectural prototype of the joint multi-grained fusion.
        """
        with torch.no_grad():
            # 1. Token-level fusion
            p_token = self.token_fusion(asr_logits, llm_logits)
            
            # 2. Phrase-level fusion
            # Assume keyword_ids is a list of tensors for each keyword + k0
            # k0 (NORMAL_TOKEN) is always the first item
            phrase_reprs = []
            for kw_ids in keyword_ids:
                phrase_reprs.append(self.keyword_encoder(kw_ids))
            phrase_reprs = torch.stack(phrase_reprs, dim=0) # (num_keywords, phrase_dim)
            
            p_phrase = self.phrase_fusion(llm_hidden, asr_hidden, phrase_reprs)
            
            # 3. Joint Multi-Grained Fusion
            # p_joi(z_i) = p_phr(k0) * p_tok(z_i) for z_i in V
            # p_joi(z_i) = p_phr(z_i) for z_i in K
            
            p_k0 = p_phrase[:, 0].unsqueeze(-1)
            p_joint_tokens = p_k0 * p_token
            p_joint_keywords = p_phrase[:, 1:]
            
            # The joint space combines vocab probabilities and keyword probabilities
            # Z = V U K
            # beam search would run on this combined probability distribution
            
            return p_joint_tokens, p_joint_keywords

    @torch.no_grad()
    def transcribe(
        self,
        audio: np.ndarray,
        keywords: Optional[List[str]] = None,
        mode: str = "joint",
        sample_rate: int = 16000,
        debug: bool = False,
        return_baseline: bool = False,
    ) -> Dict:
        """Transcribe audio with multi-grained architecture greedy decoding."""
        baseline_text = None
        if return_baseline or mode == "baseline" or keywords is None or len(keywords) == 0:
            baseline_text = self.asr.transcribe(audio, sample_rate)
        if mode == "baseline" or keywords is None or len(keywords) == 0:
            return {"text": baseline_text, "mode": "baseline", "baseline_text": baseline_text or ""}

        inputs = self.asr.processor(audio, sampling_rate=sample_rate, return_tensors="pt")
        input_features = inputs.input_features.to(self.device)
        encoder_outputs = self.asr.model.get_encoder()(input_features)
        
        forced_decoder_ids = self.asr.model.generation_config.forced_decoder_ids or []
        prompt_ids = [token_id for _, token_id in forced_decoder_ids if token_id is not None]
        decoder_input_ids = [self.asr.model.config.decoder_start_token_id] + prompt_ids
        
        k0_ids = torch.tensor([[0]]).to(self.device) 
        keyword_ids = [k0_ids]
        for kw in keywords:
            ids = self.asr.processor.tokenizer.encode(kw, add_special_tokens=False)
            if not ids: ids = [0]
            keyword_ids.append(torch.tensor([ids]).to(self.device))
            
        output_ids = []
        llm_generated_text = ""
        MAX_DECISIONS = 100
        prompt = self.llm.build_prompt(keywords)
        
        lm_head = getattr(self.asr.model, "proj_out", None) or getattr(self.asr.model, "lm_head", None)
        if lm_head is None:
            vocab_size = self.asr.model.config.vocab_size
            lm_head = nn.Linear(self.asr.model.config.d_model, vocab_size, bias=False).to(self.device)
            
        for step in range(MAX_DECISIONS):
            input_tensor = torch.tensor([decoder_input_ids], device=self.device)
            decoder_outputs = self.asr.model.model.decoder(
                input_ids=input_tensor,
                encoder_hidden_states=encoder_outputs.last_hidden_state
            )
            asr_hidden = decoder_outputs.last_hidden_state[:, -1, :] 
            asr_logits = lm_head(asr_hidden) 
            
            llm_logits, llm_hidden = self.llm.get_logits_and_hidden(prompt, llm_generated_text)
            llm_logits = llm_logits.squeeze(1) 
            llm_hidden = llm_hidden.squeeze(1) 
            
            v_asr = asr_logits.shape[-1]
            v_llm = llm_logits.shape[-1]
            if v_llm < v_asr:
                llm_logits = F.pad(llm_logits, (0, v_asr - v_llm), value=-100)
            elif v_llm > v_asr:
                llm_logits = llm_logits[:, :v_asr]
                
            p_joint_tokens, p_joint_keywords = self.demo_joint_decoding_step(
                asr_logits, asr_hidden, llm_logits, llm_hidden, keyword_ids
            )
            
            best_token = torch.argmax(p_joint_tokens[0]).item()
            best_keyword = torch.argmax(p_joint_keywords[0]).item() if p_joint_keywords.shape[1] > 0 else -1
            
            val_token = p_joint_tokens[0, best_token].item()
            val_keyword = p_joint_keywords[0, best_keyword].item() if best_keyword != -1 else -1.0
            
            if val_keyword > val_token:
                kw = keywords[best_keyword]
                ids = self.asr.processor.tokenizer.encode(kw, add_special_tokens=False)
                decoder_input_ids.extend(ids)
                output_ids.extend(ids)
                llm_generated_text += " " + kw
            else:
                decoder_input_ids.append(best_token)
                output_ids.append(best_token)
                token_str = self.asr.processor.tokenizer.decode([best_token], skip_special_tokens=True)
                llm_generated_text += token_str
                
                if best_token == self.asr.processor.tokenizer.eos_token_id:
                    break
                    
        biased_text = self.asr.processor.tokenizer.decode(output_ids, skip_special_tokens=True)
        return {"text": biased_text, "mode": "multigrained_architecture", "baseline_text": baseline_text or ""}

    def transcribe_dataset(
        self,
        dataset,
        keywords: Optional[List[str]] = None,
        mode: str = "baseline",
        show_progress: bool = True,
        debug: bool = False,
    ) -> List[Dict]:
        """Transcribe all samples in a dataset."""
        results = []
        iterator = tqdm(dataset, desc=f"Multigrained (Architecture)") if show_progress else dataset

        for i, sample in enumerate(iterator):
            audio = sample["audio"]["array"]
            sr = sample["audio"]["sampling_rate"]

            result = self.transcribe(
                audio=audio,
                keywords=keywords,
                mode=mode,
                sample_rate=sr,
                debug=debug and i < 3,
                return_baseline=True
            )
            result["reference"] = sample["text"]
            result["words"] = sample["words"]
            result["labels"] = sample["labels"]
            results.append(result)

        return results

    def demo_architecture(self, keywords: List[str]):
        """
        Demo the architecture by simulating a single decoding step.
        IMPORTANT: This uses UNTRAINED (randomly initialized) layers.
        The outputs have no semantic meaning and are only meant to show
        tensor shape and data flow through the architecture.
        """
        print("\n=== Multi-Grained Fusion Demo ===")
        print(f"Keywords: {keywords}")
        
        vocab_size = 50000
        llm_dim = self.llm.model.config.hidden_size
        asr_dim = self.asr.model.config.d_model
        
        # Simulated logits and hidden states
        asr_logits = torch.randn(1, vocab_size).to(self.device)
        llm_logits = torch.randn(1, vocab_size).to(self.device)
        
        asr_hidden = torch.randn(1, asr_dim).to(self.device)
        llm_hidden = torch.randn(1, llm_dim).to(self.device)
        
        k0_ids = torch.tensor([[0]]).to(self.device) 
        keyword_ids = [k0_ids] + [torch.randint(0, vocab_size, (1, 5)).to(self.device) for _ in keywords]
        
        print("\n[Step 1] Token Fusion (Uncertainty-weighted)")
        p_token = self.token_fusion(asr_logits, llm_logits)
        top_tokens = torch.topk(p_token, k=3).indices[0].tolist()
        print(f"Top V tokens: {top_tokens}")
        
        print("\n[Step 2] Keyword Encoder & Phrase Fusion")
        p_joint_tokens, p_joint_keywords = self.demo_joint_decoding_step(
            asr_logits, asr_hidden, llm_logits, llm_hidden, keyword_ids
        )
        
        print(f"k0 (NORMAL) prob: {p_joint_tokens.sum().item():.4f}")
        for i, kw in enumerate(keywords):
            print(f"Keyword '{kw}': {p_joint_keywords[0, i].item():.4f}")
            
        print("\n[Step 3] Joint Space (Z = V ∪ K)")
        print(f"V size: {vocab_size}, K size: {len(keywords)}")
        print("Note: In the full paper, a beam search would run on this combined probability distribution.")
        print("=========================================\n")

