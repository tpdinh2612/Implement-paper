# Contextual ASR for Vietnamese Medical NER - Re-implementation

Re-implementation and comparison of **3 contextual ASR methods** applied to the Vietnamese medical domain using the **VietMed-NER** dataset.

## Papers Implemented

| # | Paper | Key Idea | Year |
|---|-------|----------|------|
| 1 | **Multi-grained Fusion** | Token-level + Phrase-level fusion with LLMs | Jul 2025 |
| 2 | **CopyNE** | Copy mechanism for named entities from NE dictionary | ACL 2024 |
| 3 | **PEEKING-FUTURE** | Multi-Token Prediction for contextual biasing | Dec 2025 |

## Project Structure

```
re implement/
├── shared/                         # Shared modules (used by all 3 papers)
│   ├── config.py                   # Centralized configuration
│   ├── data_loader.py              # VietMed-NER dataset loading & entity extraction
│   ├── asr_backbone.py             # Whisper ASR model wrapper
│   └── metrics.py                  # Evaluation metrics (CER, WER, B-CER, NER F1, etc.)
│
├── methods/                        # Paper-specific implementations
│   ├── multigrained/               # Multi-grained Fusion
│   │   └── model.py                # Token-level + phrase-level + joint fusion
│   ├── copyne/                     # CopyNE
│   │   └── model.py                # NE copy mechanism with confidence threshold
│   └── peeking_future/                      # PEEKING-FUTURE
│       └── model.py                # Multi-token prediction entity scoring
│
├── run_all_eval.py                 # Unified evaluation script (all 3 methods)
├── run_evaluation.py               # Legacy evaluation (multigrained only)
├── requirements.txt                # Python dependencies
├── README.md                       # This file
│
├── CopyNE.pdf                      # Original paper
├── Multigrained Fusion.pdf         # Original paper
├── PEEKING-FUTURE.pdf                       # Original paper
└── outputs/                        # Evaluation results (JSON)
```

### What's Shared vs Paper-Specific

| Component | Shared | Paper-Specific |
|-----------|--------|----------------|
| Whisper ASR backbone | ✅ `shared/asr_backbone.py` | - |
| VietMed-NER data loading | ✅ `shared/data_loader.py` | - |
| Evaluation metrics | ✅ `shared/metrics.py` | - |
| Configuration | ✅ `shared/config.py` | Each has its own `@dataclass` |
| Token-level LLM fusion | - | `methods/multigrained/` |
| Phrase-level keyword copy | - | `methods/multigrained/` |
| NE copy mechanism | - | `methods/copyne/` |
| Multi-token prediction | - | `methods/peeking_future/` |

## Method Summaries

### 1. Multi-grained Fusion
- **Token-level**: Fuse ASR + LLM logits with uncertainty weighting: `s_t = s_t^a + σ(u_t^a) · s_t^l`
- **Phrase-level**: LSTM keyword encoder + dot-product attention for entity copy
- **Joint**: Unified probability space `Z = V ∪ K` with fake keyword k₀
- **Full Architecture Implementation**: Includes `KeywordEncoder`, `PhraseFusion`, and `demo_joint_decoding_step` to visualize the unified search space `Z = V ∪ K` and probability fusion matching the paper.
- *Note:* Since the paper requires training the fusion components (LSTM, attention layers) and we use a zero-shot inference approach, the active inference pipeline uses a lightweight heuristic approximation for keyword scoring, while the full architecture classes are provided for educational/debug purposes.

### 2. CopyNE
- Entities treated as **indivisible wholes** (not individual tokens)
- **NE Encoder**: 3-layer LSTM encodes entities to representations `z_i`
- **Copy mechanism**: At each step, predict token from vocab OR copy entity
- **Unified search space**: `Q(i) = Pc(∅)·P(i)` for vocab, `Q(i) = Pc(i)` for entities
- **Confidence threshold γ = 0.9** to filter low-confidence copies
- **Full Architecture Implementation**: Complete PyTorch components for `CopyAttention`, `CopyNEDecoder`, `CopyNELoss`, and the `CopyNE` wrapper matching the ACL 2024 paper exactly.
- **Unified Search Space Q & Beam Search Demo**: Includes `get_unified_search_space` and `demo_decoding_step` to visualize how the model chooses between individual tokens and whole entities at each decoding step.
- *Note:* Since pre-trained weights for the custom decoder are unavailable, the inference script (`CopyNEInference`) uses a lightweight deterministic heuristic approximation of the beam search to reproduce the metrics without requiring training. The architecture classes are provided for educational/debug purposes.

### 3. PEEKING-FUTURE (Peeking into the Future)
- **Multi-Token Prediction**: K=4 heads predict K future tokens simultaneously
- **Entity Scoring**: Collect logits for entity subwords across K heads, score with FFN
- **Biasing weight λ** to boost/suppress context biasing
- **No entity encoder needed** - uses raw logits directly
- **Full Architecture Implementation (Level 1)**: Complete PyTorch components for `MTPModule`, `EntityScorer`, and a manual greedy decoding loop `PeekingFutureInference.transcribe` that visualizes the tensor flow (h_d -> MTP logits -> Pe -> Unified Search Q) matching Eq. 4-12 of the paper.
- *Note:* Since pre-trained weights for the MTP and Scorer modules are unavailable, the inference script runs a ToyPeekingFuture with randomly initialized PEEKING-FUTURE modules on top of a pretrained Whisper backbone. This is an educational implementation meant to demonstrate the true architectural decoding loop, but metrics will reflect random entity scoring until the PEEKING-FUTURE modules are trained.

## Evaluation Metrics

| Metric | Description | Direction |
|--------|-------------|-----------|
| CER | Character Error Rate (overall) | ↓ |
| WER | Word Error Rate (overall) | ↓ |
| B-CER | Biased CER (keyword/entity spans only) | ↓ |
| U-CER | Unbiased CER (non-keyword spans only) | ↓ |
| Recall | Keyword Recall (% entities fully recognized) | ↑ |
| NER F1 | Entity-level F1 (micro-averaged) | ↑ |

## Quick Start

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run All Methods (Quick Test)
```bash
python run_all_eval.py --methods baseline copyne peeking_future --max_samples 20 --device cpu --no_llm
```

### Run With LLM (Multigrained)
```bash
python run_all_eval.py --methods baseline copyne peeking_future multigrained --max_samples 20 --device cuda
```

### Full Evaluation
```bash
python run_all_eval.py --methods all --max_samples 50 --device cuda
```

## Dataset

**VietMed-NER** (NAACL 2025): Vietnamese medical spoken NER
- 🔗 [HuggingFace](https://huggingface.co/datasets/leduckhai/VietMed-NER)
- 18 entity types (DISEASESYMTOM, DRUG, ORGAN, etc.)
- ~3500 test samples with audio + text + BIO NER labels

## References

1. **Multi-grained Fusion**: "Improving Contextual ASR via Multi-grained Fusion with LLMs" (arXiv:2507.12252)
2. **CopyNE**: "Better Contextual ASR by Copying Named Entities" (Zhou et al., ACL 2024)
3. **PEEKING-FUTURE**: "Peeking into the Future for Contextual Biasing" (arXiv:2512.17657)
4. **VietMed-NER**: "Medical Spoken Named Entity Recognition" (NAACL 2025)
