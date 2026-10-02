# 🚀 Hướng Dẫn Chạy Inference — 3 Contextual ASR Methods trên VietMed-NER

## Mục Lục

1. [Yêu Cầu Hệ Thống](#1-yêu-cầu-hệ-thống)
2. [Cài Đặt](#2-cài-đặt)
3. [Cấu Trúc Dự Án](#3-cấu-trúc-dự-án)
4. [Chạy Nhanh (Quick Test)](#4-chạy-nhanh-quick-test)
5. [Chạy Từng Method](#5-chạy-từng-method)
6. [Chạy So Sánh Toàn Bộ](#6-chạy-so-sánh-toàn-bộ)
7. [Giải Thích Arguments](#7-giải-thích-arguments)
8. [Kết Quả Đầu Ra](#8-kết-quả-đầu-ra)
9. [Metrics Giải Thích](#9-metrics-giải-thích)
10. [Troubleshooting](#10-troubleshooting)
11. [Mô Tả 3 Methods](#11-mô-tả-3-methods)

---

## 1. Yêu Cầu Hệ Thống

### Phần cứng

| Chế độ | RAM | VRAM (GPU) | Thời gian ước tính |
|---|---|---|---|
| CPU only (`--device cpu --no_llm`) | ≥ 8 GB | Không cần | ~5-10s/sample |
| GPU, không LLM (`--device cuda --no_llm`) | ≥ 8 GB | ≥ 4 GB | ~1-2s/sample |
| GPU, có LLM (`--device cuda`) | ≥ 16 GB | ≥ 6 GB | ~2-4s/sample |

### Phần mềm

- **Python**: 3.9 trở lên
- **CUDA**: 11.8+ (nếu dùng GPU)
- **Hệ điều hành**: Windows / Linux / macOS
- **Internet**: Cần cho lần chạy đầu tiên (download model + dataset)

---

## 2. Cài Đặt

### Bước 1: Tạo môi trường ảo (khuyến nghị)

```bash
cd "e:\re implement"

# Dùng venv
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/macOS
```

```bash
# 1. Tạo môi trường conda với Python 3.10
conda create -n vietmed_asr python=3.10 -y
conda activate vietmed_asr

# 2. Cài PyTorch cùng CUDA Toolkit (ví dụ CUDA 11.8)
conda install pytorch torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# 3. Cài đặt các dependencies còn lại
pip install -r requirements.txt
```

### Bước 2: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

> **Lưu ý**: Nếu dùng GPU, cài PyTorch phiên bản CUDA trước:
> ```bash
> pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
> pip install -r requirements.txt
> ```

### Bước 3: Kiểm tra cài đặt

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
python -c "from transformers import WhisperProcessor; print('Transformers OK')"
python -c "from datasets import load_dataset; print('Datasets OK')"
```

---

## 3. Cấu Trúc Dự Án

```
re implement/
├── shared/                      # Module dùng chung
│   ├── config.py                # Config tập trung (ASR, LLM, CopyNE, PEEKING-FUTURE)
│   ├── asr_backbone.py          # Whisper ASR wrapper
│   ├── data_loader.py           # Load VietMed-NER + trích xuất entities
│   └── metrics.py               # Tính CER/WER/B-CER/U-CER/Recall/NER-F1
│
├── methods/                     # 3 paper implementations
│   ├── copyne/model.py          # CopyNE — copy named entities
│   ├── multigrained/model.py    # Multi-grained Fusion — token + phrase level
│   └── peeking_future/model.py           # PEEKING-FUTURE — multi-token prediction
│
├── run_all_eval.py              # ⭐ Script chính: chạy và so sánh tất cả methods
├── quick_test.py                # Test nhanh pipeline (5 samples)
├── requirements.txt             # Dependencies
├── cache/                       # Cache dataset (tự tạo)
└── outputs/                     # Kết quả JSON (tự tạo)
```

---

## 4. Chạy Nhanh (Quick Test)

Kiểm tra pipeline hoạt động đúng với 5 samples:

```bash
cd "e:\re implement"
python run_all_eval.py --methods baseline --max_samples 5 --device cpu
```

---

## 5. Chạy Từng Method

### 5.1 Baseline (Whisper only — không contextual biasing)

```bash
python run_all_eval.py --methods baseline --max_samples 20 --device cpu
```

### 5.2 CopyNE (Copy Named Entities)

```bash
python run_all_eval.py --methods copyne --max_samples 20 --device cpu
```

### 5.3 PEEKING-FUTURE (Multi-Token Prediction Biasing)

```bash
python run_all_eval.py --methods peeking_future --max_samples 20 --device cpu
```

### 5.4 Multi-grained Fusion (cần LLM)

```bash
# Cần GPU hoặc nhiều RAM (download Qwen2-0.5B ~1GB)
python run_all_eval.py --methods multigrained --max_samples 20 --device cuda
```

> ⚠️ Nếu không có GPU hoặc không muốn download LLM:
> ```bash
> python run_all_eval.py --methods multigrained --max_samples 20 --device cpu
> ```
> Sẽ chạy chậm hơn nhưng vẫn hoạt động.

---

## 6. Chạy So Sánh Toàn Bộ

### 6.1 Tất cả methods KHÔNG có LLM (nhanh hơn)

```bash
python run_all_eval.py --methods baseline copyne peeking_future --max_samples 20 --device cpu --no_llm
```

### 6.2 Tất cả methods CÓ LLM (đầy đủ nhất)

```bash
python run_all_eval.py --methods baseline copyne peeking_future multigrained --max_samples 50 --device cuda
```

### 6.3 Full evaluation (tất cả samples)

```bash
python run_all_eval.py --methods all --device cuda
```

> ⏱️ Full evaluation trên ~3500 samples có thể mất **1-3 giờ** tùy hardware.

---

## 7. Giải Thích Arguments

| Argument | Mô tả | Mặc định |
|---|---|---|
| `--methods` | Methods cần chạy: `baseline`, `copyne`, `peeking_future`, `multigrained`, `all` | `baseline copyne peeking_future multigrained` |
| `--split` | Dataset split: `train`, `validation`, `test` | `test` |
| `--max_samples` | Số samples tối đa (để test nhanh). Bỏ trống = chạy hết. | `20` |
| `--whisper_model` | Tên model Whisper | `openai/whisper-small` |
| `--llm_model` | Tên model LLM (cho Multigrained) | `Qwen/Qwen2-0.5B` |
| `--no_llm` | Bỏ qua LLM (skip Multigrained nếu cần LLM) | `False` |
| `--max_keywords` | Số keywords tối đa trong dictionary | `50` |
| `--output_dir` | Thư mục lưu kết quả | `./outputs` |
| `--device` | Thiết bị: `cuda` hoặc `cpu` | `cuda` |

---

## 8. Kết Quả Đầu Ra

### 8.1 Console output

Khi chạy xong, bạn sẽ thấy bảng so sánh:

```
================================================================================
  COMPARISON TABLE: All Methods on VietMed-NER
================================================================================
  Method               CER↓     B-CER↓   U-CER↓       R↑     WER↓   NER-F1↑
--------------------------------------------------------------------------------
  baseline            XX.XX     XX.XX     XX.XX     XX.XX    XX.XX     XX.XX
  copyne              XX.XX     XX.XX     XX.XX     XX.XX    XX.XX     XX.XX
  peeking_future               XX.XX     XX.XX     XX.XX     XX.XX    XX.XX     XX.XX
  multigrained        XX.XX     XX.XX     XX.XX     XX.XX    XX.XX     XX.XX
================================================================================
```

### 8.2 File JSON

Kết quả chi tiết được lưu tại:

```
outputs/unified_results_YYYYMMDD_HHMMSS.json
```

Cấu trúc file JSON:

```json
{
  "config": { "whisper_model": "...", "methods": [...] },
  "dataset_stats": { "num_samples": 3500, "entity_char_ratio": 0.12 },
  "keywords": ["keyword1", "keyword2", "..."],
  "metrics": {
    "baseline": { "CER": 15.2, "B-CER": 25.3, "U-CER": 12.1, "Recall": 65.0 },
    "copyne":   { "CER": 14.1, "B-CER": 20.1, "U-CER": 12.0, "Recall": 72.0 }
  },
  "sample_results": {
    "baseline": [
      { "reference": "bệnh nhân bị sốt cao", "hypothesis": "bệnh nhân bị số cao" }
    ]
  }
}
```

---

## 9. Metrics Giải Thích

| Metric | Tên đầy đủ | Ý nghĩa | Hướng tối ưu |
|---|---|---|---|
| **CER** | Character Error Rate | Tỷ lệ lỗi ký tự tổng thể | ↓ thấp = tốt |
| **WER** | Word Error Rate | Tỷ lệ lỗi từ tổng thể | ↓ thấp = tốt |
| **B-CER** | Biased CER | CER chỉ trên keyword spans (entities) | ↓ thấp = tốt |
| **U-CER** | Unbiased CER | CER trên non-keyword spans | ↓ thấp = tốt |
| **Recall** | Keyword Recall | % entities được nhận dạng đúng hoàn toàn | ↑ cao = tốt |
| **NER F1** | Entity-level F1 | F1-score micro-averaged theo entity type | ↑ cao = tốt |

### Mối quan hệ

- **B-CER** quan trọng nhất cho contextual ASR — đo lỗi trên entity/keyword
- **U-CER** kiểm tra method có làm giảm chất lượng non-keyword không
- **Recall** cho biết bao nhiêu % entity được nhận dạng hoàn chỉnh
- CopyNE, PEEKING-FUTURE nên cải thiện B-CER và Recall so với baseline
- Multi-grained nên cải thiện cả B-CER lẫn U-CER

---

## 10. Troubleshooting

### ❌ `ModuleNotFoundError: No module named 'shared'`

**Fix**: Chạy từ thư mục gốc:
```bash
cd "e:\re implement"
python run_all_eval.py ...
```

### ❌ `torch.cuda.OutOfMemoryError`

**Fix**: Chuyển sang CPU hoặc giảm samples:
```bash
python run_all_eval.py --device cpu --no_llm --max_samples 10
```

### ❌ `ConnectionError` khi load dataset

**Nguyên nhân**: Lần đầu cần internet để download dataset + model.

**Fix**: Đảm bảo có internet. Dataset cache trong `./cache/`.

### ❌ `ImportError: No module named 'soundfile'`

```bash
pip install soundfile
# Linux: sudo apt-get install libsndfile1
```

### ❌ `quick_test.py` hoặc `run_evaluation.py` lỗi import

Đây là legacy scripts. Dùng `run_all_eval.py` thay thế.

### ❌ Model download quá lâu

Whisper-small (~244MB) + Qwen2-0.5B (~1GB) cần download lần đầu. Sau khi cache sẽ load nhanh.

---

## 11. Mô Tả 3 Methods

### Multi-grained Fusion (Zhou et al., Jul 2025)

```
Audio → Whisper → logits + hidden
Keywords → LLM prompt → logits + hidden
Token-level:  s = s_asr + σ(entropy) · s_llm
Phrase-level: q = Linear([h_llm; h_asr]), p = softmax(q · r_i)
Joint:        p(z) = p_phr(k0)·p_tok(z) for z∈Vocab
              p(z) = p_phr(z)            for z∈Keywords
```
*(Code đi kèm `KeywordEncoder`, `PhraseFusion` và `demo_joint_decoding_step` để trực quan hóa cách không gian xác suất được hợp nhất thành Z = V ∪ K)*

### CopyNE (Zhou et al., ACL 2024)

```
Entities → 3-layer LSTM → representations z_i
At each step: P(token from vocab) OR P(copy entity_i)
Unified Search Space Q: combines Vocab and Entity probabilities
Beam Search: selects top candidates from Q (token or whole entity)
```
*(Code đi kèm `demo_decoding_step` để trực quan hóa quá trình CopyNE chọn Token vs Entity trong không gian tìm kiếm Q)*

### PEEKING-FUTURE (Selvakumar et al., Dec 2025)

```
K=4 MTP heads → predict 4 future tokens simultaneously
Entity scoring: f_θ(logit_1...logit_K) → score
Q(entity) = λ · Pe(entity), λ=4.4
No entity encoder needed — uses raw logits
```
*(Code được implement lại ở dạng Level 1 Architectural Reimplementation (ToyPeekingFuture) chạy decoder loop với Whisper backbone, `MTPModule`, `EntityScorer`, `Pe(null)`, pruning `γ` và Unified Search. Sẽ in debug shape tensor ở 3 samples đầu để minh họa luồng Eq 4-12 trong paper).*

---

## 12. Ví Dụ Copy-Paste

### Test nhanh CPU (2 phút)

```bash
cd "e:\re implement"
pip install -r requirements.txt
python run_all_eval.py --methods baseline copyne peeking_future --max_samples 10 --device cpu --no_llm
```

### Đánh giá 50 samples GPU

```bash
python run_all_eval.py --methods baseline copyne peeking_future multigrained --max_samples 50 --device cuda
```

### Full evaluation

```bash
python run_all_eval.py --methods all --device cuda --output_dir ./outputs
```

---

> 📝 **Ghi chú quan trọng**: Cả 3 phương pháp (CopyNE, PEEKING-FUTURE, Multi-grained) đều được implement dưới dạng **Level 1 Architectural Reimplementation** chạy manual greedy decoding. Do các trọng số (weights) của các module bổ sung (như MTP, Scorer, NEEncoder, PhraseFusion) chưa được train trên VietMed mà chỉ khởi tạo ngẫu nhiên, điểm benchmark của cả 3 phương pháp sẽ rất thấp và output có thể vô nghĩa (chỉ dùng để minh họa tensor flow và luồng kiến trúc).
