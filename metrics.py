"""
Evaluation Metrics Module.

Implements all metrics from the paper:
1. CER (Character Error Rate) - for Vietnamese
2. WER (Word Error Rate)
3. B-CER / B-WER (Biased - keyword spans only)
4. U-CER / U-WER (Unbiased - non-keyword spans only)
5. Keyword Recall (R) - percentage of keywords fully recognized
6. NER-level metrics: Precision, Recall, F1 (from VietMed-NER paper)

References:
- Paper Section 4.3.2 (Metrics)
- VietMed-NER paper (arxiv:2406.13337) for NER-specific metrics
"""

import re
import unicodedata
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np

try:
    from jiwer import wer as compute_wer_jiwer
    from jiwer import cer as compute_cer_jiwer
    HAS_JIWER = True
except ImportError:
    HAS_JIWER = False


# ============================================================================
# Text Normalization (following Whisper's normalizer)
# ============================================================================

def normalize_text(text: str) -> str:
    """
    Normalize text for metric computation.
    Following Whisper's text normalizer approach.
    """
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Normalize unicode
    text = unicodedata.normalize("NFC", text)
    # Remove punctuation (keeping Vietnamese diacritics)
    text = re.sub(r'[^\w\s]', '', text)
    # Remove extra whitespace again
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ============================================================================
# Character Error Rate (CER) 
# ============================================================================

def edit_distance(s1: str, s2: str) -> int:
    """Compute Levenshtein edit distance between two strings."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    
    return dp[m][n]


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate.
    CER = edit_distance(ref_chars, hyp_chars) / len(ref_chars)
    """
    ref = normalize_text(reference).replace(" ", "")
    hyp = normalize_text(hypothesis).replace(" ", "")
    
    if len(ref) == 0:
        return 0.0 if len(hyp) == 0 else 1.0
    
    dist = edit_distance(ref, hyp)
    return dist / len(ref)


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate.
    WER = edit_distance(ref_words, hyp_words) / len(ref_words)
    """
    ref_words = normalize_text(reference).split()
    hyp_words = normalize_text(hypothesis).split()
    
    if len(ref_words) == 0:
        return 0.0 if len(hyp_words) == 0 else 1.0
    
    dist = edit_distance(ref_words, hyp_words)
    return dist / len(ref_words)


# ============================================================================
# Biased / Unbiased CER/WER (keyword-specific metrics)
# ============================================================================

def extract_keyword_text(
    reference: str,
    words: List[str],
    labels: List[str],
) -> Tuple[str, str]:
    """
    Split reference text into keyword and non-keyword parts.
    
    Args:
        reference: full reference text
        words: list of words
        labels: list of BIO labels
        
    Returns:
        (keyword_text, non_keyword_text) concatenated strings
    """
    keyword_words = []
    non_keyword_words = []
    
    for word, label in zip(words, labels):
        if label.startswith("B-") or label.startswith("I-"):
            keyword_words.append(word)
        else:
            non_keyword_words.append(word)
    
    return " ".join(keyword_words), " ".join(non_keyword_words)


def align_and_extract_spans(
    reference: str,
    hypothesis: str,
    words: List[str],
    labels: List[str],
) -> Tuple[str, str, str, str]:
    """
    Extract keyword and non-keyword portions from both reference and hypothesis.
    
    For the hypothesis, we use word-level alignment to determine which words
    correspond to keyword vs non-keyword spans.
    
    Returns:
        (ref_keyword, hyp_keyword, ref_nonkeyword, hyp_nonkeyword)
    """
    ref_keyword, ref_nonkeyword = extract_keyword_text(reference, words, labels)
    
    # For hypothesis, we need to identify which parts correspond to keywords
    # Simple approach: use word position alignment
    ref_words_list = normalize_text(reference).split()
    hyp_words_list = normalize_text(hypothesis).split()
    
    # Build keyword word set from reference with positions
    keyword_positions = set()
    for i, label in enumerate(labels):
        if label.startswith("B-") or label.startswith("I-"):
            keyword_positions.add(i)
    
    # Align hypothesis words to reference words using dynamic programming
    alignment = _align_words(ref_words_list, hyp_words_list)
    
    hyp_keyword_words = []
    hyp_nonkeyword_words = []
    
    for hyp_idx, ref_idx in enumerate(alignment):
        if hyp_idx < len(hyp_words_list):
            word = hyp_words_list[hyp_idx]
            if ref_idx is not None and ref_idx in keyword_positions:
                hyp_keyword_words.append(word)
            else:
                hyp_nonkeyword_words.append(word)
    
    hyp_keyword = " ".join(hyp_keyword_words)
    hyp_nonkeyword = " ".join(hyp_nonkeyword_words)
    
    return ref_keyword, hyp_keyword, ref_nonkeyword, hyp_nonkeyword


def _align_words(ref_words: List[str], hyp_words: List[str]) -> List[Optional[int]]:
    """
    Align hypothesis words to reference words using edit distance alignment.
    Returns a list where alignment[i] = j means hyp word i aligns to ref word j.
    """
    m, n = len(ref_words), len(hyp_words)
    
    # DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[""] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
        if i > 0:
            ops[i][0] = "D"  # deletion
    for j in range(n + 1):
        dp[0][j] = j
        if j > 0:
            ops[0][j] = "I"  # insertion
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
                ops[i][j] = "M"  # match
            else:
                costs = [
                    (dp[i-1][j-1] + 1, "S"),  # substitution
                    (dp[i-1][j] + 1, "D"),     # deletion
                    (dp[i][j-1] + 1, "I"),     # insertion
                ]
                dp[i][j], ops[i][j] = min(costs, key=lambda x: x[0])
    
    # Backtrack to find alignment
    alignment = [None] * n
    i, j = m, n
    while i > 0 or j > 0:
        op = ops[i][j]
        if op == "M" or op == "S":
            alignment[j-1] = i - 1
            i -= 1
            j -= 1
        elif op == "D":
            i -= 1
        elif op == "I":
            alignment[j-1] = None
            j -= 1
        else:
            break
    
    return alignment


def compute_biased_cer(
    reference: str,
    hypothesis: str,
    words: List[str],
    labels: List[str],
) -> float:
    """
    Compute B-CER: CER on keyword spans only.
    """
    ref_kw, hyp_kw, _, _ = align_and_extract_spans(
        reference, hypothesis, words, labels
    )
    if not ref_kw.strip():
        return 0.0
    return compute_cer(ref_kw, hyp_kw)


def compute_unbiased_cer(
    reference: str,
    hypothesis: str,
    words: List[str],
    labels: List[str],
) -> float:
    """
    Compute U-CER: CER on non-keyword spans only.
    """
    _, _, ref_nkw, hyp_nkw = align_and_extract_spans(
        reference, hypothesis, words, labels
    )
    if not ref_nkw.strip():
        return 0.0
    return compute_cer(ref_nkw, hyp_nkw)


# ============================================================================
# Keyword Recall
# ============================================================================

def compute_keyword_recall(
    reference: str,
    hypothesis: str,
    words: List[str],
    labels: List[str],
) -> Tuple[int, int]:
    """
    Compute keyword recall: percentage of keywords fully and correctly recognized.
    
    A keyword (entity) is considered correctly recognized if its entire text
    appears in the hypothesis.
    
    Returns:
        (num_correct, num_total) keywords
    """
    # Extract entities
    entities = []
    current_entity = None
    
    for i, (word, label) in enumerate(zip(words, labels)):
        if label.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            current_entity = {"text": word, "type": label[2:]}
        elif label.startswith("I-") and current_entity is not None:
            current_entity["text"] += " " + word
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None
    if current_entity is not None:
        entities.append(current_entity)
    
    if len(entities) == 0:
        return 0, 0
    
    # Check each entity
    hyp_normalized = normalize_text(hypothesis)
    num_correct = 0
    for entity in entities:
        entity_text = normalize_text(entity["text"])
        if entity_text in hyp_normalized:
            num_correct += 1
    
    return num_correct, len(entities)


# ============================================================================
# NER-level Metrics (from VietMed-NER paper)
# ============================================================================

def extract_entities_from_bio(
    words: List[str], labels: List[str]
) -> List[Tuple[str, str, int, int]]:
    """
    Extract entities from BIO-tagged sequence.
    
    Returns:
        List of (entity_text, entity_type, start_idx, end_idx)
    """
    entities = []
    current = None
    
    for i, (word, label) in enumerate(zip(words, labels)):
        if label.startswith("B-"):
            if current is not None:
                entities.append(current)
            etype = label[2:]
            current = (word, etype, i, i)
        elif label.startswith("I-") and current is not None:
            etype = label[2:]
            if etype == current[1]:
                current = (current[0] + " " + word, current[1], current[2], i)
            else:
                entities.append(current)
                current = (word, etype, i, i)
        else:
            if current is not None:
                entities.append(current)
                current = None
    
    if current is not None:
        entities.append(current)
    
    return entities


def compute_ner_metrics(
    all_ref_words: List[List[str]],
    all_ref_labels: List[List[str]],
    all_hyp_texts: List[str],
) -> Dict:
    """
    Compute NER-level Precision, Recall, F1.
    
    Since we're doing ASR → NER pipeline evaluation, we match entities
    between reference and hypothesis based on text overlap.
    
    Args:
        all_ref_words: list of reference word lists
        all_ref_labels: list of reference label lists
        all_hyp_texts: list of hypothesis transcriptions
        
    Returns:
        Dict with per-type and overall P, R, F1
    """
    type_tp = defaultdict(int)
    type_fp = defaultdict(int)
    type_fn = defaultdict(int)
    
    for ref_words, ref_labels, hyp_text in zip(
        all_ref_words, all_ref_labels, all_hyp_texts
    ):
        ref_entities = extract_entities_from_bio(ref_words, ref_labels)
        hyp_normalized = normalize_text(hyp_text)
        
        matched_ref = set()
        
        for idx, (text, etype, start, end) in enumerate(ref_entities):
            entity_normalized = normalize_text(text)
            if entity_normalized in hyp_normalized:
                type_tp[etype] += 1
                matched_ref.add(idx)
            else:
                type_fn[etype] += 1
    
    # Compute per-type metrics
    results = {}
    all_types = set(list(type_tp.keys()) + list(type_fn.keys()) + list(type_fp.keys()))
    
    total_tp = sum(type_tp.values())
    total_fp = sum(type_fp.values())
    total_fn = sum(type_fn.values())
    
    for etype in sorted(all_types):
        tp = type_tp[etype]
        fp = type_fp[etype]
        fn = type_fn[etype]
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        results[etype] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    
    # Overall (micro-average)
    overall_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    overall_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    overall_f1 = 2 * overall_p * overall_r / (overall_p + overall_r) if (overall_p + overall_r) > 0 else 0.0
    
    results["overall"] = {
        "precision": overall_p,
        "recall": overall_r,
        "f1": overall_f1,
        "support": total_tp + total_fn,
    }
    
    return results


# ============================================================================
# Aggregate Metrics
# ============================================================================

def compute_all_metrics(
    results: List[Dict],
) -> Dict:
    """
    Compute all metrics from paper over a list of results.
    
    Args:
        results: list of dicts with keys:
            - 'reference': reference text
            - 'text': hypothesis text
            - 'words': reference word list
            - 'labels': reference BIO label list
            
    Returns:
        Dict with all computed metrics
    """
    # Accumulators
    total_ref_chars = 0
    total_edit_chars = 0
    total_ref_words = 0
    total_edit_words = 0
    
    total_kw_ref_chars = 0
    total_kw_edit_chars = 0
    total_nkw_ref_chars = 0
    total_nkw_edit_chars = 0
    
    total_kw_correct = 0
    total_kw_total = 0
    
    all_ref_words = []
    all_ref_labels = []
    all_hyp_texts = []
    
    for r in results:
        ref = r["reference"]
        hyp = r["text"]
        words = r["words"]
        labels = r["labels"]
        
        # Overall CER
        ref_norm = normalize_text(ref).replace(" ", "")
        hyp_norm = normalize_text(hyp).replace(" ", "")
        ed = edit_distance(ref_norm, hyp_norm)
        total_ref_chars += len(ref_norm)
        total_edit_chars += ed
        
        # Overall WER
        ref_words_list = normalize_text(ref).split()
        hyp_words_list = normalize_text(hyp).split()
        wd = edit_distance(ref_words_list, hyp_words_list)
        total_ref_words += len(ref_words_list)
        total_edit_words += wd
        
        # Biased/Unbiased CER
        ref_kw, hyp_kw, ref_nkw, hyp_nkw = align_and_extract_spans(
            ref, hyp, words, labels
        )
        
        ref_kw_chars = normalize_text(ref_kw).replace(" ", "")
        hyp_kw_chars = normalize_text(hyp_kw).replace(" ", "")
        ref_nkw_chars = normalize_text(ref_nkw).replace(" ", "")
        hyp_nkw_chars = normalize_text(hyp_nkw).replace(" ", "")
        
        if ref_kw_chars:
            kw_ed = edit_distance(ref_kw_chars, hyp_kw_chars)
            total_kw_ref_chars += len(ref_kw_chars)
            total_kw_edit_chars += kw_ed
        
        if ref_nkw_chars:
            nkw_ed = edit_distance(ref_nkw_chars, hyp_nkw_chars)
            total_nkw_ref_chars += len(ref_nkw_chars)
            total_nkw_edit_chars += nkw_ed
        
        # Keyword Recall
        kw_correct, kw_total = compute_keyword_recall(ref, hyp, words, labels)
        total_kw_correct += kw_correct
        total_kw_total += kw_total
        
        # For NER metrics
        all_ref_words.append(words)
        all_ref_labels.append(labels)
        all_hyp_texts.append(hyp)
    
    # Aggregate metrics
    metrics = {
        "CER": (total_edit_chars / total_ref_chars * 100) if total_ref_chars > 0 else 0.0,
        "WER": (total_edit_words / total_ref_words * 100) if total_ref_words > 0 else 0.0,
        "B-CER": (total_kw_edit_chars / total_kw_ref_chars * 100) if total_kw_ref_chars > 0 else 0.0,
        "U-CER": (total_nkw_edit_chars / total_nkw_ref_chars * 100) if total_nkw_ref_chars > 0 else 0.0,
        "Recall": (total_kw_correct / total_kw_total * 100) if total_kw_total > 0 else 0.0,
        "num_samples": len(results),
        "total_keywords": total_kw_total,
        "keywords_correct": total_kw_correct,
    }
    
    # NER metrics
    ner_metrics = compute_ner_metrics(all_ref_words, all_ref_labels, all_hyp_texts)
    metrics["NER"] = ner_metrics
    
    return metrics


def format_metrics_table(metrics: Dict, title: str = "Evaluation Results") -> str:
    """
    Format metrics as a readable table string (similar to paper's Table 1).
    """
    lines = []
    lines.append("=" * 70)
    lines.append(f"  {title}")
    lines.append("=" * 70)
    lines.append(f"  Samples: {metrics['num_samples']}")
    lines.append(f"  Total Keywords: {metrics['total_keywords']}")
    lines.append(f"  Keywords Correct: {metrics['keywords_correct']}")
    lines.append("-" * 70)
    lines.append(f"  {'Metric':<20} {'Value':>10}")
    lines.append("-" * 70)
    lines.append(f"  {'CER ↓':<20} {metrics['CER']:>10.2f}%")
    lines.append(f"  {'WER ↓':<20} {metrics['WER']:>10.2f}%")
    lines.append(f"  {'B-CER ↓':<20} {metrics['B-CER']:>10.2f}%")
    lines.append(f"  {'U-CER ↓':<20} {metrics['U-CER']:>10.2f}%")
    lines.append(f"  {'Recall ↑':<20} {metrics['Recall']:>10.2f}%")
    lines.append("-" * 70)
    
    # NER metrics
    if "NER" in metrics:
        ner = metrics["NER"]
        lines.append("\n  NER Entity-level Metrics:")
        lines.append("-" * 70)
        lines.append(f"  {'Entity Type':<25} {'P':>8} {'R':>8} {'F1':>8} {'Support':>8}")
        lines.append("-" * 70)
        
        for etype in sorted(ner.keys()):
            if etype == "overall":
                continue
            m = ner[etype]
            lines.append(
                f"  {etype:<25} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                f"{m['f1']:>8.4f} {m['support']:>8d}"
            )
        
        if "overall" in ner:
            lines.append("-" * 70)
            m = ner["overall"]
            lines.append(
                f"  {'OVERALL (micro)':<25} {m['precision']:>8.4f} {m['recall']:>8.4f} "
                f"{m['f1']:>8.4f} {m['support']:>8d}"
            )
    
    lines.append("=" * 70)
    return "\n".join(lines)
