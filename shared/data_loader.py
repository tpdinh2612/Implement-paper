"""
Shared Dataset Loader for VietMed-NER.
Used by all three paper implementations.
Handles loading audio, text, NER labels, keyword list construction.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm

from shared.config import DataConfig


def load_vietmed_ner(config: DataConfig, split: str = "test"):
    """Load VietMed-NER dataset from HuggingFace."""
    dataset = load_dataset(
        config.dataset_name,
        split=split,
        cache_dir=config.cache_dir,
    )
    dataset = dataset.cast_column("audio", Audio(sampling_rate=config.sample_rate))
    return dataset


def extract_entities_from_sample(sample: Dict) -> List[Dict]:
    """Extract named entities from a single sample's BIO tags."""
    words = sample["words"]
    labels = sample["labels"]
    entities = []
    current_entity = None

    for i, (word, label) in enumerate(zip(words, labels)):
        if label.startswith("B-"):
            if current_entity is not None:
                entities.append(current_entity)
            entity_type = label[2:]
            current_entity = {
                "text": word,
                "type": entity_type,
                "start_idx": i,
                "end_idx": i,
                "words": [word],
            }
        elif label.startswith("I-") and current_entity is not None:
            entity_type = label[2:]
            if entity_type == current_entity["type"]:
                current_entity["text"] += " " + word
                current_entity["end_idx"] = i
                current_entity["words"].append(word)
            else:
                entities.append(current_entity)
                current_entity = {
                    "text": word,
                    "type": entity_type,
                    "start_idx": i,
                    "end_idx": i,
                    "words": [word],
                }
        else:
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    if current_entity is not None:
        entities.append(current_entity)

    return entities


def build_keyword_list(dataset, max_keywords: int = 50) -> List[str]:
    """Build keyword list from entities in dataset (for contextual biasing)."""
    keyword_counts = defaultdict(int)
    for sample in dataset:
        entities = extract_entities_from_sample(sample)
        for entity in entities:
            keyword_counts[entity["text"].lower()] += 1

    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: -x[1])
    keywords = [kw for kw, _ in sorted_keywords[:max_keywords]]
    return keywords


def build_entity_list(dataset, max_entities: int = 50) -> List[Dict]:
    """
    Build entity list with type information (for CopyNE and PEEKING-FUTURE).
    Returns list of dicts with 'text', 'type', 'count'.
    """
    entity_info = defaultdict(lambda: {"count": 0, "type": None})
    for sample in dataset:
        entities = extract_entities_from_sample(sample)
        for entity in entities:
            key = entity["text"].lower()
            entity_info[key]["count"] += 1
            entity_info[key]["type"] = entity["type"]

    sorted_entities = sorted(entity_info.items(), key=lambda x: -x[1]["count"])
    return [
        {"text": text, "type": info["type"], "count": info["count"]}
        for text, info in sorted_entities[:max_entities]
    ]


def get_keyword_spans(
    reference_text: str,
    entities: List[Dict],
    words: List[str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Get character-level spans for keyword and non-keyword segments."""
    word_char_positions = []
    pos = 0
    for word in words:
        start = reference_text.find(word, pos)
        if start == -1:
            start = pos
        end = start + len(word)
        word_char_positions.append((start, end))
        pos = end

    keyword_char_spans = []
    for entity in entities:
        if entity["start_idx"] < len(word_char_positions) and entity["end_idx"] < len(word_char_positions):
            start_char = word_char_positions[entity["start_idx"]][0]
            end_char = word_char_positions[entity["end_idx"]][1]
            keyword_char_spans.append((start_char, end_char))

    non_keyword_spans = []
    prev_end = 0
    for start, end in sorted(keyword_char_spans):
        if prev_end < start:
            non_keyword_spans.append((prev_end, start))
        prev_end = max(prev_end, end)
    if prev_end < len(reference_text):
        non_keyword_spans.append((prev_end, len(reference_text)))

    return keyword_char_spans, non_keyword_spans


def compute_entity_char_ratio(dataset) -> float:
    """Compute the ratio of entity characters to total characters."""
    total_chars = 0
    entity_chars = 0
    for sample in dataset:
        text = sample["text"]
        total_chars += len(text.replace(" ", ""))
        entities = extract_entities_from_sample(sample)
        for entity in entities:
            entity_chars += len(entity["text"].replace(" ", ""))
    return entity_chars / total_chars if total_chars > 0 else 0.0


class VietMedNERDataProcessor:
    """Processor wrapping dataset loading and entity extraction."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._datasets = {}

    def load_split(self, split: str):
        """Load and cache a dataset split."""
        if split not in self._datasets:
            print(f"Loading VietMed-NER {split} split...")
            self._datasets[split] = load_vietmed_ner(self.config, split)
            print(f"  Loaded {len(self._datasets[split])} samples")
        return self._datasets[split]

    def get_all_entities(self, split: str) -> List[List[Dict]]:
        """Get entities for all samples in a split."""
        dataset = self.load_split(split)
        all_entities = []
        for sample in tqdm(dataset, desc=f"Extracting entities ({split})"):
            entities = extract_entities_from_sample(sample)
            all_entities.append(entities)
        return all_entities

    def get_keyword_list(self, split: str, max_keywords: int = 50) -> List[str]:
        """Build keyword list from a split."""
        dataset = self.load_split(split)
        return build_keyword_list(dataset, max_keywords)

    def get_entity_list(self, split: str, max_entities: int = 50) -> List[Dict]:
        """Build entity list with type info from a split."""
        dataset = self.load_split(split)
        return build_entity_list(dataset, max_entities)

    def get_stats(self, split: str) -> Dict:
        """Get dataset statistics."""
        dataset = self.load_split(split)
        entity_ratio = compute_entity_char_ratio(dataset)
        all_entities = self.get_all_entities(split)

        entity_type_counts = defaultdict(int)
        total_entities = 0
        for entities in all_entities:
            for e in entities:
                entity_type_counts[e["type"]] += 1
                total_entities += 1

        return {
            "num_samples": len(dataset),
            "entity_char_ratio": entity_ratio,
            "total_entities": total_entities,
            "entity_type_counts": dict(entity_type_counts),
        }
