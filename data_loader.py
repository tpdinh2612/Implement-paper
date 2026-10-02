"""
Dataset loader and preprocessor for VietMed-NER dataset.
Handles loading audio, text, and NER labels from HuggingFace datasets.
Extracts keyword lists (named entities) for contextual biasing.
"""

import re
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import torch
import numpy as np
from datasets import load_dataset, Audio
from tqdm import tqdm

from config import DataConfig


def load_vietmed_ner(config: DataConfig, split: str = "test"):
    """
    Load the VietMed-NER dataset from HuggingFace.
    
    Args:
        config: DataConfig with dataset name and cache dir
        split: 'train', 'validation', or 'test'
    
    Returns:
        HuggingFace Dataset object with audio resampled to 16kHz
    """
    dataset = load_dataset(
        config.dataset_name,
        split=split,
        cache_dir=config.cache_dir,
    )
    # Resample audio to 16kHz (Whisper requirement)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=config.sample_rate))
    return dataset


def extract_entities_from_sample(sample: Dict) -> List[Dict]:
    """
    Extract named entities from a single sample's BIO tags.
    
    Args:
        sample: dict with 'words', 'labels' keys
        
    Returns:
        List of entity dicts with 'text', 'type', 'start_idx', 'end_idx'
    """
    words = sample["words"]
    labels = sample["labels"]
    entities = []
    current_entity = None

    for i, (word, label) in enumerate(zip(words, labels)):
        if label.startswith("B-"):
            # Save previous entity if exists
            if current_entity is not None:
                entities.append(current_entity)
            # Start new entity
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
                # Type mismatch, save current and start new
                entities.append(current_entity)
                current_entity = {
                    "text": word,
                    "type": entity_type,
                    "start_idx": i,
                    "end_idx": i,
                    "words": [word],
                }
        else:
            # O tag or I without matching B
            if current_entity is not None:
                entities.append(current_entity)
                current_entity = None

    # Don't forget last entity
    if current_entity is not None:
        entities.append(current_entity)

    return entities


def build_keyword_list(dataset, max_keywords: int = 50) -> List[str]:
    """
    Build a keyword list from all entities in the dataset.
    Used for contextual biasing (keyword dictionary).
    
    Args:
        dataset: HuggingFace dataset
        max_keywords: maximum number of keywords
        
    Returns:
        List of unique keyword strings
    """
    keyword_counts = defaultdict(int)
    for sample in dataset:
        entities = extract_entities_from_sample(sample)
        for entity in entities:
            keyword_counts[entity["text"].lower()] += 1

    # Sort by frequency and take top-k
    sorted_keywords = sorted(keyword_counts.items(), key=lambda x: -x[1])
    keywords = [kw for kw, _ in sorted_keywords[:max_keywords]]
    return keywords


def get_keyword_spans(
    reference_text: str,
    entities: List[Dict],
    words: List[str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """
    Get character-level spans for keyword and non-keyword segments.
    
    Args:
        reference_text: full reference transcription text
        entities: list of entity dicts
        words: list of words in the reference
        
    Returns:
        (keyword_spans, non_keyword_spans) as character-level (start, end) tuples
    """
    # Build word-to-char position mapping
    word_char_positions = []
    pos = 0
    for word in words:
        start = reference_text.find(word, pos)
        if start == -1:
            start = pos
        end = start + len(word)
        word_char_positions.append((start, end))
        pos = end

    # Get keyword character spans
    keyword_char_spans = []
    for entity in entities:
        start_char = word_char_positions[entity["start_idx"]][0]
        end_char = word_char_positions[entity["end_idx"]][1]
        keyword_char_spans.append((start_char, end_char))

    # Get non-keyword spans (complement)
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
    """
    Compute the ratio of entity characters to total characters.
    This is the 'Entity Char Ratio' metric from the paper.
    """
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
    """
    Processor class that wraps dataset loading and entity extraction
    for the full evaluation pipeline.
    """
    
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
