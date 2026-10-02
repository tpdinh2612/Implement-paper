"""
Quick Test Script - Run a small evaluation to verify the pipeline works.
Uses only 5 samples from the test set for fast validation.

Usage:
    python quick_test.py
"""

import sys
import os
import torch
import warnings
warnings.filterwarnings("ignore")


def main():
    print("=" * 70)
    print("  Quick Test: ASR Pipeline Verification")
    print("=" * 70)
    
    # Check GPU availability
    if torch.cuda.is_available():
        device = "cuda"
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    else:
        device = "cpu"
        print("  Running on CPU (slower)")
    print()
    
    # Step 1: Load dataset
    print("[1/4] Loading VietMed-NER dataset (test split)...")
    from data_loader import VietMedNERDataProcessor, extract_entities_from_sample
    from config import DataConfig
    
    data_config = DataConfig()
    processor = VietMedNERDataProcessor(data_config)
    dataset = processor.load_split("test")
    print(f"  Dataset loaded: {len(dataset)} samples")
    
    # Show first sample
    sample = dataset[0]
    print(f"\n  Sample 0:")
    print(f"    Text: {sample['text']}")
    print(f"    Words: {sample['words'][:10]}...")
    print(f"    Labels: {sample['labels'][:10]}...")
    print(f"    Duration: {sample['duration']}s")
    
    entities = extract_entities_from_sample(sample)
    print(f"    Entities ({len(entities)}):")
    for e in entities:
        print(f"      [{e['type']}] {e['text']}")
    
    # Dataset stats
    stats = processor.get_stats("test")
    print(f"\n  Entity Char Ratio: {stats['entity_char_ratio']*100:.2f}%")
    print(f"  Total Entities: {stats['total_entities']}")
    print(f"  Entity Types: {len(stats['entity_type_counts'])}")
    
    # Step 2: Build keyword list
    print("\n[2/4] Building keyword list...")
    keywords = processor.get_keyword_list("test", max_keywords=50)
    print(f"  Keywords: {len(keywords)} items")
    print(f"  Top 10: {keywords[:10]}")
    
    # Step 3: Run ASR inference on 5 samples
    print(f"\n[3/4] Running Whisper ASR inference (5 samples)...")
    from asr_model import WhisperASR
    from config import ASRConfig
    
    asr_config = ASRConfig(
        model_name="openai/whisper-small",
        device=device,
    )
    asr = WhisperASR(asr_config)
    
    # Take first 5 samples
    test_samples = dataset.select(range(5))
    results = []
    
    for i, sample in enumerate(test_samples):
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        
        hyp = asr.transcribe(audio, sr)
        
        print(f"\n  Sample {i}:")
        print(f"    REF: {sample['text']}")
        print(f"    HYP: {hyp}")
        
        results.append({
            "reference": sample["text"],
            "text": hyp,
            "words": sample["words"],
            "labels": sample["labels"],
        })
    
    # Step 4: Compute metrics
    print(f"\n[4/4] Computing metrics...")
    from metrics import compute_all_metrics, format_metrics_table
    
    metrics = compute_all_metrics(results)
    table = format_metrics_table(metrics, title="Quick Test Results (5 samples)")
    print("\n" + table)
    
    print("\n" + "=" * 70)
    print("  ✓ Pipeline verification complete!")
    print("  To run full evaluation:")
    print("    python run_evaluation.py --mode baseline --split test")
    print("    python run_evaluation.py --mode all --split test --max_samples 50")
    print("=" * 70)


if __name__ == "__main__":
    main()
