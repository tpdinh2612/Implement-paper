"""
Main Evaluation Script for ASR on VietMed-NER Dataset.

Re-implementation of "Improving Contextual ASR via Multi-grained Fusion with LLMs"
Applied to Vietnamese medical spoken NER dataset.

Usage:
    python run_evaluation.py [--mode baseline|token|phrase|joint] [--split test]
                             [--max_samples N] [--no_llm]
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

import torch

from config import FullConfig, ASRConfig, LLMConfig, FusionConfig, DataConfig, EvalConfig
from data_loader import VietMedNERDataProcessor, extract_entities_from_sample
from asr_model import WhisperASR, MultiGrainedFusionASR
from metrics import compute_all_metrics, format_metrics_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate ASR on VietMed-NER dataset"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="baseline",
        choices=["baseline", "token", "phrase", "joint", "all"],
        help="Inference mode: baseline (Whisper only), token (+ LLM fusion), "
             "phrase (+ keyword copy), joint (both), all (run all modes)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "validation", "test"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to evaluate (for quick testing)",
    )
    parser.add_argument(
        "--whisper_model",
        type=str,
        default="openai/whisper-small",
        help="Whisper model name/path",
    )
    parser.add_argument(
        "--llm_model",
        type=str,
        default="Qwen/Qwen2-0.5B",
        help="LLM model name/path (for token-level fusion)",
    )
    parser.add_argument(
        "--no_llm",
        action="store_true",
        help="Skip loading LLM (only run baseline/phrase modes)",
    )
    parser.add_argument(
        "--max_keywords",
        type=int,
        default=50,
        help="Maximum keywords in the contextual dictionary",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./outputs",
        help="Directory to save results",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for inference",
    )
    return parser.parse_args()


def run_baseline_evaluation(
    asr: WhisperASR,
    dataset,
    data_processor: VietMedNERDataProcessor,
    split: str,
    max_samples: int = None,
) -> dict:
    """
    Run baseline Whisper evaluation (no contextual biasing).
    """
    print("\n" + "=" * 70)
    print("  Running BASELINE (Whisper) evaluation")
    print("=" * 70)
    
    results = []
    samples = dataset
    if max_samples:
        samples = dataset.select(range(min(max_samples, len(dataset))))
    
    from tqdm import tqdm
    for i, sample in enumerate(tqdm(samples, desc="Baseline inference")):
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        
        hyp = asr.transcribe(audio, sr)
        
        results.append({
            "reference": sample["text"],
            "text": hyp,
            "words": sample["words"],
            "labels": sample["labels"],
        })
    
    metrics = compute_all_metrics(results)
    return metrics, results


def run_fusion_evaluation(
    fusion_model: MultiGrainedFusionASR,
    dataset,
    keywords: list,
    mode: str,
    max_samples: int = None,
) -> dict:
    """
    Run contextual biasing evaluation.
    """
    mode_name = {
        "token": "TOKEN-LEVEL FUSION",
        "phrase": "PHRASE-LEVEL FUSION",
        "joint": "JOINT MULTI-GRAINED FUSION",
    }.get(mode, mode.upper())
    
    print("\n" + "=" * 70)
    print(f"  Running {mode_name} evaluation")
    print(f"  Keywords: {len(keywords)} items")
    print("=" * 70)
    
    samples = dataset
    if max_samples:
        samples = dataset.select(range(min(max_samples, len(dataset))))
    
    results = fusion_model.transcribe_dataset(
        samples, keywords=keywords, mode=mode
    )
    
    metrics = compute_all_metrics(results)
    return metrics, results


def main():
    args = parse_args()
    
    # Setup
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print("=" * 70)
    print("  ASR Evaluation on VietMed-NER Dataset")
    print("  Re-implementation of Multi-grained Fusion with LLMs")
    print("=" * 70)
    print(f"  Mode: {args.mode}")
    print(f"  Split: {args.split}")
    print(f"  Whisper: {args.whisper_model}")
    print(f"  LLM: {args.llm_model}")
    print(f"  Device: {args.device}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print(f"  Max keywords: {args.max_keywords}")
    print("=" * 70)
    
    # Load dataset
    data_config = DataConfig()
    data_processor = VietMedNERDataProcessor(data_config)
    dataset = data_processor.load_split(args.split)
    
    # Print dataset stats
    stats = data_processor.get_stats(args.split)
    print(f"\nDataset Statistics ({args.split}):")
    print(f"  Samples: {stats['num_samples']}")
    print(f"  Entity Char Ratio: {stats['entity_char_ratio']:.4f} ({stats['entity_char_ratio']*100:.2f}%)")
    print(f"  Total Entities: {stats['total_entities']}")
    print(f"  Entity Types: {len(stats['entity_type_counts'])}")
    for etype, count in sorted(stats['entity_type_counts'].items(), key=lambda x: -x[1]):
        print(f"    {etype}: {count}")
    
    # Build keyword list
    keywords = data_processor.get_keyword_list(args.split, args.max_keywords)
    print(f"\nKeyword list ({len(keywords)} keywords):")
    for i, kw in enumerate(keywords[:10]):
        print(f"  [{i}] {kw}")
    if len(keywords) > 10:
        print(f"  ... and {len(keywords) - 10} more")
    
    # Determine which modes to run
    modes_to_run = []
    if args.mode == "all":
        modes_to_run = ["baseline", "phrase"]
        if not args.no_llm:
            modes_to_run.extend(["token", "joint"])
    else:
        modes_to_run = [args.mode]
    
    # Initialize models
    asr_config = ASRConfig(
        model_name=args.whisper_model,
        device=args.device,
    )
    
    all_metrics = {}
    all_results = {}
    
    for mode in modes_to_run:
        start_time = time.time()
        
        if mode == "baseline":
            asr = WhisperASR(asr_config)
            metrics, results = run_baseline_evaluation(
                asr, dataset, data_processor, args.split, args.max_samples
            )
        elif mode in ("token", "phrase", "joint"):
            if mode in ("token", "joint") and args.no_llm:
                print(f"\nSkipping {mode} mode (--no_llm specified)")
                continue
            
            llm_config = LLMConfig(
                model_name=args.llm_model,
                device=args.device,
            )
            fusion_config = FusionConfig(
                token_fusion_enabled=(mode in ("token", "joint")),
                phrase_fusion_enabled=(mode in ("phrase", "joint")),
            )
            fusion_model = MultiGrainedFusionASR(
                asr_config, llm_config, fusion_config
            )
            metrics, results = run_fusion_evaluation(
                fusion_model, dataset, keywords, mode, args.max_samples
            )
        
        elapsed = time.time() - start_time
        metrics["inference_time_seconds"] = elapsed
        metrics["mode"] = mode
        
        # Print results
        table = format_metrics_table(metrics, title=f"{mode.upper()} Mode Results")
        print("\n" + table)
        print(f"\n  Inference time: {elapsed:.1f}s ({elapsed/len(results):.2f}s/sample)")
        
        all_metrics[mode] = metrics
        all_results[mode] = results
    
    # Save results
    output_file = os.path.join(
        args.output_dir, f"results_{args.split}_{timestamp}.json"
    )
    
    # Prepare serializable results
    save_data = {
        "config": {
            "whisper_model": args.whisper_model,
            "llm_model": args.llm_model,
            "split": args.split,
            "max_samples": args.max_samples,
            "max_keywords": args.max_keywords,
            "device": args.device,
        },
        "dataset_stats": {k: v for k, v in stats.items() if k != "entity_type_counts"},
        "entity_type_counts": stats["entity_type_counts"],
        "keywords": keywords,
        "metrics": {},
        "sample_results": {},
    }
    
    for mode, metrics in all_metrics.items():
        # Convert numpy types to Python types for JSON serialization
        save_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float, str)):
                save_metrics[k] = v
            elif isinstance(v, dict):
                save_metrics[k] = {
                    k2: {k3: float(v3) if isinstance(v3, float) else v3 
                         for k3, v3 in v2.items()} if isinstance(v2, dict) else v2
                    for k2, v2 in v.items()
                }
        save_data["metrics"][mode] = save_metrics
        
        # Save first 20 sample results for inspection
        save_data["sample_results"][mode] = [
            {
                "reference": r["reference"],
                "hypothesis": r["text"],
            }
            for r in all_results[mode][:20]
        ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*70}")
    print(f"  Results saved to: {output_file}")
    print(f"{'='*70}")
    
    # Print comparison table if multiple modes
    if len(all_metrics) > 1:
        print("\n" + "=" * 70)
        print("  COMPARISON TABLE (similar to Paper Table 1)")
        print("=" * 70)
        header = f"  {'Mode':<15} {'CER↓':>8} {'B-CER↓':>8} {'U-CER↓':>8} {'R↑':>8} {'WER↓':>8}"
        print(header)
        print("-" * 70)
        for mode, m in all_metrics.items():
            row = (
                f"  {mode:<15} {m['CER']:>8.2f} {m['B-CER']:>8.2f} "
                f"{m['U-CER']:>8.2f} {m['Recall']:>8.2f} {m['WER']:>8.2f}"
            )
            print(row)
        print("=" * 70)


if __name__ == "__main__":
    main()
