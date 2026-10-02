"""
Unified Evaluation Script for all 3 methods on VietMed-NER.

Runs inference and computes metrics for:
1. Baseline Whisper (shared)
2. Multi-grained Fusion (Multigrained paper)
3. CopyNE (CopyNE paper)
4. PEEKING-FUTURE (PEEKING-FUTURE paper)

Usage:
    python run_all_eval.py --max_samples 20 --device cpu
    python run_all_eval.py --methods baseline copyne peeking_future multigrained --max_samples 50
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from shared.config import (
    FullConfig, ASRConfig, LLMConfig, FusionConfig,
    CopyNEConfig, PeekingFutureConfig, DataConfig, EvalConfig,
)
from shared.data_loader import VietMedNERDataProcessor, extract_entities_from_sample
from shared.asr_backbone import WhisperASR
from shared.metrics import compute_all_metrics, format_metrics_table


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate all 3 ASR methods on VietMed-NER"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["baseline", "copyne", "peeking-future", "multigrained"],
        choices=["baseline", "copyne", "peeking-future", "multigrained", "all"],
        help="Methods to evaluate",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max_samples", type=int, default=20,
                        help="Max samples for quick eval (default 20)")
    parser.add_argument("--whisper_model", type=str, default="openai/whisper-small")
    parser.add_argument("--llm_model", type=str, default="Qwen/Qwen2-0.5B")
    parser.add_argument("--no_llm", action="store_true",
                        help="Skip LLM loading for multigrained")
    parser.add_argument("--max_keywords", type=int, default=50)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    return parser.parse_args()


def run_baseline(asr, dataset, max_samples=None):
    """Run baseline Whisper evaluation."""
    from tqdm import tqdm
    print("\n" + "=" * 70)
    print("  [1] BASELINE (Whisper)")
    print("=" * 70)

    samples = dataset
    if max_samples:
        samples = dataset.select(range(min(max_samples, len(dataset))))

    results = []
    for sample in tqdm(samples, desc="Baseline"):
        audio = sample["audio"]["array"]
        sr = sample["audio"]["sampling_rate"]
        hyp = asr.transcribe(audio, sr)
        results.append({
            "reference": sample["text"],
            "text": hyp,
            "words": sample["words"],
            "labels": sample["labels"],
        })

    return compute_all_metrics(results), results


def run_copyne(asr_config, dataset, entity_list, max_samples=None):
    """Run CopyNE evaluation."""
    from methods.copyne.model import CopyNEInference
    print("\n" + "=" * 70)
    print("  [2] CopyNE (Copy Named Entities)")
    print("=" * 70)

    copyne_config = CopyNEConfig()
    copyne = CopyNEInference(asr_config, copyne_config)

    samples = dataset
    if max_samples:
        samples = dataset.select(range(min(max_samples, len(dataset))))

    entity_texts = [e["text"] for e in entity_list] if isinstance(entity_list[0], dict) else entity_list
    results = copyne.transcribe_dataset(samples, entity_list=entity_texts)
    return compute_all_metrics(results), results


def run_peeking_future(asr_config, dataset, entity_list, max_samples=None):
    """Run PEEKING-FUTURE evaluation."""
    from methods.peeking_future.model import PeekingFutureInference
    print("\n" + "=" * 70)
    print("  [3] PEEKING-FUTURE (Multi-Token Prediction Biasing)")
    print("=" * 70)

    peeking_future_config = PeekingFutureConfig()
    peeking_future = PeekingFutureInference(asr_config, peeking_future_config)

    samples = dataset
    if max_samples:
        samples = dataset.select(range(min(max_samples, len(dataset))))

    entity_texts = [e["text"] for e in entity_list] if isinstance(entity_list[0], dict) else entity_list
    results = peeking_future.transcribe_dataset(samples, entity_list=entity_texts)
    return compute_all_metrics(results), results


def run_multigrained(asr_config, llm_config, fusion_config, dataset, keywords, max_samples=None, mode="joint"):
    """Run Multi-grained Fusion evaluation."""
    from methods.multigrained.model import MultiGrainedFusionASR
    print("\n" + "=" * 70)
    print(f"  [4] Multi-grained Fusion (mode={mode})")
    print("=" * 70)

    fusion_model = MultiGrainedFusionASR(asr_config, llm_config, fusion_config)

    samples = dataset
    if max_samples:
        samples = dataset.select(range(min(max_samples, len(dataset))))

    results = fusion_model.transcribe_dataset(samples, keywords=keywords, mode=mode)
    return compute_all_metrics(results), results


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if "all" in args.methods:
        methods = ["baseline", "copyne", "peeking-future", "multigrained"]
    else:
        methods = args.methods

    print("=" * 70)
    print("  UNIFIED EVALUATION: 3 Contextual ASR Methods on VietMed-NER")
    print("=" * 70)
    print(f"  Methods: {', '.join(methods)}")
    print(f"  Split: {args.split}")
    print(f"  Whisper: {args.whisper_model}")
    print(f"  Device: {args.device}")
    print(f"  Max samples: {args.max_samples or 'all'}")
    print("=" * 70)

    # Load dataset
    data_config = DataConfig()
    data_processor = VietMedNERDataProcessor(data_config)
    dataset = data_processor.load_split(args.split)

    stats = data_processor.get_stats(args.split)
    print(f"\nDataset Stats ({args.split}):")
    print(f"  Samples: {stats['num_samples']}")
    print(f"  Entity Char Ratio: {stats['entity_char_ratio']*100:.2f}%")
    print(f"  Total Entities: {stats['total_entities']}")

    # Build keyword/entity lists
    keywords = data_processor.get_keyword_list(args.split, args.max_keywords)
    entity_list = data_processor.get_entity_list(args.split, args.max_keywords)
    print(f"\n  Keywords: {len(keywords)}, Entities: {len(entity_list)}")

    # ASR config
    asr_config = ASRConfig(model_name=args.whisper_model, device=args.device)

    all_metrics = {}
    all_results = {}

    # Run each method
    for method in methods:
        start_time = time.time()

        if method == "baseline":
            asr = WhisperASR(asr_config)
            metrics, results = run_baseline(asr, dataset, args.max_samples)

        elif method == "copyne":
            metrics, results = run_copyne(asr_config, dataset, entity_list, args.max_samples)

        elif method == "peeking-future":
            metrics, results = run_peeking_future(asr_config, dataset, entity_list, args.max_samples)

        elif method == "multigrained":
            if args.no_llm:
                print("\nSkipping multigrained (--no_llm specified)")
                continue
            llm_config = LLMConfig(model_name=args.llm_model, device=args.device)
            fusion_config = FusionConfig()
            metrics, results = run_multigrained(
                asr_config, llm_config, fusion_config,
                dataset, keywords, args.max_samples, mode="joint"
            )

        elapsed = time.time() - start_time
        metrics["inference_time_seconds"] = elapsed
        metrics["method"] = method

        # Print results
        table = format_metrics_table(metrics, title=f"{method.upper()} Results")
        print("\n" + table)
        print(f"  Inference time: {elapsed:.1f}s ({elapsed/max(len(results),1):.2f}s/sample)")

        all_metrics[method] = metrics
        all_results[method] = results

    # Print comparison table
    if len(all_metrics) > 1:
        print("\n" + "=" * 80)
        print("  COMPARISON TABLE: All Methods on VietMed-NER")
        print("=" * 80)
        header = f"  {'Method':<20} {'CER↓':>8} {'B-CER↓':>8} {'U-CER↓':>8} {'R↑':>8} {'WER↓':>8} {'NER-F1↑':>8}"
        print(header)
        print("-" * 80)
        for method, m in all_metrics.items():
            ner_f1 = m.get("NER", {}).get("overall", {}).get("f1", 0.0) * 100
            row = (
                f"  {method:<20} {m['CER']:>8.2f} {m['B-CER']:>8.2f} "
                f"{m['U-CER']:>8.2f} {m['Recall']:>8.2f} {m['WER']:>8.2f} {ner_f1:>8.2f}"
            )
            print(row)
        print("=" * 80)

    # Save results
    output_file = os.path.join(args.output_dir, f"unified_results_{timestamp}.json")

    save_data = {
        "config": {
            "whisper_model": args.whisper_model,
            "llm_model": args.llm_model,
            "split": args.split,
            "max_samples": args.max_samples,
            "max_keywords": args.max_keywords,
            "device": args.device,
            "methods": methods,
        },
        "dataset_stats": stats,
        "keywords": keywords[:20],
        "metrics": {},
        "sample_results": {},
    }

    for method, metrics in all_metrics.items():
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
        save_data["metrics"][method] = save_metrics

        save_data["sample_results"][method] = [
            {"reference": r["reference"], "hypothesis": r["text"]}
            for r in all_results[method][:10]
        ]

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\n  Results saved to: {output_file}")

    return all_metrics


if __name__ == "__main__":
    main()
