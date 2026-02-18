#!/usr/bin/env python3
"""
End-to-end pipeline:
- Calibrate on train split
- Validate on test split
- Generate figures and metrics table

Runs for HaluEval 2.0 and TruthfulQA (latest by default).
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Optional

# Ensure repo root is on sys.path when running from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import halueval_loader
import truthfulqa_loader
import calibrate_thresholds
import validate


def ensure_halueval_splits() -> None:
    train_path = Path("./data/halueval/splits/train.json")
    test_path = Path("./data/halueval/splits/test.json")
    if train_path.exists() and test_path.exists():
        return
    dataset_paths = halueval_loader.download_halueval()
    all_samples = halueval_loader.load_and_sample(
        dataset_paths,
        n_samples=config.DATASET_SAMPLE_SIZE,
        seed=config.DATASET_RANDOM_SEED,
    )
    train_data, test_data = halueval_loader.split_train_test(
        all_samples,
        train_ratio=config.TRAIN_TEST_SPLIT_RATIO,
        seed=config.DATASET_RANDOM_SEED,
    )
    train_path.parent.mkdir(parents=True, exist_ok=True)
    halueval_loader.save_split(train_data, str(train_path))
    halueval_loader.save_split(test_data, str(test_path))


def ensure_truthfulqa_splits(update: bool, url: Optional[str]) -> None:
    train_path = Path("./data/truthfulqa/splits/train.json")
    test_path = Path("./data/truthfulqa/splits/test.json")
    if train_path.exists() and test_path.exists() and not update:
        return
    csv_path = truthfulqa_loader.download_truthfulqa(url=url or config.TRUTHFULQA_URL)
    all_samples = truthfulqa_loader.load_and_sample(csv_path, n_samples=1000, seed=42)
    train_data, test_data = truthfulqa_loader.split_train_test(all_samples, train_ratio=0.5, seed=42)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    truthfulqa_loader.save_split(train_data, str(train_path))
    truthfulqa_loader.save_split(test_data, str(test_path))


def pick_models(include_large: bool) -> List[dict]:
    models = []
    for entry in config.MODEL_CONFIGS:
        if entry["model_type"] == "llava":
            continue
        if not include_large and entry["name"] in {
            "llama_3.3_70b",
            "qwen_2.5_32b",
            "deepseek_r1_distill_qwen_32b",
        }:
            continue
        models.append(entry)
    return models


def calibrate_for_dataset(entry: dict, dataset: str, train_path: str, n_samples: int, max_tokens: int) -> str:
    from mlx_lm import load
    import hidden_state_collector
    import model_adapters

    model_path = entry["path"]
    model_type = entry["model_type"]
    model_name = entry["name"]

    print("=" * 80)
    print(f"Calibration: {model_name} on {dataset}")
    print("=" * 80)

    print(f"Loading model: {model_path}...")
    model, tokenizer = load(model_path)
    collector = hidden_state_collector.HiddenStateCollector()
    adapter = model_adapters.create_adapter(model, collector, model_type=model_type)

    if dataset == "halueval":
        train_data = halueval_loader.load_split(train_path)
    else:
        train_data = truthfulqa_loader.load_split(train_path)

    if n_samples and n_samples < len(train_data):
        import random
        random.seed(42)
        train_data = random.sample(train_data, n_samples)

    calibrator = calibrate_thresholds.ThresholdCalibrator(
        adapter=adapter,
        tokenizer=tokenizer,
        train_data=train_data,
        max_tokens=max_tokens,
    )

    hbar_s_scores, pri_scores, labels = calibrator.precompute_scores(verbose=True)
    result = calibrator.calibrate_joint(hbar_s_scores, pri_scores, labels, verbose=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{model_name}_{dataset}_{timestamp}_n{len(train_data)}.json"
    output_path = str(Path("./calibrated_params") / output_filename)
    calibrator.save_params(result, output_path)
    return output_path


def run_validation_for_dataset(
    entry: dict,
    dataset: str,
    test_path: str,
    calibrated_params_path: str,
    max_tokens: int,
    n_samples: Optional[int],
) -> str:
    model_path = entry["path"]
    model_type = entry["model_type"]
    model_name = entry["name"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"./results/validation_{model_name}_{dataset}_{timestamp}.json"

    validate.run_validation(
        model_path=model_path,
        model_type=model_type,
        test_data_path=test_path,
        calibrated_params_path=calibrated_params_path,
        max_tokens=max_tokens,
        n_samples=n_samples,
        seed=42,
        output_path=output_path,
        dataset=dataset,
        truthqa_update=False,
    )
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full calibration/validation/figures pipeline")
    parser.add_argument("--include-large", action="store_true", help="Include >=32B models")
    parser.add_argument("--hf-cache", default="/tmp/hf_cache", help="HF cache root (default: /tmp/hf_cache)")
    parser.add_argument("--hf-token", default=None, help="HF token (optional; uses HF_TOKEN env if set)")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max generation length")
    parser.add_argument("--train-samples", type=int, default=200, help="Number of train samples per model")
    parser.add_argument("--test-samples", type=int, default=None, help="Number of test samples per model")
    parser.add_argument("--truthqa-update", action="store_true", help="Redownload and resplit TruthfulQA (default on)")
    parser.add_argument("--no-truthqa-update", action="store_true", help="Skip TruthfulQA redownload")
    parser.add_argument("--truthqa-url", default=None, help="Optional TruthfulQA CSV URL")
    parser.add_argument("--output-dir", default="./figures", help="Figures output dir")
    args = parser.parse_args()

    # Force HF cache to a writable location to avoid /Volumes permissions
    os.environ.setdefault("HF_HOME", args.hf_cache)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(args.hf_cache) / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(Path(args.hf_cache) / "xet"))
    os.environ.setdefault("XET_CACHE_HOME", str(Path(args.hf_cache) / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token

    ensure_halueval_splits()
    truthqa_update = True if not args.no_truthqa_update else False
    if args.truthqa_update:
        truthqa_update = True
    ensure_truthfulqa_splits(update=truthqa_update, url=args.truthqa_url)

    models = pick_models(include_large=args.include_large)
    if not models:
        raise SystemExit("No models selected. Check config.MODEL_CONFIGS.")

    all_results = []

    for dataset, train_path, test_path in [
        ("halueval", "./data/halueval/splits/train.json", "./data/halueval/splits/test.json"),
        ("truthfulqa", "./data/truthfulqa/splits/train.json", "./data/truthfulqa/splits/test.json"),
    ]:
        for entry in models:
            calib_path = calibrate_for_dataset(
                entry=entry,
                dataset=dataset,
                train_path=train_path,
                n_samples=args.train_samples,
                max_tokens=args.max_tokens,
            )
            result_path = run_validation_for_dataset(
                entry=entry,
                dataset=dataset,
                test_path=test_path,
                calibrated_params_path=calib_path,
                max_tokens=args.max_tokens,
                n_samples=args.test_samples,
            )
            all_results.append(result_path)

    # Generate figures + metrics table
    print("\nGenerating figures...")
    fig_cmd = [
        "python",
        str(Path(__file__).resolve().parents[1] / "generate_figures.py"),
        "--output-dir",
        args.output_dir,
        "--results",
        *all_results,
    ]
    subprocess.run(fig_cmd, check=True)


if __name__ == "__main__":
    main()
