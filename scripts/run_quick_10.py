#!/usr/bin/env python3
"""
Quick calibration + validation on 10 balanced samples per model.

Default models (decoder LMs):
- Llama 3.2 3B
- Mistral 7B
- Qwen 2.5 7B
- Dolphin 2.9.4 Llama 3.1 8B
- Phi-3 Mini
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List

# Ensure repo root is on sys.path when running from anywhere
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
import halueval_loader
import truthfulqa_loader
import calibrate_thresholds
import validate


DEFAULT_MODELS = [
    "llama_3.2_3b",
    "mistral_7b",
    "qwen_2.5_7b",
    "dolphin_2.9.4_llama3.1_8b",
    "qwen3_coder_30b_a3b",
]


def set_hf_env(cache_root: str, token: str | None) -> None:
    os.environ.setdefault("HF_HOME", cache_root)
    os.environ.setdefault("HF_HUB_CACHE", str(Path(cache_root) / "hub"))
    os.environ.setdefault("HF_XET_CACHE", str(Path(cache_root) / "xet"))
    os.environ.setdefault("XET_CACHE_HOME", str(Path(cache_root) / "xet"))
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    if token:
        os.environ["HF_TOKEN"] = token


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


def ensure_truthfulqa_splits(update: bool) -> None:
    train_path = Path("./data/truthfulqa/splits/train.json")
    test_path = Path("./data/truthfulqa/splits/test.json")
    if train_path.exists() and test_path.exists() and not update:
        return
    csv_path = truthfulqa_loader.download_truthfulqa(url=config.TRUTHFULQA_URL)
    all_samples = truthfulqa_loader.load_and_sample(csv_path, n_samples=1000, seed=42)
    train_data, test_data = truthfulqa_loader.split_train_test(all_samples, train_ratio=0.5, seed=42)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    truthfulqa_loader.save_split(train_data, str(train_path))
    truthfulqa_loader.save_split(test_data, str(test_path))


def balanced_sample(data: List[dict], n: int, seed: int) -> List[dict]:
    import random
    random.seed(seed)
    pos = [d for d in data if d.get("label") == 1]
    neg = [d for d in data if d.get("label") == 0]
    half = n // 2
    if len(pos) < half or len(neg) < half:
        raise ValueError("Not enough samples for balanced split.")
    return random.sample(pos, half) + random.sample(neg, half)


def pick_models(model_names: List[str]) -> List[dict]:
    lookup = {m["name"]: m for m in config.MODEL_CONFIGS}
    missing = [m for m in model_names if m not in lookup]
    if missing:
        raise ValueError(f"Unknown model names: {missing}")
    return [lookup[m] for m in model_names]


def run_quick(
    models: List[dict],
    dataset: str,
    train_path: str,
    test_path: str,
    max_tokens: int,
) -> List[str]:
    from mlx_lm import load
    import hidden_state_collector
    import model_adapters

    results = []
    for entry in models:
        model_path = entry["path"]
        model_type = entry["model_type"]
        model_name = entry["name"]

        print("=" * 80)
        print(f"Quick 10: {model_name} on {dataset}")
        print("=" * 80)

        print(f"Loading model: {model_path}...")
        model, tokenizer = load(model_path)
        collector = hidden_state_collector.HiddenStateCollector()
        adapter = model_adapters.create_adapter(model, collector, model_type=model_type)

        if dataset == "halueval":
            train_data = halueval_loader.load_split(train_path)
            test_data = halueval_loader.load_split(test_path)
        else:
            train_data = truthfulqa_loader.load_split(train_path)
            test_data = truthfulqa_loader.load_split(test_path)

        train_data = balanced_sample(train_data, n=10, seed=42)
        test_data = balanced_sample(test_data, n=10, seed=123)

        calibrator = calibrate_thresholds.ThresholdCalibrator(
            adapter=adapter,
            tokenizer=tokenizer,
            train_data=train_data,
            max_tokens=max_tokens,
        )

        hbar_s_scores, pri_scores, labels = calibrator.precompute_scores(verbose=True)
        result = calibrator.calibrate_joint(hbar_s_scores, pri_scores, labels, verbose=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calib_path = str(Path("./calibrated_params") / f"{model_name}_{dataset}_quick10_{timestamp}.json")
        calibrator.save_params(result, calib_path)

        output_path = f"./results/validation_{model_name}_{dataset}_quick10_{timestamp}.json"
        validate.run_validation(
            model_path=model_path,
            model_type=model_type,
            test_data_path=test_path,
            calibrated_params_path=calib_path,
            max_tokens=max_tokens,
            n_samples=10,
            seed=42,
            output_path=output_path,
            dataset=dataset,
            truthqa_update=False,
        )
        results.append(output_path)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick 10-sample calibration+validation")
    parser.add_argument("--dataset", default="halueval", choices=["halueval", "truthfulqa"], help="Dataset")
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS, help="Model names from config.MODEL_CONFIGS")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max generation length")
    parser.add_argument("--hf-cache", default="/tmp/hf_cache", help="HF cache root")
    parser.add_argument("--hf-token", default=None, help="HF token (optional)")
    parser.add_argument("--truthqa-update", action="store_true", help="Redownload TruthfulQA")
    args = parser.parse_args()

    set_hf_env(args.hf_cache, args.hf_token)
    ensure_halueval_splits()
    if args.dataset == "truthfulqa":
        ensure_truthfulqa_splits(update=args.truthqa_update)

    models = pick_models(args.models)
    if args.dataset == "halueval":
        train_path = "./data/halueval/splits/train.json"
        test_path = "./data/halueval/splits/test.json"
    else:
        train_path = "./data/truthfulqa/splits/train.json"
        test_path = "./data/truthfulqa/splits/test.json"

    run_quick(models, args.dataset, train_path, test_path, args.max_tokens)


if __name__ == "__main__":
    main()
