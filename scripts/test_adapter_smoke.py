#!/usr/bin/env python3
"""
Smoke test MLX-LM adapters by downloading models and running a tiny forward pass.

- Skips models >=32B by default (config name based)
- Uses small generation (max_tokens=2) for speed
- Prints progress and timing per model

Usage:
  HF_HOME=/tmp/hf HF_HUB_CACHE=/tmp/hf/hub HF_HUB_DISABLE_XET=1 \
  HF_HUB_DISABLE_TELEMETRY=1 HF_HUB_ENABLE_HF_TRANSFER=1 \
  python /Users/mstrkttt/Documents/anthropic-ai-safety/scripts/test_adapter_smoke.py

Optional flags:
  --include-large   Include models >=32B (default: skipped)
  --include-llava   Attempt to load LLaVA (requires mlx-vlm; default: skipped)
  --max-tokens N    Max tokens for the smoke run (default: 2)
"""

import argparse
import time
import traceback
from typing import List

from tqdm import tqdm

import config
import model_adapters
import hidden_state_collector
import monitoring_loop

try:
    from mlx_lm import load
except Exception as e:
    raise SystemExit(f"mlx_lm not available: {e}")


SKIP_BY_NAME = {
    "llama_3.3_70b",
    "qwen_2.5_32b",
    "deepseek_r1_distill_qwen_32b",
}
SKIP_BY_TYPE = {
    "llava",
}


def run_smoke(models: List[dict], max_tokens: int) -> List[tuple]:
    results = []
    prompt = "The capital of France is"

    for entry in tqdm(models, desc="Models", unit="model"):
        name = entry["name"]
        path = entry["path"]
        model_type = entry["model_type"]

        print(f"\n=== {name} ({model_type}) ===")
        start = time.time()
        try:
            print(f"Loading {path} ...")
            model, tokenizer = load(path)
            print("Loaded.")

            collector = hidden_state_collector.HiddenStateCollector()
            adapter = model_adapters.create_adapter(model, collector, model_type=model_type)
            print(f"Adapter: {adapter.__class__.__name__} with {len(adapter.layers)} layers")

            monitor = monitoring_loop.HallucinationMonitor(
                adapter=adapter,
                tokenizer=tokenizer,
                tau=0.0,
                lambda_=1.0,
                pfail_cutoff=1.1,
                max_tokens=max_tokens,
                temperature=0.0,
                alpha_pri=0.1,
                compute_pri=True,
            )

            out = monitor.generate_with_monitoring(prompt, verbose=False, compute_score_only=True)
            print(f"hbar_s_score={out['hbar_s_score']:.4f} pri_score={out['pri_score']:.4f}")
            elapsed = time.time() - start
            print(f"✓ OK in {elapsed:.1f}s")
            results.append((name, True, "ok", elapsed))
        except Exception as e:
            elapsed = time.time() - start
            print(f"FAILED after {elapsed:.1f}s: {e}")
            traceback.print_exc()
            results.append((name, False, str(e), elapsed))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test MLX-LM adapters")
    parser.add_argument("--include-large", action="store_true", help="Include >=32B models")
    parser.add_argument("--include-llava", action="store_true", help="Attempt to load LLaVA models (mlx-vlm)")
    parser.add_argument("--max-tokens", type=int, default=2, help="Max tokens for smoke run")
    args = parser.parse_args()

    models = []
    for entry in config.MODEL_CONFIGS:
        if not args.include_large and entry["name"] in SKIP_BY_NAME:
            continue
        if not args.include_llava and entry["model_type"] in SKIP_BY_TYPE:
            continue
        models.append(entry)

    print("=" * 80)
    print("Adapter Smoke Test")
    print("=" * 80)
    print(f"Models: {len(models)}")
    print(f"Max tokens: {args.max_tokens}")
    print()

    results = run_smoke(models, max_tokens=args.max_tokens)

    print("\nSummary:")
    for name, ok, info, elapsed in results:
        status = "OK" if ok else "FAIL"
        print(f"- {name}: {status} ({elapsed:.1f}s){' - ' + info if not ok else ''}")


if __name__ == "__main__":
    main()
