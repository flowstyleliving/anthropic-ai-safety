#!/usr/bin/env python3
"""
Visualize phase transitions for the risk model.

Example usage:
  python3 visualize_phase_transitions.py \
    --calibrated ./calibrated_params/llama*.json ./calibrated_params/qwen*.json \
    --scores-file ./results/scores_llama_n500.npz \
    --outdir ./figures

  python3 visualize_phase_transitions.py \
    --collect-scores 500 \
    --model llama \
    --split test \
    --outdir ./figures
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import halueval_loader
import hidden_state_collector
import model_adapters
import monitoring_loop
import config

try:
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover - dependency availability depends on environment
    LogisticRegression = None

try:
    from mlx_lm import load as mlx_load
except Exception:  # pragma: no cover - only needed for score collection
    mlx_load = None


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def expand_paths(patterns: Iterable[str]) -> List[str]:
    paths: List[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if matches:
            paths.extend(str(p) for p in matches)
        else:
            if Path(pattern).exists():
                paths.append(pattern)
            else:
                print(f"Warning: no matches for pattern {pattern}")
    return list(dict.fromkeys(paths))


def model_display_name(path: str, params: Dict[str, Any]) -> str:
    name = params.get("model_name") or params.get("model_type") or ""
    if not name:
        stem = Path(path).stem
        lower = stem.lower()
        if "llama" in lower:
            name = "Llama"
        elif "qwen" in lower:
            name = "Qwen"
        elif "phi" in lower:
            name = "Phi-3"
        else:
            name = stem
    return str(name)


def load_calibrated_params(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    data["_path"] = path
    return data


def get_float(params: Dict[str, Any], *keys: str) -> Optional[float]:
    for key in keys:
        if key in params and params[key] is not None:
            try:
                return float(params[key])
            except (TypeError, ValueError):
                continue
    return None


def load_scores_file(path: str) -> Dict[str, np.ndarray]:
    scores: Dict[str, np.ndarray] = {}
    suffix = Path(path).suffix.lower()

    if suffix == ".npz":
        data = np.load(path)
        scores["hbar_s"] = _first_key(data, ["hbar_s_score", "hbar_s", "hbar_scores"])
        scores["pri"] = _first_key(data, ["pri_score", "pri", "pri_scores"])
        scores["label"] = _first_key(data, ["label", "labels", "y_true"])
        return scores

    if suffix == ".jsonl":
        hbar_vals: List[float] = []
        pri_vals: List[float] = []
        labels: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                hbar_vals.append(float(row.get("hbar_s", row.get("hbar_s_score", 0.0))))
                pri_vals.append(float(row.get("pri", row.get("pri_score", 0.0))))
                labels.append(int(row.get("label", 0)))
        scores["hbar_s"] = np.array(hbar_vals, dtype=float)
        scores["pri"] = np.array(pri_vals, dtype=float)
        scores["label"] = np.array(labels, dtype=int)
        return scores

    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "scores_by_sample" in data:
            samples = data["scores_by_sample"]
        elif isinstance(data, list):
            samples = data
        else:
            raise ValueError(f"Unsupported JSON format for scores: {path}")
        hbar_vals = [float(s.get("hbar_s", s.get("hbar_s_score", 0.0))) for s in samples]
        pri_vals = [float(s.get("pri", s.get("pri_score", 0.0))) for s in samples]
        labels = [int(s.get("label", 0)) for s in samples]
        scores["hbar_s"] = np.array(hbar_vals, dtype=float)
        scores["pri"] = np.array(pri_vals, dtype=float)
        scores["label"] = np.array(labels, dtype=int)
        return scores

    raise ValueError(f"Unsupported scores file: {path}")


def _first_key(data: Any, keys: List[str]) -> np.ndarray:
    for key in keys:
        if key in data:
            return np.array(data[key])
    return np.array([])


def detect_direction(scores: Optional[Dict[str, np.ndarray]]) -> str:
    if not scores:
        return "lower"
    labels = scores.get("label")
    hbar_s = scores.get("hbar_s")
    if labels is None or hbar_s is None or len(labels) == 0:
        return "lower"
    hal = hbar_s[labels == 1]
    cor = hbar_s[labels == 0]
    if len(hal) == 0 or len(cor) == 0:
        return "lower"
    return "lower" if np.mean(hal) < np.mean(cor) else "higher"


def fit_lambda_tau(
    scores: Dict[str, np.ndarray], direction: str
) -> Optional[Tuple[float, float]]:
    if LogisticRegression is None:
        print("Warning: scikit-learn not available; cannot fit lambda. Using fallback.")
        return None
    labels = scores.get("label")
    hbar_s = scores.get("hbar_s")
    if labels is None or hbar_s is None or len(labels) == 0:
        return None
    if len(np.unique(labels)) < 2:
        print("Warning: only one class in labels; cannot fit lambda. Using fallback.")
        return None

    x = hbar_s.reshape(-1, 1)
    if direction == "lower":
        x = -x

    model = LogisticRegression(solver="lbfgs", max_iter=1000)
    model.fit(x, labels)
    beta1 = float(model.coef_[0][0])
    beta0 = float(model.intercept_[0])
    if beta1 == 0.0:
        return None

    if direction == "lower":
        lambda_ = beta1
        tau = beta0 / beta1
    else:
        lambda_ = beta1
        tau = -beta0 / beta1

    if lambda_ <= 0:
        print("Warning: fitted lambda is non-positive; using absolute value.")
        lambda_ = abs(lambda_)
    return lambda_, tau


def prepare_model_specs(
    calibrated_paths: List[str],
    scores: Optional[Dict[str, np.ndarray]],
    direction: str,
) -> List[Dict[str, Any]]:
    specs: List[Dict[str, Any]] = []
    for path in calibrated_paths:
        params = load_calibrated_params(path)
        tau = get_float(params, "tau_hbar", "tau")
        lambda_ = get_float(params, "lambda_", "lambda")
        if lambda_ is None or tau is None:
            fit = fit_lambda_tau(scores, direction) if scores else None
            if fit is not None:
                lambda_fit, tau_fit = fit
                lambda_ = lambda_ if lambda_ is not None else lambda_fit
                tau = tau if tau is not None else tau_fit
        if lambda_ is None:
            print(f"Warning: missing lambda for {path}; using fallback 5.0")
            lambda_ = 5.0
        if tau is None:
            print(f"Warning: missing tau for {path}; using fallback 0.0")
            tau = 0.0

        specs.append(
            {
                "name": model_display_name(path, params),
                "path": path,
                "tau": float(tau),
                "lambda": float(lambda_),
                "auroc_hbar": get_float(params, "auroc_hbar"),
                "auroc_pri": get_float(params, "auroc_pri"),
            }
        )
    return specs


def resolve_x_range(
    specs: List[Dict[str, Any]], scores: Optional[Dict[str, np.ndarray]]
) -> Tuple[float, float]:
    if scores and scores.get("hbar_s") is not None and len(scores["hbar_s"]) > 0:
        min_x = float(np.min(scores["hbar_s"]))
        max_x = float(np.max(scores["hbar_s"]))
        pad = max((max_x - min_x) * 0.05, 0.25)
        return min_x - pad, max_x + pad
    taus = [s["tau"] for s in specs if s.get("tau") is not None]
    if taus:
        min_x = min(taus) - 2.0
        max_x = max(taus) + 2.0
        return min_x, max_x
    return -2.0, 4.0


def format_label(spec: Dict[str, Any]) -> str:
    label = f"{spec['name']}  tau={spec['tau']:.3f}  lambda={spec['lambda']:.2f}"
    if spec.get("auroc_pri") is not None:
        label += f"  AUROC_PRI={spec['auroc_pri']:.3f}"
    if spec.get("auroc_hbar") is not None:
        label += f"  AUROC_hbar={spec['auroc_hbar']:.3f}"
    return label


def plot_main_phase_transition(
    specs: List[Dict[str, Any]],
    scores: Optional[Dict[str, np.ndarray]],
    direction: str,
    outdir: Path,
) -> Path:
    x_min, x_max = resolve_x_range(specs, scores)
    x = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(8.5, 5.5), dpi=150)
    for spec in specs:
        tau = spec["tau"]
        lambda_ = spec["lambda"]
        if direction == "lower":
            y = sigmoid(lambda_ * (tau - x))
        else:
            y = sigmoid(lambda_ * (x - tau))
        (line,) = plt.plot(x, y, linewidth=2.0, label=format_label(spec))
        plt.axvline(tau, linestyle="--", linewidth=1.0, alpha=0.6, color=line.get_color())

    plt.xlabel("hbar_s score")
    plt.ylabel("Failure risk (p_fail)")
    plt.title("Phase transition of failure risk vs hbar_s")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=8)
    plt.tight_layout()

    output_path = outdir / "fig_phase_transition_main.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_lambda_sweep(
    specs: List[Dict[str, Any]],
    scores: Optional[Dict[str, np.ndarray]],
    direction: str,
    outdir: Path,
) -> Optional[Path]:
    if not specs:
        print("Warning: no calibrated models provided; skipping lambda sweep.")
        return None
    base = specs[0]
    tau = base["tau"]
    x_min, x_max = resolve_x_range([base], scores)
    x = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(7.5, 5.0), dpi=150)
    for lambda_ in [2, 5, 10, 20]:
        if direction == "lower":
            y = sigmoid(lambda_ * (tau - x))
        else:
            y = sigmoid(lambda_ * (x - tau))
        plt.plot(x, y, linewidth=2.0, label=f"lambda={lambda_}")
    plt.axvline(tau, linestyle="--", linewidth=1.0, alpha=0.6, color="black")
    plt.xlabel("hbar_s score")
    plt.ylabel("Failure risk (p_fail)")
    plt.title(f"Lambda sweep (tau={tau:.3f})")
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=9)
    plt.tight_layout()

    output_path = outdir / "fig_phase_transition_lambda_sweep.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def plot_data_overlay(
    specs: List[Dict[str, Any]],
    scores: Optional[Dict[str, np.ndarray]],
    direction: str,
    outdir: Path,
) -> Optional[Path]:
    if not scores or scores.get("hbar_s") is None or scores.get("label") is None:
        print("Warning: no scores file provided; skipping data overlay plot.")
        return None

    x_min, x_max = resolve_x_range(specs, scores)
    x_grid = np.linspace(x_min, x_max, 500)

    plt.figure(figsize=(8.0, 5.0), dpi=150)
    rng = np.random.default_rng(0)
    labels = scores["label"].astype(float)
    jitter = rng.uniform(-0.06, 0.06, size=labels.shape)
    plt.scatter(
        scores["hbar_s"],
        labels + jitter,
        s=10,
        alpha=0.35,
        color="#4C72B0",
        label="Samples",
    )

    tau = None
    lambda_ = None
    if specs:
        tau = specs[0]["tau"]
        lambda_ = specs[0]["lambda"]
    else:
        fit = fit_lambda_tau(scores, direction)
        if fit is not None:
            lambda_, tau = fit
    if tau is not None and lambda_ is not None:
        if direction == "lower":
            y = sigmoid(lambda_ * (tau - x_grid))
        else:
            y = sigmoid(lambda_ * (x_grid - tau))
        plt.plot(x_grid, y, color="#C44E52", linewidth=2.0, label="Theoretical p_fail")

    plt.xlabel("hbar_s score")
    plt.ylabel("Label (jittered)")
    plt.title("Data overlay: samples vs theoretical p_fail")
    plt.ylim(-0.2, 1.2)
    plt.grid(True, alpha=0.2)
    plt.legend(fontsize=9)
    plt.tight_layout()

    output_path = outdir / "fig_phase_transition_data_overlay.png"
    plt.savefig(output_path, dpi=300)
    plt.close()
    return output_path


def collect_scores(
    model_type: str,
    split: str,
    n_samples: int,
    outdir: Path,
) -> Tuple[Path, Path]:
    if mlx_load is None:
        raise RuntimeError("mlx_lm not available; cannot collect scores.")

    model_path_map: Dict[str, str] = {}
    for entry in getattr(config, "MODEL_CONFIGS", []):
        name = entry.get("name", "").lower()
        path = entry.get("path")
        if not path:
            continue
        if "llama" in name:
            model_path_map.setdefault("llama", path)
        if "qwen" in name:
            model_path_map.setdefault("qwen", path)
        if "phi" in name:
            model_path_map.setdefault("phi3", path)
            model_path_map.setdefault("phi-3", path)
        if "mistral" in name:
            model_path_map.setdefault("mistral", path)
        if "smollm" in name:
            model_path_map.setdefault("smollm", path)
    model_path_map.setdefault("llama", "mlx-community/Llama-3.2-3B-Instruct-4bit")
    model_path_map.setdefault("qwen", "mlx-community/Qwen2.5-7B-Instruct-4bit")
    model_path_map.setdefault("phi3", "mlx-community/Phi-3-mini-128k-instruct-4bit")
    model_path_map.setdefault("phi-3", model_path_map["phi3"])
    model_path_map.setdefault("mistral", "mlx-community/Mistral-7B-Instruct-v0.3-4bit")
    model_path_map.setdefault("smollm", "mlx-community/SmolLM-360M-Instruct")
    if model_type not in model_path_map:
        raise ValueError(f"Unsupported model type: {model_type}")

    data_path = Path(f"./data/halueval/splits/{split}.json")
    if not data_path.exists():
        raise FileNotFoundError(f"Split not found: {data_path}")

    print(f"Loading model: {model_path_map[model_type]}...")
    model, tokenizer = mlx_load(model_path_map[model_type])
    collector = hidden_state_collector.HiddenStateCollector()
    adapter = model_adapters.create_adapter(model, collector, model_type=model_type)

    test_data = halueval_loader.load_split(str(data_path))
    if n_samples < len(test_data):
        rng = np.random.default_rng(42)
        indices = rng.choice(len(test_data), size=n_samples, replace=False)
        test_data = [test_data[i] for i in indices]
    print(f"Collecting scores for {len(test_data)} samples.")

    monitor = monitoring_loop.HallucinationMonitor(
        adapter=adapter,
        tokenizer=tokenizer,
        tau=0.0,
        lambda_=1.0,
        pfail_cutoff=1.1,
        max_tokens=20,
        temperature=0.0,
        alpha_pri=0.1,
        compute_pri=True,
    )

    outdir.mkdir(parents=True, exist_ok=True)
    jsonl_path = outdir / f"scores_{model_type}_n{len(test_data)}_{split}.jsonl"
    npz_path = outdir / f"scores_{model_type}_n{len(test_data)}_{split}.npz"

    hbar_vals: List[float] = []
    pri_vals: List[float] = []
    labels: List[int] = []

    import gc

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for idx, sample in enumerate(test_data):
            try:
                result = monitor.generate_with_monitoring(
                    sample["prompt"],
                    verbose=False,
                    compute_score_only=True,
                )
                hbar_s = float(result["hbar_s_score"])
                pri = float(result["pri_score"])
                label = int(sample["label"])
            except Exception as exc:
                hbar_s = 0.0
                pri = 0.0
                label = int(sample.get("label", 0))
                print(f"Warning: error on sample {sample.get('id', idx)}: {exc}")

            hbar_vals.append(hbar_s)
            pri_vals.append(pri)
            labels.append(label)
            f.write(
                json.dumps(
                    {
                        "sample_id": sample.get("id", f"sample_{idx:04d}"),
                        "label": label,
                        "hbar_s": hbar_s,
                        "pri": pri,
                    }
                )
                + "\n"
            )

            if (idx + 1) % 50 == 0:
                f.flush()
                gc.collect()
                print(f"  checkpointed {idx + 1} samples")

    np.savez(
        npz_path,
        hbar_s=np.array(hbar_vals, dtype=float),
        pri=np.array(pri_vals, dtype=float),
        label=np.array(labels, dtype=int),
    )

    print(f"Saved scores: {jsonl_path}")
    print(f"Saved scores: {npz_path}")
    return jsonl_path, npz_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Phase transition visualization")
    parser.add_argument(
        "--calibrated",
        nargs="*",
        default=[],
        help="Paths or globs for calibrated params JSON files.",
    )
    parser.add_argument(
        "--scores-file",
        type=str,
        default=None,
        help="Existing scores file (.npz, .jsonl, or validation .json).",
    )
    parser.add_argument(
        "--collect-scores",
        type=int,
        default=None,
        help="Collect N samples with compute_score_only=True.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="llama",
        help="Model type for score collection (llama, qwen, phi3).",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split for score collection (train or test).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./figures",
        help="Output directory for plots and collected scores.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    scores: Optional[Dict[str, np.ndarray]] = None
    if args.collect_scores:
        _, npz_path = collect_scores(args.model, args.split, args.collect_scores, Path("./results"))
        scores = load_scores_file(str(npz_path))
    elif args.scores_file:
        scores = load_scores_file(args.scores_file)

    calibrated_paths = expand_paths(args.calibrated)
    direction = detect_direction(scores)

    specs = prepare_model_specs(calibrated_paths, scores, direction) if calibrated_paths else []

    if not specs:
        print("Warning: no calibrated params provided; plots will use scores-fit only where possible.")

    main_path = plot_main_phase_transition(specs, scores, direction, outdir)
    lambda_path = plot_lambda_sweep(specs, scores, direction, outdir)
    overlay_path = plot_data_overlay(specs, scores, direction, outdir)

    print(f"Saved plot: {main_path}")
    if lambda_path:
        print(f"Saved plot: {lambda_path}")
    if overlay_path:
        print(f"Saved plot: {overlay_path}")


if __name__ == "__main__":
    main()
