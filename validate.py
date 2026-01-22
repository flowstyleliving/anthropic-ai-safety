"""
Validation script for hallucination detection on test split.

Loads calibrated parameters and evaluates on held-out test data with comprehensive metrics:
- AUROC (Area Under ROC Curve)
- PR-AUC (Precision-Recall AUC)  
- Precision @ Recall ≥ 0.9 at calibrated thresholds
- Best precision @ Recall ≥ 0.9 from test-set threshold sweep
- Confusion matrices
- Quadrant analysis

IMPORTANT: ℏₛ is INVERTED on HaluEval (hallucinations have lower ℏₛ).
           Rule: "hallucination if ℏₛ ≤ τ_hbar" or equivalently use -hbar_s for sklearn.
"""

import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    confusion_matrix
)
from tqdm import tqdm
import mlx.core as mx
from mlx_lm import load

import model_adapters
import hidden_state_collector
import monitoring_loop
import halueval_loader


def load_calibrated_params(params_path: str) -> Dict[str, Any]:
    """
    Load calibrated parameters from JSON file.
    
    Args:
        params_path: Path to calibration JSON
        
    Returns:
        Dict with tau_hbar, tau_pri, and other calibration results
    """
    with open(params_path, 'r') as f:
        params = json.load(f)
    return params


def precompute_test_scores(
    adapter: model_adapters.ModelAdapter,
    tokenizer: Any,
    test_data: List[Dict[str, Any]],
    max_tokens: int = 20,
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[str, Any]]]:
    """
    Run generation on test split and compute ℏₛ and PRI scores.
    
    Args:
        adapter: ModelAdapter instance
        tokenizer: Tokenizer
        test_data: List of test samples
        max_tokens: Max generation length
        verbose: Print progress
        
    Returns:
        Tuple of (hbar_s_scores, pri_scores, labels, scores_by_sample) 
        where scores_by_sample is list of dicts with per-sample data
    """
    # Create monitor with halting DISABLED
    monitor = monitoring_loop.HallucinationMonitor(
        adapter=adapter,
        tokenizer=tokenizer,
        tau=0.0,
        lambda_=1.0,
        pfail_cutoff=1.1,  # Disable halting (>1.0)
        max_tokens=max_tokens,
        temperature=0.0,  # Greedy
        alpha_pri=0.1,
        compute_pri=True
    )
    
    hbar_s_scores = []
    pri_scores = []
    labels = []
    scores_by_sample = []  # NEW: Track per-sample data
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"Computing Scores on Test Set ({len(test_data)} samples)")
        print(f"{'='*80}\n")
    
    import gc
    for i, sample in enumerate(tqdm(test_data, desc="Generating", disable=not verbose)):
        try:
            result = monitor.generate_with_monitoring(
                sample["prompt"],
                verbose=False,
                compute_score_only=True
            )
            
            hbar_s = result["hbar_s_score"]
            pri = result["pri_score"]
            label = sample["label"]
            
            hbar_s_scores.append(hbar_s)
            pri_scores.append(pri)
            labels.append(label)
            
            # NEW: Save per-sample data for figure generation
            scores_by_sample.append({
                "sample_id": sample.get("id", f"sample_{i:04d}"),
                "label": int(label),
                "hbar_s": float(hbar_s),
                "pri": float(pri)
            })
            
            del result
            
            if (i + 1) % 50 == 0:
                gc.collect()
                if verbose:
                    print(f"  [{i+1}/{len(test_data)}] Memory cleanup")
        
        except Exception as e:
            if verbose:
                print(f"Error on sample {sample.get('id', i)}: {e}")
            hbar_s_scores.append(0.0)
            pri_scores.append(0.0)
            labels.append(sample["label"])
            
            # NEW: Still save error samples
            scores_by_sample.append({
                "sample_id": sample.get("id", f"sample_{i:04d}"),
                "label": int(sample["label"]),
                "hbar_s": 0.0,
                "pri": 0.0,
                "error": str(e)
            })
    
    return np.array(hbar_s_scores), np.array(pri_scores), np.array(labels), scores_by_sample


def evaluate_at_calibrated_thresholds(
    hbar_s_scores: np.ndarray,
    pri_scores: np.ndarray,
    labels: np.ndarray,
    tau_hbar: float,
    tau_pri: float,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate performance using calibrated thresholds from training.
    
    Args:
        hbar_s_scores: ℏₛ scores on test set
        pri_scores: PRI scores on test set
        labels: Ground truth labels
        tau_hbar: Calibrated ℏₛ threshold (hallucination if ℏₛ ≤ τ)
        tau_pri: Calibrated PRI threshold (hallucination if PRI ≥ τ)
        verbose: Print results
        
    Returns:
        Dict with metrics for each model
    """
    if verbose:
        print(f"\n{'='*80}")
        print("Evaluation at Calibrated Thresholds")
        print(f"{'='*80}\n")
        print(f"τ_hbar: {tau_hbar:.4f} (hallucination if ℏₛ ≤ τ)")
        print(f"τ_pri:  {tau_pri:.4f} (hallucination if PRI ≥ τ)")
        print()
    
    # Model 1: ℏₛ alone (INVERTED: lower = more hallucination)
    y_pred_hbar = (hbar_s_scores <= tau_hbar).astype(int)
    precision_hbar = precision_score(labels, y_pred_hbar, zero_division=0)
    recall_hbar = recall_score(labels, y_pred_hbar, zero_division=0)
    cm_hbar = confusion_matrix(labels, y_pred_hbar)
    
    # Model 2: PRI alone (higher = more hallucination)
    y_pred_pri = (pri_scores >= tau_pri).astype(int)
    precision_pri = precision_score(labels, y_pred_pri, zero_division=0)
    recall_pri = recall_score(labels, y_pred_pri, zero_division=0)
    cm_pri = confusion_matrix(labels, y_pred_pri)
    
    if verbose:
        print("ℏₛ alone:")
        print(f"  Precision: {precision_hbar:.4f}")
        print(f"  Recall:    {recall_hbar:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    {cm_hbar[0]} [TN, FP]")
        print(f"    {cm_hbar[1]} [FN, TP]")
        print()
        
        print("PRI alone:")
        print(f"  Precision: {precision_pri:.4f}")
        print(f"  Recall:    {recall_pri:.4f}")
        print(f"  Confusion Matrix:")
        print(f"    {cm_pri[0]} [TN, FP]")
        print(f"    {cm_pri[1]} [FN, TP]")
        print()
    
    return {
        "hbar_alone": {
            "precision": float(precision_hbar),
            "recall": float(recall_hbar),
            "confusion_matrix": cm_hbar.tolist()
        },
        "pri_alone": {
            "precision": float(precision_pri),
            "recall": float(recall_pri),
            "confusion_matrix": cm_pri.tolist()
        }
    }


def evaluate_with_threshold_sweep(
    hbar_s_scores: np.ndarray,
    pri_scores: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Sweep thresholds on test set to find best precision @ recall ≥ 0.9.
    
    This shows if calibrated thresholds are optimal or if test-specific tuning helps.
    
    Args:
        hbar_s_scores: ℏₛ scores on test set
        pri_scores: PRI scores on test set
        labels: Ground truth labels
        verbose: Print results
        
    Returns:
        Dict with best thresholds and metrics
    """
    if verbose:
        print(f"\n{'='*80}")
        print("Threshold Sweep on Test Set (Best @ Recall ≥ 0.9)")
        print(f"{'='*80}\n")
    
    min_recall = 0.9
    
    # ℏₛ alone (INVERTED: negate for sklearn)
    precision_hbar, recall_hbar, thresholds_hbar = precision_recall_curve(
        labels, -hbar_s_scores
    )
    
    valid_idx = recall_hbar >= min_recall
    if valid_idx.any():
        best_prec_hbar = precision_hbar[valid_idx].max()
        best_idx = np.where((recall_hbar >= min_recall) & (precision_hbar == best_prec_hbar))[0][0]
        best_tau_hbar = -thresholds_hbar[best_idx]  # Un-negate
    else:
        best_prec_hbar = 0.0
        best_tau_hbar = hbar_s_scores.min()
    
    # PRI alone
    precision_pri, recall_pri, thresholds_pri = precision_recall_curve(
        labels, pri_scores
    )
    
    valid_idx = recall_pri >= min_recall
    if valid_idx.any():
        best_prec_pri = precision_pri[valid_idx].max()
        best_idx = np.where((recall_pri >= min_recall) & (precision_pri == best_prec_pri))[0][0]
        best_tau_pri = thresholds_pri[best_idx]
    else:
        best_prec_pri = 0.0
        best_tau_pri = pri_scores.max()
    
    if verbose:
        print("ℏₛ alone:")
        print(f"  Best τ on test: {best_tau_hbar:.4f}")
        print(f"  Precision @ Recall≥0.9: {best_prec_hbar:.4f}")
        print()
        
        print("PRI alone:")
        print(f"  Best τ on test: {best_tau_pri:.4f}")
        print(f"  Precision @ Recall≥0.9: {best_prec_pri:.4f}")
        print()
    
    return {
        "hbar_alone": {
            "best_tau_test": float(best_tau_hbar),
            "best_precision_test": float(best_prec_hbar)
        },
        "pri_alone": {
            "best_tau_test": float(best_tau_pri),
            "best_precision_test": float(best_prec_pri)
        }
    }


def compute_auroc_and_prauc(
    hbar_s_scores: np.ndarray,
    pri_scores: np.ndarray,
    labels: np.ndarray,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Compute AUROC and PR-AUC for both signals.
    
    Args:
        hbar_s_scores: ℏₛ scores
        pri_scores: PRI scores
        labels: Ground truth labels
        verbose: Print results
        
    Returns:
        Dict with auroc and pr_auc for each signal
    """
    # ℏₛ (INVERTED: negate for sklearn)
    auroc_hbar = roc_auc_score(labels, -hbar_s_scores)
    pr_auc_hbar = average_precision_score(labels, -hbar_s_scores)
    
    # PRI (normal direction)
    auroc_pri = roc_auc_score(labels, pri_scores)
    pr_auc_pri = average_precision_score(labels, pri_scores)
    
    if verbose:
        print(f"\n{'='*80}")
        print("AUROC and PR-AUC Metrics")
        print(f"{'='*80}\n")
        print("ℏₛ alone:")
        print(f"  AUROC:  {auroc_hbar:.4f}")
        print(f"  PR-AUC: {pr_auc_hbar:.4f}")
        print()
        print("PRI alone:")
        print(f"  AUROC:  {auroc_pri:.4f}")
        print(f"  PR-AUC: {pr_auc_pri:.4f}")
        print()
    
    return {
        "hbar_alone": {
            "auroc": float(auroc_hbar),
            "pr_auc": float(pr_auc_hbar)
        },
        "pri_alone": {
            "auroc": float(auroc_pri),
            "pr_auc": float(pr_auc_pri)
        }
    }


def quadrant_analysis(
    hbar_s_scores: np.ndarray,
    pri_scores: np.ndarray,
    labels: np.ndarray,
    tau_hbar: float,
    tau_pri: float,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Analyze quadrants at calibrated thresholds.
    
    Args:
        hbar_s_scores: ℏₛ scores
        pri_scores: PRI scores
        labels: Ground truth labels
        tau_hbar: ℏₛ threshold
        tau_pri: PRI threshold
        verbose: Print results
        
    Returns:
        Dict with quadrant statistics
    """
    if verbose:
        print(f"\n{'='*80}")
        print("Quadrant Analysis (Calibrated Thresholds)")
        print(f"{'='*80}\n")
    
    # Define quadrants (ℏₛ is INVERTED)
    q1 = (hbar_s_scores > tau_hbar) & (pri_scores < tau_pri)   # Both OK
    q2 = (hbar_s_scores <= tau_hbar) & (pri_scores >= tau_pri) # Both flag
    q3 = (hbar_s_scores > tau_hbar) & (pri_scores >= tau_pri)  # ℏₛ OK, PRI flags
    q4 = (hbar_s_scores <= tau_hbar) & (pri_scores < tau_pri)  # ℏₛ flags, PRI OK
    
    quadrants = {}
    
    for q_idx, q_mask, q_name in [
        (1, q1, "Q1: ℏₛ OK, PRI OK (Low Risk)"),
        (2, q2, "Q2: ℏₛ flags, PRI flags (High Risk)"),
        (3, q3, "Q3: ℏₛ OK, PRI flags (PRI-only detection)"),
        (4, q4, "Q4: ℏₛ flags, PRI OK (ℏₛ-only detection)")
    ]:
        n_total = q_mask.sum()
        n_hal = (labels[q_mask] == 1).sum() if n_total > 0 else 0
        precision_q = n_hal / n_total if n_total > 0 else 0.0
        
        if verbose:
            print(f"{q_name}")
            print(f"  Count: {n_total} ({n_total/len(labels)*100:.1f}%)")
            print(f"  Hallucination rate: {precision_q:.3f}")
            print()
        
        quadrants[f"q{q_idx}"] = {
            "count": int(n_total),
            "hallucination_rate": float(precision_q)
        }
    
    return quadrants


def run_validation(
    model_path: str,
    model_type: str,
    test_data_path: str,
    calibrated_params_path: str,
    max_tokens: int = 20,
    n_samples: Optional[int] = None,
    seed: int = 42,
    output_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Main validation pipeline.
    
    Args:
        model_path: Path to MLX model
        model_type: Model type (llama, qwen, phi3)
        test_data_path: Path to test split JSON
        calibrated_params_path: Path to calibration results JSON
        max_tokens: Max generation length
        n_samples: Optional number of test samples (for quick testing)
        seed: Random seed
        output_path: Optional path to save results JSON
        
    Returns:
        Dict with all validation metrics
    """
    print("=" * 80)
    print("Validation on Test Split")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model: {model_path}...")
    model, tokenizer = load(model_path)
    print(f"✓ Model loaded")
    print()
    
    # Create adapter
    print("Creating adapter...")
    collector = hidden_state_collector.HiddenStateCollector()
    adapter = model_adapters.create_adapter(model, collector, model_type=model_type)
    print(f"✓ Adapter created: {len(adapter.layers)} layers")
    print()
    
    # Load test data
    print(f"Loading test data from {test_data_path}...")
    test_data = halueval_loader.load_split(test_data_path)
    
    # Sample if requested
    if n_samples is not None and n_samples < len(test_data):
        import random
        random.seed(seed)
        test_data = random.sample(test_data, n_samples)
        print(f"Sampled {n_samples} examples for validation (seed={seed})")
    print()
    
    # Load calibrated params
    print(f"Loading calibrated parameters from {calibrated_params_path}...")
    calib_params = load_calibrated_params(calibrated_params_path)
    tau_hbar = calib_params["tau_hbar"]
    tau_pri = calib_params["tau_pri"]
    print(f"✓ Loaded: τ_hbar={tau_hbar:.4f}, τ_pri={tau_pri:.4f}")
    
    # Precompute scores
    hbar_s_scores, pri_scores, labels, scores_by_sample = precompute_test_scores(
        adapter, tokenizer, test_data, max_tokens, verbose=True
    )
    
    # Score statistics
    print(f"\n{'='*80}")
    print("Test Set Score Statistics")
    print(f"{'='*80}\n")
    
    hal_idx = labels == 1
    cor_idx = labels == 0
    
    print(f"ℏₛ (Semantic Uncertainty):")
    print(f"  Hallucinated: mean={hbar_s_scores[hal_idx].mean():.3f}, std={hbar_s_scores[hal_idx].std():.3f}")
    print(f"  Correct:      mean={hbar_s_scores[cor_idx].mean():.3f}, std={hbar_s_scores[cor_idx].std():.3f}")
    print(f"  Mean diff:    {hbar_s_scores[hal_idx].mean() - hbar_s_scores[cor_idx].mean():.3f}")
    
    print(f"\nPRI (Predictive Rupture Index):")
    print(f"  Hallucinated: mean={pri_scores[hal_idx].mean():.3f}, std={pri_scores[hal_idx].std():.3f}")
    print(f"  Correct:      mean={pri_scores[cor_idx].mean():.3f}, std={pri_scores[cor_idx].std():.3f}")
    print(f"  Mean diff:    {pri_scores[hal_idx].mean() - pri_scores[cor_idx].mean():.3f}")
    
    # Compute all metrics
    auroc_prauc = compute_auroc_and_prauc(hbar_s_scores, pri_scores, labels, verbose=True)
    
    calibrated_metrics = evaluate_at_calibrated_thresholds(
        hbar_s_scores, pri_scores, labels, tau_hbar, tau_pri, verbose=True
    )
    
    sweep_metrics = evaluate_with_threshold_sweep(
        hbar_s_scores, pri_scores, labels, verbose=True
    )
    
    quadrants = quadrant_analysis(
        hbar_s_scores, pri_scores, labels, tau_hbar, tau_pri, verbose=True
    )
    
    # Compile results
    results = {
        "model_path": model_path,
        "model_type": model_type,
        "test_data_path": test_data_path,
        "calibrated_params_path": calibrated_params_path,
        "n_test_samples": len(test_data),
        "max_tokens": max_tokens,
        "calibrated_thresholds": {
            "tau_hbar": float(tau_hbar),
            "tau_pri": float(tau_pri)
        },
        "auroc_and_prauc": auroc_prauc,
        "performance_at_calibrated_thresholds": calibrated_metrics,
        "best_on_test_sweep": sweep_metrics,
        "quadrant_analysis": quadrants,
        "scores_by_sample": scores_by_sample  # NEW: Per-sample scores for figures
    }
    
    # Save if requested
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n✓ Results saved to {output_path}")
    
    print()
    print("=" * 80)
    print("✓ Validation Complete!")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate hallucination detection on test set")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit", help="Model path")
    parser.add_argument("--model-type", default="llama", choices=["llama", "qwen", "phi3"], help="Model type")
    parser.add_argument("--test-data", default="./data/halueval/splits/test.json", help="Test data path")
    parser.add_argument("--calibrated-params", default="./calibrated_params/llama_3.2_3b_20260119_213225_n200.json", 
                       help="Path to calibration results JSON")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max generation length")
    parser.add_argument("--n-samples", type=int, default=None, help="Number of test samples (None = all)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output", default="./results/validation_results.json", help="Output path for results")
    
    args = parser.parse_args()
    
    run_validation(
        model_path=args.model,
        model_type=args.model_type,
        test_data_path=args.test_data,
        calibrated_params_path=args.calibrated_params,
        max_tokens=args.max_tokens,
        n_samples=args.n_samples,
        seed=args.seed,
        output_path=args.output
    )