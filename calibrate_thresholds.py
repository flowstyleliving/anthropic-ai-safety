"""
Threshold calibration for (τ, λ) parameters.

Performs grid search on training data to find optimal parameters
that maximize AUROC for hallucination detection.
"""

import json
import os
from typing import Dict, List, Tuple, Any, Optional
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm
import mlx.core as mx
from mlx_lm import load

import model_adapters
import hidden_state_collector
import monitoring_loop
import halueval_loader


class ThresholdCalibrator:
    """
    Grid search calibrator for (τ, λ) parameters.
    
    Evaluates parameter combinations on training data to find optimal
    thresholds for hallucination detection.
    """
    
    def __init__(
        self,
        adapter: model_adapters.ModelAdapter,
        tokenizer: Any,
        train_data: List[Dict[str, Any]],
        max_tokens: int = 50
    ):
        """
        Initialize calibrator.
        
        Args:
            adapter: ModelAdapter instance
            tokenizer: Tokenizer for encoding/decoding
            train_data: List of training samples
            max_tokens: Maximum generation length per sample
        """
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.train_data = train_data
        self.max_tokens = max_tokens
    
    def precompute_scores(
        self,
        verbose: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pass 1: Run generation once per sample and compute ℏₛ scores.
        
        This is the expensive pass that runs the model. We do it once and store
        only the scalar scores, avoiding repeated generation for each τ value.
        
        Args:
            verbose: Print progress
            
        Returns:
            Tuple of (scores, labels) as numpy arrays
        """
        # Create monitor with halting DISABLED (pfail_cutoff=1.1 > 1.0)
        monitor = monitoring_loop.HallucinationMonitor(
            adapter=self.adapter,
            tokenizer=self.tokenizer,
            tau=0.0,  # Not used in score-only mode
            lambda_=1.0,  # Not used in score-only mode
            pfail_cutoff=1.1,  # Disable halting (>1.0)
            max_tokens=self.max_tokens,
            temperature=0.0  # Greedy for consistency
        )
        
        scores = []
        labels = []
        
        if verbose:
            print(f"Pass 1: Computing ℏₛ scores for {len(self.train_data)} samples...")
            print()
        
        import gc
        for i, sample in enumerate(tqdm(self.train_data, desc="Computing scores", disable=not verbose)):
            try:
                # Use lightweight score-only mode (no trajectory storage)
                result = monitor.generate_with_monitoring(
                    sample["prompt"],
                    verbose=False,
                    compute_score_only=True
                )
                
                scores.append(result["score"])
                labels.append(sample["label"])
                
                # Cleanup
                del result
                
                # Periodic garbage collection
                if (i + 1) % 50 == 0:
                    gc.collect()
            
            except Exception as e:
                if verbose:
                    print(f"Error on sample {sample.get('id', i)}: {e}")
                # On error, assume no hallucination detected
                scores.append(0.0)
                labels.append(sample["label"])
        
        scores = np.array(scores)
        labels = np.array(labels)
        
        if verbose:
            print()
            print(f"✓ Computed {len(scores)} scores")
            print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")
            print(f"  Hallucination rate: {labels.mean():.2%}")
            print()
            
            # Sanity check: Test median threshold with inverted direction
            median_threshold = np.median(scores)
            y_pred_median = (scores <= median_threshold).astype(int)
            precision_median = precision_score(labels, y_pred_median, zero_division=0)
            recall_median = recall_score(labels, y_pred_median, zero_division=0)
            print("Sanity Check (score ≤ median as hallucination classifier):")
            print(f"  Median threshold: {median_threshold:.3f}")
            print(f"  Precision: {precision_median:.3f}")
            print(f"  Recall: {recall_median:.3f}")
            if precision_median > 0.52:
                print(f"  ✓ Direction fix confirmed (precision > 0.52)")
            else:
                print(f"  ⚠ Direction may need review (precision ≤ 0.52)")
            print()
        
        return scores, labels
    
    def calibrate_tau(
        self,
        tau_range: Optional[Tuple[float, float, float]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Mode A calibration: two-pass approach for efficiency.
        
        Pass 1: Run generation once per sample, store ℏₛ scores (expensive)
        Pass 2: Sweep τ over stored scores to find optimal threshold (cheap)
        
        This eliminates redundant model execution and scales well to large
        sample sizes and fine-grained τ sweeps.
        
        Args:
            tau_range: Optional (start, stop, step) for tau values.
                      If None, uses data-driven quantile-based range.
            verbose: Print progress
            
        Returns:
            Dict with best_tau and all results
        """
        if verbose:
            print("=" * 80)
            print("Mode A Calibration: Two-Pass Approach")
            print("=" * 80)
            print()
        
        # Pass 1: Precompute scores (expensive, done once)
        scores, labels = self.precompute_scores(verbose=verbose)
        
        # Diagnostic: Score statistics by label
        if verbose:
            hal_scores = scores[labels == 1]
            cor_scores = scores[labels == 0]
            
            print("Score Statistics by Label:")
            print(f"  Hallucinated (label=1): mean={hal_scores.mean():.3f}, std={hal_scores.std():.3f}, n={len(hal_scores)}")
            print(f"  Correct (label=0):      mean={cor_scores.mean():.3f}, std={cor_scores.std():.3f}, n={len(cor_scores)}")
            print(f"  Mean difference:        {hal_scores.mean() - cor_scores.mean():.3f}")
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((hal_scores.std()**2 + cor_scores.std()**2) / 2)
            cohens_d = (hal_scores.mean() - cor_scores.mean()) / pooled_std if pooled_std > 0 else 0.0
            print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
            print()
            
            # Quadrant logging: Inspect extreme examples
            print("Quadrant Analysis (5 examples each):")
            print()
            
            # Get indices for each quadrant
            hal_idx = np.where(labels == 1)[0]
            cor_idx = np.where(labels == 0)[0]
            
            # Sort by score
            hal_sorted = hal_idx[np.argsort(scores[hal_idx])]
            cor_sorted = cor_idx[np.argsort(scores[cor_idx])]
            
            # Hallucinated with lowest scores (most confident hallucinations)
            print("1. Hallucinated with LOWEST ℏₛ (confident hallucinations):")
            for idx in hal_sorted[:5]:
                sample = self.train_data[idx]
                print(f"   ID: {sample['id']}, score={scores[idx]:.3f}")
                print(f"   Prompt: {sample['prompt'][:150]}...")
                print()
            
            # Hallucinated with highest scores (uncertain hallucinations)
            print("2. Hallucinated with HIGHEST ℏₛ (uncertain hallucinations):")
            for idx in hal_sorted[-5:]:
                sample = self.train_data[idx]
                print(f"   ID: {sample['id']}, score={scores[idx]:.3f}")
                print(f"   Prompt: {sample['prompt'][:150]}...")
                print()
            
            # Correct with lowest scores (confident correct)
            print("3. Correct with LOWEST ℏₛ (confident correct):")
            for idx in cor_sorted[:5]:
                sample = self.train_data[idx]
                print(f"   ID: {sample['id']}, score={scores[idx]:.3f}")
                print(f"   Prompt: {sample['prompt'][:150]}...")
                print()
            
            # Correct with highest scores (uncertain correct)
            print("4. Correct with HIGHEST ℏₛ (uncertain correct):")
            for idx in cor_sorted[-5:]:
                sample = self.train_data[idx]
                print(f"   ID: {sample['id']}, score={scores[idx]:.3f}")
                print(f"   Prompt: {sample['prompt'][:150]}...")
                print()
        
        # Pass 2: Sweep τ over precomputed scores (cheap)
        if tau_range is None:
            # Data-driven: use quantiles of score distribution (FULL range)
            tau_values = np.quantile(scores, np.linspace(0.01, 0.99, 99))
            # Ensure ascending order for monotonic recall increase (easier debugging)
            tau_values = np.sort(tau_values)
            if verbose:
                print("Pass 2: Using data-driven τ sweep (quantile-based, full range)")
                print(f"  τ range: [{tau_values.min():.3f}, {tau_values.max():.3f}]")
                print(f"  {len(tau_values)} values to evaluate")
        else:
            tau_start, tau_stop, tau_step = tau_range
            tau_values = np.arange(tau_start, tau_stop + tau_step, tau_step)
            # Ensure ascending order for monotonic recall increase (easier debugging)
            tau_values = np.sort(tau_values)
            if verbose:
                print("Pass 2: Using specified τ sweep")
                print(f"  τ range: [{tau_start}, {tau_stop}] step {tau_step}")
                print(f"  {len(tau_values)} values to evaluate")
        
        if verbose:
            print()
        
        # Sweep τ to find optimal threshold
        best_precision = -1.0
        best_tau = None
        all_results = []
        min_recall_target = 0.9  # Safety-oriented: catch ≥90% of hallucinations
        
        # Compute AUROC once (independent of τ)
        try:
            auroc = roc_auc_score(labels, scores) if len(np.unique(labels)) >= 2 else 0.0
        except:
            auroc = 0.0
        
        # Compute PR-AUC once (independent of τ)
        try:
            from sklearn.metrics import average_precision_score
            pr_auc = average_precision_score(labels, scores) if len(np.unique(labels)) >= 2 else 0.0
        except:
            pr_auc = 0.0
        
        if verbose:
            print(f"Overall AUROC: {auroc:.4f}")
            print(f"Overall PR-AUC: {pr_auc:.4f}")
            print()
            print("Sweeping τ to maximize precision @ recall ≥ {:.0%}:".format(min_recall_target))
            print()
        
        for tau in tau_values:
            # Hallucination = 1 when uncertainty score is LOW (confident hallucination)
            # INVERTED: Lower scores indicate MORE uncertainty/hallucination
            # (Hallucinations can be "confidently wrong" → sharper distributions → lower ℏₛ)
            y_pred = (scores <= tau).astype(int)
            
            # Compute threshold-dependent metrics
            try:
                precision = precision_score(labels, y_pred, zero_division=0)
                recall = recall_score(labels, y_pred, zero_division=0)
                f1 = f1_score(labels, y_pred, zero_division=0)
            except:
                precision = recall = f1 = 0.0
            
            result = {
                "tau": float(tau),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1)
            }
            all_results.append(result)
            
            # Selection criterion: maximize precision subject to recall ≥ min_recall_target
            if recall >= min_recall_target:
                if precision > best_precision:
                    best_precision = precision
                    best_tau = float(tau)
            
            if verbose:
                marker = " ←" if tau == best_tau else ""
                print(f"  τ={tau:.3f}: Prec={precision:.3f}, Rec={recall:.3f}, F1={f1:.3f}{marker}")
        
        if verbose:
            print()
        
        # Fallback if no τ meets recall target
        if best_tau is None:
            if verbose:
                print(f"⚠ Warning: No τ achieved recall ≥ {min_recall_target}")
                print("Falling back to best F1 score...")
            best_f1 = max(all_results, key=lambda x: x["f1"])
            best_tau = best_f1["tau"]
            best_precision = best_f1["precision"]
            if verbose:
                print(f"Fallback τ: {best_tau:.3f} (F1={best_f1['f1']:.3f})")
        else:
            if verbose:
                print(f"✓ Best τ: {best_tau:.3f}")
                print(f"  Precision: {best_precision:.3f} (at recall ≥ {min_recall_target})")
        
        if verbose:
            print()
        
        return {
            "best_tau": best_tau,
            "best_precision": best_precision,
            "min_recall_target": min_recall_target,
            "auroc": float(auroc),
            "pr_auc": float(pr_auc),
            "all_results": all_results,
            "mode": "A"  # Mode A: direct ℏₛ thresholding
        }
    
    def save_params(self, result: Dict[str, Any], output_path: str) -> None:
        """
        Save calibrated parameters to JSON.
        
        Args:
            result: Output from grid_search()
            output_path: Path to save JSON file
        """
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        print(f"Saved calibrated parameters to {output_path}")


def load_calibrated_params(model_name: str, params_dir: str = "./calibrated_params") -> Dict[str, float]:
    """
    Load calibrated (τ, λ) from JSON.
    
    Args:
        model_name: Model identifier (e.g., "llama_3.2_3b")
        params_dir: Directory containing calibration files
        
    Returns:
        Dict with tau and lambda values
    """
    params_path = Path(params_dir) / f"{model_name}.json"
    
    if not params_path.exists():
        raise FileNotFoundError(f"Calibration file not found: {params_path}")
    
    with open(params_path, 'r') as f:
        data = json.load(f)
    
    return data["best_params"]


if __name__ == "__main__":
    """
    Example: Calibrate Llama 3.2 3B on training data.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Calibrate hallucination detection thresholds")
    parser.add_argument("--model", default="mlx-community/Llama-3.2-3B-Instruct-4bit", help="Model path")
    parser.add_argument("--model-name", default="llama_3.2_3b", help="Model name for saving")
    parser.add_argument("--model-type", default="llama", choices=["llama", "qwen", "phi3"], help="Model type")
    parser.add_argument("--train-data", default="./data/halueval/splits/train.json", help="Training data path")
    parser.add_argument("--tau-start", type=float, default=None, help="Optional: Tau start value (if None, uses data-driven quantiles)")
    parser.add_argument("--tau-stop", type=float, default=None, help="Optional: Tau stop value")
    parser.add_argument("--tau-step", type=float, default=0.1, help="Tau step size (only used if tau-start/stop specified)")
    parser.add_argument("--max-tokens", type=int, default=20, help="Max generation length (lower for calibration efficiency)")
    parser.add_argument("--n-samples", type=int, default=200, help="Number of training samples")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--output-dir", default="./calibrated_params", help="Output directory")
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Threshold Calibration")
    print("=" * 80)
    print()
    
    # Load model
    print(f"Loading model: {args.model}...")
    model, tokenizer = load(args.model)
    print(f"✓ Model loaded")
    print()
    
    # Create adapter
    print("Creating adapter...")
    collector = hidden_state_collector.HiddenStateCollector()
    adapter = model_adapters.create_adapter(model, collector, model_type=args.model_type)
    print(f"✓ Adapter created: {len(adapter.layers)} layers")
    print()
    
    # Load training data
    print(f"Loading training data from {args.train_data}...")
    train_data = halueval_loader.load_split(args.train_data)
    
    # Set random seed for reproducibility
    import random
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)
    
    # Sample for quick testing
    if args.n_samples < len(train_data):
        train_data = random.sample(train_data, args.n_samples)
        print(f"Sampled {args.n_samples} examples for calibration (seed={args.seed})")
    print()
    
    # Create calibrator
    calibrator = ThresholdCalibrator(
        adapter=adapter,
        tokenizer=tokenizer,
        train_data=train_data,
        max_tokens=args.max_tokens
    )
    
    # Determine τ range (data-driven or manual)
    if args.tau_start is not None and args.tau_stop is not None:
        tau_range = (args.tau_start, args.tau_stop, args.tau_step)
        print(f"Using manual τ range: [{args.tau_start}, {args.tau_stop}] step {args.tau_step}")
    else:
        tau_range = None  # Will use data-driven quantiles
        print("Using data-driven τ range (quantile-based)")
    
    # Run Mode A calibration (τ sweep on ℏₛ)
    print("Starting Mode A calibration...")
    print()
    result = calibrator.calibrate_tau(
        tau_range=tau_range,
        verbose=True
    )
    
    # Save results with timestamp for unique files per run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{args.model_name}_{timestamp}_n{args.n_samples}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    calibrator.save_params(result, output_path)
    
    print()
    print("✓ Calibration complete!")