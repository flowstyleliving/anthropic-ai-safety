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
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Pass 1: Run generation once per sample and compute ℏₛ AND PRI scores.
        
        This is the expensive pass that runs the model. We do it once and store
        both scalar scores, avoiding repeated generation.
        
        Args:
            verbose: Print progress
            
        Returns:
            Tuple of (hbar_s_scores, pri_scores, labels) as numpy arrays
        """
        # Create monitor with halting DISABLED (pfail_cutoff=1.1 > 1.0)
        monitor = monitoring_loop.HallucinationMonitor(
            adapter=self.adapter,
            tokenizer=self.tokenizer,
            tau=0.0,  # Not used in score-only mode
            lambda_=1.0,  # Not used in score-only mode
            pfail_cutoff=1.1,  # Disable halting (>1.0)
            max_tokens=self.max_tokens,
            temperature=0.0,  # Greedy for consistency
            alpha_pri=0.1,  # PRI parameter
            compute_pri=True  # Enable PRI computation
        )
        
        hbar_s_scores = []
        pri_scores = []
        labels = []
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Pass 1: Precomputing Scores ({len(self.train_data)} samples)")
            print(f"{'='*80}\n")
        
        import gc
        for i, sample in enumerate(tqdm(self.train_data, desc="Generating", disable=not verbose)):
            try:
                # Use lightweight score-only mode (no trajectory storage)
                result = monitor.generate_with_monitoring(
                    sample["prompt"],
                    verbose=False,
                    compute_score_only=True
                )
                
                hbar_s_scores.append(result["hbar_s_score"])
                pri_scores.append(result["pri_score"])
                labels.append(sample["label"])
                
                # Cleanup
                del result
                
                # Periodic garbage collection
                if (i + 1) % 50 == 0:
                    gc.collect()
                    if verbose:
                        print(f"  [{i+1}/{len(self.train_data)}] Memory cleanup")
            
            except Exception as e:
                if verbose:
                    print(f"Error on sample {sample.get('id', i)}: {e}")
                # On error, use default values
                hbar_s_scores.append(0.0)
                pri_scores.append(0.0)
                labels.append(sample["label"])
        
        hbar_s_scores = np.array(hbar_s_scores)
        pri_scores = np.array(pri_scores)
        labels = np.array(labels)
        
        if verbose:
            print(f"\n{'='*80}")
            print("Score Statistics by Label")
            print(f"{'='*80}\n")
            
            hal_idx = labels == 1
            cor_idx = labels == 0
            
            # ℏₛ statistics
            print(f"ℏₛ (Semantic Uncertainty):")
            print(f"  Hallucinated (label=1): mean={hbar_s_scores[hal_idx].mean():.3f}, "
                  f"std={hbar_s_scores[hal_idx].std():.3f}, n={hal_idx.sum()}")
            print(f"  Correct (label=0):      mean={hbar_s_scores[cor_idx].mean():.3f}, "
                  f"std={hbar_s_scores[cor_idx].std():.3f}, n={cor_idx.sum()}")
            
            mean_diff = hbar_s_scores[hal_idx].mean() - hbar_s_scores[cor_idx].mean()
            pooled_std = np.sqrt((hbar_s_scores[hal_idx].std()**2 + hbar_s_scores[cor_idx].std()**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0
            
            print(f"  Mean difference: {mean_diff:.3f}")
            print(f"  Effect size (Cohen's d): {cohens_d:.3f}")
            
            # PRI statistics
            print(f"\nPRI (Predictive Rupture Index):")
            print(f"  Hallucinated (label=1): mean={pri_scores[hal_idx].mean():.3f}, "
                  f"std={pri_scores[hal_idx].std():.3f}")
            print(f"  Correct (label=0):      mean={pri_scores[cor_idx].mean():.3f}, "
                  f"std={pri_scores[cor_idx].std():.3f}")
            
            mean_diff_pri = pri_scores[hal_idx].mean() - pri_scores[cor_idx].mean()
            pooled_std_pri = np.sqrt((pri_scores[hal_idx].std()**2 + pri_scores[cor_idx].std()**2) / 2)
            cohens_d_pri = mean_diff_pri / pooled_std_pri if pooled_std_pri > 0 else 0.0
            
            print(f"  Mean difference: {mean_diff_pri:.3f}")
            print(f"  Effect size (Cohen's d): {cohens_d_pri:.3f}")
            
            # Signal correlation
            correlation = np.corrcoef(hbar_s_scores, pri_scores)[0, 1]
            print(f"\nSignal Correlation (ℏₛ vs PRI): {correlation:.3f}")
            
            if abs(correlation) < 0.3:
                print("  ✓ Signals are weakly correlated (good: orthogonal information)")
            elif abs(correlation) < 0.6:
                print("  ~ Signals are moderately correlated (some overlap)")
            else:
                print("  ⚠ Signals are highly correlated (may be redundant)")
            
            print(f"\n{'='*80}\n")
        
        return hbar_s_scores, pri_scores, labels
    
    def calibrate_joint(
        self,
        hbar_s_scores: np.ndarray,
        pri_scores: np.ndarray,
        labels: np.ndarray,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Pass 2: Calibrate joint (ℏₛ, PRI) model with 5-fold CV.
        
        Evaluates:
        1. ℏₛ alone
        2. PRI alone
        3. Joint logistic regression (with 5-fold CV)
        
        Args:
            hbar_s_scores: ℏₛ scores array
            pri_scores: PRI scores array
            labels: Ground truth labels
            verbose: Print progress
            
        Returns:
            Dict with best model and quadrant analysis
        """
        from sklearn.linear_model import LogisticRegressionCV
        from sklearn.metrics import precision_recall_curve, average_precision_score
        from sklearn.model_selection import StratifiedKFold
        
        if verbose:
            print(f"{'='*80}")
            print("Pass 2: Joint Calibration")
            print(f"{'='*80}\n")
        
        # Model 1: ℏₛ alone (inverted: lower ℏₛ = higher risk)
        if verbose:
            print("Model 1: ℏₛ alone")
            print("-" * 40)
        
        # AUROC (negate for sklearn: higher score = positive class)
        auroc_hbar = roc_auc_score(labels, -hbar_s_scores)
        
        # Precision-recall curve
        precision_hbar, recall_hbar, thresholds_hbar = precision_recall_curve(
            labels, -hbar_s_scores
        )
        
        # Find best threshold at recall ≥ 0.9
        valid_idx = recall_hbar >= 0.9
        if valid_idx.any():
            best_prec_hbar = precision_hbar[valid_idx].max()
            best_idx = np.where((recall_hbar >= 0.9) & (precision_hbar == best_prec_hbar))[0][0]
            tau_hbar = -thresholds_hbar[best_idx]  # Un-negate
        else:
            best_prec_hbar = 0.0
            tau_hbar = hbar_s_scores.min()
        
        if verbose:
            print(f"  AUROC: {auroc_hbar:.4f}")
            print(f"  Best τ: {tau_hbar:.4f} (classify as hallucination if ℏₛ ≤ τ)")
            print(f"  Precision @ Recall≥0.9: {best_prec_hbar:.4f}\n")
        
        # Model 2: PRI alone (higher PRI = higher risk)
        if verbose:
            print("Model 2: PRI alone")
            print("-" * 40)
        
        auroc_pri = roc_auc_score(labels, pri_scores)
        
        precision_pri, recall_pri, thresholds_pri = precision_recall_curve(labels, pri_scores)
        
        valid_idx = recall_pri >= 0.9
        if valid_idx.any():
            best_prec_pri = precision_pri[valid_idx].max()
            best_idx = np.where((recall_pri >= 0.9) & (precision_pri == best_prec_pri))[0][0]
            tau_pri = thresholds_pri[best_idx]
        else:
            best_prec_pri = 0.0
            tau_pri = pri_scores.max()
        
        if verbose:
            print(f"  AUROC: {auroc_pri:.4f}")
            print(f"  Best τ: {tau_pri:.4f} (classify as hallucination if PRI ≥ τ)")
            print(f"  Precision @ Recall≥0.9: {best_prec_pri:.4f}\n")
        
        # Model 3: Joint with 5-fold cross-validation
        if verbose:
            print("Model 3: Joint (ℏₛ, PRI) with 5-Fold CV")
            print("-" * 40)
        
        X = np.column_stack([hbar_s_scores, pri_scores])
        
        # Logistic regression with built-in CV
        clf = LogisticRegressionCV(
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            random_state=42,
            max_iter=1000,
            scoring='roc_auc'
        )
        clf.fit(X, labels)
        
        # Get probabilities (CV predictions)
        joint_probs = clf.predict_proba(X)[:, 1]
        auroc_joint = roc_auc_score(labels, joint_probs)
        
        precision_joint, recall_joint, _ = precision_recall_curve(labels, joint_probs)
        valid_idx = recall_joint >= 0.9
        if valid_idx.any():
            best_prec_joint = precision_joint[valid_idx].max()
        else:
            best_prec_joint = 0.0
        
        if verbose:
            print(f"  AUROC (CV): {auroc_joint:.4f}")
            print(f"  Weights: w_hbar={clf.coef_[0][0]:.3f}, w_pri={clf.coef_[0][1]:.3f}, "
                  f"intercept={clf.intercept_[0]:.3f}")
            print(f"  Precision @ Recall≥0.9: {best_prec_joint:.4f}\n")
        
        # Comparison
        if verbose:
            print("=" * 80)
            print("Model Comparison")
            print("=" * 80)
            print(f"  AUROC: ℏₛ={auroc_hbar:.4f}, PRI={auroc_pri:.4f}, Joint={auroc_joint:.4f}")
            print(f"  Best model: ", end="")
            if auroc_joint > max(auroc_hbar, auroc_pri):
                print("Joint (signals are complementary)")
            elif auroc_pri > auroc_hbar:
                print("PRI alone")
            else:
                print("ℏₛ alone")
            print()
        
        # Quadrant analysis using CALIBRATED thresholds (not median)
        if verbose:
            print("=" * 80)
            print("Quadrant Analysis (using calibrated thresholds)")
            print("=" * 80)
            print(f"  τ_hbar = {tau_hbar:.3f} (classify as hallucination if ℏₛ ≤ τ)")
            print(f"  τ_pri = {tau_pri:.3f} (classify as hallucination if PRI ≥ τ)")
            print()
        
        # Define quadrants using calibrated thresholds
        q1 = (hbar_s_scores > tau_hbar) & (pri_scores < tau_pri)   # Low risk both
        q2 = (hbar_s_scores <= tau_hbar) & (pri_scores >= tau_pri) # High risk both
        q3 = (hbar_s_scores > tau_hbar) & (pri_scores >= tau_pri)  # ℏₛ OK, PRI flags
        q4 = (hbar_s_scores <= tau_hbar) & (pri_scores < tau_pri)  # ℏₛ flags, PRI OK
        
        if verbose:
            for q_idx, q_mask, q_name in [
                (1, q1, "Q1: ℏₛ OK, PRI OK (Low Risk)"),
                (2, q2, "Q2: ℏₛ flags, PRI flags (High Risk)"),
                (3, q3, "Q3: ℏₛ OK, PRI flags (PRI-only detection)"),
                (4, q4, "Q4: ℏₛ flags, PRI OK (ℏₛ-only detection)")
            ]:
                n_total = q_mask.sum()
                n_hal = (labels[q_mask] == 1).sum() if n_total > 0 else 0
                precision_q = n_hal / n_total if n_total > 0 else 0.0
                
                print(f"{q_name}")
                print(f"  Count: {n_total} ({n_total/len(labels)*100:.1f}%)")
                print(f"  Hallucination rate: {precision_q:.3f}")
                print()
        
        return {
            'auroc_hbar': float(auroc_hbar),
            'auroc_pri': float(auroc_pri),
            'auroc_joint': float(auroc_joint),
            'tau_hbar': float(tau_hbar),
            'tau_pri': float(tau_pri),
            'joint_model_weights': {
                'w_hbar': float(clf.coef_[0][0]),
                'w_pri': float(clf.coef_[0][1]),
                'intercept': float(clf.intercept_[0])
            },
            'best_precision_hbar': float(best_prec_hbar),
            'best_precision_pri': float(best_prec_pri),
            'best_precision_joint': float(best_prec_joint)
        }
    
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
    
    # NEW: Two-pass approach with dual signals
    print("=" * 80)
    print("Joint (ℏₛ + PRI) Calibration")
    print("=" * 80)
    print()
    
    # Pass 1: Precompute both ℏₛ and PRI scores
    hbar_s_scores, pri_scores, labels = calibrator.precompute_scores(verbose=True)
    
    # Pass 2: Joint calibration with 5-fold CV
    result = calibrator.calibrate_joint(
        hbar_s_scores=hbar_s_scores,
        pri_scores=pri_scores,
        labels=labels,
        verbose=True
    )
    
    # Save results with timestamp for unique files per run
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_filename = f"{args.model_name}_{timestamp}_n{args.n_samples}.json"
    output_path = os.path.join(args.output_dir, output_filename)
    calibrator.save_params(result, output_path)
    
    print()
    print("=" * 80)
    print("✓ Calibration Complete!")
    print("=" * 80)
    print(f"\nResults saved to: {output_path}")
    print(f"\nKey findings:")
    print(f"  - AUROC (ℏₛ alone):  {result['auroc_hbar']:.4f}")
    print(f"  - AUROC (PRI alone): {result['auroc_pri']:.4f}")
    print(f"  - AUROC (Joint):     {result['auroc_joint']:.4f}")
    print(f"  - τ_hbar: {result['tau_hbar']:.4f}")
    print(f"  - τ_pri:  {result['tau_pri']:.4f}")
    print()