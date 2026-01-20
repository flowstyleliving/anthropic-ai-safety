# Phase 2: PRI (Predictive Rupture Index) Implementation Plan

## Overview

This document specifies the implementation of PRI (Predictive Rupture Index), a Free Energy Principle-inspired signal to complement semantic uncertainty (ℏₛ) in hallucination detection.

**Status**: Ready for implementation  
**Target**: Improve AUROC from ~0.56 to >0.65 via joint (ℏₛ, PRI) modeling  
**Key Insight**: PRI detects confident hallucinations via representational ruptures that ℏₛ misses

---

## 1. Core Concept

### What PRI Measures

**PRI = Predictive Rupture Index** tracks dynamic internal inconsistency during generation:

> Did the model make a confident prediction that required a sudden internal representational shift?

This is the signature of confident hallucinations.

### Relationship to ℏₛ

| Signal | What It Measures | Failure Mode It Catches |
|--------|------------------|-------------------------|
| **ℏₛ** | Static epistemic uncertainty (layer disagreement + entropy) | Uncertain/exploratory states |
| **PRI** | Dynamic trajectory rupture (surprise × hidden-state jump) | Confident representational commits |

**Orthogonality hypothesis**: Low ℏₛ + High PRI = confident hallucination signature

---

## 2. Mathematical Definition

### Per-Token PRI

At generation step t, after selecting token x_t:


Surprise:    S_t = -log p(x_t | x_<t)
Hidden jump: Δh_t = 1 - cos(normalize(h_t^L), normalize(h_{t-1}^L))
PRI:         PRI_t = S_t × (1 + α × Δh_t)


**Parameters**:
- `h_t^L`: Final-layer last-token hidden state at step t
- `α ∈ [0.05, 0.2]`: Scaling factor for hidden-state jump (default 0.1, calibrate on HaluEval)

**Special case (first token)**:
- Set `Δh_0 = 0` (no prior state)
- `PRI_0 = S_0` (surprise only)
- **Rationale**: Don't discard the first commit moment entirely

### Sample-Level Score

Aggregate per-token PRI values using **spike-sensitive** statistic:


PRI_score = mean(top_k(PRI_t))  where k=5


**Alternatives** (for experimentation):
- `p95(PRI_t)` - percentile-based
- `max(PRI_t)` - most sensitive to single rupture

**NOT recommended**: `mean(PRI_t)` - too smooth, misses localized ruptures

---

## 3. Design Rationale

### Why Cosine Distance (Not L2)

**Choice**: `Δh_t = 1 - cos(v1, v2)` on unit-normalized vectors

**Justification**:
1. **Scale invariance**: Hidden state magnitudes vary across layers; normalization isolates directional change
2. **Bounded range**: Δh ∈ [0, 2] (0=aligned, 1=orthogonal, 2=opposite)
3. **Consistency**: Matches Δσ computation (normalized hidden-state dispersion)

### Why Multiplicative Form (Not Additive)

**Choice**: `PRI = surprise × (1 + α·Δh)` not `surprise + α·Δh`

**Justification**:
- **Surprise as gate**: High Δh is acceptable if token was expected (S low)
- **Rupture amplification**: Unexpected token + large jump = multiplicative risk
- **Interpretability**: PRI ≈ surprise when trajectory is stable (Δh ≈ 0)

### Why This Catches Confident Hallucinations

**Observed failure mode** (from HaluEval calibration):
- Hallucinated examples have **lower** ℏₛ than correct (mean: 1.93 vs 2.04)
- Model is confident (peaked logits) AND layers agree (low dispersion)
- **ℏₛ fails to flag these**

**PRI mechanism**:
1. Model generates fabricated fact (e.g., "Sydney" as Australia's capital)
2. Logits are peaked → S_t is low (confident)
3. BUT hidden state must **jump** to encode the false claim
4. Prior context encoded "Australia + capital query"
5. New state must encode "Sydney = capital" → representational phase transition
6. Result: High Δh_t → PRI elevated despite low surprise

**Critical note**: This assumes the jump is detectable. If the model smoothly continues a false narrative, PRI may also be low. **Treat PRI v1 as a "rupture detector," not a guaranteed "confident-hallucination detector."**

---

## 4. Implementation Specification

### Critical Fixes (From Review)

#### Fix 1: Computation Frequency
**Problem**: Cannot compute PRI every token without hidden state every token  
**Solution**: Compute PRI at existing checkpoint frequency (`check_every_k_tokens=1` during calibration)  
**Rationale**: Hidden states are already pulled for ℏₛ—reuse the same timing

#### Fix 2: API Design
**Problem**: Original design re-computed softmax and didn't import math  
**Solution**: Compute surprise once in monitoring loop, pass to PRI function  
**Clean API**: `compute_pri(surprise, delta_h, alpha)` - minimal, no redundant ops

#### Fix 3: First Token Handling
**Problem**: Setting PRI_0 = 0 discards first commit moment  
**Solution**: `PRI_0 = S_0` (surprise only, since Δh_0 = 0)  
**Rationale**: Keeps PRI meaningful from token 1 onward

---

### Phase 1: Extend `uncertainty_metrics.py`

Add four new functions for PRI computation:

```python
import math  # NEW: Import for log computation

def normalize_vector(v: mx.array, epsilon: float = 1e-8) -> mx.array:
    """
    Normalize vector to unit norm.
    
    Args:
        v: MLX array of shape [dim]
        epsilon: Small constant to prevent division by zero
        
    Returns:
        Normalized vector of shape [dim]
    """
    norm = mx.sqrt(mx.sum(v * v) + epsilon)
    return v / norm


def compute_cosine_distance(v1: mx.array, v2: mx.array, epsilon: float = 1e-8) -> float:
    """
    Compute cosine distance: 1 - cos(v1, v2).
    
    Distance is bounded in [0, 2]:
    - 0 = vectors are identical
    - 1 = vectors are orthogonal
    - 2 = vectors are opposite
    
    Args:
        v1, v2: MLX arrays of shape [dim]
        epsilon: Numerical stability constant
        
    Returns:
        Cosine distance in [0, 2]
    """
    # Normalize both vectors
    v1_norm = normalize_vector(v1, epsilon)
    v2_norm = normalize_vector(v2, epsilon)
    
    # Cosine similarity
    cos_sim = float(mx.sum(v1_norm * v2_norm).item())
    
    # Clamp to [-1, 1] to handle numerical errors
    cos_sim = max(-1.0, min(1.0, cos_sim))
    
    # Cosine distance
    cos_dist = 1.0 - cos_sim
    
    return cos_dist


def compute_surprise(probs: mx.array, selected_token: int) -> float:
    """
    Compute token surprise: -log p(x_t | x_<t).
    
    Args:
        probs: Probability distribution over vocabulary (already softmaxed)
        selected_token: Token ID that was generated
        
    Returns:
        Surprise in nats (natural log)
    """
    selected_prob = float(probs[selected_token].item())
    # Clamp to avoid log(0)
    selected_prob = max(1e-10, min(1.0, selected_prob))
    surprise = -math.log(selected_prob)
    return surprise


def compute_pri(surprise: float, delta_h: float, alpha: float = 0.1) -> float:
    """
    Compute Predictive Rupture Index: PRI = S_t × (1 + α × Δh_t).
    
    Combines token surprise with hidden-state trajectory jump to detect
    confident predictions that require internal representational shifts.
    
    Args:
        surprise: -log p(x_t | x_<t), in nats
        delta_h: Cosine distance between consecutive final-layer hidden states
        alpha: Scaling factor for hidden-state jump (default 0.1)
        
    Returns:
        PRI scalar, typically in [0, 20] for natural text
        
    Interpretation:
        - Low PRI: Expected token + stable trajectory → trust
        - High PRI: Unexpected token OR trajectory jump → flag
    """
    pri = surprise * (1.0 + alpha * delta_h)
    return pri

Key design notes:

• No redundant softmax computation
• Math import added
• First-token case handled in monitoring loop (not here)
• Clean separation of concerns: surprise computation separate from PRI

--------

### Phase 2: Extend  monitoring_loop.py 

#### 2.1 Add PRI Configuration to  __init__ 

def __init__(
    self,
    adapter: ModelAdapter,
    tokenizer: Any,
    tau: float = DEFAULT_TAU,
    lambda_: float = DEFAULT_LAMBDA,
    pfail_cutoff: float = DEFAULT_PFAIL_CUTOFF,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    check_every_k_tokens: int = DEFAULT_CHECK_EVERY_K_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    alpha_pri: float = 0.1,      # NEW: PRI scaling parameter
    compute_pri: bool = True      # NEW: Enable/disable PRI computation
):
    """Initialize hallucination monitor with PRI support."""
    # ... existing initialization ...
    self.alpha_pri = alpha_pri
    self.compute_pri_flag = compute_pri

#### 2.2 Modify  generate_with_monitoring()  to Track PRI

Key changes:

1. Add state variable:  previous_hidden_final = None 
2. Extract final-layer hidden state at checkpoints (already pulled for ℏₛ)
3. Compute surprise and PRI
4. Accumulate PRI alongside ℏₛ

Pseudocode (integrate into existing loop):

def generate_with_monitoring(
    self,
    prompt: str,
    verbose: bool = False,
    compute_score_only: bool = False
) -> dict:
    # ... existing tokenization ...
    
    # NEW: Initialize PRI tracking
    previous_hidden_final = None
    
    if compute_score_only:
        top_hbar_s = []
        top_pri = []  # NEW
        k_top = 5
    else:
        trajectory = []
    
    # Token generation loop
    for step in range(self.max_tokens):
        # Get logits and sample token
        logits = self.adapter.next_token_logits(input_ids)
        
        if self.temperature == 0.0:
            next_token = mx.argmax(logits).item()
        else:
            probs_sampling = mx.softmax(logits / self.temperature, axis=-1)
            next_token = mx.random.categorical(probs_sampling).item()
        
        # Compute metrics at checkpoints
        check_frequency = 1 if compute_score_only else self.check_every_k_tokens
        
        if (step + 1) % check_frequency == 0:
            # Existing: ℏₛ computation
            probs = mx.softmax(logits, axis=-1)
            hidden_vectors = self.adapter.collector.get_all_blocks()
            metrics = compute_all_metrics(probs, hidden_vectors, self.tau, self.lambda_)
            
            # NEW: PRI computation
            if self.compute_pri_flag:
                current_hidden_final = hidden_vectors[-1]  # Last layer = final layer
                
                # Compute surprise (using already-computed probs)
                surprise = uncertainty_metrics.compute_surprise(probs, next_token)
                
                # Compute hidden-state jump
                if previous_hidden_final is None:
                    # First token: no jump, but keep surprise
                    delta_h = 0.0
                    pri = surprise  # PRI_0 = S_0
                else:
                    delta_h = uncertainty_metrics.compute_cosine_distance(
                        current_hidden_final,
                        previous_hidden_final
                    )
                    pri = uncertainty_metrics.compute_pri(surprise, delta_h, self.alpha_pri)
                
                metrics['pri'] = pri
                metrics['surprise'] = surprise
                metrics['delta_h'] = delta_h
                
                # Update state for next iteration
                previous_hidden_final = current_hidden_final
            
            # Accumulation
            if compute_score_only:
                top_hbar_s.append(float(metrics['hbar_s']))
                top_hbar_s.sort()
                top_hbar_s = top_hbar_s[-k_top:]
                
                if self.compute_pri_flag:
                    top_pri.append(float(metrics['pri']))
                    top_pri.sort()
                    top_pri = top_pri[-k_top:]
            else:
                metrics['step'] = step
                trajectory.append(metrics)
                
                if verbose:
                    pri_str = f", PRI={metrics['pri']:.4f}" if self.compute_pri_flag else ""
                    print(f"Step {step}: P_fail={metrics['pfail']:.4f}, ℏₛ={metrics['hbar_s']:.4f}{pri_str}")
                
                # Halting logic
                should_halt, reason = self._should_halt(metrics, step)
                if should_halt:
                    halted = True
                    halt_reason = reason
                    halt_step = step
                    break
        
        # Append token and continue
        generated_tokens.append(next_token)
        # ... existing EOS check and concatenation ...
    
    # Return
    if compute_score_only:
        hbar_s_score = sum(top_hbar_s) / len(top_hbar_s) if top_hbar_s else 0.0
        pri_score = sum(top_pri) / len(top_pri) if (self.compute_pri_flag and top_pri) else 0.0
        
        return {
            "score": float(hbar_s_score),  # Backward compatibility
            "hbar_s_score": float(hbar_s_score),
            "pri_score": float(pri_score),
            "halted": halted,
            "halt_reason": halt_reason
        }
    else:
        # ... existing full result return ...

Implementation notes:

• PRI computed at same frequency as ℏₛ (reuses hidden_vectors pull)
• No redundant softmax—reuse  probs  from ℏₛ computation
• First token handled correctly: PRI_0 = S_0
• Top-k mean aggregation for spike sensitivity

--------

### Phase 3: Extend  calibrate_thresholds.py 

#### 3.1 Update  precompute_scores()  for Dual Signals

def precompute_scores(self, max_tokens=20):
    """
    Pass 1: Generate once per sample, store ℏₛ AND PRI scores.
    
    Returns:
        Tuple of (hbar_s_scores, pri_scores, labels) as np.arrays
    """
    n = len(self.train_data)
    hbar_s_scores = np.zeros(n)
    pri_scores = np.zeros(n)
    labels = np.zeros(n, dtype=int)
    
    print(f"\n{'='*80}")
    print(f"Pass 1: Precomputing Scores ({n} samples)")
    print(f"{'='*80}\n")
    
    for i, sample in enumerate(tqdm(self.train_data, desc="Generating")):
        # Generate with compute_score_only=True
        self.monitor.update_params(pfail_cutoff=1.1)  # Disable halting
        self.monitor.max_tokens = max_tokens
        
        result = self.monitor.generate_with_monitoring(
            prompt=sample['prompt'],
            verbose=False,
            compute_score_only=True
        )
        
        hbar_s_scores[i] = result['hbar_s_score']
        pri_scores[i] = result['pri_score']
        labels[i] = sample['label']
        
        # Memory cleanup every 50 samples
        if (i + 1) % 50 == 0:
            del result
            gc.collect()
            print(f"  [{i+1}/{n}] Memory cleanup")
    
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

#### 3.2 Add  calibrate_joint()  Method

def calibrate_joint(self, hbar_s_scores, pri_scores, labels):
    """
    Pass 2: Calibrate joint (ℏₛ, PRI) model with train/val split.
    
    Evaluates:
    1. ℏₛ alone
    2. PRI alone
    3. Joint logistic regression (with 5-fold CV)
    
    Returns:
        Dict with best model and quadrant analysis
    """
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.metrics import roc_auc_score, precision_recall_curve
    from sklearn.model_selection import StratifiedKFold
    
    print(f"{'='*80}")
    print("Pass 2: Joint Calibration")
    print(f"{'='*80}\n")
    
    # Model 1: ℏₛ alone (inverted: lower ℏₛ = higher risk)
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
    
    print(f"  AUROC: {auroc_hbar:.4f}")
    print(f"  Best τ: {tau_hbar:.4f} (classify as hallucination if ℏₛ ≤ τ)")
    print(f"  Precision @ Recall≥0.9: {best_prec_hbar:.4f}\n")
    
    # Model 2: PRI alone (higher PRI = higher risk)
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
    
    print(f"  AUROC: {auroc_pri:.4f}")
    print(f"  Best τ: {tau_pri:.4f} (classify as hallucination if PRI ≥ τ)")
    print(f"  Precision @ Recall≥0.9: {best_prec_pri:.4f}\n")
    
    # Model 3: Joint with 5-fold cross-validation
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
    
    print(f"  AUROC (CV): {auroc_joint:.4f}")
    print(f"  Weights: w_hbar={clf.coef_[0][0]:.3f}, w_pri={clf.coef_[0][1]:.3f}, "
          f"intercept={clf.intercept_[0]:.3f}")
    print(f"  Precision @ Recall≥0.9: {best_prec_joint:.4f}\n")
    
    # Comparison
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
    
    for q_idx, q_mask, q_name in [
        (1, q1, "Q1: ℏₛ OK, PRI OK (Low Risk)"),
        (2, q2, "Q2: ℏₛ flags, PRI flags (High Risk)"),
        (3, q3, "Q3: ℏₛ OK, PRI flags (PRI-only detection)"),
        (4, q4, "Q4: ℏₛ flags, PRI OK (ℏₛ-only detection)")
    ]:
        n_total = q_mask.sum()
        n_hal = (labels[q_mask] == 1).sum()
        precision_q = n_hal / n_total if n_total > 0 else 0.0
        
        print(f"{q_name}")
        print(f"  Count: {n_total} ({n_total/len(labels)*100:.1f}%)")
        print(f"  Hallucination rate: {precision_q:.3f}")
        print()
    
    return {
        'auroc_hbar': auroc_hbar,
        'auroc_pri': auroc_pri,
        'auroc_joint': auroc_joint,
        'tau_hbar': tau_hbar,
        'tau_pri': tau_pri,
        'joint_model': clf,
        'best_precision_hbar': best_prec_hbar,
        'best_precision_pri': best_prec_pri,
        'best_precision_joint': best_prec_joint
    }

Critical fixes incorporated:

• 5-fold CV using  LogisticRegressionCV  to avoid train/eval on same data
• Calibrated thresholds for quadrant analysis (not median)
• Clear interpretation of each quadrant

--------

## 5. Policy Framework

### Quadrant-Based Decision Rules

Using calibrated thresholds τ_h (ℏₛ) and τ_p (PRI):

Decision Tree:
├─ IF ℏₛ ≤ τ_h AND PRI ≥ τ_p  [Q2: Both flag]
│  └─ ACTION: DEFER (maximum risk)
│
├─ IF ℏₛ > τ_h AND PRI ≥ τ_p  [Q3: PRI-only]
│  └─ ACTION: VERIFY_FACT (confident but ruptured)
│
├─ IF ℏₛ ≤ τ_h AND PRI < τ_p  [Q4: ℏₛ-only]
│  └─ ACTION: SAMPLE_ALTERNATIVES (low confidence, stable)
│
└─ IF ℏₛ > τ_h AND PRI < τ_p  [Q1: Neither flag]
   └─ ACTION: ACCEPT (confident + stable)

Expected quadrant distribution (hypothesis for HaluEval):

 Quadrant │ ℏₛ   │ PRI  │ Expected Precision   │ Volume 
──────────┼──────┼──────┼──────────────────────┼────────
 Q1       │ OK   │ OK   │ 0.3-0.4 (base rate)  │ 40-50% 
 Q2       │ Flag │ Flag │ 0.7-0.8 (both agree) │ 15-25% 
 Q3       │ OK   │ Flag │ 0.6-0.7 (PRI value)  │ 15-25% 
 Q4       │ Flag │ OK   │ 0.5-0.6 (ℏₛ value)   │ 10-20% 

Key hypothesis: Q3 should be enriched for confident hallucinations that ℏₛ misses.

--------

## 6. Implementation Checklist

### Coding Phase

Step 1: Extend  uncertainty_metrics.py 

[x] Add  import math  at top
[x] Add  normalize_vector(v, epsilon) 
[x] Add  compute_cosine_distance(v1, v2, epsilon) 
[x] Add  compute_surprise(probs, selected_token) 
[x] Add  compute_pri(surprise, delta_h, alpha) 

Step 2: Extend  monitoring_loop.py 

[x] Add  alpha_pri  and  compute_pri_flag  parameters to  __init__ 
[x] Add  previous_hidden_final = None  initialization in  generate_with_monitoring() 
[x] Add  top_pri = []  accumulator for  compute_score_only  mode
[x] Compute surprise using existing  probs  (no re-softmax)
[x] Handle first token:  if previous_hidden_final is None: pri = surprise 
[x] Compute  delta_h  and  pri  for subsequent tokens
[x] Store PRI in metrics dict
[x] Update  previous_hidden_final  after each checkpoint
[x] Return  pri_score  in  compute_score_only  mode

Step 3: Extend  calibrate_thresholds.py 

[ ] Modify  precompute_scores()  to return  (hbar_s, pri, labels) 
[ ] Add diagnostic printing for PRI statistics and correlation
[ ] Add  calibrate_joint()  method with 5-fold CV
[ ] Use calibrated thresholds for quadrant analysis
[ ] Update  calibrate_tau()  to call  precompute_scores()  and  calibrate_joint() 

Step 4: Update main calibration script

[ ] Modify main block to use new two-method flow
[ ] Save both ℏₛ and PRI thresholds to JSON
[ ] Save joint model weights

--------

### Validation Phase

Calibration Run:

python calibrate_thresholds.py --n-samples 200 --max-tokens 20 --seed 42

Expected outputs:

[ ] ℏₛ statistics printed (mean, std, Cohen's d)
[ ] PRI statistics printed
[ ] Correlation coefficient (target: |r| < 0.3)
[ ] AUROC for ℏₛ alone (~0.56)
[ ] AUROC for PRI alone (hypothesis: 0.55-0.60)
[ ] AUROC for joint (target: >0.65)
[ ] Quadrant analysis with precision per quadrant

Success criteria:

[ ] AUROC(Joint) > max(AUROC(ℏₛ), AUROC(PRI)) + 0.05
[ ] Q3 (PRI-only) has precision > Q1 (neither flags)
[ ] Signals are weakly correlated (|r| < 0.4)

--------

## 7. Open Questions & Future Work

### Aggregation Tuning

• Current:  mean(top_5(PRI_t)) 
• Alternatives:  p95(PRI_t) ,  max(PRI_t) ,  mean(top_3(PRI_t)) 
• Action: Run ablation on n=50 subset

### Alpha Parameter Calibration

• Current:  α = 0.1  (default)
• Expected range: [0.05, 0.2]
• Action: Grid search if PRI shows promise

### PRI Interpretation Limits

Critical caveat: PRI v1 assumes ruptures are detectable. If the model smoothly fabricates a false narrative (low S, low Δh), PRI will also be low. This is a detection boundary, not a bug.

Future work:

• Track multi-step trajectory entropy (not just pairwise jumps)
• Incorporate KL divergence between consecutive hidden-state distributions
• Measure "semantic distance" between consecutive tokens in embedding space

### Computational Optimization

Current cost: PRI adds ~2d operations per checkpoint (negligible)

Future optimization: If later moving to per-token PRI with reduced checkpoints:

• Add  collector.get_final_layer_only()  fast path
• Avoid materializing full  hidden_vectors  list when only final layer needed

--------