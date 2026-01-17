"""
Uncertainty metrics computation using MLX.

All functions operate on MLX arrays and return Python floats for logging.
"""

from typing import List
import mlx.core as mx
import math


def compute_gini_impurity(probs: mx.array) -> float:
    """
    Compute Gini impurity: 1 - Σp²
    
    Measures concentration of probability mass. Low values indicate high concentration
    (model is confident), high values indicate dispersed probability (model is uncertain).
    
    Args:
        probs: MLX array of shape [vocab_size], normalized probabilities
        
    Returns:
        Gini impurity scalar in [0, 1]
    """
    I_cat = 1.0 - float(mx.sum(probs * probs).item())
    return I_cat


def compute_delta_mu_proxy(I_cat: float, epsilon: float = 1e-2) -> float:
    """
    Compute precision proxy: 1/√(I_cat + ε)
    
    Δμ represents inverse uncertainty - higher when probability is concentrated.
    The soft floor (epsilon = 1e-2) prevents Δμ blow-ups when distribution is ultra-peaked.
    
    Args:
        I_cat: Gini impurity value
        epsilon: Soft floor to prevent explosion (default 1e-2, caps Δμ ≈ 10)
        
    Returns:
        Precision proxy scalar (bounded, typically in [1, 10])
    """
    I_cat_eff = I_cat + epsilon
    delta_mu = 1.0 / math.sqrt(I_cat_eff)
    return delta_mu


def compute_delta_sigma_proxy(hidden_vectors: List[mx.array], epsilon: float = 1e-8) -> float:
    """
    Compute flexibility proxy: √(mean(||normalize(xᵢ) - centroid||²))
    
    Δσ measures dispersion of NORMALIZED last-token hidden states across blocks.
    Normalization removes magnitude effects, measuring only directional inconsistency.
    High dispersion indicates inconsistent internal representations (flexibility).
    
    Args:
        hidden_vectors: List of MLX arrays, each shape [dim], one per transformer block
        epsilon: Small constant to prevent division by zero in normalization
        
    Returns:
        Dispersion scalar in [0, ~2] (normalized vectors have bounded dispersion)
        
    Raises:
        ValueError: If hidden_vectors is empty or contains inconsistent shapes
    """
    if not hidden_vectors:
        raise ValueError("hidden_vectors cannot be empty")
    
    # Normalize each vector to unit norm (cosine space)
    normalized_vectors = []
    for h in hidden_vectors:
        norm = mx.sqrt(mx.sum(h * h) + epsilon)
        normalized_vectors.append(h / norm)
    
    # Stack normalized vectors: [num_blocks, dim]
    stacked = mx.stack(normalized_vectors, axis=0)
    
    # Compute centroid: [dim]
    centroid = mx.mean(stacked, axis=0)
    
    # Compute L2 distances from centroid: [num_blocks]
    differences = stacked - centroid[None, :]  # Broadcasting: [num_blocks, dim]
    squared_distances = mx.sum(differences * differences, axis=1)  # [num_blocks]
    
    # Mean squared distance
    mean_squared_distance = mx.mean(squared_distances).item()
    
    # Root mean squared distance
    delta_sigma = math.sqrt(mean_squared_distance)
    
    return delta_sigma


def compute_hbar_s(delta_mu: float, delta_sigma: float) -> float:
    """
    Compute semantic uncertainty: √(Δμ × Δσ)
    
    ℏₛ combines precision (Δμ) and flexibility (Δσ) into a single uncertainty metric.
    High ℏₛ indicates model is both uncertain (low precision) and inconsistent (high flexibility).
    
    Args:
        delta_mu: Precision proxy
        delta_sigma: Flexibility proxy
        
    Returns:
        Semantic uncertainty ℏₛ (unbounded, typically in [0, 50])
    """
    hbar_s = math.sqrt(delta_mu * delta_sigma)
    return hbar_s


def compute_pfail(hbar_s: float, tau: float, lambda_: float) -> float:
    """
    Compute failure probability using sigmoid: 1/(1 + exp(-λ(ℏₛ - τ)))
    
    Maps ℏₛ to [0, 1] probability of hallucination/failure.
    - tau (τ): threshold where P_fail = 0.5
    - lambda_ (λ): steepness of sigmoid (higher = sharper transition)
    
    Args:
        hbar_s: Semantic uncertainty value
        tau: Threshold parameter
        lambda_: Steepness parameter
        
    Returns:
        Failure probability in [0, 1]
    """
    exponent = -lambda_ * (hbar_s - tau)
    # Clamp exponent to prevent overflow
    exponent = max(-50.0, min(50.0, exponent))
    pfail = 1.0 / (1.0 + math.exp(exponent))
    return pfail


def compute_all_metrics(
    probs: mx.array,
    hidden_vectors: List[mx.array],
    tau: float,
    lambda_: float,
    epsilon: float = 1e-8
) -> dict:
    """
    Convenience function to compute all uncertainty metrics at once.
    
    Args:
        probs: MLX array of shape [vocab_size], normalized probabilities
        hidden_vectors: List of hidden state vectors from transformer blocks
        tau: Threshold parameter for P_fail
        lambda_: Steepness parameter for P_fail
        epsilon: Numerical stability constant
        
    Returns:
        Dict with keys: I_cat, delta_mu, delta_sigma, hbar_s, pfail
    """
    I_cat = compute_gini_impurity(probs)
    delta_mu = compute_delta_mu_proxy(I_cat)  # Use default epsilon=1e-4 for soft floor
    delta_sigma = compute_delta_sigma_proxy(hidden_vectors, epsilon)  # epsilon for normalization
    hbar_s = compute_hbar_s(delta_mu, delta_sigma)
    pfail = compute_pfail(hbar_s, tau, lambda_)
    
    return {
        "I_cat": I_cat,
        "delta_mu": delta_mu,
        "delta_sigma": delta_sigma,
        "hbar_s": hbar_s,
        "pfail": pfail
    }