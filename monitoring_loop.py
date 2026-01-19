"""
Generation loop with uncertainty monitoring and halting.

Orchestrates token-by-token generation while computing semantic uncertainty (ℏₛ)
and halting when P_fail exceeds threshold.
"""

from typing import Any, List, Tuple, Optional
import mlx.core as mx

import model_adapters
import uncertainty_metrics
import config

ModelAdapter = model_adapters.ModelAdapter
compute_all_metrics = uncertainty_metrics.compute_all_metrics

(
    DEFAULT_TAU,
    DEFAULT_LAMBDA,
    DEFAULT_PFAIL_CUTOFF,
    DEFAULT_MAX_TOKENS,
    DEFAULT_CHECK_EVERY_K_TOKENS,
    DEFAULT_TEMPERATURE,
    DEFAULT_EPSILON
) = (
    config.DEFAULT_TAU,
    config.DEFAULT_LAMBDA,
    config.DEFAULT_PFAIL_CUTOFF,
    config.DEFAULT_MAX_TOKENS,
    config.DEFAULT_CHECK_EVERY_K_TOKENS,
    config.DEFAULT_TEMPERATURE,
    config.DEFAULT_EPSILON
)


class HallucinationMonitor:
    """
    Generation orchestrator with uncertainty monitoring.
    
    Implements token-by-token generation with ℏₛ tracking and adaptive halting
    based on P_fail threshold.
    """
    
    def __init__(
        self,
        adapter: ModelAdapter,
        tokenizer: Any,
        tau: float = DEFAULT_TAU,
        lambda_: float = DEFAULT_LAMBDA,
        pfail_cutoff: float = DEFAULT_PFAIL_CUTOFF,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        check_every_k_tokens: int = DEFAULT_CHECK_EVERY_K_TOKENS,
        temperature: float = DEFAULT_TEMPERATURE
    ):
        """
        Initialize hallucination monitor.
        
        Args:
            adapter: ModelAdapter instance for generation
            tokenizer: Tokenizer for encoding/decoding
            tau: Uncertainty threshold (τ) where P_fail = 0.5
            lambda_: Sigmoid steepness (λ) for P_fail computation
            pfail_cutoff: Halt when P_fail exceeds this value
            max_tokens: Maximum generation length
            check_every_k_tokens: Frequency of uncertainty checks (1 = every token)
            temperature: Sampling temperature (0.0 = greedy)
        """
        self.adapter = adapter
        self.tokenizer = tokenizer
        self.tau = tau
        self.lambda_ = lambda_
        self.pfail_cutoff = pfail_cutoff
        self.max_tokens = max_tokens
        self.check_every_k_tokens = check_every_k_tokens
        self.temperature = temperature
    
    def generate_with_monitoring(
        self,
        prompt: str,
        verbose: bool = False,
        compute_score_only: bool = False
    ) -> dict:
        """
        Main generation loop with uncertainty monitoring.
        
        Args:
            prompt: Input text prompt
            verbose: If True, print detailed metrics during generation
            compute_score_only: If True, only compute and return aggregated ℏₛ score
                               (lightweight mode for calibration)
            
        Returns:
            Dict with keys:
                - If compute_score_only=True:
                    - score: float - Mean of top-k ℏₛ values
                    - halted: bool - Always False in score-only mode
                    - halt_reason: str - Reason for stopping
                - If compute_score_only=False:
                    - tokens: List[int] - Generated token IDs
                    - text: str - Decoded text
                    - halted: bool - Whether generation was halted by P_fail
                    - halt_reason: str - Reason for stopping
                    - halt_step: Optional[int] - Token step where halting occurred
                    - trajectory: List[dict] - Metrics at each checked step
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt)
        if not isinstance(input_ids, mx.array):
            input_ids = mx.array(input_ids)
        
        # Initialize generation state
        generated_tokens = []
        halted = False
        halt_reason = ""
        halt_step = None
        
        # Lightweight accumulator for calibration vs full trajectory for monitoring
        if compute_score_only:
            top_hbar_s = []  # Keep top-k ℏₛ values
            k_top = 5
        else:
            trajectory = []
        
        # Get EOS token ID
        eos_token_id = self.tokenizer.eos_token_id if hasattr(self.tokenizer, 'eos_token_id') else None
        
        if verbose:
            print(f"Starting generation with prompt: {prompt}")
            print(f"Parameters: τ={self.tau}, λ={self.lambda_}, P_fail_cutoff={self.pfail_cutoff}")
            print("-" * 80)
        
        # Token generation loop
        for step in range(self.max_tokens):
            # DIAGNOSTIC: Log sequence lengths before forward pass
            if verbose and step < 3:  # Only first 3 steps to avoid spam
                print(f"[DIAGNOSTIC] Step {step}: len(input_ids)={len(input_ids)}, len(generated_tokens)={len(generated_tokens)}")
            
            # Get logits for next token (adapter calls collector.start() internally)
            logits = self.adapter.next_token_logits(input_ids)
            
            # Sample or select next token
            if self.temperature == 0.0:
                # Greedy sampling
                next_token = mx.argmax(logits).item()
            else:
                # Temperature sampling
                probs = mx.softmax(logits / self.temperature, axis=-1)
                next_token = mx.random.categorical(probs).item()
            
            # Compute uncertainty metrics BEFORE accepting token (at specified frequency)
            # For calibration (compute_score_only), force check_every_k_tokens=1
            check_frequency = 1 if compute_score_only else self.check_every_k_tokens
            
            if (step + 1) % check_frequency == 0:
                # Get probability distribution
                probs = mx.softmax(logits, axis=-1)
                
                # Get hidden states from all blocks
                hidden_vectors = self.adapter.collector.get_all_blocks()
                
                # Compute all metrics
                metrics = compute_all_metrics(
                    probs=probs,
                    hidden_vectors=hidden_vectors,
                    tau=self.tau,
                    lambda_=self.lambda_,
                    epsilon=DEFAULT_EPSILON
                )
                
                if compute_score_only:
                    # Lightweight: accumulate only ℏₛ (keep top-k)
                    top_hbar_s.append(float(metrics['hbar_s']))
                    top_hbar_s.sort()
                    top_hbar_s = top_hbar_s[-k_top:]  # Keep only top-k
                else:
                    # Full monitoring: build trajectory
                    metrics["step"] = step
                    trajectory.append(metrics)
                    
                    if verbose:
                        print(f"Step {step}: P_fail={metrics['pfail']:.4f}, ℏₛ={metrics['hbar_s']:.4f}, "
                              f"Δμ={metrics['delta_mu']:.4f}, Δσ={metrics['delta_sigma']:.4f}")
                    
                    # Check halting condition BEFORE appending risky token
                    should_halt, reason = self._should_halt(metrics["pfail"], step)
                    if should_halt:
                        halted = True
                        halt_reason = reason
                        halt_step = step
                        if verbose:
                            print(f"HALTED at step {step}: {reason}")
                            print(f"  Risky token ({next_token}) NOT appended to output")
                        break
            
            # Only append token if we didn't halt
            generated_tokens.append(next_token)
            
            # Check for EOS
            if eos_token_id is not None and next_token == eos_token_id:
                halt_reason = "EOS token generated"
                break
            
            # Append new token to input sequence
            next_token_array = mx.array([next_token])
            input_ids = mx.concatenate([input_ids, next_token_array], axis=-1)
        
        # If loop completed without halt
        if not halted and not halt_reason:
            halt_reason = "max_tokens reached"
        
        if verbose:
            print("-" * 80)
            print(f"Generation complete: {len(generated_tokens)} tokens")
            print(f"Final reason: {halt_reason}")
        
        # Return lightweight score for calibration or full result for monitoring
        if compute_score_only:
            score = sum(top_hbar_s) / len(top_hbar_s) if top_hbar_s else 0.0
            return {
                "score": float(score),
                "halted": halted,  # Should be False when pfail_cutoff > 1.0
                "halt_reason": halt_reason
            }
        else:
            # Decode generated text
            if generated_tokens:
                generated_text = self.tokenizer.decode(generated_tokens)
            else:
                generated_text = ""
            
            return {
                "tokens": generated_tokens,
                "text": generated_text,
                "halted": halted,
                "halt_reason": halt_reason,
                "halt_step": halt_step,
                "trajectory": trajectory,
                "prompt": prompt
            }
    
    def _should_halt(self, pfail: float, step: int) -> Tuple[bool, str]:
        """
        Check if generation should halt based on P_fail.
        
        Args:
            pfail: Current failure probability
            step: Current token step
            
        Returns:
            Tuple of (should_halt: bool, reason: str)
        """
        if pfail > self.pfail_cutoff:
            return True, f"P_fail={pfail:.4f} exceeded cutoff={self.pfail_cutoff}"
        
        return False, ""
    
    def update_params(self, tau: Optional[float] = None, lambda_: Optional[float] = None, 
                     pfail_cutoff: Optional[float] = None):
        """
        Update monitoring parameters without recreating monitor.
        
        Args:
            tau: New threshold parameter
            lambda_: New steepness parameter
            pfail_cutoff: New P_fail cutoff threshold
        """
        if tau is not None:
            self.tau = tau
        if lambda_ is not None:
            self.lambda_ = lambda_
        if pfail_cutoff is not None:
            self.pfail_cutoff = pfail_cutoff