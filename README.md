# Semantic Uncertainty Hallucination Detection System

Real-time hallucination detection using semantic uncertainty (ℏₛ) monitoring during LLM generation.

## Overview

This research framework tests the hypothesis that semantic uncertainty during generation predicts hallucinations. The system monitors two signals:
- **Δμ (precision)**: Inverse of probability concentration (Gini impurity)
- **Δσ (flexibility)**: Dispersion of hidden states across transformer blocks

When the combined semantic uncertainty **ℏₛ = √(Δμ × Δσ)** exceeds a calibrated threshold, generation is halted to prevent hallucinations.

## Project Structure

```
hbar-experiment/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration constants
├── requirements.txt            # Python dependencies
├── README.md                   # This file
│
# Phase 1: Core Infrastructure
├── uncertainty_metrics.py      # ℏₛ computation functions
├── hidden_state_collector.py  # Hidden state capture during generation
├── model_adapters.py           # Model-specific forward passes (Llama, Qwen, Phi-3)
├── monitoring_loop.py          # Generation with uncertainty monitoring
├── test_phase1.py             # Phase 1 validation script
│
# Phase 2: Calibration & Validation (TODO)
├── halueval_loader.py         # HaluEval 2.0 dataset loading
├── calibrate_thresholds.py    # Grid search for optimal (τ, λ)
├── validate.py                # Test set evaluation
├── multi_model_runner.py      # Run all 3 models
│
# Phase 3: Analysis & Visualization (TODO)
├── analyze_results.py         # Statistical analysis
├── generate_figures.py        # ROC curves, trajectories, heatmaps
│
# Data and outputs
├── data/                      # Dataset cache
├── calibrated_params/         # Saved (τ, λ) per model
├── results/                   # Validation results
└── figures/                   # Generated plots
```

## Installation

### Prerequisites

- Mac Mini M4 (or Apple Silicon)
- Python 3.9+
- MLX and MLX-LM already installed

### Install Dependencies

```bash
cd hbar-experiment
pip3 install -r requirements.txt
```

## Phase 1: Core Infrastructure (✓ Complete)

### Test the System

Run the Phase 1 validation script:

```bash
cd hbar-experiment
python3 test_phase1.py
```

This will:
1. Load Llama 3.2 3B Instruct (4-bit quantized)
2. Run 3 test prompts with uncertainty monitoring
3. Display ℏₛ trajectory and P_fail values
4. Validate that the full pipeline works end-to-end

### Expected Output

```
Phase 1 End-to-End Test
================================================================================

Loading Llama 3.2 3B Instruct model...
✓ Model loaded successfully

Creating adapter and collector...
✓ Adapter created: LlamaAdapter
  Number of transformer layers: 28

Running test generations...
Test 1/3: "The capital of France is"
--------------------------------------------------------------------------------
Step 0: P_fail=0.1234, ℏₛ=1.4567, Δμ=12.34, Δσ=0.56
...
```

## Usage

### Basic Generation with Monitoring

```python
from mlx_lm import load
from hidden_state_collector import HiddenStateCollector
from model_adapters import create_adapter
from monitoring_loop import HallucinationMonitor

# Load model
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Create collector and adapter
collector = HiddenStateCollector()
adapter = create_adapter(model, collector, model_type="llama")

# Create monitor
monitor = HallucinationMonitor(
    adapter=adapter,
    tokenizer=tokenizer,
    tau=2.0,           # Threshold where P_fail = 0.5
    lambda_=5.0,       # Sigmoid steepness
    pfail_cutoff=0.85, # Halt when P_fail > 0.85
    max_tokens=512,
    temperature=0.0    # Greedy sampling
)

# Generate with monitoring
result = monitor.generate_with_monitoring(
    prompt="The capital of France is",
    verbose=True
)

print(f"Generated: {result['text']}")
print(f"Halted: {result['halted']}")
print(f"Reason: {result['halt_reason']}")
```

### Accessing Uncertainty Trajectory

```python
for step_metrics in result['trajectory']:
    print(f"Step {step_metrics['step']}:")
    print(f"  ℏₛ = {step_metrics['hbar_s']:.4f}")
    print(f"  P_fail = {step_metrics['pfail']:.4f}")
    print(f"  Δμ = {step_metrics['delta_mu']:.4f}")
    print(f"  Δσ = {step_metrics['delta_sigma']:.4f}")
```

## Key Concepts

### Semantic Uncertainty (ℏₛ)

Combines precision and flexibility into a single metric:
- **High ℏₛ**: Model is uncertain (low precision) AND inconsistent (high flexibility) → likely hallucinating
- **Low ℏₛ**: Model is confident and consistent → likely accurate

### Failure Probability (P_fail)

Maps ℏₛ to [0, 1] using sigmoid: `P_fail = 1/(1 + exp(-λ(ℏₛ - τ)))`
- **τ (tau)**: Threshold where P_fail = 0.5
- **λ (lambda)**: Steepness of transition

### Halting Logic

Generation stops when:
1. **P_fail > cutoff** (default 0.85) → hallucination predicted
2. **EOS token** generated
3. **max_tokens** reached

## Models Supported

1. **Llama 3.2 3B Instruct** (4-bit) - `mlx-community/Llama-3.2-3B-Instruct-4bit`
2. **Qwen 2.5 7B Instruct** (4-bit) - `mlx-community/Qwen2.5-7B-Instruct-4bit`
3. **Phi-3 Mini Instruct** (4-bit) - `mlx-community/Phi-3-mini-128k-instruct-4bit`

## Implementation Notes

### Phase 1 Design Decisions

1. **No KV Cache**: Manual block-by-block forward passes without cache for correctness validation (slow but correct)
2. **MLX-native**: All computations use `mlx.core.array`, not NumPy
3. **Collector Lifecycle**: Reset at each token boundary via `collector.start()`
4. **Greedy Sampling**: Temperature = 0.0 for deterministic testing

### Performance Expectations

Phase 1 is **intentionally slow** due to O(n²) recomputation without KV cache:
- Llama 3.2 3B: ~2-5 tokens/sec
- Qwen 2.5 7B: ~1-2 tokens/sec
- Phi-3 Mini: ~2-4 tokens/sec

Future optimization: Add KV cache support or adaptive monitoring frequency.

## Next Steps

### Phase 2: Calibration Pipeline (In Progress)

1. Implement `halueval_loader.py` - Download and sample HaluEval 2.0
2. Implement `calibrate_thresholds.py` - Grid search for optimal (τ, λ)
3. Implement `validate.py` - Test set evaluation with AUROC
4. Run calibration on all 3 models

### Phase 3: Analysis & Visualization

1. Implement `analyze_results.py` - Cross-model comparison
2. Implement `generate_figures.py` - ROC curves, trajectories, sensitivity
3. Generate final research outputs

## References

- **HaluEval 2.0**: Benchmark for hallucination detection
- **Semantic Uncertainty**: Kuhn et al. (2023) - Uncertainty quantification in NLP
- **MLX Framework**: Apple's ML framework for Apple Silicon

## License

Research project for Anthropic AI Safety Fellowship.