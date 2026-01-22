# Implementation Plan

## [Overview]

Build a real-time hallucination detection system using semantic uncertainty (ℏₛ) monitoring for LLM generation on Mac Mini M4 with MLX.

This implementation creates a research framework for the Anthropic AI Safety Fellowship that tests the hypothesis: semantic uncertainty during generation predicts hallucinations. The system monitors two signals during token generation: Δμ (precision from probability concentration) and Δσ (flexibility from hidden state dispersion across transformer blocks). When the combined semantic uncertainty ℏₛ = √(Δμ × Δσ) exceeds a calibrated threshold, generation is halted to prevent hallucinations. The framework will be validated on HaluEval 2.0 benchmark using three 4-bit quantized models (Llama 3.2 3B, Qwen 2.5 7B, Phi-3 Mini), with calibration performed on 5K training examples and validation on 5K test examples.

The implementation is structured in three phases: (1) Core infrastructure for hidden state collection and uncertainty computation, (2) Calibration and validation pipeline with threshold optimization, (3) Analysis and visualization. Phase 1 prioritizes correctness over speed by implementing manual block-by-block forward passes without KV cache, allowing us to capture hidden states at each transformer block during generation.

## [Types]

Core data structures for hidden state collection and uncertainty monitoring.

```python
# Type definitions for the system

from typing import Dict, List, Tuple, Optional, Any
import mlx.core as mx

# Hidden state collector types
BlockHiddenStates = Dict[int, mx.array]  # Maps layer_idx -> hidden_vector [dim]

# Uncertainty computation types
UncertaintyMetrics = Dict[str, float]  # {"I_cat": float, "delta_mu": float, "delta_sigma": float, "hbar_s": float, "pfail": float}

# Generation result types
GenerationResult = Dict[str, Any]  # {"tokens": List[int], "text": str, "halted": bool, "halt_reason": str, "halt_step": Optional[int], "trajectory": List[UncertaintyMetrics]}

# Dataset types
HaluEvalSample = Dict[str, Any]  # {"prompt": str, "label": int, "id": str, "metadata": Dict}
DatasetSplit = Tuple[List[HaluEvalSample], List[HaluEvalSample]]  # (train_data, test_data)

# Calibration types
CalibrationParams = Dict[str, float]  # {"tau": float, "lambda_": float}
CalibrationResult = Dict[str, Any]  # {"model_name": str, "params": CalibrationParams, "train_auroc": float, "train_precision": float, "train_recall": float}

# Validation types
ValidationMetrics = Dict[str, Any]  # {"auroc": float, "precision": float, "recall": float, "f1": float, "confusion_matrix": List[List[int]]}
```

## [Files]

Core implementation requires 10 new Python files in the `hbar-experiment/` directory.

**New files to create:**

1. **`hbar-experiment/hidden_state_collector.py`** - Stateful collector for capturing hidden states during generation
   - Manages per-token hidden state collection across transformer blocks
   - Resets state at each token boundary

2. **`hbar-experiment/model_adapters.py`** - Model-specific adapters for Llama, Qwen, Phi-3
   - Implements manual block-by-block forward passes without KV cache
   - Handles model structure introspection and hidden state extraction
   - Three concrete adapter classes: LlamaAdapter, QwenAdapter, Phi3Adapter

3. **`hbar-experiment/uncertainty_metrics.py`** - Pure functions for uncertainty computation
   - Implements all metric calculations (Gini impurity, Δμ, Δσ, ℏₛ, P_fail)
   - MLX-native operations with minimal NumPy conversion

4. **`hbar-experiment/monitoring_loop.py`** - Generation loop with uncertainty monitoring
   - Orchestrates token-by-token generation with halting logic
   - Tracks ℏₛ trajectory and halt conditions

5. **`hbar-experiment/halueval_loader.py`** - HaluEval 2.0 dataset loading and sampling
   - Downloads dataset from original GitHub source
   - Performs deterministic sampling (10K examples, seed=42)
   - Splits data 50/50 for train/test

6. **`hbar-experiment/calibrate_thresholds.py`** - Grid search for optimal (τ, λ) per model
   - Exhaustive grid search over tau and lambda ranges
   - Saves calibrated parameters to JSON per model

7. **`hbar-experiment/validate.py`** - Validation on test split with calibrated parameters
   - Loads calibrated params and runs test evaluation
   - Computes AUROC, precision, recall, F1, confusion matrix

8. **`hbar-experiment/multi_model_runner.py`** - Orchestrates all 3 models
   - Runs Llama, Qwen, Phi-3 with their calibrated parameters
   - Aggregates results for comparison

9. **`hbar-experiment/analyze_results.py`** - Statistical analysis of results
   - Cross-model comparison
   - Confidence interval computation

10. **`hbar-experiment/generate_figures.py`** - Visualization (matplotlib-only)
    - ROC curves per model
    - ℏₛ trajectories over token steps
    - Threshold sensitivity analysis

**Configuration files:**

11. **`hbar-experiment/config.py`** - Configuration constants
    - Model paths, default hyperparameters, file paths
    - Dataset URLs, random seeds

12. **`hbar-experiment/requirements.txt`** - Python dependencies
    - Specifies all required packages

**Supporting files:**

13. **`hbar-experiment/__init__.py`** - Package initialization
14. **`hbar-experiment/README.md`** - Documentation for the experiment directory

## [Functions]

Detailed function specifications for all modules.

### `hidden_state_collector.py`

**Class: `HiddenStateCollector`**

- `__init__(self) -> None`
  - Initialize empty block storage

- `start(self) -> None`
  - Reset internal buffer for new token generation
  - Clears `_blocks` dictionary

- `record(self, layer_idx: int, hidden_vector: mx.array) -> None`
  - Store last-token hidden vector from block `layer_idx`
  - Args: `layer_idx` (0-indexed block number), `hidden_vector` (mx.array of shape [dim])
  - Raises: ValueError if hidden_vector shape is invalid

- `get_all_blocks(self) -> List[mx.array]`
  - Return all captured hidden vectors in layer order
  - Returns: List of mx.array, each of shape [dim]
  - Used for computing Δσ dispersion

### `model_adapters.py`

**Base Class: `ModelAdapter` (abstract)**

- `__init__(self, model: Any, collector: HiddenStateCollector) -> None`
  - Store model reference and collector
  - Call `_locate_components()` for validation

- `_locate_components(self) -> None` (abstract)
  - Introspect model structure to find: layers, embed_tokens, norm, lm_head
  - Try patterns: `model.model.layers`, `model.layers`, etc.
  - Raise ValueError with clear message if structure unknown

- `_extract_last_token_hidden(self, x: mx.array) -> mx.array` (abstract)
  - Normalize shape to extract last-token vector
  - Handle shapes: [dim], [seq, dim], [batch, seq, dim]
  - Return: mx.array of shape [dim]

- `forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array` (abstract)
  - Full prefix forward pass with hidden state collection
  - Calls `collector.record()` after each transformer block
  - Returns: logits for next token (mx.array)

- `next_token_logits(self, input_ids: mx.array) -> mx.array` (abstract)
  - Wrapper around `forward_prefix_with_collection`
  - Returns: logits for next token only (mx.array of shape [vocab_size])

**Concrete Class: `LlamaAdapter`**

- `_locate_components(self) -> None`
  - Locate: `self.model.model.layers`, `self.model.model.embed_tokens`, `self.model.model.norm`, `self.model.lm_head`
  - Validate all components exist

- `_extract_last_token_hidden(self, x: mx.array) -> mx.array`
  - Implementation for Llama-specific shape handling

- `forward_prefix_with_collection(self, input_ids: mx.array) -> mx.array`
  - Embed tokens -> iterate through layers -> call collector.record() -> final norm -> lm_head
  - Returns: logits

**Concrete Class: `QwenAdapter`** - Similar structure to LlamaAdapter with Qwen-specific paths

**Concrete Class: `Phi3Adapter`** - Similar structure to LlamaAdapter with Phi-3-specific paths

### `uncertainty_metrics.py`

- `compute_gini_impurity(probs: mx.array) -> float`
  - Compute 1 - Σp² (Gini impurity)
  - Args: probs (mx.array of shape [vocab_size]), normalized probabilities
  - Returns: float scalar
  - Implementation: `1.0 - mx.sum(probs * probs).item()`

- `compute_delta_mu_proxy(I_cat: float, epsilon: float = 1e-8) -> float`
  - Compute precision proxy: 1/√(I_cat + ε)
  - Args: I_cat (Gini impurity), epsilon (numerical stability)
  - Returns: float scalar

- `compute_delta_sigma_proxy(hidden_vectors: List[mx.array]) -> float`
  - Compute flexibility proxy: √(mean(||xᵢ - centroid||²))
  - Args: hidden_vectors (List of mx.array, each shape [dim])
  - Returns: float scalar
  - Implementation: stack vectors, compute centroid, compute L2 distances, return sqrt(mean)

- `compute_hbar_s(delta_mu: float, delta_sigma: float) -> float`
  - Compute semantic uncertainty: √(Δμ × Δσ)
  - Args: delta_mu, delta_sigma
  - Returns: float scalar

- `compute_pfail(hbar_s: float, tau: float, lambda_: float) -> float`
  - Compute failure probability: 1/(1 + exp(-λ(ℏₛ - τ)))
  - Args: hbar_s, tau (threshold), lambda_ (steepness)
  - Returns: float in [0, 1]

- `compute_all_metrics(probs: mx.array, hidden_vectors: List[mx.array], tau: float, lambda_: float) -> UncertaintyMetrics`
  - Convenience function that computes all metrics at once
  - Returns: Dict with I_cat, delta_mu, delta_sigma, hbar_s, pfail

### `monitoring_loop.py`

**Class: `HallucinationMonitor`**

- `__init__(self, adapter: ModelAdapter, tokenizer: Any, tau: float, lambda_: float, pfail_cutoff: float = 0.85, max_tokens: int = 512, check_every_k_tokens: int = 1, temperature: float = 0.0) -> None`
  - Initialize with model adapter, tokenizer, and hyperparameters
  - Store configuration

- `generate_with_monitoring(self, prompt: str, verbose: bool = False) -> GenerationResult`
  - Main generation loop with uncertainty monitoring
  - Tokenize prompt -> iterate token generation -> compute metrics -> check halt condition
  - Returns: GenerationResult dict with tokens, text, halted flag, trajectory
  - Stops when: (1) P_fail > pfail_cutoff, (2) max_tokens reached, (3) EOS token generated

- `_should_halt(self, pfail: float, step: int) -> Tuple[bool, str]`
  - Check halting condition
  - Returns: (should_halt: bool, reason: str)

### `halueval_loader.py`

- `download_halueval(cache_dir: str = "./data/halueval") -> str`
  - Download HaluEval 2.0 from original GitHub source
  - Args: cache_dir (where to store dataset)
  - Returns: path to downloaded dataset
  - Implementation: Use requests to download, handle existing files

- `load_and_sample(dataset_path: str, n_samples: int = 10000, seed: int = 42) -> List[HaluEvalSample]`
  - Load dataset and perform deterministic sampling
  - Args: dataset_path, n_samples, seed
  - Returns: List of HaluEvalSample dicts
  - Implementation: Load JSON, random.seed(seed), random.sample()

- `split_train_test(data: List[HaluEvalSample], train_ratio: float = 0.5, seed: int = 42) -> DatasetSplit`
  - Split data into train/test
  - Args: data, train_ratio (0.5 for 50/50), seed
  - Returns: (train_data, test_data) tuple

### `calibrate_thresholds.py`

**Class: `ThresholdCalibrator`**

- `__init__(self, adapter: ModelAdapter, tokenizer: Any, train_data: List[HaluEvalSample]) -> None`
  - Initialize with model adapter and training data

- `grid_search(self, tau_range: Tuple[float, float, float], lambda_range: Tuple[float, float, float], pfail_cutoff: float = 0.85) -> CalibrationResult`
  - Exhaustive grid search for optimal (τ, λ)
  - Args: tau_range (start, stop, step), lambda_range (start, stop, step)
  - Returns: CalibrationResult with best params and metrics
  - Implementation: nested loops over tau and lambda, evaluate on train_data, select best AUROC

- `evaluate_params(self, tau: float, lambda_: float, pfail_cutoff: float = 0.85) -> ValidationMetrics`
  - Evaluate specific (τ, λ) on training data
  - Returns: AUROC, precision, recall, F1
  - Implementation: run monitoring loop on all train samples, compute sklearn metrics

- `save_params(self, result: CalibrationResult, output_path: str) -> None`
  - Save calibrated parameters to JSON
  - Args: result, output_path (e.g., "calibrated_params/llama_3.2_3b.json")

### `validate.py`

- `load_calibrated_params(model_name: str, params_dir: str = "./calibrated_params") -> CalibrationParams`
  - Load calibrated (τ, λ) from JSON
  - Args: model_name (e.g., "llama_3.2_3b"), params_dir
  - Returns: CalibrationParams dict

- `run_validation(adapter: ModelAdapter, tokenizer: Any, test_data: List[HaluEvalSample], params: CalibrationParams, pfail_cutoff: float = 0.85) -> ValidationMetrics`
  - Run validation on test split with calibrated params
  - Args: adapter, tokenizer, test_data, params
  - Returns: ValidationMetrics with AUROC, precision, recall, F1, confusion matrix
  - Implementation: run monitoring loop on test samples, compute metrics

### `multi_model_runner.py`

- `load_model_and_tokenizer(model_path: str) -> Tuple[Any, Any]`
  - Load MLX model and tokenizer using mlx_lm.load()
  - Args: model_path (e.g., "mlx-community/Llama-3.2-3B-Instruct-4bit")
  - Returns: (model, tokenizer)

- `run_all_models(test_data: List[HaluEvalSample], model_configs: List[Dict[str, str]]) -> Dict[str, ValidationMetrics]`
  - Run all 3 models with their calibrated parameters
  - Args: test_data, model_configs (list of {"name": str, "path": str} dicts)
  - Returns: Dict mapping model_name -> ValidationMetrics
  - Implementation: iterate models, load, create adapter, load params, validate

### `analyze_results.py`

- `aggregate_stats(results: Dict[str, ValidationMetrics]) -> pd.DataFrame`
  - Aggregate metrics across models into DataFrame
  - Args: results (output from run_all_models)
  - Returns: pandas DataFrame with rows=models, cols=metrics

- `compare_models(results: Dict[str, ValidationMetrics]) -> Dict[str, Any]`
  - Statistical comparison of model performance
  - Returns: Dict with best_model, performance_ranking, etc.

- `compute_confidence_intervals(results: Dict[str, Any], bootstrap_n: int = 1000) -> Dict[str, Any]`
  - Bootstrap confidence intervals for metrics
  - Args: results (includes trajectories), bootstrap_n
  - Returns: Dict with 95% CIs per model per metric

### `generate_figures.py`

- `plot_roc_curves(results: Dict[str, Any], output_path: str = "./figures/roc_curves.png") -> None`
  - Generate ROC curves for all models on same plot
  - Args: results (includes true labels and predictions), output_path
  - Implementation: matplotlib-only, one curve per model

- `plot_hbar_trajectories(monitoring_logs: Dict[str, List[GenerationResult]], output_path: str = "./figures/hbar_trajectories.png") -> None`
  - Plot ℏₛ values over token steps for sample generations
  - Args: monitoring_logs (per-model trajectories), output_path
  - Implementation: matplotlib line plots

- `plot_threshold_sensitivity(calibration_logs: Dict[str, Any], output_path: str = "./figures/threshold_sensitivity.png") -> None`
  - Heatmap of AUROC vs (τ, λ) grid
  - Args: calibration_logs (grid search results), output_path
  - Implementation: matplotlib imshow/contourf

## [Classes]

Detailed class specifications with inheritance and key methods.

### `hidden_state_collector.HiddenStateCollector`
- **Purpose**: Stateful collector for hidden states during generation
- **Attributes**:
  - `_blocks: Dict[int, mx.array]` - Maps layer index to hidden vector
- **Methods**: `start()`, `record()`, `get_all_blocks()`
- **Lifecycle**: Reset at each token boundary via `start()`

### `model_adapters.ModelAdapter` (abstract base)
- **Purpose**: Abstract interface for model-specific adapters
- **Attributes**:
  - `model: Any` - MLX model reference
  - `collector: HiddenStateCollector` - Shared collector instance
  - `layers: List` - Transformer blocks
  - `embed_tokens: Any` - Embedding layer
  - `norm: Any` - Final normalization
  - `lm_head: Any` - Output projection
- **Methods**: `_locate_components()`, `_extract_last_token_hidden()`, `forward_prefix_with_collection()`, `next_token_logits()`
- **Inheritance**: Base class for Llama, Qwen, Phi-3 adapters

### `model_adapters.LlamaAdapter` (concrete)
- **Inherits**: `ModelAdapter`
- **Purpose**: Llama 3.2 3B specific implementation
- **Model structure**: `model.model.layers`, `model.model.embed_tokens`, `model.model.norm`, `model.lm_head`
- **Key method**: `forward_prefix_with_collection()` - Block-by-block traversal with collection

### `model_adapters.QwenAdapter` (concrete)
- **Inherits**: `ModelAdapter`
- **Purpose**: Qwen 2.5 7B specific implementation
- **Model structure**: Qwen-specific paths (introspect to find)

### `model_adapters.Phi3Adapter` (concrete)
- **Inherits**: `ModelAdapter`
- **Purpose**: Phi-3 Mini specific implementation
- **Model structure**: Phi-3-specific paths (introspect to find)

### `monitoring_loop.HallucinationMonitor`
- **Purpose**: Generation orchestrator with uncertainty monitoring
- **Attributes**:
  - `adapter: ModelAdapter` - Model adapter instance
  - `tokenizer: Any` - Tokenizer for encoding/decoding
  - `tau: float` - Uncertainty threshold
  - `lambda_: float` - Sigmoid steepness
  - `pfail_cutoff: float` - Halt threshold (default 0.85)
  - `max_tokens: int` - Maximum generation length
  - `check_every_k_tokens: int` - Monitoring frequency
  - `temperature: float` - Sampling temperature (0.0 for greedy)
- **Methods**: `generate_with_monitoring()`, `_should_halt()`
- **Output**: `GenerationResult` dict with trajectory

### `calibrate_thresholds.ThresholdCalibrator`
- **Purpose**: Grid search for optimal (τ, λ) parameters
- **Attributes**:
  - `adapter: ModelAdapter` - Model adapter
  - `tokenizer: Any` - Tokenizer
  - `train_data: List[HaluEvalSample]` - Training split
- **Methods**: `grid_search()`, `evaluate_params()`, `save_params()`
- **Output**: `CalibrationResult` with optimal params

## [Dependencies]

Python package requirements for the project.

**Already installed (confirmed):**
- `mlx==0.29.1` - MLX core framework
- `mlx-lm==0.28.0` - MLX language models
- `mlx-metal==0.29.1` - MLX Metal backend

**New dependencies to install:**
```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
tqdm>=4.62.0
requests>=2.26.0
```

**Rationale:**
- `numpy`: Minimal use for final metric conversions and sklearn compatibility
- `pandas`: Data aggregation and result analysis
- `scikit-learn`: AUROC, precision, recall, confusion matrix computation
- `matplotlib`: Visualization (ROC curves, trajectories, heatmaps) - NO seaborn
- `tqdm`: Progress bars for long-running calibration
- `requests`: HaluEval dataset download

**Installation command:**
```bash
pip3 install numpy pandas scikit-learn matplotlib tqdm requests
```

## [Testing]

Testing strategy and validation approach.

**Unit tests** (create `hbar-experiment/tests/` directory):

1. **`test_uncertainty_metrics.py`**
   - Test `compute_gini_impurity()` with known probability distributions
   - Test `compute_delta_mu_proxy()` edge cases (near-zero I_cat)
   - Test `compute_delta_sigma_proxy()` with synthetic hidden vectors
   - Test `compute_hbar_s()` and `compute_pfail()` numerical correctness
   - Validate sigmoid behavior at tau boundaries

2. **`test_hidden_state_collector.py`**
   - Test `start()` resets state correctly
   - Test `record()` stores vectors with correct indexing
   - Test `get_all_blocks()` returns ordered list
   - Test error handling for invalid shapes

3. **`test_model_adapters.py`**
   - Test `_locate_components()` for Llama adapter with known model
   - Test `_extract_last_token_hidden()` with different shape inputs
   - Integration test: single token generation with collection
   - Validate collector receives correct number of layer calls

4. **`test_monitoring_loop.py`**
   - Test halt condition triggers correctly
   - Test trajectory logging
   - Test max_tokens limit enforcement
   - Mock adapter to test monitoring logic in isolation

**Integration tests:**

5. **`test_end_to_end.py`**
   - Load small model (Llama 3.2 3B)
   - Generate on toy prompt with known hallucination characteristics
   - Verify P_fail exceeds threshold for hallucination case
   - Verify full pipeline runs without errors

**Validation tests:**

6. **`test_halueval_loader.py`**
   - Test deterministic sampling produces same 10K samples with seed=42
   - Test train/test split is reproducible
   - Verify data format matches expected structure

7. **`test_calibration.py`**
   - Test grid search produces valid parameter ranges
   - Test optimal params save/load correctly
   - Verify AUROC computation matches sklearn

**Manual validation:**
- Run Phase 1 on known hallucination example from literature
- Verify ℏₛ increases during hallucination segment
- Visual inspection of trajectory plots

## [Implementation Order]

Logical sequence of implementation steps to minimize conflicts and ensure successful integration.

**Phase 1: Core Infrastructure (Days 1-3)**

1. **Setup project structure**
   - Create `hbar-experiment/` directory
   - Create `__init__.py`, `config.py`, `requirements.txt`
   - Install dependencies
   - Initialize git tracking for experiment directory

2. **Implement `uncertainty_metrics.py`**
   - Pure functions, no dependencies on other modules
   - Write unit tests concurrently
   - Validate numerical correctness with known inputs

3. **Implement `hidden_state_collector.py`**
   - Simple stateful class
   - Write unit tests for lifecycle (start/record/get)
   - No model dependencies yet

4. **Implement `model_adapters.py` - Llama adapter only**
   - Define `ModelAdapter` abstract base class
   - Implement `LlamaAdapter` concrete class
   - Test with actual Llama 3.2 3B model loaded from MLX
   - Verify hidden state shapes and collection
   - **Critical validation**: Single-token generation with collection, print shapes

5. **Implement `monitoring_loop.py`**
   - Integrate adapter + collector + metrics
   - Test on toy prompt (e.g., "The capital of France is")
   - Verify trajectory logging and halt logic
   - **Validation**: Generate with high/low thresholds, observe behavior

**Phase 2: Calibration & Validation Pipeline (Days 4-6)**

6. **Implement `halueval_loader.py`**
   - Download dataset from GitHub
   - Implement sampling and splitting
   - Write tests for deterministic behavior
   - **Validation**: Print first 5 samples to verify format

7. **Extend `model_adapters.py` - Add Qwen and Phi-3**
   - Implement `QwenAdapter` by introspecting Qwen model structure
   - Implement `Phi3Adapter` by introspecting Phi-3 model structure
   - Test each adapter independently with single generation
   - **Critical**: Verify layer counts match expected architecture

8. **Implement `calibrate_thresholds.py`**
   - Grid search implementation with tqdm progress bars
   - Start with coarse grid (tau: 0.5-5.0 step 0.5, lambda: 1.0-10.0 step 1.0)
   - Test on small subset of training data first (100 samples)
   - Save/load functionality for calibrated params
   - **Validation**: Run calibration on Llama with 100 samples, verify best params saved

9. **Implement `validate.py`**
   - Load calibrated params
   - Run validation on test split
   - Compute all metrics with sklearn
   - **Validation**: Run on Llama with calibrated params, verify AUROC > 0.5

10. **Implement `multi_model_runner.py`**
    - Orchestrate all 3 models
    - Error handling for model loading failures
    - **Validation**: Run all models on 10 test samples, verify no crashes

**Phase 3: Analysis & Visualization (Days 7-8)**

11. **Implement `analyze_results.py`**
    - Aggregate results into pandas DataFrame
    - Cross-model comparison
    - Bootstrap confidence intervals

12. **Implement `generate_figures.py`**
    - ROC curves with matplotlib
    - ℏₛ trajectory plots
    - Threshold sensitivity heatmaps
    - **Validation**: Generate all figures, visual inspection

13. **Full pipeline execution**
    - Run calibration on all 3 models with full 5K training split
    - Run validation on all 3 models with full 5K test split
    - Generate all analysis outputs and figures
    - Document results in experiment report

14. **Refinement and optimization**
    - Fine-tune grid search ranges based on initial results
    - Add adaptive monitoring frequency if needed
    - Performance profiling and bottleneck identification
    - Document limitations and future work

**Critical checkpoints:**
- After step 4: Single model generates with hidden state collection working
- After step 5: Monitoring loop halts generation correctly
- After step 8: Calibration produces reasonable (τ, λ) values
- After step 10: All 3 models run end-to-end without ]\]\
- After step 13: Full results ready for analysis and publication

**Estimated timeline:**
- Phase 1: 3 days (focused on correctness)
- Phase 2: 3 days (includes calibration runtime)
- Phase 3: 2 days (analysis and visualization)
- Total: ~8 days for complete implementation and validation