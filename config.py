"""
Configuration constants for the hallucination detection experiment.
"""

# Model configurations
MODEL_CONFIGS = [
    {
        "name": "llama_3.2_3b",
        "path": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "display_name": "Llama 3.2 3B Instruct"
    },
    {
        "name": "qwen_2.5_7b",
        "path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "display_name": "Qwen 2.5 7B Instruct"
    },
    {
        "name": "phi_3_mini",
        "path": "mlx-community/Phi-3-mini-128k-instruct-4bit",
        "display_name": "Phi-3 Mini Instruct"
    }
]

# Dataset configuration
HALUEVAL_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
HALUEVAL_CACHE_DIR = "./data/halueval"
DATASET_SAMPLE_SIZE = 10000
DATASET_RANDOM_SEED = 42
TRAIN_TEST_SPLIT_RATIO = 0.5

# Uncertainty metric defaults
DEFAULT_EPSILON = 1e-8  # Numerical stability for delta_mu computation

# Monitoring defaults
DEFAULT_TAU = 2.0
DEFAULT_LAMBDA = 5.0
DEFAULT_PFAIL_CUTOFF = 0.85
DEFAULT_MAX_TOKENS = 512
DEFAULT_CHECK_EVERY_K_TOKENS = 1
DEFAULT_TEMPERATURE = 0.0  # Greedy sampling

# Calibration grid search ranges
CALIBRATION_TAU_RANGE = (0.1, 5.0, 0.1)  # (start, stop, step)
CALIBRATION_LAMBDA_RANGE = (0.5, 10.0, 0.5)  # (start, stop, step)
CALIBRATION_COARSE_TAU_RANGE = (0.5, 5.0, 0.5)  # For faster initial search
CALIBRATION_COARSE_LAMBDA_RANGE = (1.0, 10.0, 1.0)

# Output directories
CALIBRATED_PARAMS_DIR = "./calibrated_params"
RESULTS_DIR = "./results"
FIGURES_DIR = "./figures"

# Bootstrap parameters for confidence intervals
BOOTSTRAP_N_SAMPLES = 1000
CONFIDENCE_LEVEL = 0.95