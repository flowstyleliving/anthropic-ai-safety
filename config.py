"""
Configuration constants for the hallucination detection experiment.
"""

# Model configurations
MODEL_CONFIGS = [
    {
        "name": "llama_3.2_3b",
        "path": "mlx-community/Llama-3.2-3B-Instruct-4bit",
        "display_name": "Llama 3.2 3B Instruct",
        "model_type": "llama"
    },
    {
        "name": "llama_3.1_8b",
        "path": "mlx-community/Llama-3.1-8B-Instruct-4bit",
        "display_name": "Llama 3.1 8B Instruct",
        "model_type": "llama"
    },
    {
        "name": "llama_3.3_70b",
        "path": "mlx-community/Llama-3.3-70B-Instruct-4bit",
        "display_name": "Llama 3.3 70B Instruct",
        "model_type": "llama"
    },
    {
        "name": "qwen_2.5_7b",
        "path": "mlx-community/Qwen2.5-7B-Instruct-4bit",
        "display_name": "Qwen 2.5 7B Instruct",
        "model_type": "qwen"
    },
    {
        "name": "qwen_2.5_14b",
        "path": "mlx-community/Qwen2.5-14B-Instruct-4bit",
        "display_name": "Qwen 2.5 14B Instruct",
        "model_type": "qwen"
    },
    {
        "name": "qwen_2.5_32b",
        "path": "mlx-community/Qwen2.5-32B-Instruct-4bit",
        "display_name": "Qwen 2.5 32B Instruct",
        "model_type": "qwen"
    },
    {
        "name": "phi_3_mini",
        "path": "mlx-community/Phi-3-mini-128k-instruct-4bit",
        "display_name": "Phi-3 Mini Instruct",
        "model_type": "phi3"
    },
    {
        "name": "mistral_7b",
        "path": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
        "display_name": "Mistral 7B Instruct",
        "model_type": "mistral"
    },
    {
        "name": "deepseek_r1_distill_qwen_32b",
        "path": "mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit",
        "display_name": "DeepSeek R1 Distill Qwen 32B",
        "model_type": "qwen"
    },
    {
        "name": "smollm_360m",
        "path": "mlx-community/SmolLM-360M-Instruct",
        "display_name": "SmolLM 360M Instruct",
        "model_type": "smollm"
    },
    {
        "name": "dolphin_2.9.4_llama3.1_8b",
        "path": "mlx-community/dolphin-2.9.4-llama3.1-8b",
        "display_name": "Dolphin 2.9.4 Llama 3.1 8B",
        "model_type": "llama"
    },
    {
        "name": "qwen3_coder_30b_a3b",
        "path": "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit",
        "display_name": "Qwen3 Coder 30B A3B Instruct",
        "model_type": "qwen"
    },
    {
        "name": "llava_phi3_mini",
        "path": "mlx-community/llava-phi-3-mini-4bit",
        "display_name": "LLaVA Phi-3 Mini (MLX-VLM)",
        "model_type": "llava"
    }
]

# Embedding / encoder models (non-MLX-LM)
EMBEDDING_MODEL_CONFIGS = [
    {
        "name": "all_minilm_l6_v2",
        "path": "mlx-community/all-MiniLM-L6-v2-bf16",
        "display_name": "BERT-style MiniLM L6 v2 (embeddings)",
        "model_type": "bert"
    }
]

# Dataset configuration
HALUEVAL_URL = "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json"
HALUEVAL_CACHE_DIR = "./data/halueval"
TRUTHFULQA_URL = "https://raw.githubusercontent.com/sylinrl/TruthfulQA/main/TruthfulQA.csv"
TRUTHFULQA_CACHE_DIR = "./data/truthfulqa"
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
