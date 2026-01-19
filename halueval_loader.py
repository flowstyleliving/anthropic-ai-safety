"""
HaluEval 2.0 dataset loading and sampling.

Downloads dataset from original GitHub source, performs deterministic sampling,
and splits data for calibration and validation.
"""

import json
import os
import random
from typing import List, Dict, Any, Tuple
from pathlib import Path
import requests
from tqdm import tqdm


# HaluEval 2.0 GitHub URLs
HALUEVAL_URLS = {
    "qa": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/qa_data.json",
    "dialogue": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/dialogue_data.json",
    "summarization": "https://raw.githubusercontent.com/RUCAIBox/HaluEval/main/data/summarization_data.json"
}


def download_file(url: str, output_path: str) -> None:
    """
    Download file from URL with progress bar.
    
    Args:
        url: Source URL
        output_path: Destination file path
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(output_path, 'wb') as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_halueval(cache_dir: str = "./data/halueval") -> Dict[str, str]:
    """
    Download HaluEval 2.0 dataset from GitHub.
    
    Args:
        cache_dir: Directory to store downloaded files
        
    Returns:
        Dict mapping task type to file path
        
    Raises:
        requests.RequestException: If download fails
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    downloaded_files = {}
    
    for task_type, url in HALUEVAL_URLS.items():
        output_path = cache_path / f"{task_type}_data.json"
        
        # Skip if already downloaded
        if output_path.exists():
            print(f"✓ {task_type} data already cached at {output_path}")
            downloaded_files[task_type] = str(output_path)
            continue
        
        print(f"Downloading {task_type} data from {url}...")
        try:
            download_file(url, str(output_path))
            downloaded_files[task_type] = str(output_path)
            print(f"✓ Downloaded to {output_path}")
        except Exception as e:
            print(f"✗ Failed to download {task_type}: {e}")
            raise
    
    return downloaded_files


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    Load JSONL file (one JSON object per line) and return list of samples.
    
    Args:
        file_path: Path to JSONL file
        
    Returns:
        List of sample dictionaries
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON line: {e}")
                    continue
    
    return data


def format_sample(sample: Dict[str, Any], task_type: str, sample_id: int, is_hallucinated: bool) -> Dict[str, Any]:
    """
    Format HaluEval sample into standardized structure.
    
    HaluEval provides both correct and hallucinated answers per sample.
    This function creates ONE formatted sample (either correct or hallucinated).
    
    **JUDGMENT TASK**: The prompt asks the model to judge if the answer is hallucinated.
    We measure uncertainty during this judgment, not during free-form continuation.
    
    Args:
        sample: Raw sample from HaluEval JSON
        task_type: One of "qa", "dialogue", "summarization"
        sample_id: Unique sample identifier
        is_hallucinated: True to use hallucinated answer, False for correct
        
    Returns:
        Formatted sample with keys: prompt, label, id, metadata
    """
    # Build judgment prompt based on task type
    if task_type == "qa":
        question = sample.get("question", "")
        context = sample.get("knowledge", "")
        answer = sample.get("hallucinated_answer" if is_hallucinated else "right_answer", "")
        prompt = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"Proposed Answer: {answer}\n\n"
            f"Is this answer faithful to the context? Answer 'Yes' or 'No':"
        )
    elif task_type == "dialogue":
        dialogue = sample.get("dialogue", "")
        response = sample.get("hallucinated_response" if is_hallucinated else "right_response", "")
        prompt = (
            f"Dialogue: {dialogue}\n"
            f"Response: {response}\n\n"
            f"Is this response appropriate and factual given the dialogue? Answer 'Yes' or 'No':"
        )
    elif task_type == "summarization":
        document = sample.get("document", "")
        summary = sample.get("hallucinated_summary" if is_hallucinated else "right_summary", "")
        prompt = (
            f"Document: {document}\n"
            f"Summary: {summary}\n\n"
            f"Is this summary faithful to the document? Answer 'Yes' or 'No':"
        )
    else:
        prompt = str(sample)
    
    # Label: 1 = hallucination, 0 = correct
    label = 1 if is_hallucinated else 0
    suffix = "hal" if is_hallucinated else "cor"
    
    return {
        "prompt": prompt,
        "label": label,
        "id": f"{task_type}_{sample_id}_{suffix}",
        "task_type": task_type,
        "metadata": sample  # Keep original for reference
    }


def load_and_sample(
    dataset_paths: Dict[str, str],
    n_samples: int = 10000,
    seed: int = 42
) -> List[Dict[str, Any]]:
    """
    Load HaluEval dataset and perform deterministic sampling.
    
    Args:
        dataset_paths: Dict mapping task type to file path
        n_samples: Total number of samples to draw (distributed across tasks)
        seed: Random seed for reproducibility
        
    Returns:
        List of formatted samples
    """
    random.seed(seed)
    
    all_samples = []
    samples_per_task = n_samples // len(dataset_paths)
    
    for task_type, file_path in dataset_paths.items():
        print(f"Loading {task_type} data from {file_path}...")
        raw_data = load_json_file(file_path)
        print(f"  Loaded {len(raw_data)} samples")
        
        # Each raw sample has BOTH correct and hallucinated versions
        # We want samples_per_task PAIRS, so sample half as many raw samples
        n_raw_samples = samples_per_task // 2
        
        # Sample deterministically
        if len(raw_data) > n_raw_samples:
            sampled = random.sample(raw_data, n_raw_samples)
        else:
            sampled = raw_data
        
        # Create BOTH correct and hallucinated versions of each sample
        formatted = []
        for i, sample in enumerate(sampled):
            # Add correct version (label=0)
            formatted.append(format_sample(sample, task_type, i, is_hallucinated=False))
            # Add hallucinated version (label=1)
            formatted.append(format_sample(sample, task_type, i, is_hallucinated=True))
        
        all_samples.extend(formatted)
        print(f"  Created {len(formatted)} samples ({len(formatted)//2} correct + {len(formatted)//2} hallucinated)")
    
    # Shuffle combined dataset
    random.shuffle(all_samples)
    
    print(f"\nTotal samples: {len(all_samples)}")
    return all_samples


def split_train_test(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.5,
    seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train and test sets.
    
    Args:
        data: List of samples
        train_ratio: Fraction for training (default 0.5 for 50/50 split)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_data, test_data)
    """
    random.seed(seed)
    
    # Shuffle for good measure
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    # Split
    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]
    
    # Print statistics
    train_pos = sum(1 for s in train_data if s["label"] == 1)
    test_pos = sum(1 for s in test_data if s["label"] == 1)
    
    print(f"\nTrain set: {len(train_data)} samples ({train_pos} hallucinations, {len(train_data)-train_pos} correct)")
    print(f"Test set: {len(test_data)} samples ({test_pos} hallucinations, {test_pos} correct)")
    
    return train_data, test_data


def save_split(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save data split to JSON file.
    
    Args:
        data: List of samples
        output_path: Output file path
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {output_path}")


def load_split(input_path: str) -> List[Dict[str, Any]]:
    """
    Load data split from JSON file.
    
    Args:
        input_path: Input file path
        
    Returns:
        List of samples
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {input_path}")
    return data


if __name__ == "__main__":
    """
    Example usage: Download dataset, sample, and split.
    """
    # Download dataset
    print("=" * 80)
    print("HaluEval 2.0 Dataset Preparation")
    print("=" * 80)
    print()
    
    dataset_paths = download_halueval()
    print()
    
    # Sample and split
    all_samples = load_and_sample(dataset_paths, n_samples=10000, seed=42)
    train_data, test_data = split_train_test(all_samples, train_ratio=0.5, seed=42)
    
    # Save splits
    output_dir = Path("./data/halueval/splits")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    save_split(train_data, str(output_dir / "train.json"))
    save_split(test_data, str(output_dir / "test.json"))
    
    print()
    print("✓ Dataset preparation complete!")
    print(f"  Train: {output_dir / 'train.json'}")
    print(f"  Test: {output_dir / 'test.json'}")