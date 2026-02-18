"""
TruthfulQA dataset loading and sampling.

Downloads TruthfulQA.csv from GitHub, parses into standardized samples,
and splits data for calibration and validation.
"""

import csv
import json
import os
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path

import requests
from tqdm import tqdm

import config


def download_file(url: str, output_path: str) -> None:
    """
    Download file from URL with progress bar.

    Args:
        url: Source URL
        output_path: Destination file path
    """
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        desc=os.path.basename(output_path),
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_truthfulqa(
    cache_dir: str = config.TRUTHFULQA_CACHE_DIR,
    url: str = config.TRUTHFULQA_URL,
) -> str:
    """
    Download TruthfulQA dataset CSV from GitHub.

    Args:
        cache_dir: Directory to store downloaded file
        url: Source CSV URL

    Returns:
        Path to downloaded CSV
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)

    output_path = cache_path / "TruthfulQA.csv"
    if output_path.exists():
        print(f"✓ TruthfulQA already cached at {output_path}")
        return str(output_path)

    print(f"Downloading TruthfulQA from {url}...")
    download_file(url, str(output_path))
    print(f"✓ Downloaded to {output_path}")
    return str(output_path)


def load_csv_rows(file_path: str) -> List[Dict[str, str]]:
    """
    Load CSV file into a list of dict rows.

    Args:
        file_path: Path to CSV

    Returns:
        List of row dictionaries
    """
    with open(file_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _split_answers(raw: Optional[str]) -> List[str]:
    if not raw:
        return []
    raw = raw.strip()
    if not raw:
        return []
    if ";" in raw:
        parts = [p.strip() for p in raw.split(";")]
    else:
        parts = [p.strip() for p in raw.split("\n")]
    return [p for p in parts if p]


def _first_nonempty(values: List[Optional[str]]) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        value = value.strip()
        if value:
            return value
    return None


def _get_answer_from_row(
    row: Dict[str, str],
    primary_key: str,
    fallback_key: str,
    rng: random.Random,
) -> Optional[str]:
    primary = row.get(primary_key)
    primary = primary.strip() if primary else None
    if primary:
        return primary

    fallback = row.get(fallback_key)
    candidates = _split_answers(fallback)
    if candidates:
        return rng.choice(candidates)
    return None


def format_sample(
    row: Dict[str, str],
    sample_id: int,
    is_hallucinated: bool,
    rng: random.Random,
) -> Dict[str, Any]:
    """
    Format TruthfulQA row into standardized structure.

    We create ONE formatted sample (either correct or hallucinated).

    Args:
        row: CSV row dict
        sample_id: Unique identifier
        is_hallucinated: True for incorrect answer, False for correct
        rng: Random instance for deterministic sampling

    Returns:
        Formatted sample with keys: prompt, label, id, task_type, metadata
    """
    question = row.get("Question") or row.get("question") or ""

    if is_hallucinated:
        answer = _get_answer_from_row(
            row=row,
            primary_key="Best Incorrect Answer",
            fallback_key="Incorrect Answers",
            rng=rng,
        )
    else:
        answer = _get_answer_from_row(
            row=row,
            primary_key="Best Answer",
            fallback_key="Correct Answers",
            rng=rng,
        )

    if not answer:
        raise ValueError("Missing answer in TruthfulQA row; check CSV columns.")

    prompt = (
        f"Question: {question}\n"
        f"Proposed Answer: {answer}\n\n"
        f"Is this answer truthful? Answer 'Yes' or 'No':"
    )

    label = 1 if is_hallucinated else 0
    suffix = "hal" if is_hallucinated else "cor"

    return {
        "prompt": prompt,
        "label": label,
        "id": f"truthfulqa_{sample_id}_{suffix}",
        "task_type": "truthfulqa",
        "metadata": row,
    }


def load_and_sample(
    file_path: str,
    n_samples: int = 1000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    Load TruthfulQA dataset and perform deterministic sampling.

    Args:
        file_path: Path to TruthfulQA.csv
        n_samples: Total number of samples to draw (correct + incorrect)
        seed: Random seed for reproducibility

    Returns:
        List of formatted samples
    """
    rng = random.Random(seed)
    rows = load_csv_rows(file_path)
    print(f"Loaded {len(rows)} TruthfulQA questions")

    if n_samples is None or n_samples <= 0:
        n_raw_samples = len(rows)
    else:
        n_raw_samples = min(len(rows), max(1, n_samples // 2))

    sampled = rng.sample(rows, n_raw_samples) if len(rows) > n_raw_samples else rows

    formatted: List[Dict[str, Any]] = []
    for i, row in enumerate(sampled):
        formatted.append(format_sample(row, i, is_hallucinated=False, rng=rng))
        formatted.append(format_sample(row, i, is_hallucinated=True, rng=rng))

    rng.shuffle(formatted)
    print(f"Created {len(formatted)} samples ({len(formatted)//2} correct + {len(formatted)//2} hallucinated)")
    return formatted


def split_train_test(
    data: List[Dict[str, Any]],
    train_ratio: float = 0.5,
    seed: int = 42,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Split data into train and test sets.

    Args:
        data: List of samples
        train_ratio: Fraction for training
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_data, test_data)
    """
    rng = random.Random(seed)
    shuffled = data.copy()
    rng.shuffle(shuffled)

    split_idx = int(len(shuffled) * train_ratio)
    train_data = shuffled[:split_idx]
    test_data = shuffled[split_idx:]

    train_pos = sum(1 for s in train_data if s["label"] == 1)
    test_pos = sum(1 for s in test_data if s["label"] == 1)

    print(f"\nTrain set: {len(train_data)} samples ({train_pos} hallucinations, {len(train_data)-train_pos} correct)")
    print(f"Test set: {len(test_data)} samples ({test_pos} hallucinations, {len(test_data)-test_pos} correct)")
    return train_data, test_data


def save_split(data: List[Dict[str, Any]], output_path: str) -> None:
    """
    Save data split to JSON file.

    Args:
        data: List of samples
        output_path: Output file path
    """
    with open(output_path, "w", encoding="utf-8") as f:
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
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Loaded {len(data)} samples from {input_path}")
    return data


if __name__ == "__main__":
    """
    Example usage: Download TruthfulQA, sample, and split.
    """
    print("=" * 80)
    print("TruthfulQA Dataset Preparation")
    print("=" * 80)
    print()

    csv_path = download_truthfulqa()
    print()

    all_samples = load_and_sample(csv_path, n_samples=1000, seed=42)
    train_data, test_data = split_train_test(all_samples, train_ratio=0.5, seed=42)

    output_dir = Path("./data/truthfulqa/splits")
    output_dir.mkdir(parents=True, exist_ok=True)

    save_split(train_data, str(output_dir / "train.json"))
    save_split(test_data, str(output_dir / "test.json"))

    print()
    print("✓ Dataset preparation complete!")
    print(f"  Train: {output_dir / 'train.json'}")
    print(f"  Test:  {output_dir / 'test.json'}")
