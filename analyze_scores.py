"""
Quick diagnostic: Analyze score separation by label.
"""

import json
import numpy as np

# Load calibration result (if available)
try:
    with open('calibrated_params/llama_3.2_3b.json', 'r') as f:
        data = json.load(f)
    
    # Extract scores from all_results (they're embedded in tau sweep)
    # We need to re-run precompute to get raw scores...
    print("Note: Need raw scores from precompute_scores() to analyze properly")
    print("Run calibration first to generate score data")
except:
    print("No calibration data found")

# Alternative: Add to calibrate_thresholds.py
print("\n" + "="*80)
print("ADD THIS TO calibrate_tau() AFTER precompute_scores():")
print("="*80)
print("""
# Diagnostic: Score statistics by label
hal_scores = scores[labels == 1]
cor_scores = scores[labels == 0]

print(f"Score Statistics by Label:")
print(f"  Hallucinated (label=1): mean={hal_scores.mean():.3f}, std={hal_scores.std():.3f}, n={len(hal_scores)}")
print(f"  Correct (label=0):      mean={cor_scores.mean():.3f}, std={cor_scores.std():.3f}, n={len(cor_scores)}")
print(f"  Mean difference: {hal_scores.mean() - cor_scores.mean():.3f}")
print(f"  Effect size (Cohen's d): {(hal_scores.mean() - cor_scores.mean()) / np.sqrt((hal_scores.std()**2 + cor_scores.std()**2) / 2):.3f}")
print()
""")