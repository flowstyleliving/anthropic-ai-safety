# Calibration Optimization - Fix Summary

## Problem
The `calibrate_thresholds.py` script was crashing due to:
1. **Redundant generation**: Running model for each τ value (13 × 100 samples = 1,300 generations)
2. **Quadratic memory growth**: Input sequences growing without KV cache
3. **Trajectory storage**: Large nested dicts with arrays accumulating in memory

**Sample size location**: Line 333 in `calibrate_thresholds.py`
```python
parser.add_argument("--n-samples", type=int, default=100, ...)
```

---

## Solution: Two-Pass Calibration

### **Pass 1: Precompute Scores (Expensive, Done Once)**
- Run generation once per sample
- Use lightweight `compute_score_only=True` mode
- Store only scalar ℏₛ scores (mean of top-5 values)
- No trajectory storage
- Memory cleanup every 50 samples

### **Pass 2: Sweep τ (Cheap)**
- Pure numpy operations on stored scores
- Evaluate 40+ τ values in seconds
- Data-driven τ range from score quantiles

**Performance improvement**: 
- Old: `O(#tau × #samples × generation_cost)`
- New: `O(#samples × generation_cost) + O(#tau × #samples)`
- ~13x speedup for 13 τ values

---

## Changes Made

### 1. `monitoring_loop.py`
Added `compute_score_only` parameter to `generate_with_monitoring()`:
- When `True`: accumulates only top-k ℏₛ values (no trajectory)
- Forces `check_every_k_tokens=1` for consistent calibration
- Returns single scalar `score` instead of full trajectory

### 2. `calibrate_thresholds.py`
- **New method**: `precompute_scores()` - Pass 1 with memory cleanup
- **Refactored**: `calibrate_tau()` - Two-pass approach with data-driven τ
- **Removed**: `evaluate_tau()` - No longer needed
- Added `gc.collect()` every 50 samples
- Quantile-based τ sweep: `np.quantile(scores, np.linspace(0.6, 0.99, 40))`

---

## Usage

### Quick Test (10-20 samples)
```bash
python calibrate_thresholds.py \
  --n-samples 10 \
  --max-tokens 20
```

### Full Calibration (100+ samples)
```bash
python calibrate_thresholds.py \
  --n-samples 100 \
  --max-tokens 20
```

### Large Scale (500+ samples)
```bash
python calibrate_thresholds.py \
  --n-samples 500 \
  --max-tokens 20
```

### Optional: Specify τ range (instead of auto-detection)
```bash
python calibrate_thresholds.py \
  --tau-start 2.0 \
  --tau-stop 4.0 \
  --tau-step 0.1
```

---

## Key Improvements

✅ **No redundant generation**: Model runs once per sample  
✅ **Minimal memory footprint**: Only scalar scores stored  
✅ **Data-driven τ sweep**: Automatically finds meaningful range  
✅ **Memory cleanup**: `gc.collect()` + `del result` after each sample  
✅ **Scalable**: Can now handle 500+ samples without crashing  

---

## Recommended Settings

| Scenario | n-samples | max-tokens | Notes |
|----------|-----------|------------|-------|
| Quick test | 10-20 | 20 | Debug/verify |
| Standard | 100-200 | 20 | Good balance |
| High quality | 500+ | 20-30 | Production |

**Why max-tokens=20?**
- Calibration focuses on early hallucination signals
- Reduces quadratic attention cost
- Still captures sufficient ℏₛ statistics (top-5 values)

---

## Verification Checklist

After running calibration:
- [ ] No trajectory dicts stored during Pass 1
- [ ] Score range printed (e.g., [1.2, 4.5])
- [ ] Hallucination rate printed (e.g., 45%)
- [ ] AUROC/PR-AUC computed once (independent of τ)
- [ ] Best τ selected with precision/recall metrics
- [ ] JSON saved to `./calibrated_params/`

---

## Troubleshooting

**If still seeing memory issues:**
1. Reduce `--n-samples` to 50
2. Reduce `--max-tokens` to 15
3. Check that `compute_score_only=True` is being used
4. Verify no trajectory storage with `print(result.keys())`

**If τ range seems wrong:**
- The script auto-detects from score quantiles
- Check "Score range" output in Pass 1
- Manually specify `--tau-start/stop` if needed

---

## Technical Notes

- **Top-k score**: Uses mean of top-5 ℏₛ values (more stable than p95 for max_tokens=20)
- **Halting disabled**: `pfail_cutoff=1.1` ensures full generation trajectories
- **Greedy sampling**: `temperature=0.0` for reproducible scores
- **Precision target**: Maximizes precision at recall ≥ 90% (safety-oriented)