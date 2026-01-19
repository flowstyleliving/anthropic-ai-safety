# Task Alignment Fix - Summary

## Problem Diagnosed

Your initial calibration showed **AUROC ≈ 0.51** (near-random), indicating the uncertainty metric wasn't separating hallucinations from correct answers. The root cause was **task misalignment**:

### Old Prompts (Continuation Task)
```
Context: ...
Question: ...
Answer: [given answer]
```
→ Model continues text after the answer  
→ Measuring: "How uncertain is the model about continuing this paragraph?"  
→ **Not aligned with hallucination detection**

### New Prompts (Judgment Task)
```
Context: ...
Question: ...
Proposed Answer: [given answer]

Is this answer faithful to the context? Answer 'Yes' or 'No':
```
→ Model judges if the answer is hallucinated  
→ Measuring: "How uncertain is the model about the correctness of this answer?"  
→ **Aligned with hallucination detection**

---

## Changes Made

### 1. **Fixed Prompting** (`halueval_loader.py`)
- Changed `format_sample()` to create judgment prompts
- Three task types:
  - **QA**: "Is this answer faithful to the context?"
  - **Dialogue**: "Is this response appropriate and factual?"
  - **Summarization**: "Is this summary faithful to the document?"
- All end with "Answer 'Yes' or 'No':"

### 2. **Added Diagnostics** (`calibrate_thresholds.py`)
Now prints score statistics by label:
```python
Hallucinated (label=1): mean=X, std=Y, n=Z
Correct (label=0):      mean=A, std=B, n=C
Mean difference:        D
Effect size (Cohen's d): E
```

**Effect size interpretation:**
- d < 0.2: Negligible separation (current suspected state)
- d ≈ 0.5: Medium separation (target)
- d > 0.8: Strong separation (ideal)

### 3. **Expanded τ Sweep**
Changed from `np.linspace(0.6, 0.99, 40)` to `np.linspace(0.01, 0.99, 99)`
- Old: Only swept upper quantiles → guaranteed low recall
- New: Full range → complete precision-recall curve

### 4. **Regenerated Dataset**
- Ran `python halueval_loader.py`
- Created 4,998 training samples with new prompts
- Balanced: ~50% hallucinated, ~50% correct

---

## Next Steps

### Immediate: Re-run Calibration

```bash
python calibrate_thresholds.py --n-samples 200 --max-tokens 20 --seed 42
```

**What to look for:**

1. **Score Statistics**
   ```
   Hallucinated (label=1): mean=2.8, std=0.5, n=103
   Correct (label=0):      mean=1.9, std=0.4, n=97
   Mean difference:        0.9        ← Should be > 0.3
   Effect size (Cohen's d): 1.95      ← Should be > 0.5
   ```

2. **AUROC**
   - Current: ~0.51 (random)
   - Target: > 0.65 (useful)
   - Good: > 0.75 (strong signal)

3. **Recall Achievement**
   - Old: No τ achieved recall ≥ 0.9
   - New: Should find τ with recall ≥ 0.9 (if signal exists)

### If AUROC is Still Low (~0.5-0.6)

This suggests the signal is weak even with correct task alignment. Possible causes:

**A. Metric Issue**
- Current: Mean of top-5 ℏₛ values
- Try: Max ℏₛ, or percentile-based scores
- Try: Different uncertainty metrics (Δμ, Δσ separately)

**B. Generation Length**
- Current: max_tokens=20
- Try: Longer generation (30-50 tokens) to capture more uncertainty

**C. Model Limitation**
- Llama 3.2 3B might not have sufficient knowledge for judgment task
- Try: Larger model (7B or 13B)

**D. Dataset Quality**
- HaluEval labels might not match model's judgment
- Inspect: Samples where model is "confidently wrong"

### If AUROC Improves (> 0.65)

🎉 Success! The task alignment worked. Next steps:

1. **Find optimal τ**
   - Use the calibrated threshold from output
   - Should now achieve recall ≥ 0.9 with reasonable precision

2. **Test on held-out data**
   ```bash
   # Run evaluation on test set
   python evaluate.py --test-data data/halueval/splits/test.json
   ```

3. **Analyze failure modes**
   - Which task types work best? (QA vs dialogue vs summarization)
   - Where does the model fail?

---

## Technical Details

### Prompt Examples

**QA Sample (Hallucinated)**
```
Context: The Rieder Automatic Rifle was a fully automatic Lee–Enfield...
Question: The Rieder Automatic Rifle was a model made by...
Proposed Answer: [hallucinated answer]

Is this answer faithful to the context? Answer 'Yes' or 'No':
```

**Expected behavior:**
- Hallucinated: Model uncertain → high ℏₛ → predicts "No" (but with high uncertainty)
- Correct: Model confident → low ℏₛ → predicts "Yes" (with low uncertainty)

### Why This Matters

The calibration measures **epistemic uncertainty during decision-making**, not during free-form generation. The model must:
1. Parse the context
2. Evaluate the proposed answer
3. Make a binary judgment

Uncertainty signals during this process correlate with factual errors.

---

## Files Modified

1. **halueval_loader.py** - Fixed prompt formatting
2. **calibrate_thresholds.py** - Added diagnostics, expanded τ sweep
3. **data/halueval/splits/train.json** - Regenerated with new prompts
4. **data/halueval/splits/test.json** - Regenerated with new prompts

---

## Quick Verification Command

```bash
# Check a few samples
python3 -c "
import json
with open('data/halueval/splits/train.json') as f:
    samples = json.load(f)
    
# Should end with judgment question
print(samples[0]['prompt'][-100:])
"
```

Should print something ending in: `"...Answer 'Yes' or 'No':"`

---

## Summary

**Before:** Continuation task → AUROC ~0.51 (random)  
**After:** Judgment task → AUROC expected to improve  

**Key insight:** You weren't measuring hallucination uncertainty—you were measuring continuation uncertainty. Now you are.

Run the calibration again and check the score statistics to confirm signal separation! 🎯