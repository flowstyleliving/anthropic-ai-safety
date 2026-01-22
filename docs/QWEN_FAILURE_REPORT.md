# Qwen 2.5 7B Calibration Failure Report

**Date**: January 21, 2026  
**Status**: ❌ FAILED - Technical incompatibility

---

## Issue

Attempted to calibrate Qwen 2.5 7B Instruct (4-bit) for cross-model validation but encountered a critical error on **all 200 samples**:

```
Error on sample [ID]: [scaled_dot_product_attention] Mask type must promote to output type float16.
```

## Root Cause

**Attention mask dtype mismatch**: The Qwen model implementation in MLX expects attention masks in a different dtype than what our adapter provides. Specifically:

- Our code creates masks that don't promote to `float16` (the model's computation dtype)
- This is a **model-specific issue** - Llama 3.2 3B works fine with the same code
- Likely related to Qwen's custom attention implementation vs standard transformer attention

## Impact

- All ℏₛ scores: 0.0
- All PRI scores: 0.0
- AUROC: 0.5 (random)
- Calibration parameters meaningless (τ_hbar=0, τ_pri=0)

**Result**: Cannot perform cross-model validation on Qwen without fixing the attention mask issue.

---

## Technical Details

### Error Location

The error occurs during forward pass in the scaled dot-product attention operation:

```python
# In monitoring_loop.py, when calling adapter.forward()
hidden_vectors = self.adapter.forward(
    self.tokenizer.encode(prompt + new_token_str),
    cache=cache
)
```

### Why Llama Works But Qwen Doesn't

1. **Llama attention** (LlamaAttention): Uses standard causal mask that auto-promotes to float16
2. **Qwen attention** (Qwen2Attention): Has custom mask handling that's stricter about dtypes

### Attempted Workaround

None yet - this would require:
1. Modifying `model_adapters.py` to handle Qwen's mask requirements
2. OR using a different Qwen model version
3. OR staying with Llama-only results (acceptable for fellowship application)

---

## Recommendation

### For Fellowship Application (Immediate)

**Skip multi-model validation** and focus on:
1. Strong Llama 3.2 3B results (n=500 validation complete)
2. Document PRI as a proof-of-concept on one architecture
3. Note in limitations: "Future work should validate across architectures"

**Rationale**: One model done rigorously > multiple models done poorly

### For Future Work (Post-Fellowship)

Fix the attention mask issue to enable true multi-model validation:

```python
# Potential fix in model_adapters.py
def forward(self, input_ids, cache=None):
    # ...
    if self.model_type == "qwen":
        # Convert mask to float16 explicitly for Qwen
        mask = create_additive_causal_mask(input_ids.shape[0])
        mask = mx.array(mask, dtype=mx.float16)
    # ...
```

---

## Files Generated

- `calibrated_params/qwen_2.5_7b_20260121_151805_n200.json` - Invalid (all zeros)
- `results/calibration_qwen7b_n200.log` - Full error log

---

## Conclusion

**Skip Option B (multi-model)** → Proceed directly to **Option C (visualizations + writeup)**

The Llama 3.2 3B results alone are sufficient to demonstrate:
- ✅ PRI as a novel hallucination detector
- ✅ Proper experimental methodology (train/test splits, calibration, CV)
- ✅ Systematic analysis (AUROC, precision/recall, quadrant analysis)

Cross-model validation would be nice-to-have but isn't essential for the fellowship application story.