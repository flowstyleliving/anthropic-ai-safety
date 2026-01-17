"""
Test logit parity between model.__call__ and manual block traversal.
"""
import mlx.core as mx
from mlx_lm import load
from mlx_lm.models.llama import create_attention_mask

print("Loading model...")
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Test with proper mask and cache
prompt = "The capital of France is"
print(f"\nTesting: '{prompt}'")
print("=" * 60)

input_ids = mx.array(tokenizer.encode(prompt))
if input_ids.ndim == 1:
    input_ids = input_ids[None, :]

print(f"Input shape: {input_ids.shape}")

# 1. Built-in
logits_builtin = model(input_ids)
print(f"Built-in logits shape: {logits_builtin.shape}")

# 2. Manual with proper mask and cache
print("\nManual traversal with mask and cache...")
x = model.model.embed_tokens(input_ids)

# Create cache (list of None for each layer)
cache = [None] * len(model.model.layers)

# Create attention mask
mask = create_attention_mask(x, cache[0])
print(f"Mask: {mask}")

for layer, c in zip(model.model.layers, cache):
    x = layer(x, mask, cache=c)

x = model.model.norm(x)
manual_logits = model.model.embed_tokens.as_linear(x)
print(f"Manual logits shape: {manual_logits.shape}")

# Compare
print(f"\nBuilt-in logits [0, -1, :5]: {logits_builtin[0, -1, :5]}")
print(f"Manual logits [0, -1, :5]: {manual_logits[0, -1, :5]}")

diff = mx.abs(logits_builtin - manual_logits).max()
print(f"Max difference: {diff.item()}")

if diff.item() < 1e-2:  # Allow for float16 precision
    print("✓ PARITY ACHIEVED!")
else:
    print("✗ Parity check failed")

# Test on second prompt
print("\n" + "=" * 60)
prompt2 = "2 + 2 equals"
print(f"\nTesting: '{prompt2}'")
input_ids2 = mx.array(tokenizer.encode(prompt2))
if input_ids2.ndim == 1:
    input_ids2 = input_ids2[None, :]

logits_builtin2 = model(input_ids2)
x2 = model.model.embed_tokens(input_ids2)
cache2 = [None] * len(model.model.layers)
mask2 = create_attention_mask(x2, cache2[0])

for layer, c in zip(model.model.layers, cache2):
    x2 = layer(x2, mask2, cache=c)

x2 = model.model.norm(x2)
manual_logits2 = model.model.embed_tokens.as_linear(x2)

diff2 = mx.abs(logits_builtin2 - manual_logits2).max()
print(f"Max difference: {diff2.item()}")

if diff2.item() < 1e-2:
    print("✓ PARITY ACHIEVED!")
else:
    print("✗ Parity check failed")