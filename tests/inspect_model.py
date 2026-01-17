"""Quick script to inspect Llama model structure."""
from mlx_lm import load

print("Loading model...")
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

print("\nModel keys (dictionary interface):")
if hasattr(model, 'keys'):
    for key in model.keys():
        print(f"  {key}: {type(model[key])}")

print("\nmodel.model keys:")
if hasattr(model, 'model') and hasattr(model.model, 'keys'):
    for key in model.model.keys():
        val = model.model[key]
        print(f"  {key}: {type(val)}")

print("\nTrying direct access:")
print(f"  model['model']: {model.get('model', 'NOT FOUND')}")
print(f"  model['lm_head']: {model.get('lm_head', 'NOT FOUND')}")

print("\nChecking __call__ method:")
print(f"  Callable: {callable(model)}")

# Try to understand how the model generates logits
print("\nLet's test a forward pass:")
import mlx.core as mx
test_ids = mx.array([[1, 2, 3]])
try:
    output = model(test_ids)
    print(f"  model(test_ids) output shape: {output.shape if hasattr(output, 'shape') else type(output)}")
    print(f"  Output type: {type(output)}")
except Exception as e:
    print(f"  Error: {e}")