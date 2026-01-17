"""Check where the Llama model is cached."""
from mlx_lm.utils import get_model_path
import os

model_name = "mlx-community/Llama-3.2-3B-Instruct-4bit"
path_tuple = get_model_path(model_name)
path = str(path_tuple[0]) if isinstance(path_tuple, tuple) else str(path_tuple)

print(f"Model: {model_name}")
print(f"Cache path: {path}")
print(f"Exists: {os.path.exists(path)}")

if os.path.exists(path):
    # Get size
    import subprocess
    result = subprocess.run(['du', '-sh', path], capture_output=True, text=True)
    print(f"Size: {result.stdout.strip()}")
    
    # List files
    print(f"\nFiles in {path}:")
    for f in os.listdir(path):
        fpath = os.path.join(path, f)
        size = os.path.getsize(fpath) / (1024*1024)  # MB
        print(f"  {f}: {size:.1f} MB")
else:
    print("Model not cached yet - will be downloaded on first use")