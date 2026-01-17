"""
Phase 1 end-to-end test script.

Tests the core infrastructure with Llama 3.2 3B model on a simple prompt.
Validates that hidden state collection, uncertainty computation, and monitoring work correctly.
"""

import sys
import os
import mlx.core as mx
from mlx_lm import load

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hidden_state_collector
import model_adapters
import monitoring_loop

HiddenStateCollector = hidden_state_collector.HiddenStateCollector
create_adapter = model_adapters.create_adapter
HallucinationMonitor = monitoring_loop.HallucinationMonitor


def test_phase1():
    """Run Phase 1 validation test."""
    
    print("=" * 80)
    print("Phase 1 End-to-End Test")
    print("=" * 80)
    print()
    
    # Load model and tokenizer
    print("Loading Llama 3.2 3B Instruct model...")
    model_path = "mlx-community/Llama-3.2-3B-Instruct-4bit"
    
    try:
        model, tokenizer = load(model_path)
        print(f"✓ Model loaded successfully")
        print(f"  Model type: {type(model)}")
        print()
    except Exception as e:
        print(f"✗ Failed to load model: {e}")
        return
    
    # Create collector and adapter
    print("Creating adapter and collector...")
    try:
        collector = HiddenStateCollector()
        adapter = create_adapter(model, collector, model_type="llama")
        print(f"✓ Adapter created: {type(adapter).__name__}")
        print(f"  Number of transformer layers: {len(adapter.layers)}")
        print()
    except Exception as e:
        print(f"✗ Failed to create adapter: {e}")
        return
    
    # Create monitor with conservative parameters for testing
    print("Creating hallucination monitor...")
    monitor = HallucinationMonitor(
        adapter=adapter,
        tokenizer=tokenizer,
        tau=2.0,
        lambda_=5.0,
        pfail_cutoff=0.85,
        max_tokens=20,  # Short for testing
        temperature=0.0  # Greedy
    )
    print("✓ Monitor created")
    print()
    
    # Test prompts
    test_prompts = [
        "The capital of France is",
        "2 + 2 equals",
        "The first president of the United States was"
    ]
    
    print("Running test generations...")
    print("=" * 80)
    print()
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}/{len(test_prompts)}: \"{prompt}\"")
        print("-" * 80)
        
        try:
            result = monitor.generate_with_monitoring(prompt, verbose=True)
            
            print()
            print("Results:")
            print(f"  Generated text: {result['text']}")
            print(f"  Tokens generated: {len(result['tokens'])}")
            print(f"  Halted: {result['halted']}")
            print(f"  Halt reason: {result['halt_reason']}")
            if result['halt_step'] is not None:
                print(f"  Halt step: {result['halt_step']}")
            print(f"  Trajectory points: {len(result['trajectory'])}")
            
            if result['trajectory']:
                print()
                print("  Final metrics:")
                final_metrics = result['trajectory'][-1]
                print(f"    ℏₛ = {final_metrics['hbar_s']:.4f}")
                print(f"    P_fail = {final_metrics['pfail']:.4f}")
                print(f"    Δμ = {final_metrics['delta_mu']:.4f}")
                print(f"    Δσ = {final_metrics['delta_sigma']:.4f}")
                print(f"    I_cat = {final_metrics['I_cat']:.4f}")
            
            print()
            print("✓ Test completed successfully")
            
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        print()
        print("=" * 80)
        print()
    
    print("Phase 1 validation complete!")
    print()
    print("Next steps:")
    print("  1. Verify hidden state shapes are correct")
    print("  2. Check that ℏₛ values are reasonable")
    print("  3. Validate that P_fail increases with uncertainty")
    print("  4. Proceed to Phase 2 (dataset loading and calibration)")


if __name__ == "__main__":
    test_phase1()