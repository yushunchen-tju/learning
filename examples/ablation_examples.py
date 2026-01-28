#!/usr/bin/env python
"""
Quick Example: FFT Pathformer Ablation Study

This script demonstrates how to quickly run ablation experiments
and compare results.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from openstl.models.fft_pathformer_ablation import (
    get_ablation_model, 
    get_ablation_table,
    count_parameters
)
from configs.ablation_config import print_all_configs, print_experiment_plan


def example_1_basic_usage():
    """Example 1: Basic model creation and forward pass."""
    print("\n" + "="*80)
    print("Example 1: Basic Usage")
    print("="*80)
    
    # Create models
    full_model = get_ablation_model('full', dim=256)
    ablation_model = get_ablation_model('no_spectral_se', dim=256)
    
    # Forward pass
    x = torch.randn(2, 256, 32, 32)
    
    with torch.no_grad():
        y_full = full_model(x)
        y_ablation = ablation_model(x)
    
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Full model output: {tuple(y_full.shape)}")
    print(f"Ablation model output: {tuple(y_ablation.shape)}")
    print(f"\nFull model parameters: {count_parameters(full_model):,}")
    print(f"Ablation model parameters: {count_parameters(ablation_model):,}")


def example_2_compare_variants():
    """Example 2: Compare multiple variants."""
    print("\n" + "="*80)
    print("Example 2: Compare Multiple Variants")
    print("="*80)
    
    variants = ['full', 'no_spectral_se', 'no_spectral_selector', 'single_scale']
    
    x = torch.randn(4, 256, 32, 32)
    
    print(f"\n{'Variant':<25} {'Parameters':<15} {'Output Shape'}")
    print("-"*80)
    
    for variant in variants:
        model = get_ablation_model(variant, dim=256)
        params = count_parameters(model)
        
        with torch.no_grad():
            y = model(x)
        
        print(f"{variant:<25} {params:<15,} {tuple(y.shape)}")


def example_3_branch_usage():
    """Example 3: Analyze branch selection patterns."""
    print("\n" + "="*80)
    print("Example 3: Branch Usage Analysis")
    print("="*80)
    
    # Create model with spectral routing
    model = get_ablation_model('full', dim=256)
    model.eval()
    
    # Run multiple forward passes
    num_samples = 50
    print(f"\nRunning {num_samples} forward passes...")
    
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(1, 256, 32, 32)
            _ = model(x)
    
    # Print usage statistics
    print("\nBranch selection statistics:")
    print(model.get_usage_str())


def example_4_efficiency_comparison():
    """Example 4: Compare efficiency metrics."""
    print("\n" + "="*80)
    print("Example 4: Efficiency Comparison")
    print("="*80)
    
    import time
    
    variants = {
        'full': 'Full model with all components',
        'no_sparsity': 'All branches (no top-k)',
        'single_scale': 'Single scale only'
    }
    
    x = torch.randn(4, 256, 32, 32)
    
    print(f"\n{'Variant':<20} {'Params (M)':<12} {'Time (ms)':<12} {'Description'}")
    print("-"*80)
    
    for variant, desc in variants.items():
        model = get_ablation_model(variant, dim=256)
        model.eval()
        
        params = count_parameters(model) / 1e6
        
        # Measure time (warmup + timing)
        with torch.no_grad():
            for _ in range(10):  # Warmup
                _ = model(x)
            
            start = time.time()
            for _ in range(50):
                _ = model(x)
            end = time.time()
        
        avg_time = (end - start) / 50 * 1000  # ms
        
        print(f"{variant:<20} {params:<12.2f} {avg_time:<12.2f} {desc}")


def example_5_custom_configuration():
    """Example 5: Create models with custom configurations."""
    print("\n" + "="*80)
    print("Example 5: Custom Configuration")
    print("="*80)
    
    # Create model with custom parameters
    model = get_ablation_model(
        variant='full',
        dim=512,  # Larger dimension
        mlp_ratio=2.0,  # Smaller MLP ratio
        drop=0.1,  # With dropout
        drop_path=0.2,
        patch_sizes=[4, 8],  # Only 2 scales
        top_k=1  # Select only 1 branch
    )
    
    print(f"Model parameters: {count_parameters(model):,}")
    
    # Test with larger input
    x = torch.randn(2, 512, 64, 64)
    
    with torch.no_grad():
        y = model(x)
    
    print(f"Input shape: {tuple(x.shape)}")
    print(f"Output shape: {tuple(y.shape)}")


def example_6_show_all_info():
    """Example 6: Display all available information."""
    print("\n" + "="*80)
    print("Example 6: Display All Available Information")
    print("="*80)
    
    # Show ablation table
    print("\n1. Model Comparison Table:")
    get_ablation_table()
    
    # Show configurations
    print("\n2. Experiment Configurations:")
    print_all_configs()
    
    # Show experiment plan
    print("\n3. Recommended Experiment Plan:")
    print_experiment_plan()


def main():
    """Run all examples."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  FFT Pathformer Ablation Study - Quick Examples".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    
    examples = [
        example_1_basic_usage,
        example_2_compare_variants,
        example_3_branch_usage,
        example_4_efficiency_comparison,
        example_5_custom_configuration,
        example_6_show_all_info,
    ]
    
    for i, example_func in enumerate(examples, 1):
        try:
            example_func()
        except Exception as e:
            print(f"\n✗ Example {i} failed: {str(e)}")
            continue
    
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  All examples completed!".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80 + "\n")
    
    # Print usage instructions
    print("Next Steps:")
    print("-" * 80)
    print("1. Run full evaluation: python tools/ablation_eval.py --config all")
    print("2. Test specific variant: python tools/ablation_eval.py --config no_spectral_se")
    print("3. Analyze branches: python tools/ablation_eval.py --config full --analyze_branches")
    print("4. Read documentation: docs/ABLATION_STUDY.md")
    print("-" * 80 + "\n")


if __name__ == "__main__":
    main()
