"""
Ablation Experiment Evaluation Script

This script provides utilities for running ablation experiments,
collecting metrics, and generating comparison reports.

Usage:
    python tools/ablation_eval.py --config full
    python tools/ablation_eval.py --config all --save_results results.json
"""

import torch
import torch.nn as nn
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict

import sys
sys.path.append(str(Path(__file__).parent.parent))

from openstl.models.fft_pathformer_ablation import get_ablation_model, count_parameters
from configs.ablation_config import (
    ABLATION_CONFIGS, 
    get_ablation_config,
    get_configs_by_group,
    EXPERIMENT_GROUPS
)


# ============================================================================
# Metrics Collection
# ============================================================================

class AblationMetrics:
    """Collects and stores metrics for ablation experiments."""
    
    def __init__(self):
        self.metrics = defaultdict(dict)
    
    def add_model_metrics(self, variant_name: str, model: nn.Module, 
                         input_shape: Tuple[int, int, int, int]):
        """
        Collect metrics for a model variant.
        
        Args:
            variant_name: Name of the variant
            model: Model instance
            input_shape: (B, C, H, W) input tensor shape
        """
        device = next(model.parameters()).device
        
        # 1. Parameter count
        params = count_parameters(model)
        self.metrics[variant_name]['parameters'] = params
        
        # 2. Model size (MB)
        param_size = sum([p.nelement() * p.element_size() for p in model.parameters()])
        buffer_size = sum([b.nelement() * b.element_size() for b in model.buffers()])
        model_size_mb = (param_size + buffer_size) / 1024**2
        self.metrics[variant_name]['model_size_mb'] = model_size_mb
        
        # 3. Inference time (averaged over multiple runs)
        model.eval()
        x = torch.randn(*input_shape).to(device)
        
        # Warmup
        with torch.no_grad():
            for _ in range(10):
                _ = model(x)
        
        # Timing
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start_time = time.time()
        
        num_runs = 100
        with torch.no_grad():
            for _ in range(num_runs):
                _ = model(x)
        
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        end_time = time.time()
        
        avg_time_ms = (end_time - start_time) / num_runs * 1000
        self.metrics[variant_name]['inference_time_ms'] = avg_time_ms
        
        # 4. Throughput (samples per second)
        throughput = input_shape[0] / (avg_time_ms / 1000)
        self.metrics[variant_name]['throughput'] = throughput
        
        # 5. FLOPs (if available)
        try:
            from fvcore.nn import FlopCountAnalysis
            flops = FlopCountAnalysis(model, x)
            total_flops = flops.total()
            self.metrics[variant_name]['flops'] = total_flops
            self.metrics[variant_name]['gflops'] = total_flops / 1e9
        except:
            self.metrics[variant_name]['flops'] = "N/A"
            self.metrics[variant_name]['gflops'] = "N/A"
        
        return self.metrics[variant_name]
    
    def get_metrics(self, variant_name: str) -> Dict[str, Any]:
        """Get metrics for a specific variant."""
        return self.metrics.get(variant_name, {})
    
    def get_all_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get all collected metrics."""
        return dict(self.metrics)
    
    def compare_to_baseline(self, baseline: str = 'full') -> Dict[str, Dict[str, float]]:
        """
        Compare all variants to baseline.
        
        Returns:
            Dictionary mapping variant names to relative differences
        """
        if baseline not in self.metrics:
            raise ValueError(f"Baseline '{baseline}' not found in metrics")
        
        baseline_metrics = self.metrics[baseline]
        comparisons = {}
        
        for variant, metrics in self.metrics.items():
            if variant == baseline:
                continue
            
            comparison = {}
            for key, value in metrics.items():
                if key in baseline_metrics and isinstance(value, (int, float)):
                    baseline_val = baseline_metrics[key]
                    if baseline_val != 0:
                        # Compute percentage difference
                        diff = ((value - baseline_val) / baseline_val) * 100
                        comparison[f'{key}_diff_%'] = diff
            
            comparisons[variant] = comparison
        
        return comparisons


# ============================================================================
# Reporting
# ============================================================================

def print_metrics_table(metrics: AblationMetrics, baseline: str = 'full'):
    """Print a formatted comparison table of all metrics."""
    
    all_metrics = metrics.get_all_metrics()
    if not all_metrics:
        print("No metrics available")
        return
    
    print("\n" + "="*120)
    print("FFT Pathformer Ablation Study - Performance Metrics")
    print("="*120)
    
    # Header
    header = f"{'Variant':<25} {'Params (M)':<15} {'Size (MB)':<12} {'Time (ms)':<12} {'GFLOPs':<12} {'Throughput':<12}"
    print(header)
    print("-"*120)
    
    # Baseline first
    if baseline in all_metrics:
        m = all_metrics[baseline]
        params_m = m['parameters'] / 1e6
        print(f"{baseline:<25} {params_m:<15.2f} {m['model_size_mb']:<12.2f} "
              f"{m['inference_time_ms']:<12.2f} {m.get('gflops', 'N/A'):<12} "
              f"{m['throughput']:<12.2f}")
        print("-"*120)
    
    # Other variants
    for variant, m in all_metrics.items():
        if variant == baseline:
            continue
        params_m = m['parameters'] / 1e6
        gflops_str = f"{m['gflops']:.2f}" if isinstance(m.get('gflops'), float) else "N/A"
        print(f"{variant:<25} {params_m:<15.2f} {m['model_size_mb']:<12.2f} "
              f"{m['inference_time_ms']:<12.2f} {gflops_str:<12} "
              f"{m['throughput']:<12.2f}")
    
    print("="*120)
    
    # Comparison to baseline
    if baseline in all_metrics:
        print("\nRelative to Baseline (%):")
        print("-"*120)
        
        comparisons = metrics.compare_to_baseline(baseline)
        for variant, comp in comparisons.items():
            params_diff = comp.get('parameters_diff_%', 0)
            time_diff = comp.get('inference_time_ms_diff_%', 0)
            
            print(f"{variant:<25} Params: {params_diff:+.1f}%  "
                  f"Time: {time_diff:+.1f}%")
    
    print("="*120 + "\n")


def save_results(metrics: AblationMetrics, output_path: str):
    """Save metrics to JSON file."""
    results = {
        'metrics': metrics.get_all_metrics(),
        'comparisons': metrics.compare_to_baseline('full') if 'full' in metrics.metrics else {},
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_path}")


# ============================================================================
# Experiment Runner
# ============================================================================

def run_ablation_experiment(variant_name: str, 
                           input_shape: Tuple[int, int, int, int] = (4, 256, 32, 32),
                           device: str = 'cuda' if torch.cuda.is_available() else 'cpu') -> Dict[str, Any]:
    """
    Run a single ablation experiment.
    
    Args:
        variant_name: Name of the variant to test
        input_shape: Input tensor shape (B, C, H, W)
        device: Device to run on
    
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f"Running experiment: {variant_name}")
    print(f"{'='*60}")
    
    # Get configuration
    config = get_ablation_config(variant_name)
    print(f"Description: {config.description}")
    print(f"Expected: {config.expected_effect}")
    
    # Create model
    print("\nCreating model...")
    model = get_ablation_model(
        variant=config.variant,
        dim=config.dim,
        mlp_ratio=config.mlp_ratio,
        drop=config.drop,
        drop_path=config.drop_path,
        patch_sizes=config.patch_sizes,
        top_k=config.top_k
    )
    model = model.to(device)
    
    # Collect metrics
    print("Collecting metrics...")
    metrics_collector = AblationMetrics()
    variant_metrics = metrics_collector.add_model_metrics(variant_name, model, input_shape)
    
    # Print summary
    print("\nMetrics Summary:")
    print(f"  Parameters: {variant_metrics['parameters']:,}")
    print(f"  Model Size: {variant_metrics['model_size_mb']:.2f} MB")
    print(f"  Inference Time: {variant_metrics['inference_time_ms']:.2f} ms")
    print(f"  Throughput: {variant_metrics['throughput']:.2f} samples/sec")
    if isinstance(variant_metrics.get('gflops'), float):
        print(f"  GFLOPs: {variant_metrics['gflops']:.2f}")
    
    return variant_metrics


def run_all_experiments(variants: List[str] = None,
                       input_shape: Tuple[int, int, int, int] = (4, 256, 32, 32),
                       device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
                       save_path: str = None):
    """
    Run multiple ablation experiments and compare results.
    
    Args:
        variants: List of variant names to test (None = all variants)
        input_shape: Input tensor shape
        device: Device to run on
        save_path: Path to save results JSON (optional)
    """
    if variants is None:
        variants = list(ABLATION_CONFIGS.keys())
    
    print("\n" + "="*80)
    print(f"Running {len(variants)} Ablation Experiments")
    print("="*80)
    
    metrics = AblationMetrics()
    
    for variant in variants:
        try:
            config = get_ablation_config(variant)
            model = get_ablation_model(
                variant=config.variant,
                dim=config.dim,
                mlp_ratio=config.mlp_ratio,
                drop=config.drop,
                drop_path=config.drop_path,
                patch_sizes=config.patch_sizes,
                top_k=config.top_k
            )
            model = model.to(device)
            
            metrics.add_model_metrics(variant, model, input_shape)
            print(f"✓ Completed: {variant}")
            
        except Exception as e:
            print(f"✗ Failed: {variant} - {str(e)}")
            continue
    
    # Print comparison table
    print_metrics_table(metrics, baseline='full')
    
    # Save results if requested
    if save_path:
        save_results(metrics, save_path)
    
    return metrics


# ============================================================================
# Branch Usage Analysis
# ============================================================================

def analyze_branch_usage(variant_name: str, 
                        num_samples: int = 100,
                        input_shape: Tuple[int, int, int, int] = (1, 256, 32, 32),
                        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Analyze branch selection patterns for routing variants.
    
    Args:
        variant_name: Name of variant to analyze
        num_samples: Number of samples to test
        input_shape: Input shape (batch size will be set to 1)
        device: Device to run on
    """
    print(f"\n{'='*60}")
    print(f"Branch Usage Analysis: {variant_name}")
    print(f"{'='*60}")
    
    config = get_ablation_config(variant_name)
    model = get_ablation_model(
        variant=config.variant,
        dim=config.dim,
        patch_sizes=config.patch_sizes,
        top_k=config.top_k
    ).to(device)
    
    model.eval()
    
    # Run inference on random samples
    with torch.no_grad():
        for _ in range(num_samples):
            x = torch.randn(1, input_shape[1], input_shape[2], input_shape[3]).to(device)
            _ = model(x)
    
    # Print usage statistics
    print(f"\nBranch selection statistics over {num_samples} samples:")
    print(model.get_usage_str())
    print("="*60 + "\n")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='FFT Pathformer Ablation Evaluation')
    parser.add_argument('--config', type=str, default='full',
                       help='Ablation config name or "all" for all configs')
    parser.add_argument('--group', type=str, default=None,
                       help='Run all configs in a specific group')
    parser.add_argument('--input_size', type=int, default=32,
                       help='Input spatial size (height=width)')
    parser.add_argument('--dim', type=int, default=256,
                       help='Feature dimension')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to run on')
    parser.add_argument('--save_results', type=str, default=None,
                       help='Path to save results JSON')
    parser.add_argument('--analyze_branches', action='store_true',
                       help='Analyze branch usage patterns')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples for branch analysis')
    
    args = parser.parse_args()
    
    input_shape = (args.batch_size, args.dim, args.input_size, args.input_size)
    
    if args.config == 'all':
        # Run all experiments
        run_all_experiments(
            variants=None,
            input_shape=input_shape,
            device=args.device,
            save_path=args.save_results
        )
    elif args.group:
        # Run specific group
        group_configs = get_configs_by_group(args.group)
        run_all_experiments(
            variants=list(group_configs.keys()),
            input_shape=input_shape,
            device=args.device,
            save_path=args.save_results
        )
    else:
        # Run single experiment
        run_ablation_experiment(
            variant_name=args.config,
            input_shape=input_shape,
            device=args.device
        )
        
        # Branch analysis if requested
        if args.analyze_branches:
            analyze_branch_usage(
                variant_name=args.config,
                num_samples=args.num_samples,
                input_shape=input_shape,
                device=args.device
            )


if __name__ == "__main__":
    main()
