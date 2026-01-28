# FFT Pathformer Ablation Study Guide

## Overview

This ablation study framework provides a systematic evaluation of the FFT Pathformer model components. It allows researchers to understand the contribution of each component through controlled experiments.

## Architecture Components

The FFT Pathformer consists of the following key components:

### 1. **PatchBranch**
- Multi-scale patch processing with attention
- Uses depthwise separable convolutions for efficiency
- Supports multiple patch sizes (2x2, 4x4, 8x8, 16x16)

### 2. **SpectralSELayer** (创新点 1)
- Frequency-domain channel attention
- Computes spectral energy per channel via FFT
- Reweights channels based on frequency content

### 3. **SpectralScaleSelector** (创新点 2)
- Adaptive scale selection based on frequency analysis
- Analyzes global spectral energy distribution
- Selects top-k most appropriate scales dynamically

### 4. **FFTPathformerBlock**
- Main building block combining all components
- Sparse execution: only runs selected branches
- Includes MLP with spatial mixing

---

## Ablation Variants

### Baseline
- **Full Model**: All components enabled (reference performance)

### Component Ablations

#### 1. **Without Spectral SE** (`no_spectral_se`)
- **Removes**: SpectralSELayer
- **Purpose**: Test the contribution of frequency-based channel attention
- **Expected**: Slight performance drop
- **Research Question**: How much does spectral channel attention help?

#### 2. **Without Spectral Selector** (`no_spectral_selector`)
- **Replaces**: Spectral-based routing → Spatial-based routing
- **Purpose**: Compare frequency vs spatial routing strategies
- **Expected**: Performance drop showing importance of spectral analysis
- **Research Question**: Is frequency-domain routing better than spatial?

### Efficiency Ablations

#### 3. **Without Sparsity** (`no_sparsity`)
- **Changes**: Top-k selection → All branches
- **Purpose**: Analyze efficiency vs performance trade-off
- **Expected**: Similar/better performance but much higher cost
- **Research Question**: Is sparse execution necessary?

#### 4. **Without Depthwise Separable Conv** (`no_depthwise_sep`)
- **Replaces**: Depthwise separable → Standard convolutions
- **Purpose**: Justify the use of depthwise separable convolutions
- **Expected**: More parameters, similar performance
- **Research Question**: Do depthwise separable convs provide efficiency without sacrificing accuracy?

### Architecture Ablations

#### 5. **Single Scale** (`single_scale`)
- **Changes**: Multi-scale (4 branches) → Single scale (1 branch)
- **Purpose**: Demonstrate necessity of multi-scale design
- **Expected**: Significant performance drop
- **Research Question**: Is multi-scale processing essential?

### Routing Strategy Ablations

#### 6. **Spatial Router** (`spatial_router`)
- **Changes**: Uses spatial features instead of spectral features
- **Purpose**: Compare routing strategies
- **Expected**: Performance drop vs spectral routing
- **Research Question**: What routing strategy works best?

#### 7. **Uniform Router** (`uniform_router`)
- **Changes**: Fixed uniform weights instead of adaptive selection
- **Purpose**: Show benefit of adaptive routing
- **Expected**: Significant drop, no adaptability
- **Research Question**: Is adaptive routing necessary?

---

## Quick Start

### 1. Installation

```bash
cd /home/runner/work/learning/learning
pip install -e .
pip install fvcore  # For FLOPs calculation
```

### 2. Run Single Experiment

```bash
# Evaluate the full model
python tools/ablation_eval.py --config full

# Evaluate a specific ablation
python tools/ablation_eval.py --config no_spectral_se

# With custom settings
python tools/ablation_eval.py --config full --dim 256 --input_size 32 --batch_size 8
```

### 3. Run All Experiments

```bash
# Run all ablation experiments
python tools/ablation_eval.py --config all --save_results ablation_results.json

# Run specific group
python tools/ablation_eval.py --group component --save_results component_results.json
```

### 4. Analyze Branch Usage

```bash
# Analyze how often each scale is selected
python tools/ablation_eval.py --config full --analyze_branches --num_samples 1000
```

---

## Experiment Groups

Experiments are organized into logical groups:

- **baseline**: Full model only
- **component**: Test individual components (SE, selector)
- **efficiency**: Test efficiency optimizations (sparsity, depthwise)
- **architecture**: Test architectural choices (single vs multi-scale)
- **routing**: Test different routing strategies

---

## Python API Usage

### Basic Usage

```python
from openstl.models.fft_pathformer_ablation import get_ablation_model
import torch

# Create full model
model = get_ablation_model('full', dim=256)

# Create ablation variant
model_ablation = get_ablation_model('no_spectral_se', dim=256)

# Forward pass
x = torch.randn(4, 256, 32, 32)
y = model(x)  # Output: [4, 256, 32, 32]
```

### Compare Models

```python
from tools.ablation_eval import run_all_experiments

# Run and compare all variants
metrics = run_all_experiments(
    variants=['full', 'no_spectral_se', 'no_spectral_selector'],
    input_shape=(4, 256, 32, 32),
    device='cuda',
    save_path='results.json'
)
```

### Collect Metrics

```python
from tools.ablation_eval import AblationMetrics
from openstl.models.fft_pathformer_ablation import get_ablation_model

metrics = AblationMetrics()

# Collect metrics for multiple variants
for variant in ['full', 'no_spectral_se']:
    model = get_ablation_model(variant, dim=256)
    metrics.add_model_metrics(variant, model, input_shape=(4, 256, 32, 32))

# Compare to baseline
comparisons = metrics.compare_to_baseline('full')
print(comparisons)
```

---

## Expected Results

### Performance Hierarchy (Expected)

1. **Full Model** (Best) - Baseline reference
2. **No Depthwise Sep** - Similar accuracy, more parameters
3. **No Spectral SE** - Slight drop
4. **Spatial Router** - Moderate drop
5. **No Spectral Selector** - Moderate drop  
6. **Single Scale** - Significant drop
7. **Uniform Router** - Significant drop
8. **No Sparsity** - Similar/better but much slower

### Key Findings to Report

When writing the ablation study in your paper, report:

1. **Baseline Performance**: Establish the full model's metrics
2. **Component Contributions**: 
   - SpectralSE contribution (full vs no_spectral_se)
   - Spectral routing contribution (spectral vs spatial/uniform)
3. **Architecture Justification**:
   - Multi-scale necessity (full vs single_scale)
4. **Efficiency Analysis**:
   - Sparsity benefit (no_sparsity vs full)
   - Depthwise separable benefit (no_depthwise_sep vs full)

---

## Metrics Collected

For each variant, the following metrics are collected:

1. **Parameters**: Total trainable parameters
2. **Model Size**: Memory footprint in MB
3. **Inference Time**: Average forward pass time (ms)
4. **Throughput**: Samples per second
5. **GFLOPs**: Computational complexity (if fvcore available)
6. **Branch Usage**: Selection frequency for each scale (for routing variants)

---

## Visualization and Analysis

### Generate Comparison Table

```python
from configs.ablation_config import print_all_configs, print_experiment_plan

# Print all configurations
print_all_configs()

# Print recommended execution order
print_experiment_plan()
```

### Analyze Branch Selection Patterns

```python
from tools.ablation_eval import analyze_branch_usage

# Analyze which scales are selected most often
analyze_branch_usage('full', num_samples=1000)
```

---

## Paper Writing Guide

### Ablation Study Section Structure

```
4. Ablation Study

4.1 Experimental Setup
- Dataset: [Your dataset]
- Metrics: [Your task metrics]
- Implementation details

4.2 Component Analysis
- Table: Performance with/without each component
- Analysis: Which components contribute most?

4.3 Routing Strategy Comparison
- Table: Spectral vs Spatial vs Uniform routing
- Visualization: Branch selection patterns
- Analysis: Why spectral routing works better

4.4 Architecture Choices
- Single-scale vs Multi-scale comparison
- Analysis: Importance of multi-scale design

4.5 Efficiency Analysis
- Parameters vs Performance trade-offs
- Inference time comparison
- Depthwise separable conv justification
```

### Example Tables

**Table 1: Component Ablation**
```
| Variant              | Params (M) | Accuracy | Time (ms) |
|---------------------|-----------|----------|-----------|
| Full Model          | 10.5      | 0.920    | 12.3      |
| w/o Spectral SE     | 10.2      | 0.905    | 11.8      |
| w/o Spectral Router | 10.5      | 0.895    | 12.1      |
| Uniform Router      | 10.5      | 0.870    | 12.0      |
```

**Table 2: Efficiency Analysis**
```
| Variant           | Params (M) | GFLOPs | Time (ms) | Accuracy |
|------------------|-----------|--------|-----------|----------|
| Full Model       | 10.5      | 5.2    | 12.3      | 0.920    |
| No Sparsity      | 10.5      | 10.4   | 23.1      | 0.925    |
| No Depthwise Sep | 25.3      | 5.5    | 12.8      | 0.922    |
| Single Scale     | 3.2       | 1.8    | 5.4       | 0.850    |
```

---

## Customization

### Add New Ablation Variant

1. Edit `openstl/models/fft_pathformer_ablation.py`:
   - Add configuration in `get_ablation_model()` function

2. Edit `configs/ablation_config.py`:
   - Add new `AblationConfig` entry
   - Add to appropriate experiment group

3. Run experiment:
   ```bash
   python tools/ablation_eval.py --config your_new_variant
   ```

### Example: Add "Without MLP" Variant

```python
# In get_ablation_model():
'no_mlp': {
    'use_spectral_se': True,
    'use_spectral_selector': True,
    'use_spatial_selector': False,
    'use_uniform_selector': False,
    'use_all_branches': False,
    'use_depthwise_sep': True,
    'patch_sizes': patch_sizes,
    'use_mlp': False,  # New parameter
},
```

---

## Troubleshooting

### Out of Memory
- Reduce batch size: `--batch_size 1`
- Reduce input size: `--input_size 16`
- Use CPU: `--device cpu`

### FLOPs Calculation Fails
- Install fvcore: `pip install fvcore`
- Or run without FLOPs (will show "N/A")

### Branch Usage Shows "No data"
- Make sure model is in eval mode
- Run with enough samples: `--num_samples 1000`

---

## Citation

If you use this ablation framework in your research, please cite:

```bibtex
@article{your_paper,
  title={Your Paper Title with FFT Pathformer},
  author={Your Name},
  journal={Your Conference/Journal},
  year={2024}
}
```

---

## Questions?

For questions or issues:
1. Check this documentation
2. Review example scripts in `tools/ablation_eval.py`
3. Check configuration in `configs/ablation_config.py`
4. Review model code in `openstl/models/fft_pathformer_ablation.py`

Good luck with your ablation study! 🚀
