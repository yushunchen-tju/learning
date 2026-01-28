# Experimental Protocol

## Setup

```python
# Common settings for all experiments
dim = 256
input_size = 32
batch_size = 4
device = 'cuda'
```

## Running Experiments

### 1. Baseline
```bash
python tools/ablation_eval.py --config full --save_results results_full.json
```

### 2. Component Ablations
```bash
python tools/ablation_eval.py --group component --save_results results_component.json
```

### 3. All Experiments
```bash
python tools/ablation_eval.py --config all --save_results results_all.json
```

## Analysis

After collecting results:

1. **Parameter Efficiency**: Compare parameter counts
2. **Computational Efficiency**: Compare FLOPs and inference time
3. **Performance Impact**: Compare accuracy/loss metrics
4. **Branch Usage**: Analyze routing patterns for adaptive variants

