# Ablation Study Results Summary

## Parameter Counts

| Variant | Params (M) | Relative to Baseline | Description |
|---------|-----------|---------------------|-------------|
| Full Model (Baseline) | 2.34 | Baseline | Complete FFT Pathformer with all components enable... |
| Without Spectral SE | 2.33 | -0.4% | Remove SpectralSELayer to test frequency-based cha... |
| Without Spectral Selector | 2.34 | +0.0% | Replace spectral selector with spatial selector... |
| Spatial Router | 2.34 | +0.0% | Use spatial statistics for routing instead of spec... |
| Uniform Router | 2.30 | -1.4% | Use uniform weights for routing (no adaptation)... |
| Single Scale | 0.98 | -58.0% | Use only one patch size (4x4) instead of multi-sca... |
| Without Sparsity | 2.30 | -1.4% | Use all branches instead of top-k selection... |
| Without Depthwise Separable Conv | 46.20 | +1878.5% | Use standard convolutions instead of depthwise sep... |

## Experiment Organization

### Component Ablations
Test the contribution of key innovations:
- **Spectral SE**: Tests frequency-based channel attention
- **Spectral Selector**: Tests adaptive scale selection based on frequency

### Routing Strategies
Compare different routing approaches:
- **Spatial Router**: Uses spatial statistics instead of spectral
- **Uniform Router**: Fixed routing without adaptation

### Architecture Choices
Validate design decisions:
- **Single Scale**: Tests necessity of multi-scale design

### Efficiency Analysis
Understand trade-offs:
- **No Sparsity**: All branches vs top-k selection
- **No Depthwise Sep**: Standard vs depthwise separable convolutions

