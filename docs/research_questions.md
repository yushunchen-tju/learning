# Research Questions for Ablation Study

## Component Ablations

**RQ1: How much does frequency-domain channel attention contribute to performance?**
- Variant: `no_spectral_se`
- Description: Remove SpectralSELayer to test frequency-based channel attention
- Expected: Slight performance drop; tests importance of spectral channel attention

**RQ2: Is spectral-based routing superior to spatial-based routing?**
- Variant: `no_spectral_selector`
- Description: Replace spectral selector with spatial selector
- Expected: Performance drop; shows importance of frequency-aware routing

## Routing Strategies

**RQ3: What is the advantage of using spectral features over spatial features for routing?**
- Variant: `spatial_router`
- Description: Use spatial statistics for routing instead of spectral
- Expected: Performance drop; shows superiority of spectral routing

**RQ4: How important is adaptive routing compared to fixed routing?**
- Variant: `uniform_router`
- Description: Use uniform weights for routing (no adaptation)
- Expected: Significant drop; demonstrates need for adaptive routing

## Architecture

**RQ5: Is multi-scale processing essential for performance?**
- Variant: `single_scale`
- Description: Use only one patch size (4x4) instead of multi-scale
- Expected: Significant performance drop; demonstrates need for multi-scale

## Efficiency

**RQ6: What is the efficiency gain from sparse branch execution?**
- Variant: `no_sparsity`
- Description: Use all branches instead of top-k selection
- Expected: Similar or slightly better performance but much higher computational cost

**RQ7: Do depthwise separable convolutions provide good efficiency without sacrificing accuracy?**
- Variant: `no_depthwise_sep`
- Description: Use standard convolutions instead of depthwise separable
- Expected: Similar performance but many more parameters

