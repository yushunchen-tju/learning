"""
FFT Pathformer Ablation Study Module

This module provides comprehensive ablation variants for the FFT Pathformer model
to systematically evaluate the contribution of each component.

Ablation Experiments:
1. Full Model (Baseline): All components enabled
2. w/o Spectral SE: Remove SpectralSELayer to test frequency-based channel attention
3. w/o Spectral Selector: Remove adaptive scale selection, use uniform routing
4. w/o Sparsity: Use all branches instead of top-k selection
5. w/o Depthwise Sep Conv: Use standard convolutions in PatchBranch
6. Single Scale: Use only one patch size instead of multi-scale
7. Spatial Router: Replace spectral-based routing with spatial statistics

Usage:
    from openstl.models.fft_pathformer_ablation import get_ablation_model
    
    # Get full model
    model = get_ablation_model('full', dim=256)
    
    # Get ablation variant
    model = get_ablation_model('no_spectral_se', dim=256)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath


# ============================================================================
# Component: PatchBranch (with ablation support)
# ============================================================================

class PatchBranch(nn.Module):
    """
    Multi-scale patch processing branch with depthwise separable convolutions.
    Supports ablation by switching between depthwise separable and standard convolutions.
    """
    def __init__(self, dim, patch_size, drop=0., use_depthwise_sep=True):
        super().__init__()
        self.patch_size = patch_size
        self.use_depthwise_sep = use_depthwise_sep
        
        if use_depthwise_sep:
            # Depthwise Separable: Efficient for large patches
            self.patch_embed = nn.Sequential(
                # Depthwise: spatial
                nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size, groups=dim),
                # Pointwise: channel mixing
                nn.Conv2d(dim, dim, 1)
            )
        else:
            # Standard convolution: for ablation
            self.patch_embed = nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size)
        
        self.norm = nn.LayerNorm(dim)
        
        # Attention
        from timm.models.vision_transformer import Attention
        self.attn = Attention(dim, num_heads=8, qkv_bias=True, attn_drop=drop, proj_drop=drop)
        
        if use_depthwise_sep:
            # Depthwise Separable unembedding
            self.patch_unembed = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.ConvTranspose2d(dim, dim, kernel_size=patch_size, stride=patch_size, groups=dim)
            )
        else:
            # Standard transposed convolution
            self.patch_unembed = nn.ConvTranspose2d(dim, dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x_p = self.patch_embed(x)
        Hp, Wp = x_p.shape[2], x_p.shape[3]
        
        x_trans = x_p.flatten(2).transpose(1, 2)
        x_trans = self.norm(x_trans)
        x_trans = self.attn(x_trans)
        x_trans = x_trans.transpose(1, 2).reshape(B, C, Hp, Wp)
        
        x_out = self.patch_unembed(x_trans)
        
        if x_out.shape[-2:] != (H, W):
            x_out = F.interpolate(x_out, size=(H, W), mode='bilinear', align_corners=False)
        return x_out


# ============================================================================
# Component: SpectralSELayer
# ============================================================================

class SpectralSELayer(nn.Module):
    """
    Frequency-domain channel attention (Spectral Squeeze-and-Excitation).
    Reweights channels based on their spectral energy distribution.
    """
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Compute frequency spectrum
        fft_x = torch.fft.rfft2(x, norm='backward')
        
        # Compute spectral energy per channel
        mag_x = torch.abs(fft_x)
        # Log compression for better learning dynamics
        channel_energy = torch.log(mag_x.sum(dim=(2, 3)) + 1e-8)
        
        # Generate channel weights
        y = self.fc(channel_energy)  # (B, C)
        y = y.view(B, C, 1, 1)
        
        # Reweight
        return x * y


# ============================================================================
# Component: SpectralScaleSelector (Spectral-based Router)
# ============================================================================

class SpectralScaleSelector(nn.Module):
    """
    Frequency-aware scale selector that analyzes global spectral energy
    to dynamically select the most appropriate patch scales.
    """
    def __init__(self, dim, num_branches, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_branches = num_branches
        self.feat_dim = dim
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Gating network
        self.selector_net = nn.Sequential(
            nn.Linear(self.feat_dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_branches),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # FFT spectral analysis
        fft_x = torch.fft.rfft2(x, norm='backward')
        mag_x = torch.abs(fft_x) + 1e-8
        log_mag = torch.log(mag_x)
        
        # Extract spectral fingerprint
        spectral_feat = self.global_pool(log_mag).flatten(1)
        
        # Compute selection weights
        all_scores = self.selector_net(spectral_feat)
        
        # Top-k selection
        topk_weights, topk_indices = torch.topk(all_scores, self.top_k, dim=1)
        
        # Renormalize
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
        
        return topk_weights, topk_indices


# ============================================================================
# Component: SpatialScaleSelector (Spatial-based Router for ablation)
# ============================================================================

class SpatialScaleSelector(nn.Module):
    """
    Spatial-based scale selector for ablation study.
    Uses spatial statistics instead of frequency analysis.
    """
    def __init__(self, dim, num_branches, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_branches = num_branches
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Gating network
        self.selector_net = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_branches),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Use spatial features directly (no FFT)
        spatial_feat = self.global_pool(x).flatten(1)
        
        # Compute selection weights
        all_scores = self.selector_net(spatial_feat)
        
        # Top-k selection
        topk_weights, topk_indices = torch.topk(all_scores, self.top_k, dim=1)
        
        # Renormalize
        topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)
        
        return topk_weights, topk_indices


# ============================================================================
# Component: UniformScaleSelector (Uniform routing for ablation)
# ============================================================================

class UniformScaleSelector(nn.Module):
    """
    Uniform scale selector for ablation study.
    Always selects the same branches with equal weights.
    """
    def __init__(self, dim, num_branches, top_k=2):
        super().__init__()
        self.top_k = top_k
        self.num_branches = num_branches
        # Fixed selection: always choose first top_k branches
        self.register_buffer('fixed_indices', torch.arange(top_k).unsqueeze(0))
        self.register_buffer('fixed_weights', torch.ones(1, top_k) / top_k)

    def forward(self, x):
        B = x.shape[0]
        
        # Return uniform weights for fixed branches
        weights = self.fixed_weights.expand(B, -1)
        indices = self.fixed_indices.expand(B, -1)
        
        return weights, indices


# ============================================================================
# MixMlp Layer (Spatial-aware MLP)
# ============================================================================

class MixMlp(nn.Module):
    """
    Mix-FFN with depthwise convolution for spatial information.
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, 
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.dwconv = nn.Conv2d(hidden_features, hidden_features, 3, 1, 1, 
                                groups=hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


# ============================================================================
# Main: FFTPathformerBlock with Ablation Support
# ============================================================================

class FFTPathformerBlock(nn.Module):
    """
    Adaptive multi-scale FFT Pathformer block with ablation support.
    
    Args:
        dim: Feature dimension
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
        drop_path: DropPath rate
        patch_sizes: List of patch sizes for multi-scale processing
        use_spectral_se: Whether to use spectral channel attention
        use_spectral_selector: Whether to use spectral-based scale selection
        use_spatial_selector: Whether to use spatial-based selection (for ablation)
        use_uniform_selector: Whether to use uniform selection (for ablation)
        use_all_branches: Whether to use all branches (no sparsity)
        use_depthwise_sep: Whether to use depthwise separable convolutions
        top_k: Number of branches to select (ignored if use_all_branches=True)
    """
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0.1,
                 patch_sizes=[2, 4, 8, 16],
                 use_spectral_se=True,
                 use_spectral_selector=True,
                 use_spatial_selector=False,
                 use_uniform_selector=False,
                 use_all_branches=False,
                 use_depthwise_sep=True,
                 top_k=2):
        super().__init__()
        
        self.patch_sizes = patch_sizes
        self.num_branches = len(self.patch_sizes)
        self.use_spectral_se = use_spectral_se
        self.use_all_branches = use_all_branches
        
        # Multi-scale branches
        self.branches = nn.ModuleList([
            PatchBranch(dim, p, drop=drop, use_depthwise_sep=use_depthwise_sep) 
            for p in self.patch_sizes
        ])
        
        # Scale selector (router)
        if use_all_branches:
            # No selection needed, use all branches
            self.scale_selector = None
        elif use_uniform_selector:
            self.scale_selector = UniformScaleSelector(dim, self.num_branches, top_k)
        elif use_spatial_selector:
            self.scale_selector = SpatialScaleSelector(dim, self.num_branches, top_k)
        elif use_spectral_selector:
            self.scale_selector = SpectralScaleSelector(dim, self.num_branches, top_k)
        else:
            raise ValueError("Must specify a selector type")
        
        # Spectral SE layer (optional)
        if use_spectral_se:
            self.spectral_se = SpectralSELayer(dim)
        else:
            self.spectral_se = nn.Identity()
        
        # Standard components
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm_mlp = nn.BatchNorm2d(dim)
        self.mlp = MixMlp(in_features=dim, hidden_features=int(dim * mlp_ratio), 
                         act_layer=nn.GELU, drop=drop)
        
        # Statistics buffer
        self.register_buffer('branch_counts', torch.zeros(self.num_branches, dtype=torch.long), 
                           persistent=False)

    def forward(self, x):
        shortcut = x
        B, C, H, W = x.shape
        
        if self.use_all_branches:
            # Use all branches with equal weights
            final_output = torch.zeros_like(x)
            for branch_module in self.branches:
                branch_out = branch_module(x)
                final_output += branch_out / self.num_branches
        else:
            # Adaptive scale selection (sparse execution)
            weights, indices = self.scale_selector(x)
            
            # Statistics
            if not self.training:
                with torch.no_grad():
                    self.branch_counts += torch.bincount(indices.flatten(), 
                                                        minlength=self.num_branches)
            
            # Sparse execution: only run selected branches
            final_output = torch.zeros_like(x)
            
            for k in range(weights.shape[1]):
                w_k = weights[:, k].view(B, 1, 1, 1)
                idx_k = indices[:, k]
                
                for branch_idx, branch_module in enumerate(self.branches):
                    mask = (idx_k == branch_idx).view(B, 1, 1, 1).float()
                    
                    if mask.sum() == 0:
                        continue
                    
                    branch_out = branch_module(x)
                    final_output += branch_out * w_k * mask
        
        x = shortcut + self.drop_path(final_output)
        
        # Post-processing (SE + MLP)
        shortcut_mlp = x
        x = self.norm_mlp(x)
        x = self.spectral_se(x)  # Will be identity if disabled
        x = self.mlp(x)
        x = shortcut_mlp + self.drop_path(x)
        
        return x
    
    def get_usage_str(self):
        """Get branch usage statistics."""
        total = self.branch_counts.sum().item()
        if total == 0:
            return "No data"
        info = [f"Scale-{p}: {c.item()} ({c.item()/total*100:.1f}%)" 
                for p, c in zip(self.patch_sizes, self.branch_counts)]
        return " | ".join(info)


# ============================================================================
# Ablation Model Factory
# ============================================================================

def get_ablation_model(variant, dim=256, mlp_ratio=4., drop=0., drop_path=0.1,
                       patch_sizes=[2, 4, 8, 16], top_k=2):
    """
    Factory function to create different ablation variants.
    
    Args:
        variant: One of ['full', 'no_spectral_se', 'no_spectral_selector', 
                        'no_sparsity', 'no_depthwise_sep', 'single_scale',
                        'spatial_router', 'uniform_router']
        dim: Feature dimension
        mlp_ratio: MLP expansion ratio
        drop: Dropout rate
        drop_path: DropPath rate
        patch_sizes: List of patch sizes
        top_k: Number of branches to select
    
    Returns:
        FFTPathformerBlock instance with specified configuration
    """
    
    configs = {
        # Full model (baseline)
        'full': {
            'use_spectral_se': True,
            'use_spectral_selector': True,
            'use_spatial_selector': False,
            'use_uniform_selector': False,
            'use_all_branches': False,
            'use_depthwise_sep': True,
            'patch_sizes': patch_sizes,
        },
        
        # Ablation: Remove Spectral SE
        'no_spectral_se': {
            'use_spectral_se': False,
            'use_spectral_selector': True,
            'use_spatial_selector': False,
            'use_uniform_selector': False,
            'use_all_branches': False,
            'use_depthwise_sep': True,
            'patch_sizes': patch_sizes,
        },
        
        # Ablation: Remove spectral-based selector
        'no_spectral_selector': {
            'use_spectral_se': True,
            'use_spectral_selector': False,
            'use_spatial_selector': True,
            'use_uniform_selector': False,
            'use_all_branches': False,
            'use_depthwise_sep': True,
            'patch_sizes': patch_sizes,
        },
        
        # Ablation: No sparsity (use all branches)
        'no_sparsity': {
            'use_spectral_se': True,
            'use_spectral_selector': True,
            'use_spatial_selector': False,
            'use_uniform_selector': False,
            'use_all_branches': True,
            'use_depthwise_sep': True,
            'patch_sizes': patch_sizes,
        },
        
        # Ablation: Remove depthwise separable convolutions
        'no_depthwise_sep': {
            'use_spectral_se': True,
            'use_spectral_selector': True,
            'use_spatial_selector': False,
            'use_uniform_selector': False,
            'use_all_branches': False,
            'use_depthwise_sep': False,
            'patch_sizes': patch_sizes,
        },
        
        # Ablation: Single scale only
        'single_scale': {
            'use_spectral_se': True,
            'use_spectral_selector': True,
            'use_spatial_selector': False,
            'use_uniform_selector': False,
            'use_all_branches': False,
            'use_depthwise_sep': True,
            'patch_sizes': [4],  # Only one scale
        },
        
        # Ablation: Spatial router instead of spectral
        'spatial_router': {
            'use_spectral_se': True,
            'use_spectral_selector': False,
            'use_spatial_selector': True,
            'use_uniform_selector': False,
            'use_all_branches': False,
            'use_depthwise_sep': True,
            'patch_sizes': patch_sizes,
        },
        
        # Ablation: Uniform router (no adaptation)
        'uniform_router': {
            'use_spectral_se': True,
            'use_spectral_selector': False,
            'use_spatial_selector': False,
            'use_uniform_selector': True,
            'use_all_branches': False,
            'use_depthwise_sep': True,
            'patch_sizes': patch_sizes,
        },
    }
    
    if variant not in configs:
        raise ValueError(f"Unknown variant: {variant}. Choose from {list(configs.keys())}")
    
    config = configs[variant]
    
    # Adjust top_k if there are fewer branches than top_k
    actual_top_k = min(top_k, len(config['patch_sizes']))
    
    return FFTPathformerBlock(
        dim=dim,
        mlp_ratio=mlp_ratio,
        drop=drop,
        drop_path=drop_path,
        top_k=actual_top_k,
        **config
    )


# ============================================================================
# Utility Functions
# ============================================================================

def count_parameters(model):
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_ablation_table():
    """
    Generate a comparison table of all ablation variants.
    Returns a formatted string with parameter counts and descriptions.
    """
    variants = [
        'full', 'no_spectral_se', 'no_spectral_selector', 
        'no_sparsity', 'no_depthwise_sep', 'single_scale',
        'spatial_router', 'uniform_router'
    ]
    
    descriptions = {
        'full': 'Full model with all components',
        'no_spectral_se': 'Without spectral channel attention',
        'no_spectral_selector': 'Without spectral-based scale selection',
        'no_sparsity': 'Use all branches (no top-k selection)',
        'no_depthwise_sep': 'Standard convolutions instead of depthwise separable',
        'single_scale': 'Single scale (patch_size=4) only',
        'spatial_router': 'Spatial-based routing instead of spectral',
        'uniform_router': 'Uniform routing (no adaptation)',
    }
    
    print("\n" + "="*80)
    print("FFT Pathformer Ablation Study - Model Variants")
    print("="*80)
    print(f"{'Variant':<25} {'Parameters':<15} {'Description'}")
    print("-"*80)
    
    for variant in variants:
        model = get_ablation_model(variant, dim=256)
        params = count_parameters(model)
        desc = descriptions[variant]
        print(f"{variant:<25} {params:<15,} {desc}")
    
    print("="*80 + "\n")


if __name__ == "__main__":
    # Test all variants
    print("Testing FFT Pathformer Ablation Variants...\n")
    
    # Generate comparison table
    get_ablation_table()
    
    # Test forward pass for each variant
    x = torch.randn(2, 256, 32, 32)
    
    variants = [
        'full', 'no_spectral_se', 'no_spectral_selector', 
        'no_sparsity', 'no_depthwise_sep', 'single_scale',
        'spatial_router', 'uniform_router'
    ]
    
    print("\nForward Pass Tests:")
    print("-"*80)
    for variant in variants:
        model = get_ablation_model(variant, dim=256)
        model.eval()
        with torch.no_grad():
            y = model(x)
        print(f"✓ {variant:<25} Output shape: {tuple(y.shape)}")
    
    print("\nAll tests passed!")
