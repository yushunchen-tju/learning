"""
Configuration for FFT Pathformer Ablation Experiments

This file defines configurations for different ablation studies.
Each configuration specifies which components to enable/disable.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    variant: str  # Variant name for get_ablation_model()
    
    # Model hyperparameters
    dim: int = 256
    mlp_ratio: float = 4.0
    drop: float = 0.0
    drop_path: float = 0.1
    patch_sizes: List[int] = field(default_factory=lambda: [2, 4, 8, 16])
    top_k: int = 2
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    epochs: int = 100
    
    # Additional metadata
    ablation_group: str = "main"  # Group for organizing experiments
    expected_effect: str = ""  # Expected performance change


# ============================================================================
# Define All Ablation Experiments
# ============================================================================

ABLATION_CONFIGS = {
    # ========== Baseline ==========
    "full": AblationConfig(
        name="Full Model (Baseline)",
        description="Complete FFT Pathformer with all components enabled",
        variant="full",
        ablation_group="baseline",
        expected_effect="Best performance (reference)"
    ),
    
    # ========== Component Ablations ==========
    "no_spectral_se": AblationConfig(
        name="Without Spectral SE",
        description="Remove SpectralSELayer to test frequency-based channel attention",
        variant="no_spectral_se",
        ablation_group="component",
        expected_effect="Slight performance drop; tests importance of spectral channel attention"
    ),
    
    "no_spectral_selector": AblationConfig(
        name="Without Spectral Selector",
        description="Replace spectral selector with spatial selector",
        variant="no_spectral_selector",
        ablation_group="component",
        expected_effect="Performance drop; shows importance of frequency-aware routing"
    ),
    
    "no_sparsity": AblationConfig(
        name="Without Sparsity",
        description="Use all branches instead of top-k selection",
        variant="no_sparsity",
        ablation_group="efficiency",
        expected_effect="Similar or slightly better performance but much higher computational cost"
    ),
    
    "no_depthwise_sep": AblationConfig(
        name="Without Depthwise Separable Conv",
        description="Use standard convolutions instead of depthwise separable",
        variant="no_depthwise_sep",
        ablation_group="efficiency",
        expected_effect="Similar performance but many more parameters"
    ),
    
    # ========== Architecture Ablations ==========
    "single_scale": AblationConfig(
        name="Single Scale",
        description="Use only one patch size (4x4) instead of multi-scale",
        variant="single_scale",
        patch_sizes=[4],
        ablation_group="architecture",
        expected_effect="Significant performance drop; demonstrates need for multi-scale"
    ),
    
    # ========== Routing Strategy Ablations ==========
    "spatial_router": AblationConfig(
        name="Spatial Router",
        description="Use spatial statistics for routing instead of spectral",
        variant="spatial_router",
        ablation_group="routing",
        expected_effect="Performance drop; shows superiority of spectral routing"
    ),
    
    "uniform_router": AblationConfig(
        name="Uniform Router",
        description="Use uniform weights for routing (no adaptation)",
        variant="uniform_router",
        ablation_group="routing",
        expected_effect="Significant drop; demonstrates need for adaptive routing"
    ),
}


# ============================================================================
# Experiment Groups
# ============================================================================

EXPERIMENT_GROUPS = {
    "baseline": ["full"],
    "component": ["no_spectral_se", "no_spectral_selector"],
    "efficiency": ["no_sparsity", "no_depthwise_sep"],
    "architecture": ["single_scale"],
    "routing": ["spatial_router", "uniform_router"],
    "all": list(ABLATION_CONFIGS.keys()),
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_ablation_config(name: str) -> AblationConfig:
    """Get configuration for a specific ablation experiment."""
    if name not in ABLATION_CONFIGS:
        raise ValueError(f"Unknown ablation config: {name}. "
                        f"Available: {list(ABLATION_CONFIGS.keys())}")
    return ABLATION_CONFIGS[name]


def get_configs_by_group(group: str) -> Dict[str, AblationConfig]:
    """Get all configurations in a specific experiment group."""
    if group not in EXPERIMENT_GROUPS:
        raise ValueError(f"Unknown group: {group}. "
                        f"Available: {list(EXPERIMENT_GROUPS.keys())}")
    
    return {name: ABLATION_CONFIGS[name] for name in EXPERIMENT_GROUPS[group]}


def print_all_configs():
    """Print all ablation configurations in a formatted table."""
    print("\n" + "="*100)
    print("FFT Pathformer Ablation Experiment Configurations")
    print("="*100)
    
    for group_name, config_names in EXPERIMENT_GROUPS.items():
        if group_name == "all":
            continue
            
        print(f"\n{group_name.upper()} EXPERIMENTS:")
        print("-"*100)
        
        for config_name in config_names:
            config = ABLATION_CONFIGS[config_name]
            print(f"\n  [{config_name}]")
            print(f"  Name: {config.name}")
            print(f"  Description: {config.description}")
            print(f"  Expected Effect: {config.expected_effect}")
            print(f"  Variant: {config.variant}")
            print(f"  Patch Sizes: {config.patch_sizes}")
    
    print("\n" + "="*100 + "\n")


def get_experiment_plan():
    """
    Generate a recommended experiment plan with prioritized order.
    Returns a list of (config_name, priority, reasoning) tuples.
    """
    plan = [
        # Priority 1: Baseline
        ("full", 1, "Baseline - Must run first to establish reference performance"),
        
        # Priority 2: Core components (test key innovations)
        ("no_spectral_selector", 2, "Tests the core innovation: spectral-based routing"),
        ("no_spectral_se", 2, "Tests the spectral channel attention contribution"),
        
        # Priority 3: Routing strategies
        ("spatial_router", 3, "Compare spectral vs spatial routing strategies"),
        ("uniform_router", 3, "Show benefit of adaptive routing over fixed"),
        
        # Priority 4: Architecture choices
        ("single_scale", 4, "Demonstrate necessity of multi-scale design"),
        ("no_sparsity", 4, "Analyze efficiency vs performance trade-off"),
        
        # Priority 5: Implementation details
        ("no_depthwise_sep", 5, "Justify depthwise separable conv for efficiency"),
    ]
    
    return plan


def print_experiment_plan():
    """Print the recommended experiment execution plan."""
    plan = get_experiment_plan()
    
    print("\n" + "="*100)
    print("Recommended Ablation Experiment Execution Plan")
    print("="*100)
    
    current_priority = 0
    for config_name, priority, reasoning in plan:
        if priority != current_priority:
            current_priority = priority
            print(f"\n--- PRIORITY {priority} ---")
        
        config = ABLATION_CONFIGS[config_name]
        print(f"\n  [{config_name}] {config.name}")
        print(f"  → Reasoning: {reasoning}")
        print(f"  → Expected: {config.expected_effect}")
    
    print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    # Print all configurations
    print_all_configs()
    
    # Print experiment plan
    print_experiment_plan()
    
    # Example: Get a specific config
    print("\nExample - Getting 'no_spectral_se' config:")
    print("-"*50)
    config = get_ablation_config("no_spectral_se")
    print(f"Name: {config.name}")
    print(f"Variant: {config.variant}")
    print(f"Dim: {config.dim}")
    print(f"Patch sizes: {config.patch_sizes}")
