"""
Ablation Study Summary Tables Generator

This script generates LaTeX and Markdown tables for the ablation study
that can be directly used in papers.
"""

import torch
import sys
import importlib.util
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

# Direct import to avoid dependency issues
spec = importlib.util.spec_from_file_location(
    'fft_pathformer_ablation', 
    Path(__file__).parent.parent / 'openstl' / 'models' / 'fft_pathformer_ablation.py'
)
ablation_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation_module)

get_ablation_model = ablation_module.get_ablation_model
count_parameters = ablation_module.count_parameters

from configs.ablation_config import ABLATION_CONFIGS, EXPERIMENT_GROUPS


def generate_latex_table():
    """Generate LaTeX table for paper."""
    
    latex = r"""
\begin{table*}[t]
\centering
\caption{Ablation Study of FFT Pathformer Components. We systematically evaluate the contribution 
of each component by removing or replacing it. The full model serves as the baseline. 
Params: number of parameters (M), GFLOPs: computational complexity, Time: inference time (ms).}
\label{tab:ablation}
\begin{tabular}{llcccc}
\toprule
\textbf{Group} & \textbf{Variant} & \textbf{Params (M)} & \textbf{GFLOPs} & \textbf{Time (ms)} & \textbf{Accuracy} \\
\midrule
"""
    
    # Baseline
    latex += r"\multirow{1}{*}{Baseline} & Full Model"
    model = get_ablation_model('full', dim=256)
    params = count_parameters(model) / 1e6
    latex += f" & {params:.2f} & -- & -- & -- \\\\\n"
    latex += r"\midrule" + "\n"
    
    # Component ablations
    latex += r"\multirow{2}{*}{Component} "
    for i, variant in enumerate(['no_spectral_se', 'no_spectral_selector']):
        config = ABLATION_CONFIGS[variant]
        model = get_ablation_model(variant, dim=256)
        params = count_parameters(model) / 1e6
        
        if i > 0:
            latex += " & "
        else:
            latex += "& "
        
        latex += f"{config.name} & {params:.2f} & -- & -- & -- \\\\\n"
    
    latex += r"\midrule" + "\n"
    
    # Routing strategies
    latex += r"\multirow{2}{*}{Routing} "
    for i, variant in enumerate(['spatial_router', 'uniform_router']):
        config = ABLATION_CONFIGS[variant]
        model = get_ablation_model(variant, dim=256)
        params = count_parameters(model) / 1e6
        
        if i > 0:
            latex += " & "
        else:
            latex += "& "
        
        latex += f"{config.name} & {params:.2f} & -- & -- & -- \\\\\n"
    
    latex += r"\midrule" + "\n"
    
    # Architecture
    latex += r"\multirow{1}{*}{Architecture} & Single Scale"
    model = get_ablation_model('single_scale', dim=256)
    params = count_parameters(model) / 1e6
    latex += f" & {params:.2f} & -- & -- & -- \\\\\n"
    latex += r"\midrule" + "\n"
    
    # Efficiency
    latex += r"\multirow{2}{*}{Efficiency} "
    for i, variant in enumerate(['no_sparsity', 'no_depthwise_sep']):
        config = ABLATION_CONFIGS[variant]
        model = get_ablation_model(variant, dim=256)
        params = count_parameters(model) / 1e6
        
        if i > 0:
            latex += " & "
        else:
            latex += "& "
        
        latex += f"{config.name} & {params:.2f} & -- & -- & -- \\\\\n"
    
    latex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    
    return latex


def generate_markdown_table():
    """Generate Markdown table for README."""
    
    md = "# Ablation Study Results Summary\n\n"
    md += "## Parameter Counts\n\n"
    md += "| Variant | Params (M) | Relative to Baseline | Description |\n"
    md += "|---------|-----------|---------------------|-------------|\n"
    
    baseline_model = get_ablation_model('full', dim=256)
    baseline_params = count_parameters(baseline_model)
    
    for variant_name in ['full', 'no_spectral_se', 'no_spectral_selector', 
                        'spatial_router', 'uniform_router', 'single_scale',
                        'no_sparsity', 'no_depthwise_sep']:
        config = ABLATION_CONFIGS[variant_name]
        model = get_ablation_model(variant_name, dim=256)
        params = count_parameters(model)
        params_m = params / 1e6
        
        if variant_name == 'full':
            relative = "Baseline"
        else:
            diff_pct = ((params - baseline_params) / baseline_params) * 100
            relative = f"{diff_pct:+.1f}%"
        
        md += f"| {config.name} | {params_m:.2f} | {relative} | {config.description[:50]}... |\n"
    
    md += "\n## Experiment Organization\n\n"
    md += "### Component Ablations\n"
    md += "Test the contribution of key innovations:\n"
    md += "- **Spectral SE**: Tests frequency-based channel attention\n"
    md += "- **Spectral Selector**: Tests adaptive scale selection based on frequency\n\n"
    
    md += "### Routing Strategies\n"
    md += "Compare different routing approaches:\n"
    md += "- **Spatial Router**: Uses spatial statistics instead of spectral\n"
    md += "- **Uniform Router**: Fixed routing without adaptation\n\n"
    
    md += "### Architecture Choices\n"
    md += "Validate design decisions:\n"
    md += "- **Single Scale**: Tests necessity of multi-scale design\n\n"
    
    md += "### Efficiency Analysis\n"
    md += "Understand trade-offs:\n"
    md += "- **No Sparsity**: All branches vs top-k selection\n"
    md += "- **No Depthwise Sep**: Standard vs depthwise separable convolutions\n\n"
    
    return md


def generate_research_questions():
    """Generate list of research questions for each ablation."""
    
    rq = "# Research Questions for Ablation Study\n\n"
    
    questions = {
        "Component Ablations": {
            "no_spectral_se": "RQ1: How much does frequency-domain channel attention contribute to performance?",
            "no_spectral_selector": "RQ2: Is spectral-based routing superior to spatial-based routing?",
        },
        "Routing Strategies": {
            "spatial_router": "RQ3: What is the advantage of using spectral features over spatial features for routing?",
            "uniform_router": "RQ4: How important is adaptive routing compared to fixed routing?",
        },
        "Architecture": {
            "single_scale": "RQ5: Is multi-scale processing essential for performance?",
        },
        "Efficiency": {
            "no_sparsity": "RQ6: What is the efficiency gain from sparse branch execution?",
            "no_depthwise_sep": "RQ7: Do depthwise separable convolutions provide good efficiency without sacrificing accuracy?",
        }
    }
    
    for group, group_rqs in questions.items():
        rq += f"## {group}\n\n"
        for variant, question in group_rqs.items():
            config = ABLATION_CONFIGS[variant]
            rq += f"**{question}**\n"
            rq += f"- Variant: `{variant}`\n"
            rq += f"- Description: {config.description}\n"
            rq += f"- Expected: {config.expected_effect}\n\n"
    
    return rq


def generate_experiment_protocol():
    """Generate experimental protocol for reproducibility."""
    
    protocol = "# Experimental Protocol\n\n"
    protocol += "## Setup\n\n"
    protocol += "```python\n"
    protocol += "# Common settings for all experiments\n"
    protocol += "dim = 256\n"
    protocol += "input_size = 32\n"
    protocol += "batch_size = 4\n"
    protocol += "device = 'cuda'\n"
    protocol += "```\n\n"
    
    protocol += "## Running Experiments\n\n"
    protocol += "### 1. Baseline\n"
    protocol += "```bash\n"
    protocol += "python tools/ablation_eval.py --config full --save_results results_full.json\n"
    protocol += "```\n\n"
    
    protocol += "### 2. Component Ablations\n"
    protocol += "```bash\n"
    protocol += "python tools/ablation_eval.py --group component --save_results results_component.json\n"
    protocol += "```\n\n"
    
    protocol += "### 3. All Experiments\n"
    protocol += "```bash\n"
    protocol += "python tools/ablation_eval.py --config all --save_results results_all.json\n"
    protocol += "```\n\n"
    
    protocol += "## Analysis\n\n"
    protocol += "After collecting results:\n\n"
    protocol += "1. **Parameter Efficiency**: Compare parameter counts\n"
    protocol += "2. **Computational Efficiency**: Compare FLOPs and inference time\n"
    protocol += "3. **Performance Impact**: Compare accuracy/loss metrics\n"
    protocol += "4. **Branch Usage**: Analyze routing patterns for adaptive variants\n\n"
    
    return protocol


def main():
    """Generate all summary materials."""
    
    print("Generating ablation study summary materials...\n")
    
    # Generate LaTeX table
    latex = generate_latex_table()
    latex_file = Path(__file__).parent.parent / "docs" / "ablation_table.tex"
    with open(latex_file, 'w') as f:
        f.write(latex)
    print(f"✓ LaTeX table saved to {latex_file}")
    
    # Generate Markdown table
    markdown = generate_markdown_table()
    md_file = Path(__file__).parent.parent / "docs" / "ablation_summary.md"
    with open(md_file, 'w') as f:
        f.write(markdown)
    print(f"✓ Markdown summary saved to {md_file}")
    
    # Generate research questions
    rq = generate_research_questions()
    rq_file = Path(__file__).parent.parent / "docs" / "research_questions.md"
    with open(rq_file, 'w') as f:
        f.write(rq)
    print(f"✓ Research questions saved to {rq_file}")
    
    # Generate protocol
    protocol = generate_experiment_protocol()
    protocol_file = Path(__file__).parent.parent / "docs" / "experiment_protocol.md"
    with open(protocol_file, 'w') as f:
        f.write(protocol)
    print(f"✓ Experiment protocol saved to {protocol_file}")
    
    print("\n" + "="*80)
    print("Summary materials generated successfully!")
    print("="*80)
    print("\nFiles created:")
    print("  1. docs/ablation_table.tex - LaTeX table for paper")
    print("  2. docs/ablation_summary.md - Markdown summary")
    print("  3. docs/research_questions.md - Research questions")
    print("  4. docs/experiment_protocol.md - Experimental protocol")
    print("\nYou can now use these materials in your paper and documentation.")


if __name__ == "__main__":
    main()
