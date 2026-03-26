#!/usr/bin/env python
# coding: utf-8

# # Evaluate All Zero-Shot Baselines with MMD (Per-Plate)
# 
# This notebook evaluates all mean shift baselines using Maximum Mean Discrepancy (MMD) with per-plate processing.
# 
# ## Models to Evaluate:
# 
# 1. **State Model** - Trained perturbation prediction model (gold standard)
# 2. **Nearest-Cell-Line** - Mean shift from most similar training cell line (per plate)
# 3. **Cross-Cell-Line V1 (DMSO-corrected)** - Average mean shift across all training cell lines (per plate)
# 4. **Cross-Cell-Line V2 (Raw mean)** - Mean of training drug embeddings minus test DMSO (per plate)
# 5. **Mean Shift Ablation** - Oracle baseline using test data (per plate)
# 6. **Control Passthrough** - Do-nothing baseline (no shift applied)
# 
# ## Expected Ranking:
# 
# 1. Mean Shift Ablation (best - uses test data)
# 2. State Model
# 3. Nearest-Cell-Line or Cross-Cell-Line
# 4. Control Passthrough (worst)
# 
# ## MMD Metrics:
# 
# - **RBF MMD**: Radial basis function kernel
# - **Energy Distance**: Distance-based metric
# 
# Lower is better for both metrics.

# In[ ]:


import subprocess
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

print("Imports successful!")


# ## Configuration

# In[ ]:


# Paths
REAL_PATH = '/tahoe/zeroshot_data/real.h5ad'
PRED_STATE = '/tahoe/zeroshot_data/pred.h5ad'
PRED_NEAREST = '/tahoe/zeroshot_data/pred_nearest_cell_line_per_plate.h5ad'
PRED_CROSS_V1 = '/tahoe/zeroshot_data/pred_cross_cell_line_v1_dmso_corrected_per_plate.h5ad'
PRED_CROSS_V2 = '/tahoe/zeroshot_data/pred_cross_cell_line_v2_raw_mean_per_plate.h5ad'
PRED_ABLATION = '/tahoe/zeroshot_data/pred_mean_shift_ablation_per_plate.h5ad'
PRED_CONTROL = '/tahoe/zeroshot_data/pred_control_passthrough_per_plate.h5ad'

# MMD script (same as few-shot)
MMD_SCRIPT = '../../scripts/mmd_state_pipeline/mmd_anndata_pair.py'

# Output directories (zeroshot-specific, per-plate)
OUT_STATE = '/tahoe/zeroshot_data/mmd_state_per_plate'
OUT_NEAREST = '/tahoe/zeroshot_data/mmd_nearest_per_plate'
OUT_CROSS_V1 = '/tahoe/zeroshot_data/mmd_cross_v1_per_plate'
OUT_CROSS_V2 = '/tahoe/zeroshot_data/mmd_cross_v2_per_plate'
OUT_ABLATION = '/tahoe/zeroshot_data/mmd_ablation_per_plate'
OUT_CONTROL = '/tahoe/zeroshot_data/mmd_control_per_plate'

# MMD params
EMBED_KEY = 'mosaicfm-70m-merged'
CONTROL_PERT = 'DMSO_TF'
PERT_COL = 'drugname_drugconc'
MAX_CELLS = 250
DEVICE = 'cpu'

print("Configuration loaded")
print(f"  Real data: {REAL_PATH}")
print(f"  MMD script: {MMD_SCRIPT}")


# ## Run MMD Evaluations
# 
# Evaluate all 6 models using the same MMD script

# In[ ]:


def run_mmd(name, pred_path, output_dir):
    print(f"\n{'='*80}")
    print(f"EVALUATING: {name}")
    print(f"{'='*80}\n")
    
    cmd = [
        'python', MMD_SCRIPT,
        '--adata-real', REAL_PATH,
        '--adata-pred', pred_path,
        '--control-pert', CONTROL_PERT,
        '--pert-col', PERT_COL,
        '--embed-key', EMBED_KEY,
        '--max-cells-per-group', str(MAX_CELLS),
        '--device', DEVICE,
        '--outdir', output_dir
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    
    if result.returncode == 0:
        print(f"\n✓ {name} evaluation complete")
    else:
        print(f"\n✗ {name} failed (code {result.returncode})")
    
    return result.returncode == 0

# Run all evaluations
results = {}
results['State'] = run_mmd('State Model', PRED_STATE, OUT_STATE)
results['Nearest'] = run_mmd('Nearest-Cell-Line', PRED_NEAREST, OUT_NEAREST)
results['Cross_V1'] = run_mmd('Cross-Cell-Line V1 (DMSO-corrected)', PRED_CROSS_V1, OUT_CROSS_V1)
results['Cross_V2'] = run_mmd('Cross-Cell-Line V2 (raw mean)', PRED_CROSS_V2, OUT_CROSS_V2)
results['Ablation'] = run_mmd('Mean Shift Ablation', PRED_ABLATION, OUT_ABLATION)
results['Control'] = run_mmd('Control Passthrough', PRED_CONTROL, OUT_CONTROL)

print(f"\n{'='*80}")
print("ALL EVALUATIONS COMPLETE")
print(f"{'='*80}")


# ## Load and Compare Results

# In[ ]:


# Load summaries
with open(f"{OUT_STATE}/mmd_summary.json") as f:
    summary_state = json.load(f)
with open(f"{OUT_NEAREST}/mmd_summary.json") as f:
    summary_nearest = json.load(f)
with open(f"{OUT_CROSS_V1}/mmd_summary.json") as f:
    summary_cross_v1 = json.load(f)
with open(f"{OUT_CROSS_V2}/mmd_summary.json") as f:
    summary_cross_v2 = json.load(f)
with open(f"{OUT_ABLATION}/mmd_summary.json") as f:
    summary_ablation = json.load(f)
with open(f"{OUT_CONTROL}/mmd_summary.json") as f:
    summary_control = json.load(f)

print("Loaded all summaries")


# ## Display Rankings

# In[ ]:


print(f"\n{'='*80}")
print("ZERO-SHOT GENERALIZATION: MODEL RANKINGS (PER-PLATE)")
print(f"{'='*80}\n")

# RBF MMD ranking
rbf_scores = [
    ('State Model', summary_state['transport_mmd_rbf']['mean']),
    ('Nearest-Cell-Line', summary_nearest['transport_mmd_rbf']['mean']),
    ('Cross-Cell-Line V1 (DMSO-corrected)', summary_cross_v1['transport_mmd_rbf']['mean']),
    ('Cross-Cell-Line V2 (raw mean)', summary_cross_v2['transport_mmd_rbf']['mean']),
    ('Mean Shift Ablation', summary_ablation['transport_mmd_rbf']['mean']),
    ('Control Passthrough', summary_control['transport_mmd_rbf']['mean'])
]
rbf_scores_sorted = sorted(rbf_scores, key=lambda x: x[1])

print("RBF MMD RANKING (Lower is Better):")
print(f"{'-'*80}")
for rank, (name, score) in enumerate(rbf_scores_sorted, 1):
    arrow = '>' if rank < len(rbf_scores_sorted) else ''
    print(f"  {rank}. {name:45s} {score:.6f} {arrow}")

# Energy MMD ranking
energy_scores = [
    ('State Model', summary_state['transport_mmd_energy']['mean']),
    ('Nearest-Cell-Line', summary_nearest['transport_mmd_energy']['mean']),
    ('Cross-Cell-Line V1 (DMSO-corrected)', summary_cross_v1['transport_mmd_energy']['mean']),
    ('Cross-Cell-Line V2 (raw mean)', summary_cross_v2['transport_mmd_energy']['mean']),
    ('Mean Shift Ablation', summary_ablation['transport_mmd_energy']['mean']),
    ('Control Passthrough', summary_control['transport_mmd_energy']['mean'])
]
energy_scores_sorted = sorted(energy_scores, key=lambda x: x[1])

print(f"\nENERGY MMD RANKING (Lower is Better):")
print(f"{'-'*80}")
for rank, (name, score) in enumerate(energy_scores_sorted, 1):
    arrow = '>' if rank < len(energy_scores_sorted) else ''
    print(f"  {rank}. {name:45s} {score:.6f} {arrow}")

print(f"\n{'='*80}")


# ## Detailed Statistics

# In[ ]:


def print_stats(name, summary):
    print(f"\n{'-'*80}")
    print(f"{name}")
    print(f"{'-'*80}")
    print(f"RBF MMD:")
    print(f"  Mean:   {summary['transport_mmd_rbf']['mean']:.6f}")
    print(f"  Median: {summary['transport_mmd_rbf']['median']:.6f}")
    print(f"  Std:    {summary['transport_mmd_rbf']['std']:.6f}")
    print(f"Energy MMD:")
    print(f"  Mean:   {summary['transport_mmd_energy']['mean']:.6f}")
    print(f"  Median: {summary['transport_mmd_energy']['median']:.6f}")
    print(f"  Std:    {summary['transport_mmd_energy']['std']:.6f}")

print_stats("State Model", summary_state)
print_stats("Nearest-Cell-Line", summary_nearest)
print_stats("Cross-Cell-Line V1 (DMSO-corrected)", summary_cross_v1)
print_stats("Cross-Cell-Line V2 (raw mean)", summary_cross_v2)
print_stats("Mean Shift Ablation", summary_ablation)
print_stats("Control Passthrough", summary_control)


# ## Visualization

# In[ ]:


# Create comparison dataframe
comparison_data = {
    'Model': ['State', 'Nearest', 'Cross V1', 'Cross V2', 'Ablation', 'Control'],
    'RBF_Mean': [
        summary_state['transport_mmd_rbf']['mean'],
        summary_nearest['transport_mmd_rbf']['mean'],
        summary_cross_v1['transport_mmd_rbf']['mean'],
        summary_cross_v2['transport_mmd_rbf']['mean'],
        summary_ablation['transport_mmd_rbf']['mean'],
        summary_control['transport_mmd_rbf']['mean']
    ],
    'Energy_Mean': [
        summary_state['transport_mmd_energy']['mean'],
        summary_nearest['transport_mmd_energy']['mean'],
        summary_cross_v1['transport_mmd_energy']['mean'],
        summary_cross_v2['transport_mmd_energy']['mean'],
        summary_ablation['transport_mmd_energy']['mean'],
        summary_control['transport_mmd_energy']['mean']
    ]
}
df_comp = pd.DataFrame(comparison_data)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# RBF
axes[0].barh(df_comp['Model'], df_comp['RBF_Mean'])
axes[0].set_xlabel('RBF MMD (Lower is Better)')
axes[0].set_title('Zero-Shot (Per-Plate): RBF MMD Comparison')
axes[0].grid(axis='x', alpha=0.3)

# Energy
axes[1].barh(df_comp['Model'], df_comp['Energy_Mean'])
axes[1].set_xlabel('Energy MMD (Lower is Better)')
axes[1].set_title('Zero-Shot (Per-Plate): Energy MMD Comparison')
axes[1].grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('/tahoe/zeroshot_data/mmd_comparison_per_plate.png', dpi=150)
print("\n✓ Saved plot to /tahoe/zeroshot_data/mmd_comparison_per_plate.png")
plt.show()

