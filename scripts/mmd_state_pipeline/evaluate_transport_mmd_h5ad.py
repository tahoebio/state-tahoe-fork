#!/usr/bin/env python3
"""
Evaluate transport performance using MMD between transported and observed embeddings (h5ad version).

This script:
1. Loads test dataset (observed drug-treated cells) from HuggingFace dataset
2. Loads transported embeddings from h5ad file (state tx predictions)
3. Loads DMSO controls from h5ad file
4. Computes MMD between observed vs transported for each combination  
5. Analyzes transport performance across drug_dose and cell_line conditions

This is the h5ad equivalent of evaluate_transport_mmd.py for state tx results.

IMPORTANT: Embedding Key Assumptions:
- Transported data (predictions): Uses .obsm['model_preds'] (output from state tx infer)
- DMSO controls: Uses .obsm['X_hvg'] (should match input data used for training)
- Test dataset: Uses 'mosaicfm-70m-merged' (from HuggingFace dataset)
"""

import json
import pandas as pd
import numpy as np
import torch
import scanpy as sc
from datasets import load_from_disk
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration
CONFIG = {
    # Data paths
    'test_dataset': "/tahoe/drive_3/ANALYSIS/analysis_190/Data/20250618.tahoe_embeddings_70M.test.dataset",
    'transported_data': "dmso_controls_predicted_all_perturbations.20250710.h5ad",
    'dmso_controls_data': "dmso_controls.h5ad",
    
    # MMD parameters
    'embedding_key': 'model_preds',  # Key in transported data obsm (predictions from state tx)
    'dmso_embedding_key': 'X_hvg',  # Key in DMSO controls obsm (should match input data)
    'test_embedding_key': 'mosaicfm-70m-merged',  # Key in test dataset (from HF dataset)
    'mmd_kernel_mul': 2.0,
    'mmd_kernel_num': 5,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    
    # Analysis parameters
    'min_cells_for_mmd': 50,  # Minimum cells needed for reliable MMD
    'max_cells_per_combination': 1500,  # Match transport sampling
    'max_combinations_to_test': 0,  # Set to 0 to run all combinations, or positive number to limit for testing
    
    # Output
    'output_dir': Path("mmd_evaluation_h5ad.20250710"),
    'results_file': "mmd_results.parquet",
    'summary_file': "mmd_summary.json"
}

def gaussian_kernel(source: torch.Tensor, target: torch.Tensor, kernel_mul=2.0, kernel_num=5) -> torch.Tensor:
    """Compute Gaussian kernel between source and target tensors."""
    total = torch.cat([source, target], dim=0)
    L2_distance = torch.cdist(total, total, p=2).pow(2)
    
    n_samples = total.size(0)
    bandwidth = torch.sum(L2_distance) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    
    exponents = torch.arange(kernel_num, device=L2_distance.device, dtype=torch.float32)
    bandwidth_list = bandwidth * torch.pow(kernel_mul, exponents)
    bandwidth_list = bandwidth_list.view(kernel_num, 1, 1)
    
    kernel_vals = torch.exp(-L2_distance.unsqueeze(0) / bandwidth_list)
    return kernel_vals.sum(0)

def mmd_loss(source: torch.Tensor, target: torch.Tensor, kernel_mul=2.0, kernel_num=5) -> float:
    """Compute MMD loss between source and target distributions."""
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num)
    n_source = source.size(0)
    
    XX = kernels[:n_source, :n_source]
    YY = kernels[n_source:, n_source:]
    XY = kernels[:n_source, n_source:]
    YX = kernels[n_source:, :n_source]
    
    loss = torch.mean(XX) + torch.mean(YY) - torch.mean(XY) - torch.mean(YX)
    return loss.item()

def load_test_data():
    """Load and process test dataset."""
    print("Loading test dataset...")
    test_dataset = load_from_disk(CONFIG['test_dataset'])
    
    # Convert to pandas for efficient processing
    test_df = pd.DataFrame({
        'drug_dose_id': test_dataset['drug_dose_id'],
        'cell_id': test_dataset['cell_id'],
        'index': range(len(test_dataset))
    })
    
    print(f"✓ Test dataset loaded: {len(test_dataset):,} samples")
    return test_dataset, test_df

def load_transported_data():
    """Load transported embeddings from h5ad file."""
    print("Loading transported embeddings...")
    
    if not Path(CONFIG['transported_data']).exists():
        raise FileNotFoundError(f"Transported data not found: {CONFIG['transported_data']}")
    
    adata = sc.read_h5ad(CONFIG['transported_data'])
    
    # Extract relevant data
    transported_df = pd.DataFrame({
        'drug_dose': adata.obs['drug_dose'].values,
        'cell_line': adata.obs['cell_line'].values,
        'cell_id': adata.obs['cell_id'].values,
        'drug_dose_id': adata.obs['drug_dose_id'].values,
        'embedding': [emb for emb in adata.obsm[CONFIG['embedding_key']]]
    })
    
    print(f"✓ Transported data loaded: {len(transported_df):,} samples")
    print(f"  Unique drug_dose combinations: {adata.obs['drug_dose'].nunique()}")
    print(f"  Unique cell_line combinations: {adata.obs['cell_line'].nunique()}")
    
    return transported_df

def load_dmso_controls():
    """Load DMSO control samples for baseline comparison."""
    print("Loading DMSO control samples...")
    
    if not Path(CONFIG['dmso_controls_data']).exists():
        raise FileNotFoundError(f"DMSO controls not found: {CONFIG['dmso_controls_data']}")
    
    adata = sc.read_h5ad(CONFIG['dmso_controls_data'])
    
    # Extract relevant data
    dmso_df = pd.DataFrame({
        'cell_line': adata.obs['cell_line'].values,
        'cell_id': adata.obs['cell_id'].values,
        'drug_dose': adata.obs['drug_dose'].values,
        'embedding': [emb for emb in adata.obsm[CONFIG['dmso_embedding_key']]]
    })
    
    print(f"✓ DMSO controls loaded: {len(dmso_df):,} samples")
    print(f"  Unique cell_line combinations: {adata.obs['cell_line'].nunique()}")
    
    return dmso_df

def create_vocab_mapping(test_dataset, transported_df):
    """Create vocabulary mappings from the data."""
    print("Creating vocabulary mappings...")
    
    # Load the vocabulary file to get proper drug_dose_id mappings
    with open('/tahoe/drive_3/ANALYSIS/analysis_190/Data/drugdose_cell_line_vocab.json') as f:
        vocab = json.load(f)
    
    drug_dose_map = vocab['drug_dose_map']
    cell_map = vocab['cell_map']
    
    # Create reverse mappings
    id_to_drug_dose = {v: k for k, v in drug_dose_map.items()}
    id_to_cell = {v: k for k, v in cell_map.items()}
    
    print(f"✓ Vocabulary mappings created:")
    print(f"  Drug-dose combinations: {len(drug_dose_map)}")
    print(f"  Cell combinations: {len(cell_map)}")
    
    return drug_dose_map, cell_map, id_to_drug_dose, id_to_cell

def create_evaluation_plan(test_df: pd.DataFrame, transported_df: pd.DataFrame, id_to_drug_dose: dict, id_to_cell: dict):
    """Create plan for MMD evaluation based on available data."""
    print("Creating evaluation plan...")
    
    # Get test combinations with sufficient data
    test_combinations = (test_df
                        .groupby(['drug_dose_id', 'cell_id'])
                        .size()
                        .reset_index(name='n_test_cells')
                        .query(f'n_test_cells >= {CONFIG["min_cells_for_mmd"]}'))
    
    # Add readable names
    test_combinations['drug_dose'] = test_combinations['drug_dose_id'].map(id_to_drug_dose)
    test_combinations['cell_line'] = test_combinations['cell_id'].map(id_to_cell)
    
    # Filter out combinations where mapping failed (NaN values)
    test_combinations = test_combinations.dropna(subset=['drug_dose', 'cell_line'])
    
    # Get transported combinations
    transported_combinations = (transported_df
                               .groupby(['drug_dose', 'cell_line'])
                               .size()
                               .reset_index(name='n_transported_cells'))
    
    # Find overlapping combinations - match on drug_dose and cell_line strings
    evaluation_plan = test_combinations.merge(
        transported_combinations, 
        on=['drug_dose', 'cell_line'], 
        how='inner'
    )
    
    print(f"✓ Evaluation plan: {len(evaluation_plan)} combinations")
    print(f"  Test combinations with ≥{CONFIG['min_cells_for_mmd']} cells: {len(test_combinations)}")
    print(f"  Transported combinations: {len(transported_combinations)}")
    print(f"  Overlapping combinations: {len(evaluation_plan)}")
    
    # Debug: show some examples
    if len(evaluation_plan) > 0:
        print(f"\nSample overlapping combinations:")
        print(evaluation_plan[['drug_dose', 'cell_line', 'n_test_cells', 'n_transported_cells']].head())
    else:
        print("\nNo overlapping combinations found. Debugging...")
        print("Sample test combinations:")
        print(test_combinations[['drug_dose', 'cell_line']].head())
        print("Sample transported combinations:")
        print(transported_combinations[['drug_dose', 'cell_line']].head())
    
    return evaluation_plan

def compute_mmd_for_combination(
    drug_dose: str, 
    cell_line: str,
    test_dataset,
    test_df: pd.DataFrame,
    transported_df: pd.DataFrame,
    dmso_df: pd.DataFrame,
    drug_dose_map: dict,
    cell_map: dict
) -> Tuple[float, float, int, int, int]:
    """Compute both baseline and transport MMD for a single drug_dose × cell_line combination."""
    
    # Get IDs
    drug_dose_id = drug_dose_map[drug_dose]
    cell_id = cell_map[cell_line]
    
    # Get test (observed) samples
    test_indices = test_df[
        (test_df['drug_dose_id'] == drug_dose_id) & 
        (test_df['cell_id'] == cell_id)
    ]['index'].tolist()
    
    if len(test_indices) > CONFIG['max_cells_per_combination']:
        # Randomly sample to match transported size
        test_indices = np.random.choice(test_indices, CONFIG['max_cells_per_combination'], replace=False).tolist()
    
    # Extract test embeddings
    test_embeddings = []
    for idx in test_indices:
        embedding = test_dataset[idx][CONFIG['test_embedding_key']]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        test_embeddings.append(embedding)
    
    test_tensor = torch.stack(test_embeddings).to(CONFIG['device'])
    
    # Get DMSO control samples for baseline
    dmso_samples = dmso_df[dmso_df['cell_line'] == cell_line]
    
    if len(dmso_samples) == 0:
        raise ValueError(f"No DMSO controls found for cell line: {cell_line}")
    
    # Sample DMSO to match test size (for fair comparison)
    n_dmso_to_use = min(len(dmso_samples), len(test_tensor))
    if len(dmso_samples) > n_dmso_to_use:
        dmso_samples = dmso_samples.sample(n_dmso_to_use, random_state=42)
    
    dmso_embeddings = []
    for _, row in dmso_samples.iterrows():
        embedding = torch.tensor(row['embedding'], dtype=torch.float32)
        dmso_embeddings.append(embedding)
    
    dmso_tensor = torch.stack(dmso_embeddings).to(CONFIG['device'])
    
    # Compute baseline MMD (DMSO vs observed drug-treated)
    baseline_mmd = mmd_loss(
        dmso_tensor,
        test_tensor,
        kernel_mul=CONFIG['mmd_kernel_mul'],
        kernel_num=CONFIG['mmd_kernel_num']
    )
    
    # Get transported (simulated) samples
    transported_samples = transported_df[
        (transported_df['drug_dose'] == drug_dose) & 
        (transported_df['cell_line'] == cell_line)
    ]
    
    transported_embeddings = []
    for _, row in transported_samples.iterrows():
        embedding = torch.tensor(row['embedding'], dtype=torch.float32)
        transported_embeddings.append(embedding)
    
    transported_tensor = torch.stack(transported_embeddings).to(CONFIG['device'])
    
    # Compute transport MMD (transported vs observed drug-treated)
    transport_mmd = mmd_loss(
        transported_tensor, 
        test_tensor,
        kernel_mul=CONFIG['mmd_kernel_mul'],
        kernel_num=CONFIG['mmd_kernel_num']
    )
    
    return baseline_mmd, transport_mmd, len(test_tensor), len(dmso_tensor), len(transported_tensor)

def evaluate_all_combinations():
    """Evaluate baseline and transport MMD for all available combinations."""
    print("="*60)
    print("MMD EVALUATION (Baseline + Transport) - H5AD Version")
    print("="*60)
    
    # Load data
    test_dataset, test_df = load_test_data()
    transported_df = load_transported_data()
    dmso_df = load_dmso_controls()
    
    # Create vocabulary mappings
    drug_dose_map, cell_map, id_to_drug_dose, id_to_cell = create_vocab_mapping(test_dataset, transported_df)
    
    # Create evaluation plan
    evaluation_plan = create_evaluation_plan(test_df, transported_df, id_to_drug_dose, id_to_cell)
    
    # Set random seed for reproducible sampling
    np.random.seed(42)
    
    # Limit to a subset for testing
    if 'max_combinations_to_test' in CONFIG and CONFIG['max_combinations_to_test'] > 0:
        evaluation_plan = evaluation_plan.head(CONFIG['max_combinations_to_test'])
        print(f"Testing on subset of {len(evaluation_plan)} combinations...")
    
    # Evaluate each combination
    print(f"\nComputing baseline and transport MMD for {len(evaluation_plan)} combinations...")
    results = []
    
    for _, row in tqdm(evaluation_plan.iterrows(), total=len(evaluation_plan), desc="Computing MMD"):
        drug_dose = row['drug_dose']
        cell_line = row['cell_line']
        
        try:
            baseline_mmd, transport_mmd, n_test, n_dmso, n_transported = compute_mmd_for_combination(
                drug_dose, cell_line, test_dataset, test_df, transported_df, dmso_df, drug_dose_map, cell_map
            )
            
            # Compute improvement metrics
            improvement_ratio = baseline_mmd / transport_mmd if transport_mmd > 0 else float('inf')
            improvement_delta = baseline_mmd - transport_mmd
            
            results.append({
                'drug_dose': drug_dose,
                'cell_line': cell_line,
                'baseline_mmd': baseline_mmd,
                'transport_mmd': transport_mmd,
                'improvement_ratio': improvement_ratio,
                'improvement_delta': improvement_delta,
                'n_test_cells': n_test,
                'n_dmso_cells': n_dmso,
                'n_transported_cells': n_transported
            })
            
        except Exception as e:
            print(f"⚠ Failed {drug_dose} + {cell_line}: {e}")
            continue
    
    print(f"✓ Completed MMD evaluation for {len(results)} combinations")
    return results

def analyze_results(results: List[dict]):
    """Analyze and summarize baseline vs transport MMD results."""
    print("\nAnalyzing MMD results...")
    
    results_df = pd.DataFrame(results)
    
    # Overall statistics for both baseline and transport
    overall_stats = {
        'n_combinations': len(results_df),
        # Baseline MMD stats
        'mean_baseline_mmd': results_df['baseline_mmd'].mean(),
        'median_baseline_mmd': results_df['baseline_mmd'].median(),
        'std_baseline_mmd': results_df['baseline_mmd'].std(),
        # Transport MMD stats
        'mean_transport_mmd': results_df['transport_mmd'].mean(),
        'median_transport_mmd': results_df['transport_mmd'].median(),
        'std_transport_mmd': results_df['transport_mmd'].std(),
        # Improvement stats
        'mean_improvement_ratio': results_df['improvement_ratio'].mean(),
        'median_improvement_ratio': results_df['improvement_ratio'].median(),
        'mean_improvement_delta': results_df['improvement_delta'].mean(),
        'median_improvement_delta': results_df['improvement_delta'].median(),
        # Overall ranges
        'min_baseline_mmd': results_df['baseline_mmd'].min(),
        'max_baseline_mmd': results_df['baseline_mmd'].max(),
        'min_transport_mmd': results_df['transport_mmd'].min(),
        'max_transport_mmd': results_df['transport_mmd'].max()
    }
    
    # Per cell line analysis
    cell_line_stats = (results_df
                      .groupby('cell_line')
                      .agg({
                          'baseline_mmd': 'mean',
                          'transport_mmd': 'mean',
                          'improvement_ratio': 'mean',
                          'improvement_delta': 'mean',
                          'drug_dose': 'count'
                      })
                      .rename(columns={
                          'baseline_mmd': 'mean_baseline_mmd',
                          'transport_mmd': 'mean_transport_mmd',
                          'improvement_ratio': 'mean_improvement_ratio',
                          'improvement_delta': 'mean_improvement_delta',
                          'drug_dose': 'n_combinations'
                      })
                      .sort_values('mean_improvement_ratio', ascending=False))
    
    # Per drug analysis (best improvements)
    drug_stats = (results_df
                 .groupby('drug_dose')
                 .agg({
                     'baseline_mmd': 'mean',
                     'transport_mmd': 'mean',
                     'improvement_ratio': 'mean',
                     'improvement_delta': 'mean',
                     'cell_line': 'count'
                 })
                 .rename(columns={
                     'baseline_mmd': 'mean_baseline_mmd',
                     'transport_mmd': 'mean_transport_mmd',
                     'improvement_ratio': 'mean_improvement_ratio',
                     'improvement_delta': 'mean_improvement_delta',
                     'cell_line': 'n_combinations'
                 })
                 .sort_values('mean_improvement_ratio', ascending=False))
    
    print(f"\n{'='*60}")
    print("MMD EVALUATION RESULTS (Baseline vs Transport) - H5AD Version")
    print("="*60)
    
    print(f"Overall Performance:")
    print(f"  Combinations evaluated: {overall_stats['n_combinations']:,}")
    print(f"  ")
    print(f"  Baseline MMD (DMSO vs Drug):")
    print(f"    Mean: {overall_stats['mean_baseline_mmd']:.6f}")
    print(f"    Median: {overall_stats['median_baseline_mmd']:.6f}")
    print(f"    Range: [{overall_stats['min_baseline_mmd']:.6f}, {overall_stats['max_baseline_mmd']:.6f}]")
    print(f"  ")
    print(f"  Transport MMD (Transported vs Drug):")
    print(f"    Mean: {overall_stats['mean_transport_mmd']:.6f}")
    print(f"    Median: {overall_stats['median_transport_mmd']:.6f}")
    print(f"    Range: [{overall_stats['min_transport_mmd']:.6f}, {overall_stats['max_transport_mmd']:.6f}]")
    print(f"  ")
    print(f"  Transport Improvement:")
    print(f"    Mean ratio (baseline/transport): {overall_stats['mean_improvement_ratio']:.3f}x")
    print(f"    Median ratio: {overall_stats['median_improvement_ratio']:.3f}x")
    print(f"    Mean delta (baseline - transport): {overall_stats['mean_improvement_delta']:.6f}")
    
    print(f"\nBest Cell Lines (highest improvement ratio):")
    for cell_line in cell_line_stats.head(5).index:
        row = cell_line_stats.loc[cell_line]
        print(f"  {cell_line}: {row['mean_improvement_ratio']:.3f}x improvement")
        print(f"    Baseline: {row['mean_baseline_mmd']:.6f} → Transport: {row['mean_transport_mmd']:.6f}")
    
    print(f"\nWorst Cell Lines (lowest improvement ratio):")
    for cell_line in cell_line_stats.tail(5).index:
        row = cell_line_stats.loc[cell_line]
        print(f"  {cell_line}: {row['mean_improvement_ratio']:.3f}x improvement")
        print(f"    Baseline: {row['mean_baseline_mmd']:.6f} → Transport: {row['mean_transport_mmd']:.6f}")
    
    print(f"\nBest Drugs (highest improvement ratio):")
    for drug_dose in drug_stats.head(5).index:
        row = drug_stats.loc[drug_dose]
        print(f"  {drug_dose}: {row['mean_improvement_ratio']:.3f}x improvement ({row['n_combinations']} combinations)")
    
    print(f"\nWorst Drugs (lowest improvement ratio):")
    for drug_dose in drug_stats.tail(5).index:
        row = drug_stats.loc[drug_dose]
        print(f"  {drug_dose}: {row['mean_improvement_ratio']:.3f}x improvement ({row['n_combinations']} combinations)")
    
    return overall_stats, results_df, cell_line_stats, drug_stats

def save_results(results: List[dict], overall_stats: dict):
    """Save evaluation results."""
    print(f"\nSaving results...")
    
    CONFIG['output_dir'].mkdir(exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = CONFIG['output_dir'] / CONFIG['results_file']
    results_df.to_parquet(results_path)
    
    # Save summary statistics
    summary_path = CONFIG['output_dir'] / CONFIG['summary_file']
    with open(summary_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Summary saved to: {summary_path}")
    
    return results_path

def main():
    """Main evaluation function."""
    try:
        # Run evaluation
        results = evaluate_all_combinations()
        
        # Analyze results
        overall_stats, results_df, cell_line_stats, drug_stats = analyze_results(results)
        
        # Save results
        results_path = save_results(results, overall_stats)
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"✓ Baseline + Transport MMD evaluation completed for {len(results)} combinations")
        print(f"✓ Results saved to: {results_path}")
        print("✓ Transport quality assessment with baseline comparison complete!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()