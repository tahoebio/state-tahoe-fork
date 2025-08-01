#!/usr/bin/env python3
"""
Evaluate transport performance using MMD between transported and observed embeddings (h5ad test version).

This script:
1. Loads test dataset (observed drug-treated cells) from h5ad file
2. Loads transported embeddings from h5ad file (state tx predictions)
3. Loads DMSO controls from h5ad file
4. Computes MMD between observed vs transported for each combination  
5. Analyzes transport performance across drug_dose and cell_line conditions

This is the h5ad test dataset version of evaluate_transport_mmd_h5ad.py.
"""

import json
import pandas as pd
import numpy as np
import torch
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Tuple
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from geomloss import SamplesLoss

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate transport performance using MMD between transported and observed embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python evaluate_transport_mmd_h5ad_test.py \\
    --test-dataset test_subset.h5ad \\
    --transported-data predictions.h5ad \\
    --dmso-controls dmso_controls.h5ad

  # With custom parameters
  python evaluate_transport_mmd_h5ad_test.py \\
    --test-dataset test_subset.h5ad \\
    --transported-data predictions.h5ad \\
    --dmso-controls dmso_controls.h5ad \\
    --embedding-key X_hvg \\
    --output-dir results_hvg \\
    --min-cells 10
        """
    )
    
    # Required arguments
    parser.add_argument('--test-dataset', required=True,
                        help='Path to test_subset.h5ad file (ground truth drug-treated cells)')
    parser.add_argument('--transported-data', required=True,
                        help='Path to predictions h5ad file (state transport predictions)')
    parser.add_argument('--dmso-controls', required=True,
                        help='Path to dmso_controls.h5ad file (DMSO control cells)')
    
    # Optional embedding keys
    parser.add_argument('--embedding-key', default='X_hvg',
                        help='Embedding key in obsm for all datasets (default: X_hvg)')
    parser.add_argument('--test-embedding-key', default=None,
                        help='Embedding key in test dataset obsm (default: same as --embedding-key)')
    parser.add_argument('--dmso-embedding-key', default=None,
                        help='Embedding key in DMSO controls obsm (default: same as --embedding-key)')
    parser.add_argument('--transported-embedding-key', default='model_preds',
                        help='Embedding key in transported data obsm (default: model_preds - the actual predictions)')
    
    # Output options
    parser.add_argument('--output-dir', default='mmd_evaluation_results',
                        help='Output directory for results (default: mmd_evaluation_results)')
    parser.add_argument('--results-file', default='mmd_results.parquet',
                        help='Results file name (default: mmd_results.parquet)')
    parser.add_argument('--summary-file', default='mmd_summary.json',
                        help='Summary file name (default: mmd_summary.json)')
    
    # Analysis parameters
    parser.add_argument('--min-cells', type=int, default=50,
                        help='Minimum cells needed for reliable MMD (default: 50)')
    parser.add_argument('--max-cells', type=int, default=1500,
                        help='Maximum cells per combination (default: 1500)')
    parser.add_argument('--max-combinations', type=int, default=0,
                        help='Limit combinations for testing (default: 0 = all)')
    parser.add_argument('--min-tech-dup-cells', type=int, default=100,
                        help='Minimum cells needed for technical duplicate computation (default: 100)')
    
    # MMD parameters
    parser.add_argument('--mmd-kernel-mul', type=float, default=2.0,
                        help='MMD kernel multiplier (default: 2.0)')
    parser.add_argument('--mmd-kernel-num', type=int, default=5,
                        help='MMD kernel number (default: 5)')
    parser.add_argument('--device', default='auto',
                        help='Device for computation (auto, cuda, cpu) (default: auto)')
    
    # Energy kernel parameters
    parser.add_argument('--energy-blur', type=float, default=0.05,
                        help='Energy kernel blur parameter (default: 0.05)')
    parser.add_argument('--energy-scaling', type=float, default=0.5,
                        help='Energy kernel scaling parameter (default: 0.5)')
    parser.add_argument('--energy-backend', default='auto',
                        help='Energy kernel backend (auto, tensorized, online) (default: auto)')
    
    args = parser.parse_args()
    
    # Set embedding keys if not specified
    if args.test_embedding_key is None:
        args.test_embedding_key = args.embedding_key
    if args.dmso_embedding_key is None:
        args.dmso_embedding_key = args.embedding_key
    # transported_embedding_key has its own default ('model_preds'), don't override
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

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

def energy_mmd_loss(source: torch.Tensor, target: torch.Tensor, blur=0.05, scaling=0.5, backend='auto') -> float:
    """Compute Energy kernel MMD using geomloss SamplesLoss."""
    energy_loss = SamplesLoss(loss="energy", blur=blur, scaling=scaling, backend=backend)
    return energy_loss(source, target).item()

def load_test_data(args):
    """Load and process test dataset from h5ad file."""
    print("Loading test dataset...")
    
    if not Path(args.test_dataset).exists():
        raise FileNotFoundError(f"Test dataset not found: {args.test_dataset}")
    
    adata = sc.read_h5ad(args.test_dataset)
    
    # Convert to pandas for efficient processing
    test_df = pd.DataFrame({
        'drug_dose': adata.obs['drug_dose'].values,
        'cell_line': adata.obs['cell_line'].values,
        'index': range(len(adata))
    })
    
    print(f"✓ Test dataset loaded: {len(adata):,} samples")
    print(f"  Unique drug_dose combinations: {adata.obs['drug_dose'].nunique()}")
    print(f"  Unique cell_line combinations: {adata.obs['cell_line'].nunique()}")
    print(f"  Obsm keys: {list(adata.obsm.keys())}")
    
    return adata, test_df

def load_transported_data(args):
    """Load transported embeddings from h5ad file."""
    print("Loading transported embeddings...")
    
    if not Path(args.transported_data).exists():
        raise FileNotFoundError(f"Transported data not found: {args.transported_data}")
    
    adata = sc.read_h5ad(args.transported_data)
    
    # Extract relevant data
    transported_df = pd.DataFrame({
        'drug_dose': adata.obs['drug_dose'].values,
        'cell_line': adata.obs['cell_line'].values,
        'embedding': [emb for emb in adata.obsm[args.transported_embedding_key]]
    })
    
    print(f"✓ Transported data loaded: {len(transported_df):,} samples")
    print(f"  Unique drug_dose combinations: {adata.obs['drug_dose'].nunique()}")
    print(f"  Unique cell_line combinations: {adata.obs['cell_line'].nunique()}")
    print(f"  Obsm keys: {list(adata.obsm.keys())}")
    
    return transported_df

def load_dmso_controls(args):
    """Load DMSO control samples for baseline comparison."""
    print("Loading DMSO control samples...")
    
    if not Path(args.dmso_controls).exists():
        raise FileNotFoundError(f"DMSO controls not found: {args.dmso_controls}")
    
    adata = sc.read_h5ad(args.dmso_controls)
    
    # Extract relevant data
    dmso_df = pd.DataFrame({
        'cell_line': adata.obs['cell_line'].values,
        'drug_dose': adata.obs['drug_dose'].values,
        'embedding': [emb for emb in adata.obsm[args.dmso_embedding_key]]
    })
    
    print(f"✓ DMSO controls loaded: {len(dmso_df):,} samples")
    print(f"  Unique cell_line combinations: {adata.obs['cell_line'].nunique()}")
    print(f"  Obsm keys: {list(adata.obsm.keys())}")
    
    return dmso_df

def create_evaluation_plan(test_df: pd.DataFrame, transported_df: pd.DataFrame, args):
    """Create plan for MMD evaluation based on available data."""
    print("Creating evaluation plan...")
    
    # Get test combinations with sufficient data
    test_combinations = (test_df
                        .groupby(['drug_dose', 'cell_line'])
                        .size()
                        .reset_index(name='n_test_cells')
                        .query(f'n_test_cells >= {args.min_cells}'))
    
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
    print(f"  Test combinations with ≥{args.min_cells} cells: {len(test_combinations)}")
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
    test_adata,
    test_df: pd.DataFrame,
    transported_df: pd.DataFrame,
    dmso_df: pd.DataFrame,
    args
) -> Tuple[float, float, float, float, int, int, int]:
    """Compute both baseline and transport MMD for a single drug_dose × cell_line combination using both RBF and Energy kernels."""
    
    # Get test (observed) samples
    test_indices = test_df[
        (test_df['drug_dose'] == drug_dose) & 
        (test_df['cell_line'] == cell_line)
    ]['index'].tolist()
    
    if len(test_indices) > args.max_cells:
        # Randomly sample to match transported size
        test_indices = np.random.choice(test_indices, args.max_cells, replace=False).tolist()
    
    # Extract test embeddings
    test_embeddings = []
    for idx in test_indices:
        embedding = test_adata.obsm[args.test_embedding_key][idx]
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        test_embeddings.append(embedding)
    
    test_tensor = torch.stack(test_embeddings).to(args.device)
    
    # Compute technical duplicate MMD if we have enough cells
    technical_duplicate_mmd_rbf = None
    technical_duplicate_mmd_energy = None
    n_rep1 = 0
    n_rep2 = 0
    
    if len(test_tensor) >= args.min_tech_dup_cells:
        try:
            # Randomly split test indices into two groups
            np.random.seed(42)  # For reproducibility
            shuffled_indices = np.random.permutation(len(test_tensor))
            split_point = len(shuffled_indices) // 2
            
            rep1_indices = shuffled_indices[:split_point]
            rep2_indices = shuffled_indices[split_point:]
            
            rep1_tensor = test_tensor[rep1_indices]
            rep2_tensor = test_tensor[rep2_indices]
            
            n_rep1 = len(rep1_tensor)
            n_rep2 = len(rep2_tensor)
            
            # Compute technical duplicate MMD - RBF kernel
            technical_duplicate_mmd_rbf = mmd_loss(
                rep1_tensor,
                rep2_tensor,
                kernel_mul=args.mmd_kernel_mul,
                kernel_num=args.mmd_kernel_num
            )
            
            # Compute technical duplicate MMD - Energy kernel
            technical_duplicate_mmd_energy = energy_mmd_loss(
                rep1_tensor,
                rep2_tensor,
                blur=args.energy_blur,
                scaling=args.energy_scaling,
                backend=args.energy_backend
            )
            
        except Exception as e:
            print(f"⚠ Failed to compute technical duplicate for {drug_dose} + {cell_line}: {e}")
            technical_duplicate_mmd_rbf = None
            technical_duplicate_mmd_energy = None
            n_rep1 = 0
            n_rep2 = 0
    
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
    
    dmso_tensor = torch.stack(dmso_embeddings).to(args.device)
    
    # Compute baseline MMD (DMSO vs observed drug-treated) - RBF kernel
    baseline_mmd_rbf = mmd_loss(
        dmso_tensor,
        test_tensor,
        kernel_mul=args.mmd_kernel_mul,
        kernel_num=args.mmd_kernel_num
    )
    
    # Compute baseline MMD (DMSO vs observed drug-treated) - Energy kernel
    baseline_mmd_energy = energy_mmd_loss(
        dmso_tensor,
        test_tensor,
        blur=args.energy_blur,
        scaling=args.energy_scaling,
        backend=args.energy_backend
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
    
    transported_tensor = torch.stack(transported_embeddings).to(args.device)
    
    # Compute transport MMD (transported vs observed drug-treated) - RBF kernel
    transport_mmd_rbf = mmd_loss(
        transported_tensor, 
        test_tensor,
        kernel_mul=args.mmd_kernel_mul,
        kernel_num=args.mmd_kernel_num
    )
    
    # Compute transport MMD (transported vs observed drug-treated) - Energy kernel
    transport_mmd_energy = energy_mmd_loss(
        transported_tensor,
        test_tensor,
        blur=args.energy_blur,
        scaling=args.energy_scaling,
        backend=args.energy_backend
    )
    
    return (baseline_mmd_rbf, transport_mmd_rbf, baseline_mmd_energy, transport_mmd_energy, 
            technical_duplicate_mmd_rbf, technical_duplicate_mmd_energy,
            len(test_tensor), len(dmso_tensor), len(transported_tensor), n_rep1, n_rep2)

def evaluate_all_combinations(args):
    """Evaluate baseline and transport MMD for all available combinations."""
    print("="*60)
    print("DUAL-KERNEL MMD EVALUATION (RBF + Energy) - H5AD Test Version")
    print("="*60)
    
    # Load data
    test_adata, test_df = load_test_data(args)
    transported_df = load_transported_data(args)
    dmso_df = load_dmso_controls(args)
    
    # Create evaluation plan
    evaluation_plan = create_evaluation_plan(test_df, transported_df, args)
    
    # Set random seed for reproducible sampling
    np.random.seed(42)
    
    # Limit to a subset for testing
    if args.max_combinations > 0:
        evaluation_plan = evaluation_plan.head(args.max_combinations)
        print(f"Testing on subset of {len(evaluation_plan)} combinations...")
    
    # Evaluate each combination
    print(f"\nComputing dual-kernel MMD (RBF + Energy) for {len(evaluation_plan)} combinations...")
    results = []
    
    for _, row in tqdm(evaluation_plan.iterrows(), total=len(evaluation_plan), desc="Computing dual-kernel MMD"):
        drug_dose = row['drug_dose']
        cell_line = row['cell_line']
        
        try:
            baseline_mmd_rbf, transport_mmd_rbf, baseline_mmd_energy, transport_mmd_energy, \
            technical_duplicate_mmd_rbf, technical_duplicate_mmd_energy, \
            n_test, n_dmso, n_transported, n_rep1, n_rep2 = compute_mmd_for_combination(
                drug_dose, cell_line, test_adata, test_df, transported_df, dmso_df, args
            )
            
            # Compute improvement metrics for RBF kernel
            improvement_ratio_rbf = baseline_mmd_rbf / transport_mmd_rbf if transport_mmd_rbf > 0 else float('inf')
            improvement_delta_rbf = baseline_mmd_rbf - transport_mmd_rbf
            
            # Compute improvement metrics for Energy kernel
            improvement_ratio_energy = baseline_mmd_energy / transport_mmd_energy if transport_mmd_energy > 0 else float('inf')
            improvement_delta_energy = baseline_mmd_energy - transport_mmd_energy
            
            results.append({
                'drug_dose': drug_dose,
                'cell_line': cell_line,
                # RBF kernel results (existing)
                'baseline_mmd': baseline_mmd_rbf,
                'transport_mmd': transport_mmd_rbf,
                'technical_duplicate_mmd': technical_duplicate_mmd_rbf,
                'improvement_ratio': improvement_ratio_rbf,
                'improvement_delta': improvement_delta_rbf,
                # Energy kernel results (new)
                'baseline_mmd_energy': baseline_mmd_energy,
                'transport_mmd_energy': transport_mmd_energy,
                'technical_duplicate_mmd_energy': technical_duplicate_mmd_energy,
                'improvement_ratio_energy': improvement_ratio_energy,
                'improvement_delta_energy': improvement_delta_energy,
                # Cell count information
                'n_test_cells': n_test,
                'n_dmso_cells': n_dmso,
                'n_transported_cells': n_transported,
                'n_rep1_cells': n_rep1,
                'n_rep2_cells': n_rep2,
                'has_technical_duplicate': technical_duplicate_mmd_rbf is not None
            })
            
        except Exception as e:
            print(f"⚠ Failed {drug_dose} + {cell_line}: {e}")
            continue
    
    print(f"✓ Completed dual-kernel MMD evaluation for {len(results)} combinations")
    return results

def analyze_results(results: List[dict]):
    """Analyze and summarize dual-kernel MMD results (RBF + Energy)."""
    print("\nAnalyzing dual-kernel MMD results...")
    
    results_df = pd.DataFrame(results)
    
    # Filter results that have technical duplicate data for statistics
    tech_dup_results = results_df[results_df['has_technical_duplicate']]
    
    # Overall statistics for both baseline and transport
    overall_stats = {
        'n_combinations': len(results_df),
        # RBF kernel - Baseline MMD stats
        'mean_baseline_mmd': results_df['baseline_mmd'].mean(),
        'median_baseline_mmd': results_df['baseline_mmd'].median(),
        'std_baseline_mmd': results_df['baseline_mmd'].std(),
        # RBF kernel - Transport MMD stats
        'mean_transport_mmd': results_df['transport_mmd'].mean(),
        'median_transport_mmd': results_df['transport_mmd'].median(),
        'std_transport_mmd': results_df['transport_mmd'].std(),
        # RBF kernel - Improvement stats
        'mean_improvement_ratio': results_df['improvement_ratio'].mean(),
        'median_improvement_ratio': results_df['improvement_ratio'].median(),
        'mean_improvement_delta': results_df['improvement_delta'].mean(),
        'median_improvement_delta': results_df['improvement_delta'].median(),
        # RBF kernel - Overall ranges
        'min_baseline_mmd': results_df['baseline_mmd'].min(),
        'max_baseline_mmd': results_df['baseline_mmd'].max(),
        'min_transport_mmd': results_df['transport_mmd'].min(),
        'max_transport_mmd': results_df['transport_mmd'].max(),
        # Energy kernel - Baseline MMD stats
        'mean_baseline_mmd_energy': results_df['baseline_mmd_energy'].mean(),
        'median_baseline_mmd_energy': results_df['baseline_mmd_energy'].median(),
        'std_baseline_mmd_energy': results_df['baseline_mmd_energy'].std(),
        # Energy kernel - Transport MMD stats
        'mean_transport_mmd_energy': results_df['transport_mmd_energy'].mean(),
        'median_transport_mmd_energy': results_df['transport_mmd_energy'].median(),
        'std_transport_mmd_energy': results_df['transport_mmd_energy'].std(),
        # Energy kernel - Improvement stats
        'mean_improvement_ratio_energy': results_df['improvement_ratio_energy'].mean(),
        'median_improvement_ratio_energy': results_df['improvement_ratio_energy'].median(),
        'mean_improvement_delta_energy': results_df['improvement_delta_energy'].mean(),
        'median_improvement_delta_energy': results_df['improvement_delta_energy'].median(),
        # Energy kernel - Overall ranges
        'min_baseline_mmd_energy': results_df['baseline_mmd_energy'].min(),
        'max_baseline_mmd_energy': results_df['baseline_mmd_energy'].max(),
        'min_transport_mmd_energy': results_df['transport_mmd_energy'].min(),
        'max_transport_mmd_energy': results_df['transport_mmd_energy'].max(),
        # Technical duplicate statistics
        'n_combinations_with_tech_dup': len(tech_dup_results),
        'tech_dup_coverage': len(tech_dup_results) / len(results_df) if len(results_df) > 0 else 0,
        # Technical duplicate - RBF kernel stats
        'mean_technical_duplicate_mmd': tech_dup_results['technical_duplicate_mmd'].mean() if len(tech_dup_results) > 0 else None,
        'median_technical_duplicate_mmd': tech_dup_results['technical_duplicate_mmd'].median() if len(tech_dup_results) > 0 else None,
        'std_technical_duplicate_mmd': tech_dup_results['technical_duplicate_mmd'].std() if len(tech_dup_results) > 0 else None,
        'min_technical_duplicate_mmd': tech_dup_results['technical_duplicate_mmd'].min() if len(tech_dup_results) > 0 else None,
        'max_technical_duplicate_mmd': tech_dup_results['technical_duplicate_mmd'].max() if len(tech_dup_results) > 0 else None,
        # Technical duplicate - Energy kernel stats
        'mean_technical_duplicate_mmd_energy': tech_dup_results['technical_duplicate_mmd_energy'].mean() if len(tech_dup_results) > 0 else None,
        'median_technical_duplicate_mmd_energy': tech_dup_results['technical_duplicate_mmd_energy'].median() if len(tech_dup_results) > 0 else None,
        'std_technical_duplicate_mmd_energy': tech_dup_results['technical_duplicate_mmd_energy'].std() if len(tech_dup_results) > 0 else None,
        'min_technical_duplicate_mmd_energy': tech_dup_results['technical_duplicate_mmd_energy'].min() if len(tech_dup_results) > 0 else None,
        'max_technical_duplicate_mmd_energy': tech_dup_results['technical_duplicate_mmd_energy'].max() if len(tech_dup_results) > 0 else None
    }
    
    # Per cell line analysis
    cell_line_stats = (results_df
                      .groupby('cell_line')
                      .agg({
                          'baseline_mmd': 'mean',
                          'transport_mmd': 'mean',
                          'technical_duplicate_mmd': 'mean',
                          'improvement_ratio': 'mean',
                          'improvement_delta': 'mean',
                          'baseline_mmd_energy': 'mean',
                          'transport_mmd_energy': 'mean',
                          'technical_duplicate_mmd_energy': 'mean',
                          'improvement_ratio_energy': 'mean',
                          'improvement_delta_energy': 'mean',
                          'drug_dose': 'count',
                          'has_technical_duplicate': 'sum'
                      })
                      .rename(columns={
                          'baseline_mmd': 'mean_baseline_mmd',
                          'transport_mmd': 'mean_transport_mmd',
                          'technical_duplicate_mmd': 'mean_technical_duplicate_mmd',
                          'improvement_ratio': 'mean_improvement_ratio',
                          'improvement_delta': 'mean_improvement_delta',
                          'baseline_mmd_energy': 'mean_baseline_mmd_energy',
                          'transport_mmd_energy': 'mean_transport_mmd_energy',
                          'technical_duplicate_mmd_energy': 'mean_technical_duplicate_mmd_energy',
                          'improvement_ratio_energy': 'mean_improvement_ratio_energy',
                          'improvement_delta_energy': 'mean_improvement_delta_energy',
                          'drug_dose': 'n_combinations',
                          'has_technical_duplicate': 'n_tech_dup_combinations'
                      })
                      .sort_values('mean_improvement_ratio', ascending=False))
    
    # Per drug analysis (best improvements)
    drug_stats = (results_df
                 .groupby('drug_dose')
                 .agg({
                     'baseline_mmd': 'mean',
                     'transport_mmd': 'mean',
                     'technical_duplicate_mmd': 'mean',
                     'improvement_ratio': 'mean',
                     'improvement_delta': 'mean',
                     'baseline_mmd_energy': 'mean',
                     'transport_mmd_energy': 'mean',
                     'technical_duplicate_mmd_energy': 'mean',
                     'improvement_ratio_energy': 'mean',
                     'improvement_delta_energy': 'mean',
                     'cell_line': 'count',
                     'has_technical_duplicate': 'sum'
                 })
                 .rename(columns={
                     'baseline_mmd': 'mean_baseline_mmd',
                     'transport_mmd': 'mean_transport_mmd',
                     'technical_duplicate_mmd': 'mean_technical_duplicate_mmd',
                     'improvement_ratio': 'mean_improvement_ratio',
                     'improvement_delta': 'mean_improvement_delta',
                     'baseline_mmd_energy': 'mean_baseline_mmd_energy',
                     'transport_mmd_energy': 'mean_transport_mmd_energy',
                     'technical_duplicate_mmd_energy': 'mean_technical_duplicate_mmd_energy',
                     'improvement_ratio_energy': 'mean_improvement_ratio_energy',
                     'improvement_delta_energy': 'mean_improvement_delta_energy',
                     'cell_line': 'n_combinations',
                     'has_technical_duplicate': 'n_tech_dup_combinations'
                 })
                 .sort_values('mean_improvement_ratio', ascending=False))
    
    print(f"\n{'='*60}")
    print("DUAL-KERNEL MMD EVALUATION RESULTS (RBF + Energy) - H5AD Test Version")
    print("="*60)
    
    print(f"Overall Performance:")
    print(f"  Combinations evaluated: {overall_stats['n_combinations']:,}")
    print(f"  Combinations with technical duplicates: {overall_stats['n_combinations_with_tech_dup']:,} ({overall_stats['tech_dup_coverage']:.1%})")
    print(f"  ")
    print(f"  === RBF KERNEL RESULTS ===")
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
    print(f"  ")
    print(f"  === ENERGY KERNEL RESULTS ===")
    print(f"  Baseline MMD (DMSO vs Drug):")
    print(f"    Mean: {overall_stats['mean_baseline_mmd_energy']:.6f}")
    print(f"    Median: {overall_stats['median_baseline_mmd_energy']:.6f}")
    print(f"    Range: [{overall_stats['min_baseline_mmd_energy']:.6f}, {overall_stats['max_baseline_mmd_energy']:.6f}]")
    print(f"  ")
    print(f"  Transport MMD (Transported vs Drug):")
    print(f"    Mean: {overall_stats['mean_transport_mmd_energy']:.6f}")
    print(f"    Median: {overall_stats['median_transport_mmd_energy']:.6f}")
    print(f"    Range: [{overall_stats['min_transport_mmd_energy']:.6f}, {overall_stats['max_transport_mmd_energy']:.6f}]")
    print(f"  ")
    print(f"  Transport Improvement:")
    print(f"    Mean ratio (baseline/transport): {overall_stats['mean_improvement_ratio_energy']:.3f}x")
    print(f"    Median ratio: {overall_stats['median_improvement_ratio_energy']:.3f}x")
    print(f"    Mean delta (baseline - transport): {overall_stats['mean_improvement_delta_energy']:.6f}")
    
    # Technical duplicate results (if available)
    if overall_stats['n_combinations_with_tech_dup'] > 0:
        print(f"  ")
        print(f"  === TECHNICAL DUPLICATE RESULTS (Ideal Performance Target) ===")
        print(f"  Technical Duplicate MMD - RBF Kernel (Rep1 vs Rep2):")
        print(f"    Mean: {overall_stats['mean_technical_duplicate_mmd']:.6f}")
        print(f"    Median: {overall_stats['median_technical_duplicate_mmd']:.6f}")
        print(f"    Range: [{overall_stats['min_technical_duplicate_mmd']:.6f}, {overall_stats['max_technical_duplicate_mmd']:.6f}]")
        print(f"  ")
        print(f"  Technical Duplicate MMD - Energy Kernel (Rep1 vs Rep2):")
        print(f"    Mean: {overall_stats['mean_technical_duplicate_mmd_energy']:.6f}")
        print(f"    Median: {overall_stats['median_technical_duplicate_mmd_energy']:.6f}")
        print(f"    Range: [{overall_stats['min_technical_duplicate_mmd_energy']:.6f}, {overall_stats['max_technical_duplicate_mmd_energy']:.6f}]")
        print(f"  ")
        print(f"  NOTE: Technical duplicate values represent the 'noise floor' - what MMD")
        print(f"        values we'd expect for perfect predictions (technical replicates).")
        print(f"        Transport MMD values close to technical duplicate indicate excellent predictions.")
    else:
        print(f"  ")
        print(f"  === TECHNICAL DUPLICATE RESULTS ===")
        print(f"  No combinations had sufficient cells (≥{args.min_tech_dup_cells}) for technical duplicate computation.")
    
    print(f"\n=== BEST CELL LINES (RBF kernel improvement ratio) ===")
    for cell_line in cell_line_stats.head(5).index:
        row = cell_line_stats.loc[cell_line]
        print(f"  {cell_line} ({row['n_tech_dup_combinations']}/{row['n_combinations']} with tech dup):")
        print(f"    RBF: {row['mean_improvement_ratio']:.3f}x improvement")
        print(f"      Baseline: {row['mean_baseline_mmd']:.6f} → Transport: {row['mean_transport_mmd']:.6f}")
        if pd.notna(row['mean_technical_duplicate_mmd']):
            print(f"      Tech Dup: {row['mean_technical_duplicate_mmd']:.6f}")
        print(f"    Energy: {row['mean_improvement_ratio_energy']:.3f}x improvement")
        print(f"      Baseline: {row['mean_baseline_mmd_energy']:.6f} → Transport: {row['mean_transport_mmd_energy']:.6f}")
        if pd.notna(row['mean_technical_duplicate_mmd_energy']):
            print(f"      Tech Dup: {row['mean_technical_duplicate_mmd_energy']:.6f}")
    
    print(f"\n=== WORST CELL LINES (RBF kernel improvement ratio) ===")
    for cell_line in cell_line_stats.tail(5).index:
        row = cell_line_stats.loc[cell_line]
        print(f"  {cell_line} ({row['n_tech_dup_combinations']}/{row['n_combinations']} with tech dup):")
        print(f"    RBF: {row['mean_improvement_ratio']:.3f}x improvement")
        print(f"      Baseline: {row['mean_baseline_mmd']:.6f} → Transport: {row['mean_transport_mmd']:.6f}")
        if pd.notna(row['mean_technical_duplicate_mmd']):
            print(f"      Tech Dup: {row['mean_technical_duplicate_mmd']:.6f}")
        print(f"    Energy: {row['mean_improvement_ratio_energy']:.3f}x improvement")
        print(f"      Baseline: {row['mean_baseline_mmd_energy']:.6f} → Transport: {row['mean_transport_mmd_energy']:.6f}")
        if pd.notna(row['mean_technical_duplicate_mmd_energy']):
            print(f"      Tech Dup: {row['mean_technical_duplicate_mmd_energy']:.6f}")
    
    print(f"\n=== BEST DRUGS (RBF kernel improvement ratio) ===")
    for drug_dose in drug_stats.head(5).index:
        row = drug_stats.loc[drug_dose]
        print(f"  {drug_dose} ({row['n_tech_dup_combinations']}/{row['n_combinations']} with tech dup):")
        print(f"    RBF: {row['mean_improvement_ratio']:.3f}x improvement")
        print(f"    Energy: {row['mean_improvement_ratio_energy']:.3f}x improvement")
        if pd.notna(row['mean_technical_duplicate_mmd']):
            print(f"    Tech Dup RBF: {row['mean_technical_duplicate_mmd']:.6f}, Energy: {row['mean_technical_duplicate_mmd_energy']:.6f}")
    
    print(f"\n=== WORST DRUGS (RBF kernel improvement ratio) ===")
    for drug_dose in drug_stats.tail(5).index:
        row = drug_stats.loc[drug_dose]
        print(f"  {drug_dose} ({row['n_tech_dup_combinations']}/{row['n_combinations']} with tech dup):")
        print(f"    RBF: {row['mean_improvement_ratio']:.3f}x improvement")
        print(f"    Energy: {row['mean_improvement_ratio_energy']:.3f}x improvement")
        if pd.notna(row['mean_technical_duplicate_mmd']):
            print(f"    Tech Dup RBF: {row['mean_technical_duplicate_mmd']:.6f}, Energy: {row['mean_technical_duplicate_mmd_energy']:.6f}")
    
    return overall_stats, results_df, cell_line_stats, drug_stats

def save_results(results: List[dict], overall_stats: dict, args):
    """Save evaluation results."""
    print(f"\nSaving results...")
    
    Path(args.output_dir).mkdir(exist_ok=True)
    
    # Save detailed results
    results_df = pd.DataFrame(results)
    results_path = Path(args.output_dir) / args.results_file
    results_df.to_parquet(results_path)
    
    # Save summary statistics
    summary_path = Path(args.output_dir) / args.summary_file
    with open(summary_path, 'w') as f:
        json.dump(overall_stats, f, indent=2)
    
    print(f"✓ Results saved to: {results_path}")
    print(f"✓ Summary saved to: {summary_path}")
    
    return results_path

def main():
    """Main evaluation function."""
    try:
        # Parse arguments
        args = parse_args()
        
        # Run evaluation
        results = evaluate_all_combinations(args)
        
        # Analyze results
        overall_stats, results_df, cell_line_stats, drug_stats = analyze_results(results)
        
        # Save results
        results_path = save_results(results, overall_stats, args)
        
        print(f"\n{'='*60}")
        print("EVALUATION COMPLETE")
        print("="*60)
        print(f"✓ Dual-kernel MMD evaluation completed for {len(results)} combinations")
        print(f"✓ Both RBF and Energy kernel results computed")
        print(f"✓ Results saved to: {results_path}")
        print("✓ Transport quality assessment with baseline comparison complete!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()