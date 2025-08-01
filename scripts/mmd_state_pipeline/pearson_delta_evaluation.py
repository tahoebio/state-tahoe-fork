#!/usr/bin/env python3
"""
Pearson Delta evaluation metric for state prediction models.

This script implements the Pearson Delta evaluation metric which assesses the accuracy 
of predicted gene expression changes induced by perturbations. The metric quantifies 
the similarity between observed (true) and predicted perturbation-induced expression 
deltas across all tested perturbations.

Implementation steps:
1. Load predicted, test, and DMSO expanded control h5ad files (all have .obsm['X_hvg'] with log-normalized HVG data)
2. Calculate pseudobulks for each drug-cell combination in all three files
3. Compute signed expression deltas using same DMSO baseline:
   - True deltas: test - dmso_expanded (preserves up/down regulation)
   - Predicted deltas: predicted - dmso_expanded (preserves up/down regulation)
4. Calculate Pearson correlation between true and predicted deltas

The X_hvg field contains log-normalized expression data filtered to highly variable genes.
All three files have identical drug-cell combination structure for direct comparison.
"""

import json
import pandas as pd
import numpy as np
import scanpy as sc
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

# Configuration
CONFIG = {
    # Data paths - will be set via command line
    'predicted_data': None,  # Path to predicted h5ad file
    'test_data': None,       # Path to test h5ad file
    'dmso_controls_data': None,  # Path to dmso_controls_expanded_for_inference.h5ad file
    
    # Gene expression parameters
    'hvg_key': 'X_hvg',  # Key in obsm containing log-normalized HVG data
    'pred_key': 'model_preds',  # Key in obsm containing model predictions
    
    # Evaluation parameters
    'min_cells_for_pseudobulk': 10,  # Minimum cells needed for reliable pseudobulk
    'log_scale_inputs': False,  # Whether to apply log10(x+1) transformation to input values
    
    # Output
    'output_dir': Path("pearson_delta_evaluation"),
    'results_file': "pearson_delta_results.parquet",
    'summary_file': "pearson_delta_summary.json",
    'plots_dir': "plots"
}

def load_h5ad_data(file_path: str, data_type: str) -> sc.AnnData:
    """Load h5ad file and validate it has the required X_hvg field."""
    print(f"Loading {data_type} data from: {file_path}")
    
    if not Path(file_path).exists():
        raise FileNotFoundError(f"{data_type} data not found: {file_path}")
    
    adata = sc.read_h5ad(file_path)
    
    # Validate X_hvg field exists
    if CONFIG['hvg_key'] not in adata.obsm:
        raise ValueError(f"Required field '{CONFIG['hvg_key']}' not found in {data_type} data")
    
    print(f"✓ {data_type} data loaded: {adata.shape[0]:,} cells × {adata.shape[1]:,} genes")
    print(f"  HVG data shape: {adata.obsm[CONFIG['hvg_key']].shape}")
    
    if 'drug_dose' in adata.obs.columns:
        print(f"  Unique drug_dose combinations: {adata.obs['drug_dose'].nunique()}")
    if 'cell_line' in adata.obs.columns:
        print(f"  Unique cell_line combinations: {adata.obs['cell_line'].nunique()}")
    
    return adata

def create_evaluation_plan(predicted_adata: sc.AnnData, test_adata: sc.AnnData, dmso_adata: sc.AnnData) -> pd.DataFrame:
    """Create evaluation plan based on available perturbations in all three datasets."""
    print("Creating evaluation plan...")
    
    # Get unique perturbations from predicted data
    predicted_perturbations = predicted_adata.obs.groupby(['drug_dose', 'cell_line']).size().reset_index(name='n_predicted_cells')
    
    # Get unique perturbations from test data
    test_perturbations = test_adata.obs.groupby(['drug_dose', 'cell_line']).size().reset_index(name='n_test_cells')
    
    # Get unique perturbations from DMSO expanded data
    dmso_perturbations = dmso_adata.obs.groupby(['drug_dose', 'cell_line']).size().reset_index(name='n_dmso_cells')
    
    # Create evaluation plan - find common perturbations across all three datasets
    evaluation_plan = predicted_perturbations.merge(
        test_perturbations,
        on=['drug_dose', 'cell_line'],
        how='inner'
    ).merge(
        dmso_perturbations,
        on=['drug_dose', 'cell_line'],
        how='inner'
    )
    
    # Filter to combinations with sufficient cells
    evaluation_plan = evaluation_plan[
        (evaluation_plan['n_predicted_cells'] >= CONFIG['min_cells_for_pseudobulk']) &
        (evaluation_plan['n_test_cells'] >= CONFIG['min_cells_for_pseudobulk']) &
        (evaluation_plan['n_dmso_cells'] >= CONFIG['min_cells_for_pseudobulk'])
    ]
    
    print(f"✓ Evaluation plan created: {len(evaluation_plan)} perturbations")
    print(f"  Unique drugs: {evaluation_plan['drug_dose'].nunique()}")
    print(f"  Unique cell lines: {evaluation_plan['cell_line'].nunique()}")
    
    if len(evaluation_plan) == 0:
        print("⚠ No valid perturbations found with sufficient cells in all three datasets")
        
    return evaluation_plan

def calculate_pseudobulk(adata: sc.AnnData, embed_key: str, condition_col: str = 'drug_dose', 
                        cell_line_col: str = 'cell_line') -> pd.DataFrame:
    """Calculate pseudobulk expression profiles by averaging cells within groups (vectorized)."""
    print(f"Calculating pseudobulks using embedding key: {embed_key}")
    
    # Get expression data
    if embed_key not in adata.obsm:
        raise ValueError(f"Embedding key '{embed_key}' not found in obsm. Available keys: {list(adata.obsm.keys())}")
    
    expr_data = adata.obsm[embed_key]
    print(f"  Expression data shape: {expr_data.shape}")
    
    # Apply log10(x+1) transformation if enabled
    if CONFIG['log_scale_inputs']:
        print(f"  Applying log10(x+1) transformation to input values")
        # Ensure non-negative values before log transformation
        if np.any(expr_data < 0):
            print(f"  Warning: Found {np.sum(expr_data < 0)} negative values, setting to 0 before log transformation")
            expr_data = np.maximum(expr_data, 0)
        expr_data = np.log10(expr_data + 1)
        print(f"  After log scaling - min: {np.min(expr_data):.4f}, max: {np.max(expr_data):.4f}, mean: {np.mean(expr_data):.4f}")
    
    # Create combined grouping column for efficiency
    adata.obs['_group_key'] = adata.obs[condition_col].astype(str) + '__' + adata.obs[cell_line_col].astype(str)
    
    # Get unique groups
    unique_groups = adata.obs['_group_key'].unique()
    
    pseudobulks = []
    
    for group_key in tqdm(unique_groups, desc="Computing pseudobulks"):
        # Get mask for this group
        group_mask = adata.obs['_group_key'] == group_key
        
        # Extract group expression data
        group_expr = expr_data[group_mask]
        
        # Calculate mean expression (vectorized)
        mean_expr = np.mean(group_expr, axis=0)
        
        # Parse back the condition and cell_line
        condition, cell_line = group_key.split('__')
        
        pseudobulk_data = {
            condition_col: condition,
            cell_line_col: cell_line,
            'n_cells': group_mask.sum(),
            'mean_expression': mean_expr
        }
        
        pseudobulks.append(pseudobulk_data)
    
    pseudobulk_df = pd.DataFrame(pseudobulks)
    
    print(f"✓ Computed {len(pseudobulk_df)} pseudobulks")
    return pseudobulk_df

def compute_expression_deltas_direct(perturbation_pseudobulks: pd.DataFrame, 
                                   control_pseudobulks: pd.DataFrame,
                                   data_type: str) -> pd.DataFrame:
    """Compute signed expression deltas (log fold changes) using direct matching between perturbation and control pseudobulks."""
    print(f"Computing {data_type} expression deltas...")
    
    deltas = []
    
    for _, pert_row in tqdm(perturbation_pseudobulks.iterrows(), 
                           total=len(perturbation_pseudobulks), 
                           desc=f"Computing {data_type} deltas"):
        
        # Find matching control with same drug_dose and cell_line
        control_mask = ((control_pseudobulks['drug_dose'] == pert_row['drug_dose']) & 
                       (control_pseudobulks['cell_line'] == pert_row['cell_line']))
        
        if not control_mask.any():
            print(f"⚠ No control found for {pert_row['drug_dose']} + {pert_row['cell_line']}")
            continue
        
        control_row = control_pseudobulks[control_mask].iloc[0]
        
        # Calculate signed difference for each gene (log fold change)
        pert_expr = pert_row['mean_expression']
        ctrl_expr = control_row['mean_expression']
        
        # Element-wise signed difference with NaN checking (preserves up/down regulation)
        gene_deltas = pert_expr - ctrl_expr
        
        # Check for NaN values and report
        if np.any(np.isnan(gene_deltas)):
            print(f"⚠ NaN values found in deltas for {pert_row['drug_dose']} + {pert_row['cell_line']}")
            print(f"  NaN count: {np.sum(np.isnan(gene_deltas))}/{len(gene_deltas)}")
            print(f"  Pert expr NaN count: {np.sum(np.isnan(pert_expr))}")
            print(f"  Ctrl expr NaN count: {np.sum(np.isnan(ctrl_expr))}")
        
        # Store results
        delta_data = {
            'drug_dose': pert_row['drug_dose'],
            'cell_line': pert_row['cell_line'],
            'n_pert_cells': pert_row['n_cells'],
            'n_ctrl_cells': control_row['n_cells'],
            'expression_deltas': gene_deltas,
            'delta_mean': np.nanmean(gene_deltas),
            'delta_std': np.nanstd(gene_deltas),
            'delta_nan_count': np.sum(np.isnan(gene_deltas))
        }
        
        deltas.append(delta_data)
    
    deltas_df = pd.DataFrame(deltas)
    
    print(f"✓ Computed {data_type} expression deltas for {len(deltas_df)} perturbations")
    if len(deltas_df) > 0:
        print(f"  Delta stats: mean={deltas_df['delta_mean'].mean():.4f}, std={deltas_df['delta_std'].mean():.4f}")
        print(f"  Total NaN deltas: {deltas_df['delta_nan_count'].sum():,}")
    
    return deltas_df

def compute_technical_duplicate_correlation(test_adata: sc.AnnData, dmso_adata: sc.AnnData, 
                                          test_key: str, dmso_key: str) -> Tuple[float, int, int]:
    """Compute Pearson correlation between technical duplicate signed expression deltas (log fold changes)."""
    print("Computing technical duplicate correlation...")
    
    # Get combinations with sufficient cells for technical duplicates
    test_combinations = test_adata.obs.groupby(['drug_dose', 'cell_line']).size().reset_index(name='n_test_cells')
    valid_combinations = test_combinations[test_combinations['n_test_cells'] >= CONFIG['min_tech_dup_cells']]
    
    print(f"Found {len(valid_combinations)} combinations with ≥{CONFIG['min_tech_dup_cells']} cells for technical duplicates")
    
    if len(valid_combinations) == 0:
        print("⚠ No combinations have sufficient cells for technical duplicate computation")
        return 0.0, 0, 0
    
    # Calculate pseudobulks for DMSO controls
    print("Calculating DMSO pseudobulks for technical duplicate computation...")
    dmso_pseudobulks = calculate_pseudobulk(dmso_adata, dmso_key)
    
    # Collect all technical duplicate deltas
    rep1_deltas_all = []
    rep2_deltas_all = []
    n_combinations_used = 0
    
    for _, row in tqdm(valid_combinations.iterrows(), total=len(valid_combinations), 
                      desc="Computing technical duplicate deltas"):
        drug_dose = row['drug_dose']
        cell_line = row['cell_line']
        
        try:
            # Get cells for this combination
            combination_mask = ((test_adata.obs['drug_dose'] == drug_dose) & 
                              (test_adata.obs['cell_line'] == cell_line))
            combination_indices = np.where(combination_mask)[0]
            
            # Randomly split into two groups
            np.random.seed(42)  # For reproducibility
            shuffled_indices = np.random.permutation(combination_indices)
            split_point = len(shuffled_indices) // 2
            
            rep1_indices = shuffled_indices[:split_point]
            rep2_indices = shuffled_indices[split_point:]
            
            # Extract expression data for each replicate
            rep1_expr = test_adata.obsm[test_key][rep1_indices]
            rep2_expr = test_adata.obsm[test_key][rep2_indices]
            
            # Calculate pseudobulks
            rep1_pseudobulk = np.mean(rep1_expr, axis=0)
            rep2_pseudobulk = np.mean(rep2_expr, axis=0)
            
            # Find matching DMSO control
            dmso_mask = ((dmso_pseudobulks['drug_dose'] == drug_dose) & 
                        (dmso_pseudobulks['cell_line'] == cell_line))
            
            if not dmso_mask.any():
                print(f"⚠ No DMSO control found for {drug_dose} + {cell_line}")
                continue
                
            dmso_row = dmso_pseudobulks[dmso_mask].iloc[0]
            dmso_pseudobulk = dmso_row['mean_expression']
            
            # Compute signed deltas for each replicate (log fold changes)
            rep1_deltas = rep1_pseudobulk - dmso_pseudobulk
            rep2_deltas = rep2_pseudobulk - dmso_pseudobulk
            
            # Check for NaN values
            if np.any(np.isnan(rep1_deltas)) or np.any(np.isnan(rep2_deltas)):
                print(f"⚠ NaN values found in technical duplicate deltas for {drug_dose} + {cell_line}")
                continue
            
            # Add to collection
            rep1_deltas_all.extend(rep1_deltas)
            rep2_deltas_all.extend(rep2_deltas)
            n_combinations_used += 1
            
        except Exception as e:
            print(f"⚠ Failed to compute technical duplicate for {drug_dose} + {cell_line}: {e}")
            continue
    
    # Convert to numpy arrays
    rep1_deltas_all = np.array(rep1_deltas_all)
    rep2_deltas_all = np.array(rep2_deltas_all)
    
    print(f"Technical duplicate computation used {n_combinations_used} combinations")
    print(f"Technical duplicate arrays: {len(rep1_deltas_all):,} values each")
    
    # Remove NaN values
    valid_mask = ~(np.isnan(rep1_deltas_all) | np.isnan(rep2_deltas_all))
    rep1_clean = rep1_deltas_all[valid_mask]
    rep2_clean = rep2_deltas_all[valid_mask]
    
    print(f"Valid technical duplicate values: {len(rep1_clean):,}")
    
    # Compute correlation
    if len(rep1_clean) > 1:
        correlation, p_value = pearsonr(rep1_clean, rep2_clean)
        print(f"✓ Technical duplicate correlation: {correlation:.4f} (p={p_value:.2e})")
        return correlation, len(rep1_clean), n_combinations_used
    else:
        print("⚠ No valid technical duplicate pairs found")
        return 0.0, 0, 0

def compute_pearson_delta_correlation(true_deltas_df: pd.DataFrame, 
                                    pred_deltas_df: pd.DataFrame) -> Tuple[float, int]:
    """Compute Pearson correlation between true and predicted signed expression deltas (log fold changes)."""
    print("Computing Pearson Delta correlation...")
    
    # Align perturbations between true and predicted
    common_perturbations = pd.merge(
        true_deltas_df[['drug_dose', 'cell_line']],
        pred_deltas_df[['drug_dose', 'cell_line']],
        on=['drug_dose', 'cell_line'],
        how='inner'
    )
    
    print(f"Found {len(common_perturbations)} common perturbations")
    
    # Collect all delta values
    true_deltas_all = []
    pred_deltas_all = []
    
    for _, row in common_perturbations.iterrows():
        drug_dose = row['drug_dose']
        cell_line = row['cell_line']
        
        # Get true deltas
        true_mask = (true_deltas_df['drug_dose'] == drug_dose) & (true_deltas_df['cell_line'] == cell_line)
        true_row = true_deltas_df[true_mask].iloc[0]
        
        # Get predicted deltas
        pred_mask = (pred_deltas_df['drug_dose'] == drug_dose) & (pred_deltas_df['cell_line'] == cell_line)
        pred_row = pred_deltas_df[pred_mask].iloc[0]
        
        # Add all gene deltas to the collection
        true_deltas_all.extend(true_row['expression_deltas'])
        pred_deltas_all.extend(pred_row['expression_deltas'])
    
    # Convert to numpy arrays for easier handling
    true_deltas_all = np.array(true_deltas_all)
    pred_deltas_all = np.array(pred_deltas_all)
    
    # Check for NaN values
    true_nan_count = np.sum(np.isnan(true_deltas_all))
    pred_nan_count = np.sum(np.isnan(pred_deltas_all))
    
    print(f"Delta arrays: {len(true_deltas_all):,} values each")
    print(f"True deltas NaN count: {true_nan_count:,}")
    print(f"Pred deltas NaN count: {pred_nan_count:,}")
    
    # Remove NaN values
    valid_mask = ~(np.isnan(true_deltas_all) | np.isnan(pred_deltas_all))
    true_deltas_clean = true_deltas_all[valid_mask]
    pred_deltas_clean = pred_deltas_all[valid_mask]
    
    print(f"Valid (non-NaN) values: {len(true_deltas_clean):,}")
    
    # Compute Pearson correlation
    if len(true_deltas_clean) > 1:
        correlation, p_value = pearsonr(true_deltas_clean, pred_deltas_clean)
        print(f"✓ Pearson Delta correlation: {correlation:.4f} (p={p_value:.2e})")
        return correlation, len(true_deltas_clean)
    else:
        print("⚠ No valid delta pairs found after NaN removal")
        return 0.0, 0

def evaluate_pearson_delta(predicted_path: str, test_path: str, dmso_path: str):
    """Main evaluation function for Pearson Delta metric using three-file approach."""
    print("="*60)
    print("PEARSON DELTA EVALUATION (Three-file approach with correct embedding keys)")
    print("="*60)
    
    # Load data
    predicted_adata = load_h5ad_data(predicted_path, "predicted")
    test_adata = load_h5ad_data(test_path, "test")
    dmso_adata = load_h5ad_data(dmso_path, "DMSO expanded controls")
    
    # Define embedding keys
    pred_key = CONFIG['pred_key']  # 'model_preds' - actual predictions
    test_key = CONFIG['hvg_key']   # 'X_hvg' - real drug-treated cells
    dmso_key = CONFIG['hvg_key']   # 'X_hvg' - DMSO control cells
    
    print(f"Embedding keys: predicted='{pred_key}', test='{test_key}', dmso='{dmso_key}'")
    
    # Validate embedding keys exist
    if pred_key not in predicted_adata.obsm:
        raise ValueError(f"Predicted data missing key '{pred_key}'. Available keys: {list(predicted_adata.obsm.keys())}")
    if test_key not in test_adata.obsm:
        raise ValueError(f"Test data missing key '{test_key}'. Available keys: {list(test_adata.obsm.keys())}")
    if dmso_key not in dmso_adata.obsm:
        raise ValueError(f"DMSO data missing key '{dmso_key}'. Available keys: {list(dmso_adata.obsm.keys())}")
    
    # Verify dimensions match
    pred_shape = predicted_adata.obsm[pred_key].shape[1]
    test_shape = test_adata.obsm[test_key].shape[1]
    dmso_shape = dmso_adata.obsm[dmso_key].shape[1]
    
    if not (pred_shape == test_shape == dmso_shape):
        raise ValueError(f"Dimensions don't match: predicted={pred_shape}, test={test_shape}, dmso={dmso_shape}")
    
    print(f"Using {pred_shape} features")
    
    # Create evaluation plan
    evaluation_plan = create_evaluation_plan(predicted_adata, test_adata, dmso_adata)
    
    if len(evaluation_plan) == 0:
        print("No valid perturbations found for evaluation")
        return None
    
    # Calculate pseudobulks for all three datasets with correct embedding keys
    print("\nCalculating predicted pseudobulks...")
    predicted_pseudobulks = calculate_pseudobulk(predicted_adata, pred_key)
    
    print("\nCalculating test pseudobulks...")
    test_pseudobulks = calculate_pseudobulk(test_adata, test_key)
    
    print("\nCalculating DMSO expanded pseudobulks...")
    dmso_pseudobulks = calculate_pseudobulk(dmso_adata, dmso_key)
    
    # Compute expression deltas using same DMSO baseline
    print("\nComputing expression deltas...")
    predicted_deltas = compute_expression_deltas_direct(predicted_pseudobulks, dmso_pseudobulks, "predicted")
    test_deltas = compute_expression_deltas_direct(test_pseudobulks, dmso_pseudobulks, "test")
    
    # Compute Pearson Delta correlation
    print("\nComputing Pearson Delta correlation...")
    correlation, n_comparisons = compute_pearson_delta_correlation(test_deltas, predicted_deltas)
    
    # Compute technical duplicate correlation
    print("\nComputing technical duplicate correlation...")
    tech_dup_correlation, n_tech_dup_comparisons, n_tech_dup_combinations = compute_technical_duplicate_correlation(
        test_adata, dmso_adata, test_key, dmso_key
    )
    
    # Compile results
    results = {
        'pearson_delta_correlation': correlation,
        'n_comparisons': n_comparisons,
        'n_predicted_perturbations': len(predicted_deltas),
        'n_test_perturbations': len(test_deltas),
        'n_features': pred_shape,
        'min_cells_for_pseudobulk': CONFIG['min_cells_for_pseudobulk'],
        'approach': 'three_file_with_correct_embedding_keys',
        'embedding_keys': {
            'predicted': pred_key,
            'test': test_key,
            'dmso': dmso_key
        },
        # Technical duplicate metrics
        'technical_duplicate_correlation': tech_dup_correlation,
        'n_tech_dup_comparisons': n_tech_dup_comparisons,
        'n_tech_dup_combinations': n_tech_dup_combinations,
        'tech_dup_coverage': n_tech_dup_combinations / len(predicted_deltas) if len(predicted_deltas) > 0 else 0,
        'min_tech_dup_cells': CONFIG['min_tech_dup_cells'],
        # Performance metrics
        'performance_ratio': correlation / tech_dup_correlation if tech_dup_correlation > 0 else 0
    }
    
    return results, predicted_deltas, test_deltas

def save_results(results: dict, predicted_deltas: pd.DataFrame, test_deltas: pd.DataFrame):
    """Save evaluation results and data."""
    print(f"\nSaving results...")
    
    CONFIG['output_dir'].mkdir(exist_ok=True)
    
    # Save deltas (drop expression arrays but keep stats)
    predicted_summary = predicted_deltas.drop('expression_deltas', axis=1)
    test_summary = test_deltas.drop('expression_deltas', axis=1)
    
    predicted_summary_path = CONFIG['output_dir'] / "predicted_deltas_summary.parquet"
    test_summary_path = CONFIG['output_dir'] / "test_deltas_summary.parquet"
    
    predicted_summary.to_parquet(predicted_summary_path)
    test_summary.to_parquet(test_summary_path)
    
    # Save summary results (handle NaN values in JSON)
    summary_path = CONFIG['output_dir'] / CONFIG['summary_file']
    
    # Convert NaN to None for JSON serialization
    results_json = results.copy()
    for key, value in results_json.items():
        if isinstance(value, float) and np.isnan(value):
            results_json[key] = None
    
    with open(summary_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"✓ Predicted deltas summary saved to: {predicted_summary_path}")
    print(f"✓ Test deltas summary saved to: {test_summary_path}")
    print(f"✓ Summary saved to: {summary_path}")

def main():
    """Main function with command line interface."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Pearson Delta correlation between predicted and test data using expanded DMSO controls")
    parser.add_argument("--predicted", required=True, help="Path to predicted h5ad file (dmso_controls_predicted_all_perturbations.h5ad)")
    parser.add_argument("--test", required=True, help="Path to test h5ad file (test_subset.h5ad)")
    parser.add_argument("--dmso-controls", required=True, help="Path to DMSO expanded controls h5ad file (dmso_controls_expanded_for_inference.h5ad)")
    parser.add_argument("--pred-key", default="model_preds", help="Embedding key for predictions in predicted file")
    parser.add_argument("--test-key", default="X_hvg", help="Embedding key for test data")
    parser.add_argument("--dmso-key", default="X_hvg", help="Embedding key for DMSO controls")
    parser.add_argument("--output-dir", default="pearson_delta_evaluation", help="Output directory")
    parser.add_argument("--min-cells", type=int, default=10, help="Minimum cells for pseudobulk")
    parser.add_argument("--min-tech-dup-cells", type=int, default=20, help="Minimum cells for technical duplicate computation")
    parser.add_argument("--log-scale-inputs", action="store_true", help="Apply log10(x+1) transformation to input values before correlation calculation")
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['predicted_data'] = args.predicted
    CONFIG['test_data'] = args.test
    CONFIG['dmso_controls_data'] = args.dmso_controls
    CONFIG['output_dir'] = Path(args.output_dir)
    CONFIG['min_cells_for_pseudobulk'] = args.min_cells
    CONFIG['min_tech_dup_cells'] = args.min_tech_dup_cells
    CONFIG['pred_key'] = args.pred_key
    CONFIG['hvg_key'] = args.test_key  # Use test_key for both test and dmso
    CONFIG['log_scale_inputs'] = args.log_scale_inputs
    
    try:
        # Run evaluation
        result = evaluate_pearson_delta(args.predicted, args.test, args.dmso_controls)
        
        if result is None:
            print("Evaluation failed - no valid perturbations found")
            return
        
        results, predicted_deltas, test_deltas = result
        
        # Save results
        save_results(results, predicted_deltas, test_deltas)
        
        print(f"\n{'='*60}")
        print("PEARSON DELTA EVALUATION COMPLETE")
        print("="*60)
        
        # Prediction correlation results
        if np.isnan(results['pearson_delta_correlation']):
            print(f"✓ Pearson Delta Correlation: NaN (check for data issues)")
        else:
            print(f"✓ Pearson Delta Correlation: {results['pearson_delta_correlation']:.4f}")
        print(f"✓ Comparisons made: {results['n_comparisons']:,}")
        
        # Technical duplicate correlation results  
        if results['n_tech_dup_combinations'] > 0:
            print(f"✓ Technical Duplicate Correlation: {results['technical_duplicate_correlation']:.4f}")
            print(f"✓ Tech Dup Comparisons: {results['n_tech_dup_comparisons']:,}")
            print(f"✓ Tech Dup Coverage: {results['n_tech_dup_combinations']}/{results['n_predicted_perturbations']} combinations ({results['tech_dup_coverage']:.1%})")
            print(f"✓ Performance Ratio: {results['performance_ratio']:.3f} (prediction/tech_dup)")
        else:
            print(f"⚠ Technical Duplicate Correlation: No combinations with ≥{results['min_tech_dup_cells']} cells")
        
        # Data summary
        print(f"✓ Predicted perturbations: {results['n_predicted_perturbations']:,}")
        print(f"✓ Test perturbations: {results['n_test_perturbations']:,}")
        print(f"✓ Features used: {results['n_features']:,}")
        print(f"✓ Embedding keys: {results['embedding_keys']}")
        
        # Interpretation guidance
        if results['n_tech_dup_combinations'] > 0:
            print(f"\n--- INTERPRETATION GUIDE ---")
            print(f"Technical duplicate correlation ({results['technical_duplicate_correlation']:.4f}) represents ideal performance")
            if results['performance_ratio'] > 0.8:
                print(f"→ EXCELLENT: Model correlation very close to biological limit")
            elif results['performance_ratio'] > 0.5:
                print(f"→ GOOD: Model captures substantial biological signal")
            elif results['performance_ratio'] > 0.2:
                print(f"→ MODERATE: Some biological signal captured, room for improvement")
            else:
                print(f"→ POOR: Model correlation much lower than biological limit")
        
        print("✓ Results saved successfully!")
        
    except Exception as e:
        print(f"ERROR: {e}")
        raise

if __name__ == "__main__":
    main()