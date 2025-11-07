#!/usr/bin/env python3
"""
Compute MMD distributional metrics for drug response prediction using AnnData pair format.

This script:
1. Loads real (observed) and predicted AnnData objects with perturbation data
2. Extracts control and treated cells from both datasets
3. Computes dual-kernel MMD (RBF + Energy) for baseline, transport, and control comparisons
4. Supports categorical grouping and celltype splitting
5. Uses sampling for computational efficiency

Based on pearson_delta_only.py interface but with MMD distributional metrics from
evaluate_transport_mmd_h5ad_test.py.
"""

import argparse
import sys
import os
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import scanpy as sc
import polars as pl
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
from geomloss import SamplesLoss

# Set up logging with timestamps
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)

# Configure root logger
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s:%(name)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute dual-kernel MMD metrics for drug response prediction using AnnData pair format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python mmd_anndata_pair.py \\
    --adata-real observed_data.h5ad \\
    --adata-pred predicted_data.h5ad \\
    --control-pert DMSO_TF

  # With grouping and sampling
  python mmd_anndata_pair.py \\
    --adata-real observed_data.h5ad \\
    --adata-pred predicted_data.h5ad \\
    --control-pert DMSO_TF \\
    --group-by plate batch \\
    --max-cells-per-group 300 \\
    --embed-key-real X_hvg \\
    --embed-key-pred model_preds
        """
    )
    
    # Core arguments (similar to pearson_delta_only.py)
    parser.add_argument(
        "--adata-pred",
        type=str,
        required=True,
        help="Path to the predicted adata object to evaluate",
    )
    parser.add_argument(
        "--adata-real", 
        type=str,
        required=True,
        help="Path to the real adata object to evaluate against",
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default="DMSO_TF",
        help="Name of the control perturbation [default: %(default)s]",
    )
    parser.add_argument(
        "--pert-col",
        type=str,
        default="perturbation",
        help="Name of the column designated perturbations [default: %(default)s]",
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        help="Name of the column designated celltype to split results by (optional)",
    )
    parser.add_argument(
        "--embed-key",
        type=str,
        help="Key for embedded data (.obsm) in both AnnData objects (evaluated over .X otherwise)",
    )
    parser.add_argument(
        "--embed-key-pred",
        type=str,
        help="Key for embedded data (.obsm) in predicted AnnData object (overrides --embed-key for pred)",
    )
    parser.add_argument(
        "--embed-key-real",
        type=str,
        help="Key for embedded data (.obsm) in real AnnData object (overrides --embed-key for real)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./mmd_anndata_pair_results",
        help="Output directory to write results [default: %(default)s]",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        help="Additional categorical columns to group by before calculating MMD (e.g., plate batch timepoint)",
    )
    
    # Sampling parameters
    parser.add_argument(
        "--max-cells-per-group",
        type=int,
        default=250,
        help="Maximum cells to sample per condition for efficiency [default: %(default)s]",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling [default: %(default)s]",
    )
    
    # MMD parameters
    parser.add_argument(
        "--min-cells",
        type=int,
        default=20,
        help="Minimum cells needed for reliable MMD [default: %(default)s]",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for computation (auto, cuda, cpu) [default: %(default)s]",
    )
    
    # RBF kernel parameters
    parser.add_argument(
        "--mmd-kernel-mul",
        type=float,
        default=2.0,
        help="RBF MMD kernel multiplier [default: %(default)s]",
    )
    parser.add_argument(
        "--mmd-kernel-num",
        type=int,
        default=5,
        help="RBF MMD kernel number [default: %(default)s]",
    )
    
    # Energy kernel parameters  
    parser.add_argument(
        "--energy-blur",
        type=float,
        default=0.05,
        help="Energy kernel blur parameter [default: %(default)s]",
    )
    parser.add_argument(
        "--energy-scaling",
        type=float,
        default=0.5,
        help="Energy kernel scaling parameter [default: %(default)s]",
    )
    parser.add_argument(
        "--energy-backend",
        default="auto",
        help="Energy kernel backend (auto, tensorized, online) [default: %(default)s]",
    )
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    
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
    """Compute RBF MMD loss between source and target distributions."""
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


def map_control_perturbation(control_pert: str) -> str:
    """Map control perturbation names to their actual values in h5ad files."""
    # Handle special case where CLI uses "DMSO_TF" but h5ad files use the full string
    if control_pert == "DMSO_TF":
        return "[('DMSO_TF', 0.0, 'uM')]"
    return control_pert


def prepare_embeddings(adata_real, adata_pred, embed_key_real=None, embed_key_pred=None, embed_key=None):
    """Prepare embeddings so both datasets use the same key for MMD analysis."""
    # Determine which keys to use
    real_key = embed_key_real or embed_key
    pred_key = embed_key_pred or embed_key
    
    # If different keys specified, copy pred data to match real key
    if real_key and pred_key and real_key != pred_key:
        logger.info(f"Using different embedding keys: real='{real_key}', pred='{pred_key}'")
        logger.info(f"Copying pred['{pred_key}'] to pred['{real_key}'] for MMD analysis")
        
        if pred_key not in adata_pred.obsm:
            raise ValueError(f"Predicted data missing embedding key '{pred_key}'")
        if real_key not in adata_real.obsm:
            raise ValueError(f"Real data missing embedding key '{real_key}'")
            
        # Copy pred embedding to match real key name
        adata_pred.obsm[real_key] = adata_pred.obsm[pred_key]
        return real_key
    
    # Same key for both or no specific keys
    return real_key or pred_key


def sample_cells(adata, mask, max_cells, seed=42):
    """Sample cells from AnnData object based on mask and max cell limit."""
    indices = np.where(mask)[0]
    
    if len(indices) <= max_cells:
        # Use all available cells
        return indices
    else:
        # Sample without replacement
        np.random.seed(seed)
        return np.random.choice(indices, max_cells, replace=False)


def create_compound_grouping_key(adata, pert_col: str, group_by_cols: list = None):
    """Create compound grouping keys combining perturbation with additional categorical columns."""
    if group_by_cols is None:
        return adata.obs[pert_col].astype(str)
    
    # Start with perturbation column
    compound_key = adata.obs[pert_col].astype(str)
    
    # Add each grouping column separated by "::"
    for col in group_by_cols:
        if col not in adata.obs.columns:
            raise ValueError(f"Grouping column '{col}' not found in AnnData.obs")
        compound_key = compound_key + "::" + adata.obs[col].astype(str)
    
    return compound_key


def parse_compound_key(compound_key: str):
    """Parse a compound key back into perturbation and group components."""
    parts = compound_key.split("::")
    perturbation = parts[0]
    group_suffix = "::".join(parts[1:]) if len(parts) > 1 else None
    return perturbation, group_suffix


def extract_embeddings(adata, indices, embed_key, device):
    """Extract embeddings from AnnData object for given indices."""
    if embed_key is None:
        # Use .X
        if hasattr(adata.X, 'toarray'):
            matrix = adata.X[indices].toarray()
        else:
            matrix = adata.X[indices]
    else:
        # Use .obsm[embed_key]
        matrix = adata.obsm[embed_key][indices]
    
    # Convert to tensor
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)
    
    return matrix.to(device)


def compute_dual_mmd_for_combination(
    real_adata, pred_adata, perturbation, control_pert, group_suffix,
    pert_col, embed_key, max_cells, device, seed,
    mmd_kernel_mul, mmd_kernel_num, energy_blur, energy_scaling, energy_backend,
    group_by_cols=None
) -> Dict:
    """Compute dual-kernel MMD for a single perturbation-group combination."""
    
    # Create masks for this specific group
    if group_suffix:
        # Need to match the compound key exactly
        pert_key = f"{perturbation}::{group_suffix}"
        control_key = f"{control_pert}::{group_suffix}"
        
        # Create compound keys for both datasets using the provided group_by_cols
        real_compound_keys = create_compound_grouping_key(real_adata, pert_col, group_by_cols)
        pred_compound_keys = create_compound_grouping_key(pred_adata, pert_col, group_by_cols)
        
        real_pert_mask = real_compound_keys == pert_key
        real_control_mask = real_compound_keys == control_key
        pred_pert_mask = pred_compound_keys == pert_key
        pred_control_mask = pred_compound_keys == control_key
    else:
        # Simple perturbation matching (no grouping)
        real_pert_mask = real_adata.obs[pert_col] == perturbation
        real_control_mask = real_adata.obs[pert_col] == control_pert
        pred_pert_mask = pred_adata.obs[pert_col] == perturbation
        pred_control_mask = pred_adata.obs[pert_col] == control_pert
    
    # Sample cells for each condition
    real_pert_indices = sample_cells(real_adata, real_pert_mask, max_cells, seed)
    real_control_indices = sample_cells(real_adata, real_control_mask, max_cells, seed + 1)
    pred_pert_indices = sample_cells(pred_adata, pred_pert_mask, max_cells, seed + 2)
    pred_control_indices = sample_cells(pred_adata, pred_control_mask, max_cells, seed + 3)
    
    # Check minimum cell requirements
    if (len(real_pert_indices) < 10 or len(real_control_indices) < 10 or
        len(pred_pert_indices) < 10 or len(pred_control_indices) < 10):
        raise ValueError(f"Insufficient cells for reliable MMD computation")
    
    # Extract embeddings
    real_pert_emb = extract_embeddings(real_adata, real_pert_indices, embed_key, device)
    real_control_emb = extract_embeddings(real_adata, real_control_indices, embed_key, device)
    pred_pert_emb = extract_embeddings(pred_adata, pred_pert_indices, embed_key, device)
    pred_control_emb = extract_embeddings(pred_adata, pred_control_indices, embed_key, device)
    
    # Compute all MMD comparisons with both kernels
    results = {}
    
    try:
        # Baseline MMD: real_control vs real_treated
        results['baseline_mmd_rbf'] = mmd_loss(
            real_control_emb, real_pert_emb, mmd_kernel_mul, mmd_kernel_num
        )
        results['baseline_mmd_energy'] = energy_mmd_loss(
            real_control_emb, real_pert_emb, energy_blur, energy_scaling, energy_backend
        )
        
        # Transport MMD: pred_treated vs real_treated
        results['transport_mmd_rbf'] = mmd_loss(
            pred_pert_emb, real_pert_emb, mmd_kernel_mul, mmd_kernel_num
        )
        results['transport_mmd_energy'] = energy_mmd_loss(
            pred_pert_emb, real_pert_emb, energy_blur, energy_scaling, energy_backend
        )
        
        # Control MMD: pred_control vs real_control
        results['control_mmd_rbf'] = mmd_loss(
            pred_control_emb, real_control_emb, mmd_kernel_mul, mmd_kernel_num
        )
        results['control_mmd_energy'] = energy_mmd_loss(
            pred_control_emb, real_control_emb, energy_blur, energy_scaling, energy_backend
        )
        
    except Exception as e:
        logger.warning(f"MMD computation failed: {e}")
        # Return NaN values for failed computations
        for key in ['baseline_mmd_rbf', 'baseline_mmd_energy', 'transport_mmd_rbf', 
                   'transport_mmd_energy', 'control_mmd_rbf', 'control_mmd_energy']:
            results[key] = float('nan')
    
    # Add cell counts
    results.update({
        'n_real_pert': len(real_pert_indices),
        'n_real_control': len(real_control_indices),
        'n_pred_pert': len(pred_pert_indices),
        'n_pred_control': len(pred_control_indices)
    })
    
    return results


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Map control perturbation name to actual value in h5ad files
    actual_control_pert = map_control_perturbation(args.control_pert)
    if actual_control_pert != args.control_pert:
        logger.info(f"Mapping control perturbation '{args.control_pert}' -> '{actual_control_pert}'")
    
    logger.info(f"Reading predicted anndata from {args.adata_pred}")
    logger.info(f"Reading real anndata from {args.adata_real}")
    
    try:
        os.makedirs(args.outdir, exist_ok=True)
        
        # Load data
        real_adata = sc.read_h5ad(args.adata_real)
        pred_adata = sc.read_h5ad(args.adata_pred)
        
        logger.info(f"Real data shape: {real_adata.shape}")
        logger.info(f"Pred data shape: {pred_adata.shape}")
        
        # Prepare embeddings with potentially different keys
        final_embed_key = prepare_embeddings(
            real_adata, pred_adata,
            embed_key_real=args.embed_key_real,
            embed_key_pred=args.embed_key_pred,
            embed_key=args.embed_key
        )
        
        if final_embed_key:
            logger.info(f"Using embedding key: {final_embed_key}")
        else:
            logger.info("Using .X matrix for embeddings")
        
        # Set random seed
        np.random.seed(args.random_seed)
        
        # Get all perturbations (excluding control)
        real_perts = set(real_adata.obs[args.pert_col].unique())
        pred_perts = set(pred_adata.obs[args.pert_col].unique())
        common_perts = real_perts.intersection(pred_perts)
        
        # Remove control perturbation from evaluation list
        common_perts.discard(actual_control_pert)
        
        logger.info(f"Real dataset has {len(real_perts)} perturbations")
        logger.info(f"Pred dataset has {len(pred_perts)} perturbations")
        logger.info(f"Common perturbations: {len(common_perts)}")
        logger.info(f"Control perturbation: {actual_control_pert}")
        
        # Create evaluation combinations
        evaluation_combinations = []
        
        if args.group_by:
            logger.info(f"Using categorical grouping by columns: {args.group_by}")
            
            # Create compound keys
            real_compound_keys = create_compound_grouping_key(real_adata, args.pert_col, args.group_by)
            pred_compound_keys = create_compound_grouping_key(pred_adata, args.pert_col, args.group_by)
            
            # Get all unique combinations
            real_keys = set(real_compound_keys.unique())
            pred_keys = set(pred_compound_keys.unique())
            common_keys = real_keys.intersection(pred_keys)
            
            # Filter to non-control perturbations
            for key in common_keys:
                perturbation, group_suffix = parse_compound_key(key)
                if perturbation != actual_control_pert:
                    evaluation_combinations.append((perturbation, group_suffix))
                    
        else:
            # Simple case: no grouping
            for perturbation in common_perts:
                evaluation_combinations.append((perturbation, None))
        
        logger.info(f"Total combinations to evaluate: {len(evaluation_combinations)}")
        
        # Evaluate each combination
        results = []
        
        for perturbation, group_suffix in tqdm(evaluation_combinations, desc="Computing dual-kernel MMD"):
            try:
                result = compute_dual_mmd_for_combination(
                    real_adata, pred_adata, perturbation, actual_control_pert, group_suffix,
                    args.pert_col, final_embed_key, args.max_cells_per_group, args.device, args.random_seed,
                    args.mmd_kernel_mul, args.mmd_kernel_num, 
                    args.energy_blur, args.energy_scaling, args.energy_backend,
                    group_by_cols=args.group_by
                )
                
                # Add perturbation and group info
                result_row = {
                    'perturbation': perturbation,
                    'group_suffix': group_suffix if group_suffix else '',
                    **result
                }
                
                results.append(result_row)
                
            except Exception as e:
                logger.warning(f"Failed {perturbation} {group_suffix}: {e}")
                continue
        
        logger.info(f"✓ Completed dual-kernel MMD evaluation for {len(results)} combinations")
        
        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)
        
        if len(results_df) > 0:
            # Save detailed results
            results_path = os.path.join(args.outdir, "mmd_results.csv")
            results_df.to_csv(results_path, index=False)
            logger.info(f"✓ Results saved to: {results_path}")
            
            # Compute summary statistics
            summary_stats = {}
            
            for metric in ['baseline_mmd_rbf', 'baseline_mmd_energy', 'transport_mmd_rbf', 
                          'transport_mmd_energy', 'control_mmd_rbf', 'control_mmd_energy']:
                valid_values = results_df[metric].dropna()
                if len(valid_values) > 0:
                    summary_stats[metric] = {
                        'count': len(valid_values),
                        'mean': float(valid_values.mean()),
                        'median': float(valid_values.median()),
                        'std': float(valid_values.std()),
                        'min': float(valid_values.min()),
                        'max': float(valid_values.max())
                    }
            
            # Compute improvement ratios
            if 'baseline_mmd_rbf' in summary_stats and 'transport_mmd_rbf' in summary_stats:
                rbf_improvements = results_df['baseline_mmd_rbf'] / results_df['transport_mmd_rbf']
                rbf_improvements = rbf_improvements.replace([np.inf, -np.inf], np.nan).dropna()
                if len(rbf_improvements) > 0:
                    summary_stats['improvement_ratio_rbf'] = {
                        'mean': float(rbf_improvements.mean()),
                        'median': float(rbf_improvements.median())
                    }
            
            if 'baseline_mmd_energy' in summary_stats and 'transport_mmd_energy' in summary_stats:
                energy_improvements = results_df['baseline_mmd_energy'] / results_df['transport_mmd_energy']
                energy_improvements = energy_improvements.replace([np.inf, -np.inf], np.nan).dropna()
                if len(energy_improvements) > 0:
                    summary_stats['improvement_ratio_energy'] = {
                        'mean': float(energy_improvements.mean()),
                        'median': float(energy_improvements.median())
                    }
            
            # Save summary
            summary_path = os.path.join(args.outdir, "mmd_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            logger.info(f"✓ Summary saved to: {summary_path}")
            
            # Print key results
            logger.info(f"\n{'='*60}")
            logger.info("DUAL-KERNEL MMD EVALUATION RESULTS")
            logger.info("="*60)
            
            if 'transport_mmd_rbf' in summary_stats:
                rbf_stats = summary_stats['transport_mmd_rbf']
                logger.info(f"RBF Transport MMD: {rbf_stats['mean']:.6f} ± {rbf_stats['std']:.6f}")
                
            if 'transport_mmd_energy' in summary_stats:
                energy_stats = summary_stats['transport_mmd_energy']
                logger.info(f"Energy Transport MMD: {energy_stats['mean']:.6f} ± {energy_stats['std']:.6f}")
                
            if 'improvement_ratio_rbf' in summary_stats:
                rbf_improvement = summary_stats['improvement_ratio_rbf']
                logger.info(f"RBF Improvement Ratio: {rbf_improvement['mean']:.3f}x")
                
            if 'improvement_ratio_energy' in summary_stats:
                energy_improvement = summary_stats['improvement_ratio_energy']
                logger.info(f"Energy Improvement Ratio: {energy_improvement['mean']:.3f}x")
        
        else:
            logger.warning("No valid results computed!")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()