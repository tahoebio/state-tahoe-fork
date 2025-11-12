#!/usr/bin/env python3
"""
Compute kNN separability metrics for drug response prediction using single AnnData format.

This script:
1. Loads real (observed) AnnData object with perturbation data
2. Extracts control and treated cells for each perturbation
3. Computes kNN classification accuracy to measure separability
4. Supports categorical grouping and celltype splitting
5. Uses permutation testing to establish null baselines
6. Uses balanced sampling for computational efficiency

Based on mmd_anndata_pair.py interface but focused on separability analysis
using kNN classification instead of MMD distributional metrics.
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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import polars as pl
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

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
        description="Compute kNN separability metrics for drug response prediction using AnnData format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python knn_separability_anndata.py \\
    --adata observed_data.h5ad \\
    --control-pert DMSO_TF

  # With grouping and sampling
  python knn_separability_anndata.py \\
    --adata observed_data.h5ad \\
    --control-pert DMSO_TF \\
    --group-by plate batch \\
    --max-cells-per-group 300 \\
    --embed-key X_hvg \\
    --k-neighbors 15
        """
    )

    # Core arguments
    parser.add_argument(
        "--adata",
        type=str,
        required=True,
        help="Path to the real adata object to analyze",
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
        help="Key for embedded data (.obsm) in AnnData object (evaluated over .X otherwise)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./knn_separability_results",
        help="Output directory to write results [default: %(default)s]",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        help="Additional categorical columns to group by before calculating separability (e.g., plate batch timepoint)",
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

    # kNN parameters
    parser.add_argument(
        "--min-cells",
        type=int,
        default=20,
        help="Minimum cells needed for reliable classification [default: %(default)s]",
    )
    parser.add_argument(
        "--k-neighbors",
        type=int,
        help="Number of neighbors for kNN classification (auto-computed if not specified)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device for computation (auto, cuda, cpu) [default: %(default)s]",
    )

    args = parser.parse_args()

    # Set device
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    return args


def map_control_perturbation(control_pert: str) -> str:
    """Map control perturbation names to their actual values in h5ad files."""
    # Handle special case where CLI uses "DMSO_TF" but h5ad files use the full string
    if control_pert == "DMSO_TF":
        return "[('DMSO_TF', 0.0, 'uM')]"
    return control_pert


def sample_cells_balanced(adata, control_mask, treated_mask, max_cells_per_condition, seed=42):
    """Sample cells ensuring balanced numbers between control and treated conditions."""
    # Get indices for each condition
    control_indices = np.where(control_mask)[0]
    treated_indices = np.where(treated_mask)[0]

    # Determine how many cells to sample from each condition
    n_control_available = len(control_indices)
    n_treated_available = len(treated_indices)

    # Take minimum of what's available and the max allowed
    n_samples_per_condition = min(
        max_cells_per_condition,
        n_control_available,
        n_treated_available
    )

    # Sample from each condition
    np.random.seed(seed)
    if n_control_available <= n_samples_per_condition:
        sampled_control = control_indices
    else:
        sampled_control = np.random.choice(control_indices, n_samples_per_condition, replace=False)

    np.random.seed(seed + 1)
    if n_treated_available <= n_samples_per_condition:
        sampled_treated = treated_indices
    else:
        sampled_treated = np.random.choice(treated_indices, n_samples_per_condition, replace=False)

    return sampled_control, sampled_treated


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

    # Convert to numpy if it's not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)

    return matrix


def compute_dynamic_k(n_samples: int, k_override: Optional[int] = None) -> int:
    """Compute appropriate k value based on sample size."""
    if k_override is not None:
        return k_override

    # Dynamic k selection based on sample size
    if n_samples < 50:
        return min(5, n_samples // 4)
    elif n_samples < 200:
        return min(int(np.sqrt(n_samples)), 15)
    else:
        return min(int(np.sqrt(n_samples)), 25)


def compute_knn_separability(
    control_emb, treated_emb, k=None, seed=42
) -> Dict:
    """Compute kNN separability between control and treated embeddings."""

    # Ensure we have numpy arrays
    if isinstance(control_emb, torch.Tensor):
        control_emb = control_emb.cpu().numpy()
    if isinstance(treated_emb, torch.Tensor):
        treated_emb = treated_emb.cpu().numpy()

    n_control = len(control_emb)
    n_treated = len(treated_emb)

    # Balance the samples (already balanced by sampling, but ensure equal length)
    n_samples = min(n_control, n_treated)
    control_emb = control_emb[:n_samples]
    treated_emb = treated_emb[:n_samples]

    # Combine embeddings and create labels
    X = np.vstack([control_emb, treated_emb])
    y = np.hstack([np.zeros(n_samples), np.ones(n_samples)])

    # Determine k
    k_actual = compute_dynamic_k(n_samples * 2, k)

    # Ensure k is valid
    k_actual = min(k_actual, len(X) - 1)

    if k_actual < 1:
        raise ValueError(f"k={k_actual} is too small for {len(X)} samples")

    # Compute kNN accuracy for real labels
    try:
        knn = KNeighborsClassifier(n_neighbors=k_actual)
        knn.fit(X, y)

        # Predict on the same data (this is measuring separability of the space)
        y_pred = knn.predict(X)
        separability = accuracy_score(y, y_pred)

        # Compute single permuted reference
        np.random.seed(seed)
        y_perm = np.random.permutation(y)
        knn.fit(X, y_perm)
        y_pred_perm = knn.predict(X)
        permuted_accuracy = accuracy_score(y_perm, y_pred_perm)

    except Exception as e:
        logger.warning(f"kNN computation failed: {e}")
        return {
            'separability': float('nan'),
            'permuted_reference': float('nan'),
            'n_control': n_samples,
            'n_treated': n_samples,
            'k_used': k_actual
        }

    return {
        'separability': separability,
        'permuted_reference': permuted_accuracy,
        'n_control': n_samples,
        'n_treated': n_samples,
        'k_used': k_actual
    }


def compute_separability_for_combination(
    adata, perturbation, control_pert, group_suffix,
    pert_col, embed_key, max_cells, device, seed,
    k_neighbors,
    group_by_cols=None
) -> Dict:
    """Compute kNN separability for a single perturbation-group combination."""

    # Create masks for this specific group
    if group_suffix:
        # Need to match the compound key exactly
        pert_key = f"{perturbation}::{group_suffix}"
        control_key = f"{control_pert}::{group_suffix}"

        # Create compound keys for the dataset using the provided group_by_cols
        compound_keys = create_compound_grouping_key(adata, pert_col, group_by_cols)

        pert_mask = compound_keys == pert_key
        control_mask = compound_keys == control_key
    else:
        # Simple perturbation matching (no grouping)
        pert_mask = adata.obs[pert_col] == perturbation
        control_mask = adata.obs[pert_col] == control_pert

    # Sample cells for balanced comparison
    control_indices, treated_indices = sample_cells_balanced(
        adata, control_mask, pert_mask, max_cells, seed
    )

    # Check minimum cell requirements
    if (len(control_indices) < 10 or len(treated_indices) < 10):
        raise ValueError(f"Insufficient cells for reliable classification")

    # Extract embeddings
    control_emb = extract_embeddings(adata, control_indices, embed_key, device)
    treated_emb = extract_embeddings(adata, treated_indices, embed_key, device)

    # Compute kNN separability
    try:
        results = compute_knn_separability(
            control_emb, treated_emb, k_neighbors, seed
        )

    except Exception as e:
        logger.warning(f"Separability computation failed: {e}")
        # Return NaN values for failed computations
        results = {
            'separability': float('nan'),
            'permuted_reference': float('nan'),
            'n_control': len(control_indices),
            'n_treated': len(treated_indices),
            'k_used': k_neighbors or 5
        }

    return results


def main():
    """Main evaluation function."""
    args = parse_args()

    # Map control perturbation name to actual value in h5ad files
    actual_control_pert = map_control_perturbation(args.control_pert)
    if actual_control_pert != args.control_pert:
        logger.info(f"Mapping control perturbation '{args.control_pert}' -> '{actual_control_pert}'")

    logger.info(f"Reading anndata from {args.adata}")

    try:
        os.makedirs(args.outdir, exist_ok=True)

        # Load data
        adata = sc.read_h5ad(args.adata)

        logger.info(f"Data shape: {adata.shape}")

        if args.embed_key:
            logger.info(f"Using embedding key: {args.embed_key}")
            if args.embed_key not in adata.obsm:
                raise ValueError(f"Embedding key '{args.embed_key}' not found in AnnData.obsm")
        else:
            logger.info("Using .X matrix for embeddings")

        # Set random seed
        np.random.seed(args.random_seed)

        # Get all perturbations (excluding control)
        all_perts = set(adata.obs[args.pert_col].unique())

        # Remove control perturbation from evaluation list
        all_perts.discard(actual_control_pert)

        logger.info(f"Dataset has {len(all_perts) + 1} perturbations (including control)")
        logger.info(f"Evaluating {len(all_perts)} perturbations")
        logger.info(f"Control perturbation: {actual_control_pert}")

        # Create evaluation combinations
        evaluation_combinations = []

        if args.group_by:
            logger.info(f"Using categorical grouping by columns: {args.group_by}")

            # Create compound keys
            compound_keys = create_compound_grouping_key(adata, args.pert_col, args.group_by)

            # Get all unique combinations
            all_keys = set(compound_keys.unique())

            # Filter to non-control perturbations
            for key in all_keys:
                perturbation, group_suffix = parse_compound_key(key)
                if perturbation != actual_control_pert:
                    evaluation_combinations.append((perturbation, group_suffix))

        else:
            # Simple case: no grouping
            for perturbation in all_perts:
                evaluation_combinations.append((perturbation, None))

        logger.info(f"Total combinations to evaluate: {len(evaluation_combinations)}")

        # Evaluate each combination
        results = []

        for perturbation, group_suffix in tqdm(evaluation_combinations, desc="Computing kNN separability"):
            try:
                result = compute_separability_for_combination(
                    adata, perturbation, actual_control_pert, group_suffix,
                    args.pert_col, args.embed_key, args.max_cells_per_group, args.device, args.random_seed,
                    args.k_neighbors,
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

        logger.info(f"✓ Completed kNN separability evaluation for {len(results)} combinations")

        # Convert to DataFrame and save
        results_df = pd.DataFrame(results)

        if len(results_df) > 0:
            # Save detailed results
            results_path = os.path.join(args.outdir, "knn_separability_results.csv")
            results_df.to_csv(results_path, index=False)
            logger.info(f"✓ Results saved to: {results_path}")

            # Compute summary statistics
            summary_stats = {}

            for metric in ['separability', 'permuted_reference']:
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

            # Add k usage statistics
            k_values = results_df['k_used'].dropna()
            if len(k_values) > 0:
                summary_stats['k_used'] = {
                    'mean': float(k_values.mean()),
                    'median': float(k_values.median()),
                    'min': int(k_values.min()),
                    'max': int(k_values.max())
                }

            # Difference statistics (separability - permuted_reference)
            valid_sep = results_df['separability'].dropna()
            valid_perm = results_df['permuted_reference'].dropna()
            if len(valid_sep) > 0 and len(valid_perm) > 0:
                differences = valid_sep - valid_perm.reindex(valid_sep.index)
                differences = differences.dropna()
                if len(differences) > 0:
                    summary_stats['improvement_over_permuted'] = {
                        'count': len(differences),
                        'mean': float(differences.mean()),
                        'median': float(differences.median()),
                        'std': float(differences.std()),
                        'min': float(differences.min()),
                        'max': float(differences.max())
                    }

            # Save summary
            summary_path = os.path.join(args.outdir, "knn_separability_summary.json")
            with open(summary_path, 'w') as f:
                json.dump(summary_stats, f, indent=2)
            logger.info(f"✓ Summary saved to: {summary_path}")

            # Print key results
            logger.info(f"\n{'='*60}")
            logger.info("kNN SEPARABILITY EVALUATION RESULTS")
            logger.info("="*60)

            if 'separability' in summary_stats:
                sep_stats = summary_stats['separability']
                logger.info(f"Overall Separability: {sep_stats['mean']:.3f} ± {sep_stats['std']:.3f}")

            if 'permuted_reference' in summary_stats:
                perm_stats = summary_stats['permuted_reference']
                logger.info(f"Permuted Reference: {perm_stats['mean']:.3f} ± {perm_stats['std']:.3f}")

            if 'improvement_over_permuted' in summary_stats:
                imp_stats = summary_stats['improvement_over_permuted']
                logger.info(f"Improvement over Permuted: {imp_stats['mean']:.3f} ± {imp_stats['std']:.3f}")

            if 'k_used' in summary_stats:
                k_stats = summary_stats['k_used']
                logger.info(f"k values used: {k_stats['min']} - {k_stats['max']} (median: {k_stats['median']:.0f})")

        else:
            logger.warning("No valid results computed!")

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()