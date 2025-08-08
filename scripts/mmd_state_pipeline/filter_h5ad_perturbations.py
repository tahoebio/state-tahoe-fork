#!/usr/bin/env python3
"""
Efficient h5ad perturbation filtering utility.

This script filters predicted h5ad files to only include perturbations 
present in the reference/test data, solving the common perturbation 
mismatch issue in cell-eval.

Uses a simpler approach that loads data into memory in chunks to avoid
backed mode issues with very large files.
"""

import argparse
import logging
from pathlib import Path
import numpy as np
import anndata as ad
from typing import Set

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_perturbations(h5ad_path: Path, pert_col: str) -> Set[str]:
    """Extract unique perturbations from h5ad file."""
    logger.info(f"Loading perturbations from {h5ad_path}")
    
    adata = ad.read_h5ad(h5ad_path, backed='r')
    
    if pert_col not in adata.obs.columns:
        raise ValueError(f"Perturbation column '{pert_col}' not found in {h5ad_path}")
    
    perturbations = set(adata.obs[pert_col].astype(str).unique())
    
    logger.info(f"Found {len(perturbations)} unique perturbations in {h5ad_path}")
    return perturbations

def diagnose_compatibility(predicted_path: Path, reference_path: Path, pert_col: str = "drug_dose"):
    """Diagnose potential issues preventing cell-eval from working."""
    logger.info("=== DIAGNOSTIC MODE ===")
    
    # Load files in backed mode for minimal memory usage
    logger.info("Loading files for diagnostic analysis...")
    pred_adata = ad.read_h5ad(predicted_path, backed='r')
    ref_adata = ad.read_h5ad(reference_path, backed='r')
    
    # Check embedding keys
    logger.info("Checking embedding keys:")
    logger.info(f"  Predicted file obsm keys: {list(pred_adata.obsm.keys())}")
    logger.info(f"  Reference file obsm keys: {list(ref_adata.obsm.keys())}")
    
    # Check if X_hvg exists in predicted file
    has_x_hvg_pred = 'X_hvg' in pred_adata.obsm
    has_x_hvg_ref = 'X_hvg' in ref_adata.obsm
    logger.info(f"  X_hvg in predicted: {has_x_hvg_pred}")
    logger.info(f"  X_hvg in reference: {has_x_hvg_ref}")
    
    if not has_x_hvg_pred:
        logger.warning("  ⚠ WARNING: X_hvg not found in predicted file - this will cause cell-eval to fail!")
        logger.info("  Available keys in predicted file:")
        for key in pred_adata.obsm.keys():
            logger.info(f"    - {key}")
    
    # Check perturbations (sample a few to show actual names)
    logger.info("Sampling perturbation names:")
    pred_perts = pred_adata.obs[pert_col].astype(str).unique()[:5]
    ref_perts = ref_adata.obs[pert_col].astype(str).unique()[:5]
    logger.info(f"  Predicted (first 5): {list(pred_perts)}")
    logger.info(f"  Reference (first 5): {list(ref_perts)}")
    
    # Check cell line coverage
    cell_col = 'cell_line'  # Assuming this is the cell line column
    if cell_col in pred_adata.obs.columns and cell_col in ref_adata.obs.columns:
        pred_cells = set(pred_adata.obs[cell_col].astype(str).unique())
        ref_cells = set(ref_adata.obs[cell_col].astype(str).unique())
        
        logger.info("Cell line coverage:")
        logger.info(f"  Predicted file: {len(pred_cells)} cell lines")
        logger.info(f"  Reference file: {len(ref_cells)} cell lines")
        logger.info(f"  Common cell lines: {len(pred_cells & ref_cells)}")
        
        pred_only_cells = pred_cells - ref_cells
        ref_only_cells = ref_cells - pred_cells
        
        if len(pred_only_cells) > 0:
            logger.info(f"  Only in predicted: {len(pred_only_cells)} cell lines")
            if len(pred_only_cells) <= 5:
                logger.info(f"    {list(pred_only_cells)}")
                
        if len(ref_only_cells) > 0:
            logger.info(f"  Only in reference: {len(ref_only_cells)} cell lines")
            if len(ref_only_cells) <= 5:
                logger.info(f"    {list(ref_only_cells)}")
    
    # Check perturbation-cell combinations
    logger.info("Checking perturbation-cell combinations:")
    if cell_col in pred_adata.obs.columns and cell_col in ref_adata.obs.columns:
        pred_combos = set(zip(pred_adata.obs[pert_col].astype(str), pred_adata.obs[cell_col].astype(str)))
        ref_combos = set(zip(ref_adata.obs[pert_col].astype(str), ref_adata.obs[cell_col].astype(str)))
        
        logger.info(f"  Predicted combinations: {len(pred_combos)}")
        logger.info(f"  Reference combinations: {len(ref_combos)}")
        logger.info(f"  Common combinations: {len(pred_combos & ref_combos)}")
        
        pred_only_combos = pred_combos - ref_combos
        if len(pred_only_combos) > 0:
            logger.info(f"  Only in predicted: {len(pred_only_combos)} combinations")
            logger.info("    Examples:")
            for combo in list(pred_only_combos)[:3]:
                logger.info(f"      {combo[0]} + {combo[1]}")
    
    logger.info("=== END DIAGNOSTICS ===\n")

def get_combinations(h5ad_path: Path, pert_col: str = "drug_dose", cell_col: str = "cell_line"):
    """Extract unique drug-cell combinations from h5ad file."""
    logger.info(f"Loading combinations from {h5ad_path}")
    
    adata = ad.read_h5ad(h5ad_path, backed='r')
    
    if pert_col not in adata.obs.columns:
        raise ValueError(f"Perturbation column '{pert_col}' not found in {h5ad_path}")
    if cell_col not in adata.obs.columns:
        raise ValueError(f"Cell line column '{cell_col}' not found in {h5ad_path}")
    
    combinations = set(zip(
        adata.obs[pert_col].astype(str),
        adata.obs[cell_col].astype(str)
    ))
    
    logger.info(f"Found {len(combinations)} unique combinations in {h5ad_path}")
    return combinations

def filter_h5ad_by_perturbations(
    predicted_path: Path,
    reference_path: Path, 
    output_path: Path,
    pert_col: str = "drug_dose",
    cell_col: str = "cell_line"
) -> None:
    """
    Filter predicted h5ad to match reference data exactly.
    Filters by drug-cell combinations, which is what cell-eval actually needs.
    """
    
    # Run diagnostics first
    diagnose_compatibility(predicted_path, reference_path, pert_col)
    
    # Get combinations from both files
    logger.info("Step 1: Extracting combinations from reference file")
    ref_combos = get_combinations(reference_path, pert_col, cell_col)
    
    logger.info("Step 2: Extracting combinations from predicted file")
    pred_combos = get_combinations(predicted_path, pert_col, cell_col)
    
    # Analyze differences
    pred_only = pred_combos - ref_combos
    ref_only = ref_combos - pred_combos
    common = pred_combos & ref_combos
    
    logger.info(f"Combination analysis:")
    logger.info(f"  Common combinations: {len(common)}")
    logger.info(f"  Only in predicted: {len(pred_only)}")
    logger.info(f"  Only in reference: {len(ref_only)}")
    
    if len(pred_only) == 0:
        logger.info("No filtering needed - all predicted combinations are in reference")
        return
        
    if len(pred_only) > 0:
        logger.info(f"Will remove {len(pred_only)} combinations from predicted data:")
        for combo in sorted(list(pred_only)[:10]):  # Show first 10
            logger.info(f"  - {combo[0]} + {combo[1]}")
        if len(pred_only) > 10:
            logger.info(f"  ... and {len(pred_only) - 10} more")
    
    # Load predicted file into RAM
    logger.info("Step 3: Loading predicted file into RAM")
    pred_adata = ad.read_h5ad(predicted_path)
    
    # Create filter mask based on combinations
    logger.info("Step 4: Creating combination filter mask")
    pred_combinations = list(zip(
        pred_adata.obs[pert_col].astype(str),
        pred_adata.obs[cell_col].astype(str)
    ))
    
    mask = np.array([combo in ref_combos for combo in pred_combinations])
    n_keep = mask.sum()
    n_total = len(mask)
    
    logger.info(f"Filtering: keeping {n_keep:,} / {n_total:,} observations ({100*n_keep/n_total:.1f}%)")
    
    # Apply filter and save
    logger.info("Step 5: Applying filter and saving")
    filtered_adata = pred_adata[mask]
    filtered_adata.write(output_path)
    
    logger.info(f"Filtered data saved to {output_path}")
    logger.info(f"Final shape: {filtered_adata.shape}")
    
    # Verify the filtering worked
    final_combos = set(zip(
        filtered_adata.obs[pert_col].astype(str),
        filtered_adata.obs[cell_col].astype(str)
    ))
    extra_combos = final_combos - ref_combos
    
    if len(extra_combos) == 0:
        logger.info("✓ Filtering successful - no extra combinations remain")
    else:
        logger.warning(f"⚠ Warning: {len(extra_combos)} unexpected combinations still present")

def main():
    parser = argparse.ArgumentParser(
        description="Filter h5ad predictions to match reference drug-cell combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python filter_h5ad_perturbations.py \\
        --predicted dmso_controls_predicted_all_perturbations.h5ad \\
        --reference test_subset.h5ad \\
        --output dmso_controls_predicted_filtered.h5ad
        """
    )
    
    parser.add_argument(
        "--predicted", "-p",
        type=Path,
        required=True,
        help="Path to predicted h5ad file (will be filtered)"
    )
    
    parser.add_argument(
        "--reference", "-r", 
        type=Path,
        required=True,
        help="Path to reference h5ad file (defines allowed perturbations)"
    )
    
    parser.add_argument(
        "--output", "-o",
        type=Path, 
        required=False,
        help="Path for filtered output h5ad file (not required with --diagnose-only)"
    )
    
    parser.add_argument(
        "--pert-col",
        type=str,
        default="drug_dose",
        help="Column name containing perturbations (default: drug_dose)"
    )
    
    parser.add_argument(
        "--cell-col",
        type=str,
        default="cell_line",
        help="Column name containing cell lines (default: cell_line)"
    )
    
    parser.add_argument(
        "--diagnose-only",
        action="store_true",
        help="Run diagnostics only, don't filter data"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.predicted.exists():
        raise FileNotFoundError(f"Predicted file not found: {args.predicted}")
    if not args.reference.exists():
        raise FileNotFoundError(f"Reference file not found: {args.reference}")
    
    if args.diagnose_only:
        # Run diagnostics only
        diagnose_compatibility(
            predicted_path=args.predicted,
            reference_path=args.reference,
            pert_col=args.pert_col
        )
    else:
        if not args.output:
            raise ValueError("--output is required when not using --diagnose-only")
        
        # Create output directory if needed
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        # Run filtering
        filter_h5ad_by_perturbations(
            predicted_path=args.predicted,
            reference_path=args.reference,
            output_path=args.output,
            pert_col=args.pert_col,
            cell_col=args.cell_col
        )

if __name__ == "__main__":
    main()