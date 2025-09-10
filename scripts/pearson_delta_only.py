#!/usr/bin/env python3
"""
Minimal script to compute only pearson_delta metric without DE computation.
Replicates cell-eval run CLI arguments but skips differential expression.
"""

import argparse
import sys
import os
import json
from pathlib import Path
import psutil
import time

# Add the src directory to the path so we can import cell_eval
sys.path.insert(0, str(Path(__file__).parent / "src"))

import logging
import numpy as np
import anndata as ad
import polars as pl
from cell_eval.utils import split_anndata_on_celltype

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

# Set up logging with timestamps - force reconfigure
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True  # Force reconfiguration even if already configured
)
logger = logging.getLogger(__name__)

# Also configure the root logger to ensure timestamps everywhere
root_logger = logging.getLogger()
for handler in root_logger.handlers:
    handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s:%(name)s:%(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    ))


def prepare_embeddings(adata_real, adata_pred, embed_key_real=None, embed_key_pred=None, embed_key=None):
    """Prepare embeddings so both datasets use the same key for correlation analysis."""
    # Determine which keys to use
    real_key = embed_key_real or embed_key
    pred_key = embed_key_pred or embed_key
    
    # If different keys specified, copy pred data to match real key
    if real_key and pred_key and real_key != pred_key:
        logger.info(f"Using different embedding keys: real='{real_key}', pred='{pred_key}'")
        logger.info(f"Copying pred['{pred_key}'] to pred['{real_key}'] for correlation analysis")
        
        if pred_key not in adata_pred.obsm:
            raise ValueError(f"Predicted data missing embedding key '{pred_key}'")
        if real_key not in adata_real.obsm:
            raise ValueError(f"Real data missing embedding key '{real_key}'")
            
        # Copy pred embedding to match real key name
        adata_pred.obsm[real_key] = adata_pred.obsm[pred_key]
        return real_key
    
    # Same key for both or no specific keys
    return real_key or pred_key


def map_control_perturbation(control_pert: str) -> str:
    """Map control perturbation names to their actual values in h5ad files."""
    # Handle special case where CLI uses "DMSO_TF" but h5ad files use the full string
    if control_pert == "DMSO_TF":
        return "[('DMSO_TF', 0.0, 'uM')]"
    return control_pert


def filter_to_common_perturbations(real, pred, pert_col, control_pert, group_by_cols=None):
    """Filter both datasets to only include perturbations/groups present in both.
    
    Note: This function maintains the same interface (returns only real, pred) for 
    compatibility with main function. The compound keys are handled internally
    by compute_grouped_pearson_delta when needed.
    """
    if group_by_cols is None:
        # Original behavior for backward compatibility
        return _filter_to_common_perturbations_simple(real, pred, pert_col, control_pert)
    else:
        # New grouped behavior - but only return the filtered data, not the compound keys
        # The compound keys are handled separately in compute_grouped_pearson_delta
        real_filtered, pred_filtered, _, _ = _filter_to_common_compound_keys(
            real, pred, pert_col, control_pert, group_by_cols
        )
        return real_filtered, pred_filtered


def _filter_to_common_perturbations_simple(real, pred, pert_col, control_pert):
    """Original implementation: Filter both datasets to only include perturbations present in both."""
    perts_real = set(real.obs[pert_col].unique())
    perts_pred = set(pred.obs[pert_col].unique())
    
    # Find intersection of perturbations
    common_perts = perts_real.intersection(perts_pred)
    
    # Make sure control perturbation is included
    if control_pert not in common_perts:
        raise ValueError(f"Control perturbation '{control_pert}' not found in both datasets")
    
    logger.info(f"Real dataset has {len(perts_real):,} perturbations")
    logger.info(f"Pred dataset has {len(perts_pred):,} perturbations") 
    logger.info(f"Common perturbations: {len(common_perts):,}")
    
    # Filter datasets to common perturbations
    real_filtered = real[real.obs[pert_col].isin(common_perts)].copy()
    pred_filtered = pred[pred.obs[pert_col].isin(common_perts)].copy()
    
    logger.info(f"Filtered real dataset: {real_filtered.shape[0]:,} cells")
    logger.info(f"Filtered pred dataset: {pred_filtered.shape[0]:,} cells")
    
    return real_filtered, pred_filtered


def _filter_to_common_compound_keys(real, pred, pert_col, control_pert, group_by_cols):
    """Filter datasets and return compound keys separately to avoid unnecessary copying."""
    
    logger.info(f"Creating compound keys for {real.shape[0]:,} real cells...")
    real_compound_keys = create_compound_grouping_key(real, pert_col, group_by_cols)
    
    logger.info(f"Creating compound keys for {pred.shape[0]:,} pred cells...")
    pred_compound_keys = create_compound_grouping_key(pred, pert_col, group_by_cols)
    
    logger.info("Finding common compound keys...")
    keys_real = set(real_compound_keys.unique())
    keys_pred = set(pred_compound_keys.unique())
    
    # Find intersection of compound keys
    common_keys = keys_real.intersection(keys_pred)
    
    # Make sure we have at least one control perturbation key in common
    control_keys = [key for key in common_keys if key.startswith(control_pert + "::")]
    if not control_keys:
        # For backward compatibility, also check if control_pert itself is in common_keys (no grouping)
        if control_pert not in common_keys:
            raise ValueError(f"No control perturbation '{control_pert}' groups found in both datasets")
    
    logger.info(f"Real dataset has {len(keys_real):,} perturbation-group combinations")
    logger.info(f"Pred dataset has {len(keys_pred):,} perturbation-group combinations") 
    logger.info(f"Common combinations: {len(common_keys):,}")
    logger.info(f"Control groups in common: {len(control_keys):,}")
    
    # Create boolean masks directly (no copying!)
    logger.info("Creating filtering masks...")
    real_mask = real_compound_keys.isin(common_keys)
    pred_mask = pred_compound_keys.isin(common_keys)
    
    logger.info(f"Real mask filters to {real_mask.sum():,} cells")
    logger.info(f"Pred mask filters to {pred_mask.sum():,} cells")
    
    # Filter data using views (not copies)
    logger.info("Filtering datasets to common combinations...")
    real_filtered = real[real_mask]
    pred_filtered = pred[pred_mask]
    
    # Filter compound keys to match filtered data
    real_keys_filtered = real_compound_keys[real_mask]
    pred_keys_filtered = pred_compound_keys[pred_mask]
    
    logger.info(f"Filtered real dataset: {real_filtered.shape[0]:,} cells")
    logger.info(f"Filtered pred dataset: {pred_filtered.shape[0]:,} cells")
    
    # Return filtered data AND compound keys separately
    return real_filtered, pred_filtered, real_keys_filtered, pred_keys_filtered


def process_pearson_delta_results(results_dict: dict[str, float], celltype: str = None) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Process pearson_delta results into DataFrames matching cell-eval format."""
    # Convert to DataFrame with perturbation column
    data = []
    for perturbation, score in results_dict.items():
        data.append({"perturbation": perturbation, "pearson_delta": score})
    
    # Create per-perturbation results DataFrame
    results_df = pl.DataFrame(data)
    
    # Create aggregated results using describe() (matching get_agg_results)
    agg_results_df = results_df.drop("perturbation").describe()
    
    return results_df, agg_results_df


def create_compound_grouping_key(adata, pert_col: str, group_by_cols: list[str] = None):
    """Create compound grouping keys combining perturbation with additional categorical columns.
    
    Args:
        adata: AnnData object
        pert_col: Name of perturbation column 
        group_by_cols: List of additional categorical columns to group by
        
    Returns:
        Series with compound keys like "DMSO_TF::plate_A::batch_1"
    """
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
    """Parse a compound key back into perturbation and group components.
    
    Args:
        compound_key: Key like "DMSO_TF::plate_A::batch_1"
        
    Returns:
        tuple: (perturbation, group_suffix) where group_suffix is "plate_A::batch_1" or None
    """
    parts = compound_key.split("::")
    perturbation = parts[0]
    group_suffix = "::".join(parts[1:]) if len(parts) > 1 else None
    return perturbation, group_suffix


def validate_groups_have_control(adata, pert_col: str, control_pert: str, group_by_cols: list[str] = None):
    """Validate that each group has the control perturbation.
    
    Args:
        adata: AnnData object
        pert_col: Name of perturbation column
        control_pert: Name of control perturbation
        group_by_cols: List of grouping columns
        
    Returns:
        list: Valid group combinations that have control perturbation
    """
    if group_by_cols is None:
        return [None]  # Single group case
    
    # Get unique combinations of grouping columns
    group_combinations = adata.obs[group_by_cols].drop_duplicates().values
    valid_groups = []
    
    for group_combo in group_combinations:
        # Create mask for this group combination
        group_mask = (adata.obs[group_by_cols] == group_combo).all(axis=1)
        group_perturbations = set(adata.obs.loc[group_mask, pert_col].unique())
        
        if control_pert in group_perturbations:
            valid_groups.append(group_combo)
        else:
            group_str = "::".join(str(x) for x in group_combo)
            logger.warning(f"Skipping group {group_str}: missing control perturbation '{control_pert}'")
    
    logger.info(f"Found {len(valid_groups)} valid groups with control perturbation")
    return valid_groups


def compute_pearson_delta_optimized(real, pred, pert_col, control_pert, group_by_cols=None, embed_key=None):
    """Compute Pearson delta correlation with optimized pseudobulking.
    
    Handles both grouped and non-grouped analyses efficiently without
    expensive cell-eval preprocessing.
    
    Args:
        real: Real AnnData object
        pred: Predicted AnnData object
        pert_col: Name of perturbation column
        control_pert: Name of control perturbation
        group_by_cols: List of categorical columns to group by (None for non-grouped)
        embed_key: Embedding key to use (default uses .X)
        
    Returns:
        dict: Perturbation -> correlation score
    """
    if group_by_cols is None:
        # Non-grouped: treat perturbation column as the only grouping
        logger.info("Computing Pearson delta without grouping (optimized pseudobulking)...")
        
        # Filter to common perturbations (simple case)
        logger.info("Filtering to common perturbations...")
        real_filtered, pred_filtered = _filter_to_common_perturbations_simple(
            real, pred, pert_col, control_pert
        )
        
        # Create simple keys (just perturbation names, no compound grouping)
        real_keys = real_filtered.obs[pert_col].astype(str)
        pred_keys = pred_filtered.obs[pert_col].astype(str)
        
    else:
        # Grouped: use the existing compound key logic
        logger.info("Computing Pearson delta with grouping (optimized pseudobulking)...")
        real_filtered, pred_filtered, real_keys, pred_keys = _filter_to_common_compound_keys(
            real, pred, pert_col, control_pert, group_by_cols
        )
    
    # Common path for both grouped and non-grouped cases
    return _compute_pearson_delta_impl(
        real_filtered, pred_filtered, real_keys, pred_keys,
        pert_col, control_pert, group_by_cols, embed_key
    )


def _compute_pearson_delta_impl(real, pred, real_keys, pred_keys, 
                               pert_col, control_pert, group_by_cols, embed_key):
    """Implementation of Pearson delta correlation with keys passed separately.
    
    Handles both grouped (compound keys) and non-grouped (simple keys) cases.
    """
    from scipy.stats import pearsonr
    
    if group_by_cols is None:
        logger.info(f"Starting delta computation for {len(real_keys.unique()):,} unique perturbations...")
    else:
        logger.info(f"Starting delta computation for {len(real_keys.unique()):,} unique groups...")
    
    # Get data matrix
    logger.info("Extracting data matrices...")
    real_matrix = real.X if embed_key is None else real.obsm[embed_key]
    pred_matrix = pred.X if embed_key is None else pred.obsm[embed_key]
    
    logger.info(f"Real matrix shape: {real_matrix.shape}")
    logger.info(f"Pred matrix shape: {pred_matrix.shape}")
    
    # Convert sparse to dense if needed
    if hasattr(real_matrix, 'toarray'):
        logger.info("Converting sparse real matrix to dense...")
        real_matrix = real_matrix.toarray()
    if hasattr(pred_matrix, 'toarray'):
        logger.info("Converting sparse pred matrix to dense...")  
        pred_matrix = pred_matrix.toarray()
    
    # Create pseudobulks using keys passed as parameters
    logger.info("Creating pseudobulks for real data...")
    real_pseudobulks, real_cell_counts = _create_pseudobulks(real_matrix, real_keys.values)
    
    logger.info("Creating pseudobulks for predicted data...")
    pred_pseudobulks, pred_cell_counts = _create_pseudobulks(pred_matrix, pred_keys.values)
    
    logger.info(f"Created {len(real_pseudobulks):,} real pseudobulks and {len(pred_pseudobulks):,} pred pseudobulks")
    
    # Calculate deltas within groups
    logger.info("Computing deltas within groups...")
    delta_correlations = {}
    deltas_real_all = []
    deltas_pred_all = []
    
    # Group by actual perturbations to calculate overall correlation
    perturbation_groups = {}
    processed_count = 0
    skipped_count = 0
    
    for key in real_pseudobulks.keys():
        if group_by_cols is None:
            # Non-grouped case: key is just the perturbation name
            perturbation = key
            group_suffix = None
        else:
            # Grouped case: key is compound, parse it
            perturbation, group_suffix = parse_compound_key(key)
        
        if perturbation == control_pert:
            continue  # Skip control perturbation
            
        # Find matching control for this group/perturbation
        if group_suffix:
            control_key = f"{control_pert}::{group_suffix}"
        else:
            control_key = control_pert
            
        if control_key not in real_pseudobulks or control_key not in pred_pseudobulks:
            logger.warning(f"Skipping {key}: no matching control {control_key}")
            skipped_count += 1
            continue
            
        # Calculate deltas within this group/perturbation
        delta_real = real_pseudobulks[key] - real_pseudobulks[control_key]
        delta_pred = pred_pseudobulks[key] - pred_pseudobulks[control_key]
        
        # Store for per-perturbation results
        if perturbation not in perturbation_groups:
            perturbation_groups[perturbation] = {"real": [], "pred": []}
        perturbation_groups[perturbation]["real"].append(delta_real)
        perturbation_groups[perturbation]["pred"].append(delta_pred)
        
        # Store for overall correlation
        deltas_real_all.append(delta_real)
        deltas_pred_all.append(delta_pred)
        processed_count += 1
    
    logger.info(f"Processed {processed_count:,} perturbation-group combinations, skipped {skipped_count:,}")
    logger.info(f"Found {len(perturbation_groups):,} unique perturbations")
    
    # Calculate correlations per perturbation (average across all groups for that perturbation)
    # Also store detailed per-group results for traceability
    logger.info("Computing correlations per perturbation...")
    detailed_correlations = []  # Store individual (perturbation, group, correlation) tuples
    
    for key in real_pseudobulks.keys():
        if group_by_cols is None:
            # Non-grouped case: key is just the perturbation name
            perturbation = key
            group_suffix = None
            group_name = None
        else:
            # Grouped case: key is compound, parse it
            perturbation, group_suffix = parse_compound_key(key)
            group_name = group_suffix
        
        if perturbation == control_pert:
            continue  # Skip control perturbation
            
        # Find matching control for this group/perturbation
        if group_suffix:
            control_key = f"{control_pert}::{group_suffix}"
        else:
            control_key = control_pert
            
        if control_key not in real_pseudobulks or control_key not in pred_pseudobulks:
            continue
            
        # Calculate deltas within this group/perturbation
        delta_real = real_pseudobulks[key] - real_pseudobulks[control_key]
        delta_pred = pred_pseudobulks[key] - pred_pseudobulks[control_key]
        
        # Compute correlation for this specific group
        corr, _ = pearsonr(delta_real, delta_pred)
        if not np.isnan(corr):
            detailed_correlations.append({
                'perturbation': perturbation,
                'group': group_name,
                'pearson_delta': float(corr),
                'n_cells_treatment': real_cell_counts[key],
                'n_cells_control': real_cell_counts[control_key]
            })
    
    # Now compute averages per perturbation for backward compatibility
    for perturbation, deltas in perturbation_groups.items():
        if len(deltas["real"]) == 0:
            continue
            
        # Calculate correlation within each group, then average across groups
        group_correlations = []
        for real_delta, pred_delta in zip(deltas["real"], deltas["pred"]):
            corr, _ = pearsonr(real_delta, pred_delta)
            if not np.isnan(corr):
                group_correlations.append(corr)
        
        # Average correlations across groups for this perturbation
        if group_correlations:
            correlation = float(np.mean(group_correlations))
        else:
            correlation = 0.0
            
        delta_correlations[perturbation] = correlation
        
        logger.debug(f"Perturbation {perturbation}: {len(deltas['real']):,} groups, {len(group_correlations):,} valid correlations, avg = {correlation:.4f}")
    
    logger.info(f"Computed grouped Pearson delta for {len(delta_correlations):,} perturbations")
    logger.info(f"Stored {len(detailed_correlations):,} detailed group-level correlations")
    return delta_correlations, detailed_correlations


def _create_pseudobulks(matrix, compound_keys):
    """Create pseudobulks from matrix by averaging over compound keys.
    
    Returns:
        tuple: (pseudobulks_dict, cell_counts_dict) where pseudobulks_dict contains
               averaged expression values and cell_counts_dict contains number of cells
               per group
    """
    import polars as pl
    
    logger.debug(f"Creating pseudobulks from {matrix.shape[0]:,} cells x {matrix.shape[1]:,} features")
    
    # Create DataFrame with compound keys
    n_features = matrix.shape[1]
    logger.debug(f"Building DataFrame with {n_features:,} feature columns...")
    df_data = {f"feature_{i}": matrix[:, i] for i in range(n_features)}
    df_data["compound_key"] = list(compound_keys)  # Convert to list to avoid polars type mixing error
    
    df = pl.DataFrame(df_data)
    
    # Group by compound key and take mean + count
    logger.debug("Grouping by compound key and computing means and counts...")
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    pseudobulks_df = df.group_by("compound_key").agg([
        pl.col(col).mean().alias(col) for col in feature_cols
    ] + [
        pl.col("compound_key").count().alias("cell_count")
    ])
    
    logger.debug(f"Created {len(pseudobulks_df):,} pseudobulk groups")
    
    # Convert back to dictionary of arrays and separate cell counts
    logger.debug("Converting to dictionary format...")
    pseudobulks = {}
    cell_counts = {}
    for row in pseudobulks_df.iter_rows(named=True):
        key = row["compound_key"]
        values = np.array([row[col] for col in feature_cols])
        pseudobulks[key] = values
        cell_counts[key] = row["cell_count"]
    
    logger.debug(f"Pseudobulking completed: {len(pseudobulks):,} groups")
    return pseudobulks, cell_counts


def save_results_csv(results_df: pl.DataFrame, agg_results_df: pl.DataFrame, outdir: str, celltype: str = None, detailed_results: list = None):
    """Save results to CSV files matching cell-eval format."""
    if celltype:
        results_filename = f"{celltype}_results.csv"
        agg_results_filename = f"{celltype}_agg_results.csv"
        detailed_filename = f"{celltype}_detailed_results.csv"
    else:
        results_filename = "results.csv"
        agg_results_filename = "agg_results.csv"
        detailed_filename = "detailed_results.csv"
    
    results_path = os.path.join(outdir, results_filename)
    agg_results_path = os.path.join(outdir, agg_results_filename)
    detailed_path = os.path.join(outdir, detailed_filename)
    
    logger.info(f"Writing perturbation level metrics to {results_path}")
    results_df.write_csv(results_path)
    
    logger.info(f"Writing aggregate metrics to {agg_results_path}")
    agg_results_df.write_csv(agg_results_path)
    
    # Save detailed per-group results if provided
    if detailed_results:
        logger.info(f"Writing detailed per-group metrics to {detailed_path}")
        detailed_df = pl.DataFrame(detailed_results)
        detailed_df.write_csv(detailed_path)
    
    return results_path, agg_results_path


def load_celltype_group_data(h5ad_path, celltype_col, celltype_value, group_col, group_value, embed_key=None):
    """Load only data for a specific celltype AND group combination using AnnData backed mode.
    
    Args:
        h5ad_path: Path to H5AD file
        celltype_col: Column name for celltype (e.g., 'cell_line_id')
        celltype_value: Celltype to filter for
        group_col: Column name for group (e.g., 'plate')
        group_value: Group to filter for
        embed_key: Optional embedding key to load from obsm
        
    Returns:
        AnnData object with only the specified celltype-group combination
    """
    start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"    Loading {celltype_col}={celltype_value}, {group_col}={group_value}")
    logger.info(f"    Initial memory: {initial_memory:.2f} GB")
    
    # Open in backed mode to access obs without loading full data
    logger.info(f"    Opening {os.path.basename(h5ad_path)} in backed mode...")
    adata_backed = ad.read_h5ad(h5ad_path, backed='r')
    
    # Find cells matching both celltype and group
    logger.info(f"    Filtering cells...")
    mask = (adata_backed.obs[celltype_col] == celltype_value) & (adata_backed.obs[group_col] == group_value)
    matching_indices = np.where(mask)[0]
    
    if len(matching_indices) == 0:
        logger.info(f"    No cells found for this combination")
        adata_backed.file.close()
        return None
    
    logger.info(f"    Found {len(matching_indices):,} matching cells ({len(matching_indices)/adata_backed.n_obs*100:.1f}% of total)")
    
    # Load only the subset of data we need - this creates a view, then copy loads only those rows
    logger.info(f"    Loading subset data...")
    subset_start = time.time()
    
    # Create subset view, then load only specific rows into memory
    adata_subset = adata_backed[mask, :].to_memory()
    
    subset_time = time.time() - subset_start
    subset_memory = get_memory_usage()
    
    # Close the backed file
    adata_backed.file.close()
    
    logger.info(f"    Subset loaded in {subset_time:.1f}s")
    logger.info(f"    Final shape: {adata_subset.shape}")
    logger.info(f"    Memory after loading: {subset_memory:.2f} GB (+{subset_memory-initial_memory:.2f} GB)")
    
    return adata_subset


def load_group_data(h5ad_path, group_name, group_value, embed_key=None):
    """Load only the data for a specific group from an H5AD file.
    
    Uses h5py to read only the necessary indices, avoiding loading the full dataset.
    
    Args:
        h5ad_path: Path to H5AD file
        group_name: Column name to filter by (e.g., 'plate')
        group_value: Value to filter for (e.g., 'plate1')
        embed_key: Optional embedding key to load from obsm
        
    Returns:
        AnnData object with only the specified group's data
    """
    import h5py
    import pandas as pd
    from scipy import sparse
    
    start_time = time.time()
    initial_memory = get_memory_usage()
    logger.info(f"Loading {group_name}={group_value} from {os.path.basename(h5ad_path)}...")
    logger.info(f"Initial memory usage: {initial_memory:.2f} GB")
    
    with h5py.File(h5ad_path, 'r') as f:
        # Log file structure
        logger.info(f"H5AD structure - obs keys: {list(f['obs'].keys())[:5]}...")
        if 'obsm' in f:
            logger.info(f"H5AD structure - obsm keys: {list(f['obsm'].keys())}")
        
        # Read obs to find indices for this group
        logger.info(f"Reading obs metadata to identify {group_name}={group_value} cells...")
        obs_start = time.time()
        obs = pd.DataFrame()
        
        obs_group = f['obs']
        
        # Get total cells count safely
        first_key = None
        for key in obs_group.keys():
            if key != '__categories':
                first_key = key
                break
        
        total_cells = obs_group[first_key].shape[0] if first_key else 0
        logger.info(f"Total cells in file: {total_cells:,}")
        
        # Check if categories exist
        has_categories = '__categories' in obs_group
        categories_group = obs_group.get('__categories') if has_categories else None
        
        # Read all obs columns
        for key in obs_group.keys():
            if key == '__categories':
                continue
            
            # Check if this is a categorical column
            if has_categories and categories_group is not None and key in categories_group:
                # Categorical column
                logger.debug(f"  Reading categorical column: {key}")
                codes = obs_group[key][()]
                categories = categories_group[key][()]
                obs[key] = pd.Categorical.from_codes(codes, categories.astype(str))
            else:
                # Regular column  
                logger.debug(f"  Reading regular column: {key}")
                data = obs_group[key][()]  # dataset - use [()] to read all data
                obs[key] = data
        
        obs_time = time.time() - obs_start
        obs_memory = get_memory_usage()
        logger.info(f"Obs loaded in {obs_time:.1f}s, memory now: {obs_memory:.2f} GB (+{obs_memory-initial_memory:.2f} GB)")
        
        # Find indices for this group
        logger.info(f"Filtering to {group_name}={group_value}...")
        group_mask = obs[group_name] == group_value
        group_indices = np.where(group_mask)[0]
        
        if len(group_indices) == 0:
            logger.warning(f"No cells found for {group_name}={group_value}")
            return None
        
        logger.info(f"Found {len(group_indices):,} cells for {group_name}={group_value} ({len(group_indices)/total_cells*100:.1f}% of total)")
        
        # Read var
        logger.info("Reading var (gene/feature) metadata...")
        var_start = time.time()
        var = pd.DataFrame()
        if 'var' in f:
            var_group = f['var']
            
            # Get feature count safely
            first_var_key = None
            for key in var_group.keys():
                if key != '__categories':
                    first_var_key = key
                    break
            
            n_vars = var_group[first_var_key].shape[0] if first_var_key else 0
            logger.info(f"Number of features: {n_vars:,}")
            
            # Check if var categories exist
            has_var_categories = '__categories' in var_group
            var_categories_group = var_group.get('__categories') if has_var_categories else None
            
            # Read all var columns
            for key in var_group.keys():
                if key == '__categories':
                    continue
                
                # Check if this is a categorical column
                if has_var_categories and var_categories_group is not None and key in var_categories_group:
                    # Categorical column
                    codes = var_group[key][()]
                    categories = var_categories_group[key][()]
                    var[key] = pd.Categorical.from_codes(codes, categories.astype(str))
                else:
                    # Regular column
                    data = var_group[key][()]  # dataset - use [()] to read all data
                    var[key] = data
        
        var_time = time.time() - var_start
        var_memory = get_memory_usage()
        logger.info(f"Var loaded in {var_time:.1f}s, memory now: {var_memory:.2f} GB (+{var_memory-obs_memory:.2f} GB)")
        
        # Read X matrix for selected indices
        X = None
        if 'X' in f:
            x_start = time.time()
            if isinstance(f['X'], h5py.Group):
                # Sparse matrix - extract only rows we need
                logger.info("Loading sparse X matrix (CSR format)...")
                
                # Get sparse matrix metadata
                shape = tuple(f['X'].attrs['shape'])
                logger.info(f"  Original matrix shape: {shape}")
                logger.info(f"  Extracting {len(group_indices):,} rows...")
                
                # Read indptr to understand row structure
                indptr = f['X']['indptr'][()]
                logger.info(f"  Read indptr array (size: {len(indptr):,})")
                
                # Calculate which data elements we need for our selected rows
                row_starts = []
                row_ends = []
                for idx in group_indices:
                    row_starts.append(indptr[idx])
                    row_ends.append(indptr[idx + 1])
                
                # Create mapping of old data indices to new
                data_indices_needed = []
                for start, end in zip(row_starts, row_ends):
                    data_indices_needed.extend(range(start, end))
                
                data_indices_needed = np.array(data_indices_needed)
                logger.info(f"  Need {len(data_indices_needed):,} non-zero values from {f['X']['data'].size:,} total")
                
                # Read only the data and indices we need
                if len(data_indices_needed) > 0:
                    data_subset = f['X']['data'][data_indices_needed]
                    indices_subset = f['X']['indices'][data_indices_needed]
                else:
                    data_subset = np.array([])
                    indices_subset = np.array([])
                
                # Build new indptr for subset matrix
                new_indptr = [0]
                for start, end in zip(row_starts, row_ends):
                    new_indptr.append(new_indptr[-1] + (end - start))
                new_indptr = np.array(new_indptr)
                
                # Create sparse matrix with only selected rows
                sparse_memory_before = get_memory_usage()
                X = sparse.csr_matrix(
                    (data_subset, indices_subset, new_indptr), 
                    shape=(len(group_indices), shape[1])
                )
                sparse_memory_after = get_memory_usage()
                
                logger.info(f"  Sparse subset matrix created: shape {X.shape}, {X.nnz:,} non-zero values")
                logger.info(f"  Memory used for sparse matrix: {sparse_memory_after:.2f} GB (+{sparse_memory_after-sparse_memory_before:.2f} GB)")
            else:
                # Dense matrix - read only needed rows
                logger.info(f"Loading dense X matrix (selecting {len(group_indices):,} rows)...")
                X = f['X'][group_indices, :]
            
            x_time = time.time() - x_start
            x_memory = get_memory_usage()
            logger.info(f"X matrix loaded in {x_time:.1f}s, memory now: {x_memory:.2f} GB (+{x_memory-var_memory:.2f} GB)")
        
        # Read obsm embeddings if specified
        obsm = {}
        if embed_key and 'obsm' in f and embed_key in f['obsm']:
            obsm_start = time.time()
            embed_shape = f['obsm'][embed_key].shape
            logger.info(f"Loading embedding '{embed_key}' with shape {embed_shape}...")
            logger.info(f"  Selecting {len(group_indices):,} rows from embedding...")
            
            obsm[embed_key] = f['obsm'][embed_key][group_indices, :]
            
            obsm_time = time.time() - obsm_start
            obsm_memory = get_memory_usage()
            logger.info(f"Embedding loaded in {obsm_time:.1f}s, memory now: {obsm_memory:.2f} GB (+{obsm_memory-x_memory:.2f} GB)")
        
        # Create filtered obs
        logger.info("Creating filtered obs dataframe...")
        obs_filtered = obs.iloc[group_indices].reset_index(drop=True)
        
    # Create AnnData object
    logger.info("Creating AnnData object...")
    adata_start = time.time()
    adata = ad.AnnData(X=X, obs=obs_filtered, var=var)
    if obsm:
        adata.obsm = obsm
    
    adata_time = time.time() - adata_start
    final_memory = get_memory_usage()
    total_time = time.time() - start_time
    
    logger.info(f"AnnData created in {adata_time:.1f}s")
    logger.info(f"Total loading time: {total_time:.1f}s")
    logger.info(f"Final memory usage: {final_memory:.2f} GB (total increase: {final_memory-initial_memory:.2f} GB)")
    logger.info(f"Loaded AnnData shape: {adata.shape}")
    
    return adata

def process_group_separately(args, group_name, group_value, all_detailed_results):
    """Process a single group to minimize memory usage.
    
    Args:
        args: Command line arguments
        group_name: Name of the grouping column (e.g., 'plate')
        group_value: Value to filter for (e.g., 'plate1')
        all_detailed_results: List to accumulate detailed results across groups
        
    Returns:
        dict: Aggregated results for this group
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing group: {group_name} = {group_value}")
    logger.info(f"{'='*60}")
    
    # Map control perturbation
    actual_control_pert = map_control_perturbation(args.control_pert)
    
    # Determine which embedding keys to load
    real_key = args.embed_key_real or args.embed_key
    pred_key = args.embed_key_pred or args.embed_key
    
    # Load only this group's data using h5py
    real_group = load_group_data(args.adata_real, group_name, group_value, real_key)
    pred_group = load_group_data(args.adata_pred, group_name, group_value, pred_key)
    
    if real_group is None or pred_group is None:
        logger.warning(f"Skipping group {group_value}: no data found")
        return None
    
    logger.info(f"Group {group_value}: {real_group.shape[0]:,} real cells, {pred_group.shape[0]:,} pred cells")
    
    # Check if control exists in this group
    if actual_control_pert not in real_group.obs[args.pert_col].unique():
        logger.warning(f"Skipping group {group_value}: no control perturbation '{actual_control_pert}'")
        return None
    
    # Filter to common perturbations within this group
    real_group, pred_group = filter_to_common_perturbations(
        real_group, pred_group, args.pert_col, actual_control_pert, None  # No nested grouping
    )
    
    # Prepare embeddings
    final_embed_key = prepare_embeddings(
        real_group, pred_group,
        embed_key_real=args.embed_key_real,
        embed_key_pred=args.embed_key_pred,
        embed_key=args.embed_key
    )
    
    # Compute pearson_delta for this group
    logger.info(f"Computing Pearson delta for group {group_value}...")
    results, detailed_results = compute_pearson_delta_optimized(
        real_group, pred_group, args.pert_col, actual_control_pert,
        group_by_cols=None,  # Already filtered to group
        embed_key=final_embed_key
    )
    
    # Add group information to detailed results
    for detail in detailed_results:
        detail['group_name'] = group_name
        detail['group_value'] = group_value
        all_detailed_results.append(detail)
    
    # Free memory
    del real_group, pred_group
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compute pearson_delta metric without DE computation"
    )
    
    # Core arguments (similar to cell-eval run)
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
        "--allow-discrete",
        action="store_true",
        help="Allow discrete data to be evaluated (usually expected to be norm-logged inputs)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./cell_eval_pearson_delta_results",
        help="Output directory to write results [default: %(default)s]",
    )
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        help="Additional categorical columns to group by before calculating deltas (e.g., plate batch timepoint)",
    )
    parser.add_argument(
        "--use-backed",
        action="store_true",
        help="Use backed mode for reading H5AD files (memory efficient but may have compatibility issues)",
    )
    
    args = parser.parse_args()
    
    # Map control perturbation name to actual value in h5ad files
    actual_control_pert = map_control_perturbation(args.control_pert)
    if actual_control_pert != args.control_pert:
        logger.info(f"Mapping control perturbation '{args.control_pert}' -> '{actual_control_pert}'")
    
    logger.info(f"Reading predicted anndata from {args.adata_pred}")
    logger.info(f"Reading real anndata from {args.adata_real}")
    if args.embed_key:
        logger.info(f"Using embedding key: {args.embed_key}")
    if args.group_by:
        logger.info(f"Using categorical grouping by columns: {args.group_by}")
        logger.info("Deltas will be calculated within groups (e.g., per plate, batch, etc.)")
    
    try:
        os.makedirs(args.outdir, exist_ok=True)
        
        # Early validation for group-by columns if specified
        if args.group_by:
            logger.info("Validating group-by columns exist in both datasets...")
            
            # Read metadata for validation
            if args.use_backed:
                real_check = ad.read_h5ad(args.adata_real, backed='r')
                real_obs_cols = list(real_check.obs.columns)
                real_check.file.close()  # Close file handle to free resources
                
                pred_check = ad.read_h5ad(args.adata_pred, backed='r')
                pred_obs_cols = list(pred_check.obs.columns)
                pred_check.file.close()  # Close file handle to free resources
            else:
                # Load full files for validation (may use more memory)
                real_check = ad.read_h5ad(args.adata_real)
                real_obs_cols = list(real_check.obs.columns)
                
                pred_check = ad.read_h5ad(args.adata_pred)
                pred_obs_cols = list(pred_check.obs.columns)
            
            for col in args.group_by:
                if col not in real_obs_cols:
                    raise ValueError(f"Group-by column '{col}' not found in real data")
                if col not in pred_obs_cols:
                    raise ValueError(f"Group-by column '{col}' not found in predicted data")
            
            logger.info(f"All group-by columns validated: {args.group_by}")
        
        # Handle celltype splitting if specified
        if args.celltype_col is not None:
            logger.info("\n" + "="*60)
            logger.info(f"Processing with celltype splitting (column: {args.celltype_col})")
            
            # Initialize all_results_by_celltype for all cases
            all_results_by_celltype = {}
            
            # Check if we should use memory-efficient mode
            if args.use_backed and (args.group_by and len(args.group_by) == 1):
                # Memory-efficient mode for celltype + group-by
                logger.info(f"✓ MEMORY-EFFICIENT MODE: Processing each {args.group_by[0]} within each celltype separately")
                logger.info(f"Initial memory usage: {get_memory_usage():.2f} GB")
                
                # First, get unique celltypes and groups using backed mode
                logger.info(f"\nIdentifying unique celltypes and {args.group_by[0]} values...")
                real_backed = ad.read_h5ad(args.adata_real, backed='r')
                unique_celltypes = real_backed.obs[args.celltype_col].unique()
                unique_groups = real_backed.obs[args.group_by[0]].unique()
                real_backed.file.close()
                
                logger.info(f"Found {len(unique_celltypes):,} celltypes: {sorted(unique_celltypes)}")
                logger.info(f"Found {len(unique_groups):,} {args.group_by[0]} values: {sorted(unique_groups)}")
                
                # Process each celltype-group combination separately
                for ct in sorted(unique_celltypes):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Processing celltype: {ct}")
                    ct_results = {}
                    ct_detailed = []
                    
                    for group_value in sorted(unique_groups):
                        logger.info(f"\n  Processing {args.group_by[0]}={group_value} for celltype {ct}...")
                        
                        # Load only this celltype-group combination
                        real_subset = load_celltype_group_data(
                            args.adata_real, args.celltype_col, ct, 
                            args.group_by[0], group_value,
                            embed_key=args.embed_key_real or args.embed_key
                        )
                        pred_subset = load_celltype_group_data(
                            args.adata_pred, args.celltype_col, ct,
                            args.group_by[0], group_value,
                            embed_key=args.embed_key_pred or args.embed_key
                        )
                        
                        if real_subset is None or pred_subset is None:
                            logger.warning(f"    Skipping: no data found")
                            continue
                            
                        logger.info(f"    Loaded {real_subset.shape[0]:,} real cells, {pred_subset.shape[0]:,} pred cells")
                        
                        # Process this subset immediately
                        logger.info(f"    Processing Pearson delta for this subset...")
                        
                        # Filter to common perturbations within this subset
                        try:
                            real_subset, pred_subset = filter_to_common_perturbations(
                                real_subset, pred_subset, args.pert_col, actual_control_pert, None  # No nested grouping
                            )
                        except ValueError as e:
                            logger.warning(f"    Skipping: {e}")
                            # Free memory before continuing
                            del real_subset, pred_subset
                            continue
                        
                        # Prepare embeddings
                        final_embed_key = prepare_embeddings(
                            real_subset, pred_subset,
                            embed_key_real=args.embed_key_real,
                            embed_key_pred=args.embed_key_pred,
                            embed_key=args.embed_key
                        )
                        
                        # Compute pearson_delta for this subset
                        subset_results, subset_detailed = compute_pearson_delta_optimized(
                            real_subset, pred_subset, args.pert_col, actual_control_pert,
                            group_by_cols=None,  # Already filtered to celltype-group
                            embed_key=final_embed_key
                        )
                        
                        # Accumulate results for this celltype
                        for pert, score in subset_results.items():
                            if pert not in ct_results:
                                ct_results[pert] = []
                            ct_results[pert].append(score)
                        
                        # Add celltype and group info to detailed results
                        for detail in subset_detailed:
                            detail['celltype'] = ct
                            detail['group_name'] = args.group_by[0]
                            detail['group_value'] = group_value
                            ct_detailed.append(detail)
                        
                        # Free memory immediately after processing
                        del real_subset, pred_subset
                        logger.info(f"    Subset processed and freed, memory: {get_memory_usage():.2f} GB")
                
                    # Average results across groups for this celltype
                    logger.info(f"\nAveraging results across {args.group_by[0]} groups for celltype {ct}...")
                    ct_final_results = {}
                    for pert, scores in ct_results.items():
                        ct_final_results[pert] = float(np.mean(scores))
                        logger.debug(f"  {pert}: {len(scores):,} groups, avg = {ct_final_results[pert]:.4f}")
                    
                    # Store results for this celltype
                    all_results_by_celltype[ct] = (ct_final_results, ct_detailed)
                    logger.info(f"Celltype {ct} completed: {len(ct_final_results):,} perturbations, {len(ct_detailed):,} detailed results")
                    
            elif args.use_backed and not args.group_by:
                # Memory-efficient mode for celltype-only processing (no group-by)
                logger.info(f"✓ MEMORY-EFFICIENT MODE: Processing each celltype separately")
                logger.info(f"Initial memory usage: {get_memory_usage():.2f} GB")
                
                # Get unique celltypes using backed mode
                logger.info(f"\nIdentifying unique celltypes...")
                real_backed = ad.read_h5ad(args.adata_real, backed='r')
                unique_celltypes = real_backed.obs[args.celltype_col].unique()
                real_backed.file.close()
                
                logger.info(f"Found {len(unique_celltypes):,} celltypes: {sorted(unique_celltypes)}")
                
                # Process each celltype separately
                for ct in sorted(unique_celltypes):
                    logger.info(f"\n{'='*60}")
                    logger.info(f"Processing celltype: {ct}")
                    
                    # Load only this celltype using backed mode
                    logger.info(f"Loading celltype {ct} data...")
                    real_backed = ad.read_h5ad(args.adata_real, backed='r')
                    pred_backed = ad.read_h5ad(args.adata_pred, backed='r')
                    
                    # Filter to this celltype
                    real_mask = real_backed.obs[args.celltype_col] == ct
                    pred_mask = pred_backed.obs[args.celltype_col] == ct
                    
                    # Load subsets
                    real_ct = real_backed[real_mask, :].to_memory()
                    pred_ct = pred_backed[pred_mask, :].to_memory()
                    
                    # Close backed files
                    real_backed.file.close()
                    pred_backed.file.close()
                    
                    logger.info(f"  Loaded {real_ct.shape[0]:,} real cells, {pred_ct.shape[0]:,} pred cells")
                    
                    # Filter to common perturbations within this celltype
                    try:
                        real_ct, pred_ct = filter_to_common_perturbations(
                            real_ct, pred_ct, args.pert_col, actual_control_pert, None
                        )
                    except ValueError as e:
                        logger.warning(f"Skipping cell type {ct}: {e}")
                        del real_ct, pred_ct
                        continue
                    
                    # Prepare embeddings
                    final_embed_key = prepare_embeddings(
                        real_ct, pred_ct,
                        embed_key_real=args.embed_key_real,
                        embed_key_pred=args.embed_key_pred,
                        embed_key=args.embed_key
                    )
                    
                    # Compute pearson_delta for this celltype
                    logger.info("Computing Pearson delta (optimized pseudobulking)...")
                    results, detailed_results = compute_pearson_delta_optimized(
                        real_ct, pred_ct, args.pert_col, actual_control_pert,
                        group_by_cols=None,  # No grouping
                        embed_key=final_embed_key
                    )
                    
                    # Add celltype info to detailed results
                    for detail in detailed_results:
                        detail['celltype'] = ct
                    
                    # Store results for this celltype
                    all_results_by_celltype[ct] = (results, detailed_results)
                    logger.info(f"Celltype {ct} completed: {len(results):,} perturbations, {len(detailed_results):,} detailed results")
                    
                    # Free memory immediately
                    del real_ct, pred_ct
                    logger.info(f"Memory freed, current usage: {get_memory_usage():.2f} GB")
                    
            else:
                # Original behavior: load full datasets
                logger.info("⚠️  FULL DATASET MODE: Loading entire datasets into memory")
                logger.info("This will use significant memory with celltype splitting.")
                logger.info(f"Memory before loading: {get_memory_usage():.2f} GB")
                
                logger.info(f"\nLoading real data from {args.adata_real}...")
                load_start = time.time()
                real = ad.read_h5ad(args.adata_real)
                load_time = time.time() - load_start
                logger.info(f"Real data loaded in {load_time:.1f}s: shape {real.shape}, memory now: {get_memory_usage():.2f} GB")
                
                logger.info(f"\nLoading predicted data from {args.adata_pred}...")
                load_start = time.time()
                pred = ad.read_h5ad(args.adata_pred)
                load_time = time.time() - load_start
                logger.info(f"Predicted data loaded in {load_time:.1f}s: shape {pred.shape}, memory now: {get_memory_usage():.2f} GB")
            
                # Split by celltype (filtering will be done per celltype)
                logger.info(f"\nSplitting datasets by {args.celltype_col}...")
                split_start = time.time()
                real_split = split_anndata_on_celltype(real, args.celltype_col)
                pred_split = split_anndata_on_celltype(pred, args.celltype_col)
                split_time = time.time() - split_start
                logger.info(f"Split completed in {split_time:.1f}s, found {len(real_split):,} celltypes")
                logger.info(f"Memory after splitting: {get_memory_usage():.2f} GB")
                logger.info(f"Celltypes: {list(real_split.keys())}")
                
                if len(real_split) != len(pred_split):
                    raise ValueError(
                        f"Number of celltypes in real and pred anndata must match: "
                        f"{len(real_split)} != {len(pred_split)}"
                    )
                
                for ct in real_split.keys():
                    real_ct = real_split[ct]
                    pred_ct = pred_split[ct]
                    
                    # Filter to perturbations common within this cell type
                    logger.info(f"\nProcessing cell type: {ct}")
                    logger.info(f"  Shape: {real_ct.shape}, memory: {get_memory_usage():.2f} GB")
                    try:
                        real_ct, pred_ct = filter_to_common_perturbations(
                            real_ct, pred_ct, args.pert_col, actual_control_pert, args.group_by
                        )
                    except ValueError as e:
                        logger.warning(f"Skipping cell type {ct}: {e}")
                        continue
                    
                    # Prepare embeddings with potentially different keys
                    final_embed_key = prepare_embeddings(
                        real_ct, pred_ct, 
                        embed_key_real=args.embed_key_real,
                        embed_key_pred=args.embed_key_pred,
                        embed_key=args.embed_key
                    )
                    
                    # Compute pearson_delta metric for this celltype (bypassing cell-eval preprocessing)
                    logger.info("Computing Pearson delta (optimized pseudobulking)...")
                    results, detailed_results = compute_pearson_delta_optimized(
                        real_ct, pred_ct, args.pert_col, actual_control_pert, 
                        group_by_cols=args.group_by, embed_key=final_embed_key
                    )
                    
                    # Process results into DataFrames
                    results_df, agg_results_df = process_pearson_delta_results(results, ct)
                    
                    # Save CSV files for this celltype
                    save_results_csv(results_df, agg_results_df, args.outdir, ct, detailed_results)
                    
                    # Store results for this celltype
                    all_results_by_celltype[ct] = (results, detailed_results)
            
            # Final output processing and file saving
            logger.info(f"\n{'='*60}")
            logger.info("Processing final results across all celltypes...")
            
            all_agg_results = []
            
            for ct, (ct_results, ct_detailed) in all_results_by_celltype.items():
                logger.info(f"\nProcessing results for celltype: {ct}")
                
                # Process results into DataFrames
                results_df, agg_results_df = process_pearson_delta_results(ct_results, ct)
                
                # Save CSV files for this celltype
                save_results_csv(results_df, agg_results_df, args.outdir, ct, ct_detailed)
                
                # Store aggregated results for overall average
                all_agg_results.append(agg_results_df)
                
                logger.info(f"Saved results for celltype {ct}: {len(ct_results):,} perturbations")
            
            # Compute overall average across all celltypes
            if all_agg_results:
                logger.info(f"\nComputing overall average across {len(all_agg_results):,} celltypes...")
                overall_scores = []
                for agg_df in all_agg_results:
                    mean_row = agg_df.filter(pl.col("statistic") == "mean")
                    if len(mean_row) > 0:
                        overall_scores.append(float(mean_row.select("pearson_delta").item()))
                
                if overall_scores:
                    overall_mean = sum(overall_scores) / len(overall_scores)
                    logger.info(f"Overall Pearson Delta correlation across all celltypes: {overall_mean:.4f}")
            
        else:
            # Check if we're using group-by with backed mode for memory efficiency
            if args.group_by and len(args.group_by) == 1 and args.use_backed:
                # Memory-efficient mode: process each group separately
                group_col = args.group_by[0]
                logger.info(f"\nMemory-efficient mode: Processing each {group_col} separately")
                logger.info(f"Initial memory usage: {get_memory_usage():.2f} GB")
                
                # Get unique group values using backed mode
                logger.info(f"Identifying unique {group_col} values from {args.adata_real}...")
                initial_mem = get_memory_usage()
                
                real_backed = ad.read_h5ad(args.adata_real, backed='r')
                logger.info(f"Backed file opened, memory: {get_memory_usage():.2f} GB (+{get_memory_usage()-initial_mem:.2f} GB)")
                
                logger.info(f"Reading {group_col} column from obs...")
                unique_groups = real_backed.obs[group_col].unique()
                logger.info(f"Found unique values, memory: {get_memory_usage():.2f} GB")
                
                real_backed.file.close()
                logger.info(f"Backed file closed, memory: {get_memory_usage():.2f} GB")
                
                logger.info(f"Found {len(unique_groups):,} unique {group_col} values: {sorted(unique_groups)}")
                
                # Process each group separately
                all_results = {}
                all_detailed_results = []
                
                for group_value in sorted(unique_groups):
                    group_results = process_group_separately(
                        args, group_col, group_value, all_detailed_results
                    )
                    
                    if group_results:
                        # Aggregate results across groups
                        for pert, score in group_results.items():
                            if pert not in all_results:
                                all_results[pert] = []
                            all_results[pert].append(score)
                
                # Average results across groups
                logger.info("\n" + "="*60)
                logger.info("Aggregating results across all groups...")
                final_results = {}
                for pert, scores in all_results.items():
                    final_results[pert] = float(np.mean(scores))
                    logger.debug(f"Perturbation {pert}: {len(scores):,} groups, avg = {final_results[pert]:.4f}")
                
                # Process results into DataFrames
                results_df, agg_results_df = process_pearson_delta_results(final_results)
                
                # Save CSV files with detailed results
                save_results_csv(results_df, agg_results_df, args.outdir, detailed_results=all_detailed_results)
                
                # Report overall average
                mean_row = agg_results_df.filter(pl.col("statistic") == "mean")
                if len(mean_row) > 0:
                    overall_mean = float(mean_row.select("pearson_delta").item())
                    logger.info(f"\nOverall Pearson Delta correlation: {overall_mean:.4f}")
                    
            else:
                # Original behavior: process all data together
                real = ad.read_h5ad(args.adata_real)
                pred = ad.read_h5ad(args.adata_pred)
                
                # Filter to common perturbations
                real, pred = filter_to_common_perturbations(real, pred, args.pert_col, actual_control_pert, args.group_by)
                
                # Prepare embeddings with potentially different keys
                final_embed_key = prepare_embeddings(
                    real, pred,
                    embed_key_real=args.embed_key_real,
                    embed_key_pred=args.embed_key_pred,
                    embed_key=args.embed_key
                )
                
                # Compute pearson_delta metric (bypassing cell-eval preprocessing)
                logger.info("Computing Pearson delta (optimized pseudobulking)...")
                results, detailed_results = compute_pearson_delta_optimized(
                    real, pred, args.pert_col, actual_control_pert,
                    group_by_cols=args.group_by, embed_key=final_embed_key
                )
                
                # Process results into DataFrames
                results_df, agg_results_df = process_pearson_delta_results(results)
                
                # Save CSV files
                save_results_csv(results_df, agg_results_df, args.outdir, detailed_results=detailed_results)
                
                # Report overall average (single celltype case)
                mean_row = agg_results_df.filter(pl.col("statistic") == "mean")
                if len(mean_row) > 0:
                    overall_mean = float(mean_row.select("pearson_delta").item())
                    logger.info(f"Overall Pearson Delta correlation: {overall_mean:.4f}")
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()