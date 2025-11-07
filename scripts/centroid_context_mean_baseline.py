#!/usr/bin/env python3
"""
Compute Context Mean baseline using centroids with hierarchical summaries.

This script implements the CONTEXT MEAN baseline for drug response prediction 
using pre-computed centroid embeddings. It computes mean cell line effects by 
averaging deltas across perturbations for each cell line, then uses these cell-specific 
effects to predict responses on test data. Evaluation uses Pearson correlation 
between predicted and true deltas. All computations maintain strict batch boundaries 
to avoid batch effects.

Baseline Logic:
- For each cell line: average response deltas across training perturbations
- Prediction: control(cell_line) + mean_delta(cell_line)

Key features:
- Supports both single H5AD file with batch column or legacy plate directory structure
- Configurable batch column for flexible batch definitions
- Parses TOML splits for train/test assignments  
- Computes correlations directly without storing all deltas
- Generates hierarchical summaries: per cell-type per batch, per cell-type across batches, and overall
- Progress tracking with tqdm for all major operations

Usage:
    # New single-file approach with batch column
    python centroid_context_mean_baseline.py \
        --toml-file path/to/splits.toml \
        --centroids-file path/to/centroids.h5ad \
        --batch-col batch_column_name \
        --output-dir results/
        
    # Legacy plate-directory approach (deprecated)
    python centroid_context_mean_baseline.py \
        --toml-file path/to/splits.toml \
        --centroids-dir path/to/centroids/directory \
        --output-dir results/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import toml
import warnings
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Context Mean baseline using centroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--toml-file', 
        required=True, 
        help='Path to TOML split file (e.g., generalization_converted_cell_lines.toml)'
    )
    parser.add_argument(
        '--centroids-dir', 
        help='Directory containing plate centroid H5AD files (e.g., by_plate_centroids/) [DEPRECATED: use --centroids-file]'
    )
    parser.add_argument(
        '--centroids-file', 
        help='Single H5AD file containing all centroids with batch information'
    )
    parser.add_argument(
        '--cell-col', 
        default='cell_line_id', 
        help='Column name for cell type/line identifiers'
    )
    parser.add_argument(
        '--pert-col', 
        default='drugname_drugconc', 
        help='Column name for perturbation identifiers'
    )
    parser.add_argument(
        '--batch-col', 
        help='Column name for batch identifiers (optional, if not provided treats all data as single batch)'
    )
    parser.add_argument(
        '--control-pert', 
        default='DMSO_TF', 
        help='Control condition name (will be mapped to actual format)'
    )
    parser.add_argument(
        '--embed-key', 
        default='X_hvg', 
        help='Key for embeddings in .obsm'
    )
    parser.add_argument(
        '--output-dir', 
        default='./pearson_baseline_results', 
        help='Output directory for results'
    )
    parser.add_argument(
        '--ignore-batch-boundaries',
        action='store_true',
        help='Compute global mean effects across all batches, ignoring batch boundaries (experimental)'
    )
    parser.add_argument(
        '--ignore-plate-boundaries',
        action='store_true',
        help='[DEPRECATED: use --ignore-batch-boundaries] Compute global mean effects across all plates'
    )
    parser.add_argument(
        '--abs-correlation',
        action='store_true',
        help='Compute correlations using absolute values of deltas (experimental)'
    )
    
    args = parser.parse_args()
    
    # Validation and backward compatibility
    if not args.centroids_file and not args.centroids_dir:
        parser.error("Must provide either --centroids-file or --centroids-dir")
    
    if args.centroids_file and args.centroids_dir:
        parser.error("Cannot provide both --centroids-file and --centroids-dir, choose one")
    
    # Handle backward compatibility for ignore boundaries
    if args.ignore_plate_boundaries and not args.ignore_batch_boundaries:
        args.ignore_batch_boundaries = True
        print("WARNING: --ignore-plate-boundaries is deprecated, using --ignore-batch-boundaries")

    # Warn about absolute correlation mode
    if args.abs_correlation:
        print("WARNING: Using absolute value correlation mode - correlations will be computed on abs(deltas)")

    return args


def load_plate_centroids(centroids_dir):
    """Load plate files with progress tracking."""
    plate_files = list(Path(centroids_dir).glob("plate_*.h5ad"))
    
    if not plate_files:
        raise ValueError(f"No plate_*.h5ad files found in {centroids_dir}")
    
    print(f"Found {len(plate_files)} plate files")
    
    plate_data = {}
    for plate_file in tqdm(plate_files, desc="Loading plate centroids"):
        plate_name = plate_file.stem.replace('plate_', '')  # e.g., "plate1" from "plate_plate1"
        adata = sc.read_h5ad(plate_file)
        plate_data[plate_name] = adata
        print(f"  Loaded {plate_name}: {adata.shape[0]} observations, {adata.shape[1]} variables")
    
    return plate_data


def load_batch_data(centroids_file=None, centroids_dir=None, batch_col=None):
    """Load batch data with progress tracking.
    
    Args:
        centroids_file: Path to single H5AD file (new approach)
        centroids_dir: Directory with plate_*.h5ad files (legacy approach)  
        batch_col: Column name for batch identifiers (optional)
        
    Returns:
        dict: batch_name -> AnnData mapping
    """
    
    # Legacy plate-based loading
    if centroids_dir:
        print("Using legacy plate-based loading...")
        return load_plate_centroids(centroids_dir)
    
    # New single file approach
    if not centroids_file:
        raise ValueError("Must provide either centroids_file or centroids_dir")
        
    print(f"Loading single centroids file: {centroids_file}")
    
    if not os.path.exists(centroids_file):
        raise ValueError(f"Centroids file not found: {centroids_file}")
        
    print("Reading H5AD file...")
    # Only load obs and obsm since we don't need the X matrix
    import pandas as pd
    import h5py
    
    def decode_categorical_column(obs_group: h5py.Group, column_name: str) -> np.ndarray:
        """
        Decode a categorical column from H5AD obs group.
        Handles both string and categorical (codes/categories) formats.
        Also handles double-encoded categories like b"b'value'".
        """
        column_item = obs_group[column_name]
        
        if hasattr(column_item, 'dtype'):
            # Direct string/numeric column
            values = column_item[:]
            if column_item.dtype.kind in ['S', 'U']:
                return values.astype(str)
            else:
                return values
        else:
            # Categorical column with codes/categories structure
            codes = column_item['codes'][:]
            categories = column_item['categories'][:]
            
            decoded_categories = []
            for cat in categories:
                # First decode bytes to string
                if isinstance(cat, bytes):
                    cat_str = cat.decode('utf-8')
                else:
                    cat_str = str(cat)
                
                # Handle double-encoded format like "b'value'" -> "value"
                if cat_str.startswith("b'") and cat_str.endswith("'"):
                    cat_str = cat_str[2:-1]  # Remove b' and trailing '
                elif cat_str.startswith('b"') and cat_str.endswith('"'):
                    cat_str = cat_str[2:-1]  # Remove b" and trailing "
                    
                decoded_categories.append(cat_str)
            
            return np.array(decoded_categories)[codes]
    
    with h5py.File(centroids_file, 'r') as f:
        # Load obs metadata from HDF5 group structure
        obs_data = {}
        if 'obs' in f:
            obs_group = f['obs']
            for key in obs_group.keys():
                if key == '_index':
                    # Handle index specially
                    obs_data['index'] = obs_group[key][:].astype(str)
                else:
                    # Use the proper categorical decoder
                    obs_data[key] = decode_categorical_column(obs_group, key)
        
        # Create DataFrame from obs data
        obs = pd.DataFrame(obs_data)
        if 'index' in obs_data:
            obs.index = obs_data['index']
            obs = obs.drop('index', axis=1)
        
        print(f"  Loaded obs with columns: {list(obs.columns)}")
        print(f"  Sample values: {dict(obs.iloc[0])}")
        
        # Create minimal AnnData object (we'll add obsm separately) 
        adata = ad.AnnData(obs=obs)
        
        # Load only the obsm embeddings we need
        if 'obsm' in f:
            for key in f['obsm'].keys():
                adata.obsm[key] = f['obsm'][key][:]
                print(f"  Loaded obsm['{key}'] with shape: {f['obsm'][key].shape}")
    
    print(f"Loaded data: {adata.shape[0]} observations, obsm keys: {list(adata.obsm.keys())}")
    
    batch_data = {}
    
    # If no batch column specified, treat all data as single batch
    if not batch_col or batch_col not in adata.obs.columns:
        if batch_col and batch_col not in adata.obs.columns:
            print(f"WARNING: Batch column '{batch_col}' not found in data, treating as single batch")
        print("Treating all data as single batch named 'all'")
        batch_data['all'] = adata
        return batch_data
    
    # Split data by batch column
    unique_batches = adata.obs[batch_col].unique()
    print(f"Found {len(unique_batches)} unique batches in column '{batch_col}': {sorted(unique_batches)}")
    
    for batch_name in tqdm(unique_batches, desc="Splitting data by batches"):
        batch_mask = adata.obs[batch_col] == batch_name
        batch_adata = adata[batch_mask].copy()
        batch_data[str(batch_name)] = batch_adata
        print(f"  Batch {batch_name}: {batch_adata.shape[0]} observations")
    
    return batch_data


def parse_toml_splits(toml_file):
    """Parse TOML with progress tracking.
    
    Logic:
    1. If cell lines are NOT mentioned in TOML → all perturbations are training data
    2. If cell lines ARE mentioned in TOML → only val/test perturbations are held out,
       all other perturbations for those cell lines are training data
    3. If cell lines are in 'zeroshot' section → ALL perturbations are held out (complete zero-shot)
    """
    print(f"Parsing TOML file: {toml_file}")
    
    if not os.path.exists(toml_file):
        raise ValueError(f"TOML file not found: {toml_file}")
    
    data = toml.load(toml_file)
    
    explicit_splits = {}  # Only stores val/test combinations
    holdout_cells = set()  # Cell lines with explicit splits (fewshot)
    zeroshot_cells = set()  # Cell lines with complete holdout (zeroshot)
    
    # Handle zeroshot section - complete cell line holdouts
    if 'zeroshot' in data:
        cell_lines = list(data['zeroshot'].keys())
        for cell_line_key in tqdm(cell_lines, desc="Parsing zeroshot splits"):
            cell_line = cell_line_key.split('.')[-1]  # Extract CVCL_XXXX from "tahoe.CVCL_XXXX"
            zeroshot_cells.add(cell_line)
            # Mark this cell line as having zeroshot test data (all perturbations)
            explicit_splits[(cell_line, '*')] = 'test'  # Special marker for zeroshot
    
    # Handle fewshot section - partial cell line holdouts
    if 'fewshot' in data:
        cell_lines = list(data['fewshot'].keys())
        for cell_line_key in tqdm(cell_lines, desc="Parsing fewshot splits"):
            cell_line = cell_line_key.split('.')[-1]  # Extract CVCL_XXXX from "tahoe.CVCL_XXXX"
            holdout_cells.add(cell_line)
            assignments = data['fewshot'][cell_line_key]
            
            # Only store val and test - everything else is implicitly training
            for split_type in ['val', 'test']:
                if split_type in assignments:
                    for pert in assignments[split_type]:
                        explicit_splits[(cell_line, pert)] = split_type
    
    # Ensure we have at least one section
    if not ('fewshot' in data or 'zeroshot' in data):
        raise ValueError("TOML must have 'fewshot' or 'zeroshot' section")
    
    # Print split statistics
    split_counts = {}
    for split in explicit_splits.values():
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print(f"Explicit holdout combinations:")
    for split_type, count in split_counts.items():
        if split_type == 'test' and len(zeroshot_cells) > 0:
            fewshot_test = count - len(zeroshot_cells)  # Subtract zeroshot markers
            print(f"  {split_type}: {count} combinations ({fewshot_test} fewshot + {len(zeroshot_cells)} zeroshot)")
        else:
            print(f"  {split_type}: {count} combinations")
    
    print(f"Fewshot holdout cell lines: {len(holdout_cells)} ({sorted(list(holdout_cells))})")
    print(f"Zeroshot holdout cell lines: {len(zeroshot_cells)} ({sorted(list(zeroshot_cells))})")
    print(f"NOTE: All other cell lines and unlisted perturbations are training data")
    
    return explicit_splits, holdout_cells, zeroshot_cells


def evaluate_batch(batch_adata, batch_name, explicit_splits, holdout_cells, zeroshot_cells, control_pert, pert_col,
                   cell_col, embed_key, abs_correlation=False):
    """Process one batch with progress tracking."""
    
    # Map control string if needed  
    actual_control = f"[('{control_pert}', 0.0, 'uM')]" if control_pert == "DMSO_TF" else control_pert
    
    # Get controls for this batch
    controls = {}
    control_mask = batch_adata.obs[pert_col] == actual_control
    
    if not control_mask.any():
        print(f"  WARNING: No control perturbation '{actual_control}' found in batch {batch_name}")
        return []
    
    control_indices = batch_adata.obs[control_mask].index
    for i, idx in enumerate(tqdm(control_indices, 
                                desc=f"  Loading controls for {batch_name}", 
                                leave=False)):
        cell_line = batch_adata.obs.loc[idx, cell_col]
        # Get the integer position for the obsm array indexing
        int_idx = batch_adata.obs.index.get_loc(idx)
        controls[cell_line] = batch_adata.obsm[embed_key][int_idx]
    
    print(f"  Found controls for {len(controls)} cell lines")
    
    # Get unique cell lines (that have controls)
    cell_lines = list(controls.keys())
    
    # Compute mean cell line effects from training data
    cell_effects = {}
    for cell_line in tqdm(cell_lines, 
                         desc=f"  Computing training effects for {batch_name}", 
                         leave=False):
        training_deltas = []
        for idx in batch_adata.obs[batch_adata.obs[cell_col] == cell_line].index:
            pert = batch_adata.obs.loc[idx, pert_col]
            # Skip control perturbations
            if pert == actual_control:
                continue
            # Check if this is training data:
            # 1. Cell line not in zeroshot_cells (exclude ALL data from zeroshot cells) AND
            # 2. Cell line not in holdout_cells (implicit training) OR
            # 3. Cell line in holdout_cells but this perturbation not in explicit_splits (implicit training)
            is_training = (
                cell_line not in zeroshot_cells and  # Exclude ALL data from zeroshot cells
                (cell_line not in holdout_cells or 
                 explicit_splits.get((cell_line, pert)) is None)
            )
            
            if is_training:
                # Get the integer position for the obsm array indexing
                int_idx = batch_adata.obs.index.get_loc(idx)
                delta = batch_adata.obsm[embed_key][int_idx] - controls[cell_line]
                training_deltas.append(delta)
        
        if training_deltas:
            cell_effects[cell_line] = np.mean(training_deltas, axis=0)
    
    print(f"  Computed {len(cell_effects)} cell line effects from training data")
    
    # Check for zero-shot cell lines that cannot be predicted by context mean baseline
    zeroshot_test_cells = set()
    for cell_line in zeroshot_cells:
        if cell_line in controls:  # Only check cells that have controls (could be test data)
            zeroshot_test_cells.add(cell_line)
    
    if zeroshot_test_cells:
        print(f"  ERROR: Context mean baseline cannot predict for zero-shot cell lines: {sorted(zeroshot_test_cells)}")
        print(f"  REASON: Zero-shot cells have no training data to compute cell-specific mean effects")
        print(f"  SOLUTION: Use perturbation mean baseline or global mean baseline for zero-shot evaluation")
    
    # Evaluate on test set
    test_indices = []
    for idx in batch_adata.obs.index:
        cell_line = batch_adata.obs.loc[idx, cell_col]
        pert = batch_adata.obs.loc[idx, pert_col]
        # Check if this is test data: explicitly marked as 'test' in explicit_splits OR zeroshot cell
        actual_control = f"[('{control_pert}', 0.0, 'uM')]" if control_pert == "DMSO_TF" else control_pert
        is_test = (
            explicit_splits.get((cell_line, pert)) == 'test' or
            (cell_line in zeroshot_cells and pert != actual_control)
        )
        
        if (is_test and 
            cell_line in cell_effects and 
            cell_line in controls):
            test_indices.append(idx)
    
    print(f"  Found {len(test_indices)} test combinations to evaluate")
    
    correlations = []
    for idx in tqdm(test_indices, 
                    desc=f"  Computing correlations for {batch_name}", 
                    leave=False):
        cell_line = batch_adata.obs.loc[idx, cell_col]
        pert = batch_adata.obs.loc[idx, pert_col]
        
        # True delta
        # Get the integer position for the obsm array indexing  
        int_idx = batch_adata.obs.index.get_loc(idx)
        true_delta = batch_adata.obsm[embed_key][int_idx] - controls[cell_line]
        
        # Predicted delta (mean effect from cell line's training perturbations)
        pred_delta = cell_effects[cell_line]

        # Compute Pearson correlation
        if len(true_delta) > 1 and len(pred_delta) > 1:
            if abs_correlation:
                corr, p_value = pearsonr(np.abs(true_delta), np.abs(pred_delta))
            else:
                corr, p_value = pearsonr(true_delta, pred_delta)

            # Handle NaN correlations (can occur with constant vectors)
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        correlations.append({
            'batch': batch_name,
            'cell_line': cell_line,
            'perturbation': pert,
            'pearson_correlation': corr,
            'is_batch_matched': True  # Batch-specific mode: training and test from same batch
        })
    
    print(f"  Computed {len(correlations)} test correlations for {batch_name}")
    return correlations


def evaluate_global(batch_data, explicit_splits, holdout_cells, zeroshot_cells, control_pert, pert_col,
                    cell_col, embed_key, abs_correlation=False):
    """Evaluate using global mean effects across all batches (ignoring batch boundaries)."""
    
    print("Computing global mean effects across all batches...")
    
    # Map control string if needed  
    actual_control = f"[('{control_pert}', 0.0, 'uM')]" if control_pert == "DMSO_TF" else control_pert
    
    # Combine all batch data while preserving batch information
    all_data = []
    for batch_name, batch_adata in batch_data.items():
        # Add batch name to a copy of obs for tracking
        obs_copy = batch_adata.obs.copy()
        obs_copy['original_batch'] = batch_name
        
        # Create temporary combined data structure
        for idx in batch_adata.obs.index:
            int_idx = batch_adata.obs.index.get_loc(idx)
            all_data.append({
                'batch': batch_name,
                'original_index': idx,
                'int_idx': int_idx,
                'cell_line': batch_adata.obs.loc[idx, cell_col],
                'perturbation': batch_adata.obs.loc[idx, pert_col],
                'embedding': batch_adata.obsm[embed_key][int_idx],
                'adata_ref': batch_adata  # Keep reference for later access
            })
    
    print(f"  Combined {len(all_data)} observations from {len(batch_data)} batches")
    
    # Build global controls dictionary
    global_controls = {}
    control_count = 0
    for data_point in tqdm(all_data, desc="  Loading global controls", leave=False):
        if data_point['perturbation'] == actual_control:
            cell_line = data_point['cell_line']
            if cell_line not in global_controls:
                global_controls[cell_line] = []
            global_controls[cell_line].append(data_point['embedding'])
            control_count += 1
    
    # Average controls per cell line across all plates
    for cell_line in global_controls:
        global_controls[cell_line] = np.mean(global_controls[cell_line], axis=0)
    
    print(f"  Found global controls for {len(global_controls)} cell lines ({control_count} total control observations)")
    
    # Get all unique cell lines (that have controls)
    all_cell_lines = list(global_controls.keys())
    print(f"  Found {len(all_cell_lines)} unique cell lines")
    
    # Compute global mean cell line effects from ALL training data
    global_cell_effects = {}
    for cell_line in tqdm(all_cell_lines, desc="  Computing global training effects", leave=False):
        training_deltas = []
        for data_point in all_data:
            if data_point['cell_line'] == cell_line:
                pert = data_point['perturbation']
                # Skip control perturbations
                if pert == actual_control:
                    continue
                # Check if this is training data using same logic as plate-specific version
                is_training = (
                    cell_line not in zeroshot_cells and  # Exclude ALL data from zeroshot cells
                    (cell_line not in holdout_cells or 
                     explicit_splits.get((cell_line, pert)) is None)
                )
                
                if is_training:
                    delta = data_point['embedding'] - global_controls[cell_line]
                    training_deltas.append(delta)
        
        if training_deltas:
            global_cell_effects[cell_line] = np.mean(training_deltas, axis=0)
    
    print(f"  Computed {len(global_cell_effects)} global cell line effects from training data")
    
    # Check for zero-shot cell lines that cannot be predicted by context mean baseline
    global_zeroshot_test_cells = set()
    for cell_line in zeroshot_cells:
        if cell_line in global_controls:  # Only check cells that have controls (could be test data)
            global_zeroshot_test_cells.add(cell_line)
    
    if global_zeroshot_test_cells:
        print(f"  ERROR: Context mean baseline cannot predict for zero-shot cell lines: {sorted(global_zeroshot_test_cells)}")
        print(f"  REASON: Zero-shot cells have no training data to compute cell-specific mean effects")
        print(f"  SOLUTION: Use perturbation mean baseline or global mean baseline for zero-shot evaluation")
    
    # Evaluate on test set using global effects
    test_data_points = []
    for data_point in all_data:
        cell_line = data_point['cell_line']
        pert = data_point['perturbation']
        # Check if this is test data: explicitly marked as 'test' in explicit_splits OR zeroshot cell
        actual_control = f"[('{control_pert}', 0.0, 'uM')]" if control_pert == "DMSO_TF" else control_pert
        is_test = (
            explicit_splits.get((cell_line, pert)) == 'test' or
            (cell_line in zeroshot_cells and pert != actual_control)
        )
        
        if (is_test and 
            cell_line in global_cell_effects and 
            cell_line in global_controls):
            test_data_points.append(data_point)
    
    print(f"  Found {len(test_data_points)} test combinations to evaluate with global effects")
    
    # Compute correlations using global effects
    correlations = []
    for data_point in tqdm(test_data_points, desc="  Computing global correlations", leave=False):
        cell_line = data_point['cell_line']
        pert = data_point['perturbation']
        batch_name = data_point['batch']
        
        # True delta (same as before)
        true_delta = data_point['embedding'] - global_controls[cell_line]
        
        # Predicted delta (now using GLOBAL cell line effect)
        pred_delta = global_cell_effects[cell_line]

        # Compute Pearson correlation
        if len(true_delta) > 1 and len(pred_delta) > 1:
            if abs_correlation:
                corr, p_value = pearsonr(np.abs(true_delta), np.abs(pred_delta))
            else:
                corr, p_value = pearsonr(true_delta, pred_delta)

            # Handle NaN correlations (can occur with constant vectors)
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        correlations.append({
            'batch': batch_name,  # Still track batch for analysis
            'cell_line': cell_line,
            'perturbation': pert,
            'pearson_correlation': corr,
            'is_batch_matched': False  # Global mode: training from multiple batches, test from specific batch
        })
    
    print(f"  Computed {len(correlations)} test correlations using global effects")
    return correlations


def compute_hierarchical_summaries(all_correlations):
    """Compute summaries with progress tracking."""
    print("Computing hierarchical summaries...")
    df = pd.DataFrame(all_correlations)
    
    summaries = {}
    
    with tqdm(total=5, desc="Computing summaries") as pbar:
        # 1. Per cell-type per batch
        cell_batch_summary = df.groupby(['cell_line', 'batch'])['pearson_correlation'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).reset_index()
        summaries['per_cell_per_batch'] = cell_batch_summary.to_dict('records')
        pbar.set_postfix({'level': 'cell_per_batch'})
        pbar.update(1)
        
        # 2. Per cell-type across batches
        cell_summary = df.groupby('cell_line')['pearson_correlation'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).reset_index()
        summaries['per_cell_across_batches'] = cell_summary.to_dict('records')
        pbar.set_postfix({'level': 'cell_across_batches'})
        pbar.update(1)
        
        # 3. Per batch across cell types
        batch_summary = df.groupby('batch')['pearson_correlation'].agg([
            'mean', 'std', 'count', 'min', 'max'  
        ]).reset_index()
        summaries['per_batch'] = batch_summary.to_dict('records')
        pbar.set_postfix({'level': 'per_batch'})
        pbar.update(1)
        
        # 4. Batch-matching summary
        batch_matched_df = df[df['is_batch_matched'] == True] if 'is_batch_matched' in df.columns else df[0:0]  # Empty if no batch_matched column
        non_batch_matched_df = df[df['is_batch_matched'] == False] if 'is_batch_matched' in df.columns else df[0:0]  # Empty if no batch_matched column
        
        summaries['batch_matching'] = {
            'batch_matched': {
                'mean': float(batch_matched_df['pearson_correlation'].mean()) if len(batch_matched_df) > 0 else None,
                'std': float(batch_matched_df['pearson_correlation'].std()) if len(batch_matched_df) > 0 else None,
                'count': int(len(batch_matched_df)),
                'min': float(batch_matched_df['pearson_correlation'].min()) if len(batch_matched_df) > 0 else None,
                'max': float(batch_matched_df['pearson_correlation'].max()) if len(batch_matched_df) > 0 else None
            },
            'non_batch_matched': {
                'mean': float(non_batch_matched_df['pearson_correlation'].mean()) if len(non_batch_matched_df) > 0 else None,
                'std': float(non_batch_matched_df['pearson_correlation'].std()) if len(non_batch_matched_df) > 0 else None,
                'count': int(len(non_batch_matched_df)),
                'min': float(non_batch_matched_df['pearson_correlation'].min()) if len(non_batch_matched_df) > 0 else None,
                'max': float(non_batch_matched_df['pearson_correlation'].max()) if len(non_batch_matched_df) > 0 else None
            }
        }
        pbar.set_postfix({'level': 'batch_matching'})
        pbar.update(1)
        
        # 5. Overall summary
        summaries['overall'] = {
            'mean': float(df['pearson_correlation'].mean()),
            'std': float(df['pearson_correlation'].std()),
            'count': int(len(df)),
            'min': float(df['pearson_correlation'].min()),
            'max': float(df['pearson_correlation'].max()),
            'median': float(df['pearson_correlation'].median()),
            'q25': float(df['pearson_correlation'].quantile(0.25)),
            'q75': float(df['pearson_correlation'].quantile(0.75))
        }
        pbar.set_postfix({'level': 'overall'})
        pbar.update(1)
    
    return summaries, df


def main():
    """Main execution flow with overall progress tracking."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("STEP 1: Loading batch centroids")
    print("="*60)
    batch_data = load_batch_data(
        centroids_file=args.centroids_file,
        centroids_dir=args.centroids_dir,
        batch_col=args.batch_col
    )
    
    print("\n" + "="*60)
    print("STEP 2: Parsing TOML splits")
    print("="*60)
    explicit_splits, holdout_cells, zeroshot_cells = parse_toml_splits(args.toml_file)
    
    # Process batches (either batch-aware or global)
    print("\n" + "="*60)
    if args.ignore_batch_boundaries:
        print("STEP 3: Computing global effects (ignoring batch boundaries)")
        print("WARNING: Experimental mode - computing mean effects across all batches")
    else:
        print("STEP 3: Processing batches (batch-aware)")
    print("="*60)
    
    if args.ignore_batch_boundaries:
        # Global evaluation: compute mean effects across all batches
        all_correlations = evaluate_global(
            batch_data, explicit_splits, holdout_cells, zeroshot_cells,
            args.control_pert, args.pert_col,
            args.cell_col, args.embed_key, args.abs_correlation
        )
    else:
        # Batch-specific evaluation: process each batch separately (original behavior)
        all_correlations = []
        for batch_name, batch_adata in tqdm(batch_data.items(), desc="Processing batches"):
            print(f"\nProcessing batch {batch_name} ({batch_adata.shape[0]} observations)...")
            correlations = evaluate_batch(
                batch_adata, batch_name, explicit_splits, holdout_cells, zeroshot_cells,
                args.control_pert, args.pert_col,
                args.cell_col, args.embed_key, args.abs_correlation
            )
            all_correlations.extend(correlations)
    
    if not all_correlations:
        print("ERROR: No test correlations computed! Check your data and parameters.")
        return 1
    
    # Compute hierarchical summaries
    print("\n" + "="*60)
    print("STEP 4: Computing summary statistics")
    print("="*60)
    summaries, full_df = compute_hierarchical_summaries(all_correlations)
    
    # Save results
    print("\n" + "="*60)
    print("STEP 5: Saving results")
    print("="*60)
    
    # Save detailed correlations
    abs_suffix = "_abs" if args.abs_correlation else ""
    output_file = f"{args.output_dir}/context_mean{abs_suffix}_detailed_correlations.csv"
    full_df.to_csv(output_file, index=False)
    print(f"Saved detailed correlations to: {output_file}")

    # Save summaries as JSON
    summary_file = f"{args.output_dir}/context_mean{abs_suffix}_hierarchical_summaries.json"
    with open(summary_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved hierarchical summaries to: {summary_file}")
    
    # Print key statistics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Overall Statistics:")
    print(f"  Mean correlation: {summaries['overall']['mean']:.3f}")
    print(f"  Std deviation: {summaries['overall']['std']:.3f}")
    print(f"  Median: {summaries['overall']['median']:.3f}")
    print(f"  Q25-Q75: [{summaries['overall']['q25']:.3f}, {summaries['overall']['q75']:.3f}]")
    print(f"  Range: [{summaries['overall']['min']:.3f}, {summaries['overall']['max']:.3f}]")
    print(f"  N combinations: {summaries['overall']['count']}")
    
    # Print top performing cell lines
    cell_summary = pd.DataFrame(summaries['per_cell_across_batches'])
    if not cell_summary.empty:
        top_cells = cell_summary.nlargest(5, 'mean')[['cell_line', 'mean', 'count']]
        print("\nTop 5 cell lines by mean correlation:")
        for _, row in top_cells.iterrows():
            print(f"  {row['cell_line']}: {row['mean']:.3f} (n={row['count']})")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())