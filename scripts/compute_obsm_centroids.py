#!/usr/bin/env python3
"""
Compute centroids for .obsm slots based on categorical grouping columns.

Takes a .h5ad file with multiple .obsm slots and produces a new .h5ad file
with centroids computed for each group based on specified categorical columns.
Uses a memory-efficient single-pass algorithm.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from collections import defaultdict
import gc
import psutil
import logging
import time
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute centroids for .obsm slots based on categorical grouping",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_file",
        type=str,
        help="Input .h5ad file path"
    )
    
    parser.add_argument(
        "output_file", 
        type=str,
        help="Output .h5ad file path"
    )
    
    parser.add_argument(
        "--group-by",
        type=str,
        nargs="+",
        required=True,
        help="Categorical columns in .obs to group by (e.g., cell_line_id drugname_drugconc plate)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10000,
        help="Chunk size for processing large files"
    )
    
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=200.0,
        help="Memory limit in GB for deciding whether to use chunked processing"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with timestamps"
    )
    
    return parser.parse_args()


def get_file_size_gb(file_path):
    """Get file size in GB."""
    size_bytes = os.path.getsize(file_path)
    return size_bytes / (1024**3)


def estimate_memory_usage(adata, verbose=False):
    """Estimate memory usage for processing the AnnData object."""
    n_cells = adata.n_obs
    
    # Estimate memory for .obsm slots
    obsm_memory = 0
    obsm_info = {}
    
    for key, matrix in adata.obsm.items():
        memory_bytes = matrix.nbytes
        obsm_memory += memory_bytes
        obsm_info[key] = {
            'shape': matrix.shape,
            'dtype': matrix.dtype,
            'memory_mb': memory_bytes / (1024**2)
        }
    
    total_memory_gb = obsm_memory / (1024**3)
    
    if verbose:
        print(f"Memory estimation:")
        print(f"  Cells: {n_cells:,}")
        print(f"  .obsm slots: {len(adata.obsm)}")
        for key, info in obsm_info.items():
            print(f"    {key}: {info['shape']} ({info['dtype']}, {info['memory_mb']:.1f} MB)")
        print(f"  Total .obsm memory: {total_memory_gb:.2f} GB")
    
    return total_memory_gb, obsm_info


def validate_input(adata, group_columns, logger=None):
    """Validate input data and grouping columns."""
    errors = []
    warnings = []
    
    # Check if grouping columns exist
    missing_cols = [col for col in group_columns if col not in adata.obs.columns]
    if missing_cols:
        errors.append(f"Missing columns in .obs: {missing_cols}")
    
    # Check if we have .obsm data
    if len(adata.obsm) == 0:
        errors.append("No .obsm slots found in the input file")
    
    # Check for missing values in grouping columns
    total_missing = 0
    for col in group_columns:
        if col in adata.obs.columns:
            missing_count = adata.obs[col].isna().sum()
            if missing_count > 0:
                warnings.append(f"Column '{col}' has {missing_count:,} missing values (will be skipped)")
                total_missing += missing_count
    
    # Check for empty .obsm matrices
    for obsm_name, matrix in adata.obsm.items():
        if matrix.shape[0] != adata.n_obs:
            errors.append(f".obsm['{obsm_name}'] has {matrix.shape[0]} rows but expected {adata.n_obs}")
        if matrix.shape[1] == 0:
            errors.append(f".obsm['{obsm_name}'] has 0 dimensions")
    
    if errors:
        for error in errors:
            if logger:
                logger.error(error)
            else:
                print(f"ERROR: {error}", file=sys.stderr)
        return False
    
    if warnings:
        for warning in warnings:
            if logger:
                logger.warning(warning)
            else:
                print(f"WARNING: {warning}", file=sys.stderr)
    
    if logger:
        logger.info("Input validation passed:")
        logger.info(f"  Cells: {adata.n_obs:,}")
        if total_missing > 0:
            logger.info(f"  Cells with missing values: {total_missing:,} (will be skipped)")
        logger.info(f"  .obsm slots: {list(adata.obsm.keys())}")
        logger.info(f"  Grouping columns: {group_columns}")
        
        # Show group statistics
        for col in group_columns:
            unique_count = adata.obs[col].nunique()
            logger.info(f"    {col}: {unique_count} unique values")
    
    return True


def accumulate_group_statistics(adata, group_columns, logger=None):
    """
    Single-pass algorithm to accumulate statistics for each group.
    
    Returns:
        dict: {group_key: {obsm_name: {'sum': array, 'count': int}}}
    """
    if logger:
        logger.info("Starting single-pass accumulation...")
    
    # Initialize statistics dictionary
    group_stats = defaultdict(lambda: defaultdict(lambda: {'sum': None, 'count': 0}))
    
    n_cells = adata.n_obs
    progress_interval = max(1000, n_cells // 100)  # Report progress every 1% or 1000 cells
    
    # Single pass through all cells
    skipped_cells = 0
    
    # Create progress bar
    progress_bar = tqdm(total=n_cells, desc="Processing cells", unit="cells") if logger else None
    
    for cell_idx in range(n_cells):
        # Get group key from categorical columns
        group_values = []
        has_missing = False
        
        for col in group_columns:
            value = adata.obs.iloc[cell_idx][col]
            if pd.isna(value):
                has_missing = True
                break
            group_values.append(str(value))  # Convert to string for consistent hashing
        
        # Skip cells with missing values in grouping columns
        if has_missing:
            skipped_cells += 1
            continue
            
        group_key = tuple(group_values)
        
        # Process each .obsm slot
        for obsm_name, embedding_matrix in adata.obsm.items():
            cell_embedding = embedding_matrix[cell_idx]
            
            # Initialize sum array if this is the first cell for this group/obsm combination
            if group_stats[group_key][obsm_name]['sum'] is None:
                group_stats[group_key][obsm_name]['sum'] = np.zeros_like(cell_embedding, dtype=np.float64)
            
            # Accumulate
            group_stats[group_key][obsm_name]['sum'] += cell_embedding.astype(np.float64)
            group_stats[group_key][obsm_name]['count'] += 1
        
        # Update progress bar
        if progress_bar:
            n_groups = len(group_stats)
            progress_bar.set_postfix({
                'groups': n_groups,
                'skipped': skipped_cells
            })
            progress_bar.update(1)
    
    # Close progress bar
    if progress_bar:
        progress_bar.close()
    
    if logger:
        total_groups = len(group_stats)
        processed_cells = n_cells - skipped_cells
        logger.info(f"Accumulation complete: {total_groups} unique groups found")
        if skipped_cells > 0:
            logger.info(f"  Processed cells: {processed_cells:,} (skipped {skipped_cells:,} with missing values)")
        else:
            logger.info(f"  Processed cells: {processed_cells:,}")
    
    return dict(group_stats)


def compute_centroids(group_stats, logger=None):
    """
    Compute centroids from accumulated group statistics.
    
    Args:
        group_stats: Dictionary from accumulate_group_statistics
        
    Returns:
        tuple: (group_keys, centroids_dict)
            group_keys: list of group key tuples
            centroids_dict: {obsm_name: array of centroids}
    """
    if logger:
        logger.info("Computing centroids...")
    
    group_keys = list(group_stats.keys())
    n_groups = len(group_keys)
    
    if n_groups == 0:
        raise ValueError("No groups found in data")
    
    # Get .obsm slot names
    first_group = group_stats[group_keys[0]]
    obsm_names = list(first_group.keys())
    
    # Initialize centroid arrays
    centroids_dict = {}
    
    for obsm_name in obsm_names:
        # Get the dimensionality from the first group
        first_sum = first_group[obsm_name]['sum']
        embedding_dim = first_sum.shape[0] if first_sum.ndim == 1 else first_sum.shape
        
        # Create centroid array: (n_groups, embedding_dim)
        if isinstance(embedding_dim, int):
            centroids_dict[obsm_name] = np.zeros((n_groups, embedding_dim), dtype=np.float64)
        else:
            centroids_dict[obsm_name] = np.zeros((n_groups,) + embedding_dim, dtype=np.float64)
    
    # Compute centroids for each group with progress tracking
    centroid_progress = tqdm(total=n_groups, desc="Computing centroids", unit="groups") if logger and n_groups > 50 else None
    
    if logger and n_groups > 100:
        logger.info(f"Computing centroids for {n_groups:,} groups across {len(obsm_names)} .obsm slots...")
    
    for group_idx, group_key in enumerate(group_keys):
        group_data = group_stats[group_key]
        
        for obsm_name in obsm_names:
            stats = group_data[obsm_name]
            if stats['count'] > 0:
                centroid = stats['sum'] / stats['count']
                centroids_dict[obsm_name][group_idx] = centroid
            else:
                # This shouldn't happen, but handle gracefully
                if logger:
                    logger.warning(f"Group {group_key} has no cells for {obsm_name}")
                else:
                    print(f"WARNING: Group {group_key} has no cells for {obsm_name}")
        
        # Update progress bar
        if centroid_progress:
            centroid_progress.set_postfix({
                'obsm_slots': len(obsm_names)
            })
            centroid_progress.update(1)
    
    # Close progress bar
    if centroid_progress:
        centroid_progress.close()
    
    if logger:
        logger.info(f"Centroids computed for {n_groups} groups across {len(obsm_names)} .obsm slots")
        for obsm_name in obsm_names:
            shape = centroids_dict[obsm_name].shape
            logger.info(f"  {obsm_name}: {shape}")
    
    return group_keys, centroids_dict


def accumulate_group_statistics_chunked(adata, group_columns, chunk_size, logger=None):
    """
    Chunked version of single-pass algorithm for large files.
    
    Args:
        adata: AnnData object (can be backed)
        group_columns: List of grouping column names
        chunk_size: Number of cells to process per chunk
        verbose: Whether to show progress
    
    Returns:
        dict: {group_key: {obsm_name: {'sum': array, 'count': int}}}
    """
    if logger:
        logger.info(f"Starting chunked accumulation (chunk size: {chunk_size:,})...")
    
    # Initialize statistics dictionary
    group_stats = defaultdict(lambda: defaultdict(lambda: {'sum': None, 'count': 0}))
    
    n_cells = adata.n_obs
    n_chunks = (n_cells + chunk_size - 1) // chunk_size  # Ceiling division
    total_skipped_cells = 0
    
    # Create progress bar for chunks
    chunk_progress = tqdm(total=n_chunks, desc="Processing chunks", unit="chunks") if logger else None
    
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, n_cells)
        chunk_cells = end_idx - start_idx
        chunk_skipped = 0
        
        if logger:
            progress = (chunk_idx + 1) / n_chunks * 100
            logger.info(f"  Processing chunk {chunk_idx + 1}/{n_chunks} ({progress:.1f}%): cells {start_idx:,}-{end_idx-1:,}")
        
        # Process chunk of cells
        for cell_idx in range(start_idx, end_idx):
            # Get group key from categorical columns
            group_values = []
            has_missing = False
            
            for col in group_columns:
                value = adata.obs.iloc[cell_idx][col]
                if pd.isna(value):
                    has_missing = True
                    break
                group_values.append(str(value))
            
            # Skip cells with missing values in grouping columns
            if has_missing:
                chunk_skipped += 1
                continue
                
            group_key = tuple(group_values)
            
            # Process each .obsm slot
            for obsm_name, embedding_matrix in adata.obsm.items():
                cell_embedding = embedding_matrix[cell_idx]
                
                # Initialize sum array if this is the first cell for this group/obsm combination
                if group_stats[group_key][obsm_name]['sum'] is None:
                    group_stats[group_key][obsm_name]['sum'] = np.zeros_like(cell_embedding, dtype=np.float64)
                
                # Accumulate
                group_stats[group_key][obsm_name]['sum'] += cell_embedding.astype(np.float64)
                group_stats[group_key][obsm_name]['count'] += 1
        
        # Update totals
        total_skipped_cells += chunk_skipped
        
        # Update progress bar
        if chunk_progress:
            n_groups = len(group_stats)
            chunk_processed = chunk_cells - chunk_skipped
            chunk_progress.set_postfix({
                'groups': n_groups,
                'processed': f"{chunk_processed}/{chunk_cells}",
                'total_skipped': total_skipped_cells
            })
            chunk_progress.update(1)
        
        # Force garbage collection after each chunk to manage memory
        gc.collect()
    
    # Close progress bar
    if chunk_progress:
        chunk_progress.close()
    
    if logger:
        total_groups = len(group_stats)
        total_processed = n_cells - total_skipped_cells
        logger.info(f"Chunked accumulation complete: {total_groups} unique groups found")
        if total_skipped_cells > 0:
            logger.info(f"  Processed cells: {total_processed:,} (skipped {total_skipped_cells:,} with missing values)")
        else:
            logger.info(f"  Processed cells: {total_processed:,}")
    
    return dict(group_stats)


def create_output_anndata(group_keys, centroids_dict, group_columns, input_adata, logger=None):
    """
    Create output AnnData object with centroids.
    
    Args:
        group_keys: List of group key tuples
        centroids_dict: Dictionary of centroid arrays
        group_columns: List of grouping column names
        input_adata: Original AnnData object for metadata
        
    Returns:
        AnnData: New AnnData object with centroids
    """
    if logger:
        logger.info("Creating output AnnData object...")
    
    n_groups = len(group_keys)
    
    # Create .obs DataFrame for groups
    obs_data = {}
    for col_idx, col_name in enumerate(group_columns):
        obs_data[col_name] = [group_key[col_idx] for group_key in group_keys]
    
    obs_df = pd.DataFrame(obs_data)
    obs_df.index = [f"group_{i}" for i in range(n_groups)]
    
    # Create dummy .X matrix (required by AnnData)
    # Use the first .obsm slot dimensions, or create minimal matrix
    if centroids_dict:
        first_obsm = list(centroids_dict.values())[0]
        X = np.zeros((n_groups, 1), dtype=np.float32)  # Minimal .X
    else:
        X = np.zeros((n_groups, 1), dtype=np.float32)
    
    # Create AnnData object
    adata_out = sc.AnnData(X=X, obs=obs_df)
    
    # Add .obsm centroids
    for obsm_name, centroids in centroids_dict.items():
        adata_out.obsm[obsm_name] = centroids.astype(np.float32)  # Convert to float32 to save space
    
    # Copy relevant metadata from input
    if hasattr(input_adata, 'uns') and input_adata.uns:
        adata_out.uns = input_adata.uns.copy()
    
    # Add processing metadata
    adata_out.uns['centroid_computation'] = {
        'group_columns': group_columns,
        'n_input_cells': input_adata.n_obs,
        'n_output_groups': n_groups,
        'obsm_slots': list(centroids_dict.keys())
    }
    
    if logger:
        logger.info(f"Output AnnData created:")
        logger.info(f"  Groups: {n_groups}")
        logger.info(f"  .obsm slots: {list(centroids_dict.keys())}")
        logger.info(f"  Grouping columns: {group_columns}")
    
    return adata_out


def setup_logging(verbose=False):
    """Setup logging with timestamps."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    # Check file size
    file_size_gb = get_file_size_gb(args.input_file)
    logger.info(f"Input file size: {file_size_gb:.2f} GB")
    
    # Get available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(f"Available memory: {available_memory_gb:.2f} GB")
    
    # Load file with memory checking
    logger.info("Loading .h5ad file...")
    try:
        # For very large files, warn about memory usage
        if file_size_gb > args.memory_limit_gb:
            logger.warning(f"File size ({file_size_gb:.2f} GB) exceeds memory limit ({args.memory_limit_gb:.2f} GB)")
            logger.warning("Will attempt to load and use chunked processing if needed")
        
        adata = sc.read_h5ad(args.input_file)
        
        # Estimate memory usage for processing
        memory_estimate_gb, obsm_info = estimate_memory_usage(adata, verbose=args.verbose)
        
        # Validate input
        if not validate_input(adata, args.group_by, logger=logger):
            sys.exit(1)
        
        # Decide on processing strategy
        use_chunked = memory_estimate_gb > args.memory_limit_gb or file_size_gb > args.memory_limit_gb
        
        if use_chunked:
            logger.info(f"Using chunked processing (chunk size: {args.chunk_size:,} cells)")
            
            # Chunked single-pass algorithm
            group_stats = accumulate_group_statistics_chunked(adata, args.group_by, args.chunk_size, logger=logger)
            
            # Check if any groups were found
            if len(group_stats) == 0:
                logger.error("No valid groups found in the data")
                sys.exit(1)
            
            # Compute centroids
            group_keys, centroids_dict = compute_centroids(group_stats, logger=logger)
            
            # Create output AnnData
            adata_out = create_output_anndata(group_keys, centroids_dict, args.group_by, adata, logger=logger)
            
            # Save output
            logger.info(f"Saving output to {args.output_file}...")
            adata_out.write_h5ad(args.output_file)
            
            logger.info(f"Output file size: {get_file_size_gb(args.output_file):.3f} GB")
        else:
            logger.info("Using full-memory processing")
            # Load everything into memory for faster processing
            adata = adata.to_memory()
            
            # Single-pass algorithm
            group_stats = accumulate_group_statistics(adata, args.group_by, logger=logger)
            
            # Check if any groups were found
            if len(group_stats) == 0:
                logger.error("No valid groups found in the data")
                sys.exit(1)
            
            # Compute centroids
            group_keys, centroids_dict = compute_centroids(group_stats, logger=logger)
            
            # Create output AnnData
            adata_out = create_output_anndata(group_keys, centroids_dict, args.group_by, adata, logger=logger)
            
            # Save output
            logger.info(f"Saving output to {args.output_file}...")
            adata_out.write_h5ad(args.output_file)
            
            logger.info(f"Output file size: {get_file_size_gb(args.output_file):.3f} GB")
        
    except Exception as e:
        logger.error(f"Failed to load or process file: {str(e)}")
        sys.exit(1)
    
    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main()