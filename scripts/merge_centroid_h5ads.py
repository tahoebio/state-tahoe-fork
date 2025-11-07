#!/usr/bin/env python3
"""
Merge multiple centroid H5AD files by computing mean of means.

Takes a directory of centroid H5AD files and produces a merged H5AD file
where centroids for matching groups across files are averaged.

Usage:
    python merge_centroid_h5ads.py <input_dir> <output_file> [--verbose]
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import h5py
from collections import defaultdict
import gc
import psutil
import logging
import time
from tqdm import tqdm


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Merge centroid H5AD files by computing mean of means",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing centroid .h5ad files"
    )
    
    parser.add_argument(
        "output_file", 
        type=str,
        help="Output merged .h5ad file path"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output with timestamps"
    )
    
    return parser.parse_args()


def setup_logging(verbose=False):
    """Setup logging with timestamps."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def find_centroid_files(input_dir, logger=None):
    """Find all H5AD files in the input directory."""
    if not os.path.exists(input_dir):
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    h5ad_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.h5ad')])
    
    if not h5ad_files:
        raise ValueError(f"No .h5ad files found in {input_dir}")
    
    full_paths = [os.path.join(input_dir, f) for f in h5ad_files]
    
    if logger:
        logger.info(f"Found {len(h5ad_files)} H5AD files to merge")
    
    return full_paths


def load_group_keys_and_centroids(filepath, logger=None):
    """
    Load group keys and centroid data from a single H5AD file.
    
    Returns:
        tuple: (group_keys_dict, centroids_dict)
            group_keys_dict: {group_tuple: group_index}
            centroids_dict: {obsm_slot: numpy_array}
    """
    with h5py.File(filepath, 'r') as f:
        # Load group metadata
        obs = f['/obs']
        
        # Get categories for each grouping column
        cell_types = obs['cell_type/categories'][:]
        cytokines = obs['cytokine/categories'][:]
        donors = obs['donor/categories'][:]
        
        # Decode if bytes
        if cell_types.dtype.kind == 'S':
            cell_types = [c.decode('utf-8') for c in cell_types]
        if cytokines.dtype.kind == 'S':
            cytokines = [c.decode('utf-8') for c in cytokines]
        if donors.dtype.kind == 'S':
            donors = [d.decode('utf-8') for d in donors]
        
        # Get codes
        cell_type_codes = obs['cell_type/codes'][:]
        cytokine_codes = obs['cytokine/codes'][:]
        donor_codes = obs['donor/codes'][:]
        
        # Build group keys mapping
        group_keys = {}
        for i in range(len(cell_type_codes)):
            ct = cell_types[cell_type_codes[i]]
            cy = cytokines[cytokine_codes[i]]
            do = donors[donor_codes[i]]
            group_key = (ct, cy, do)
            group_keys[group_key] = i
        
        # Load centroid data from .obsm
        obsm_group = f['/obsm']
        obsm_slots = list(obsm_group.keys())
        
        centroids = {}
        for slot in obsm_slots:
            centroids[slot] = obsm_group[slot][:]  # Load all centroid data
    
    if logger:
        logger.debug(f"Loaded {len(group_keys)} groups with {len(centroids)} obsm slots from {os.path.basename(filepath)}")
    
    return group_keys, centroids


def merge_centroids_across_files(file_paths, logger=None):
    """
    Load all files and compute mean of means for matching groups.
    
    Args:
        file_paths: List of paths to centroid H5AD files
        logger: Logger instance
    
    Returns:
        tuple: (merged_group_keys, merged_centroids, contribution_counts)
    """
    if logger:
        logger.info("Loading and merging centroid data...")
    
    # Collect all data from files
    all_group_data = defaultdict(lambda: defaultdict(list))  # {group_key: {obsm_slot: [centroids_list]}}
    obsm_slots = None
    
    # Load data from all files
    for file_path in tqdm(file_paths, desc="Loading files", disable=not logger):
        group_keys, centroids = load_group_keys_and_centroids(file_path, logger)
        
        # Check obsm slots consistency
        if obsm_slots is None:
            obsm_slots = list(centroids.keys())
        elif set(obsm_slots) != set(centroids.keys()):
            raise ValueError(f"Inconsistent .obsm slots across files")
        
        # Accumulate data for each group
        for group_key, group_idx in group_keys.items():
            for slot in obsm_slots:
                all_group_data[group_key][slot].append(centroids[slot][group_idx])
    
    if logger:
        logger.info(f"Found {len(all_group_data)} unique groups across all files")
        logger.info(f"Computing mean of means for {len(obsm_slots)} .obsm slots")
    
    # Compute merged centroids
    merged_group_keys = list(all_group_data.keys())
    n_groups = len(merged_group_keys)
    
    # Initialize output arrays
    merged_centroids = {}
    contribution_counts = {}
    
    for slot in obsm_slots:
        # Get dimensions from first available centroid
        first_centroid = next(iter(all_group_data.values()))[slot][0]
        centroid_dim = first_centroid.shape[0] if first_centroid.ndim == 1 else first_centroid.shape
        
        if isinstance(centroid_dim, int):
            merged_centroids[slot] = np.zeros((n_groups, centroid_dim), dtype=np.float32)
        else:
            merged_centroids[slot] = np.zeros((n_groups,) + centroid_dim, dtype=np.float32)
        
        contribution_counts[slot] = np.zeros(n_groups, dtype=np.int32)
    
    # Compute means
    for group_idx, group_key in enumerate(tqdm(merged_group_keys, desc="Computing means", disable=not logger)):
        group_data = all_group_data[group_key]
        
        for slot in obsm_slots:
            centroids_for_group = group_data[slot]
            
            if centroids_for_group:  # Should always have at least one
                # Compute mean across files
                stacked = np.stack(centroids_for_group, axis=0)
                mean_centroid = np.mean(stacked, axis=0)
                merged_centroids[slot][group_idx] = mean_centroid.astype(np.float32)
                contribution_counts[slot][group_idx] = len(centroids_for_group)
    
    if logger:
        logger.info("Merging completed successfully")
        # Report contribution statistics
        for slot in obsm_slots:
            counts = contribution_counts[slot]
            logger.info(f"  {slot}: avg {np.mean(counts):.1f} files per group (min: {np.min(counts)}, max: {np.max(counts)})")
    
    return merged_group_keys, merged_centroids, contribution_counts


def create_merged_h5ad(group_keys, centroids, contribution_counts, output_file, input_dir, logger=None):
    """
    Create output H5AD file with merged centroids.
    
    Args:
        group_keys: List of group key tuples
        centroids: Dictionary of merged centroid arrays
        contribution_counts: Dictionary of contribution count arrays
        output_file: Path to output file
        input_dir: Input directory (for metadata)
        logger: Logger instance
    """
    if logger:
        logger.info(f"Creating merged H5AD file: {output_file}")
    
    n_groups = len(group_keys)
    obsm_slots = list(centroids.keys())
    
    # Create output directory if needed
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    with h5py.File(output_file, 'w') as f_out:
        # Create minimal X (required by AnnData spec) - empty sparse matrix
        X_group = f_out.create_group('X')
        X_group.create_dataset('data', data=np.array([], dtype=np.float32))
        X_group.create_dataset('indices', data=np.array([], dtype=np.int32))
        X_group.create_dataset('indptr', data=np.zeros(n_groups + 1, dtype=np.int32))
        
        # Create obs (group metadata)
        obs_group = f_out.create_group('obs')
        obs_index = [f'merged_group_{i}' for i in range(n_groups)]
        obs_group.create_dataset('_index', data=np.array(obs_index, dtype='S'))
        
        # Add grouping columns
        group_columns = ['cell_type', 'cytokine', 'donor']
        for col_idx, col_name in enumerate(group_columns):
            values = [group_key[col_idx] for group_key in group_keys]
            
            # Store as categorical for efficiency
            unique_values = sorted(list(set(values)))
            value_to_code = {v: i for i, v in enumerate(unique_values)}
            codes = [value_to_code[v] for v in values]
            
            col_group = obs_group.create_group(col_name)
            col_group.create_dataset('codes', data=np.array(codes, dtype=np.int32))
            
            # Store categories as strings
            cat_data = np.array(unique_values, dtype='S')
            col_group.create_dataset('categories', data=cat_data)
        
        # Add contribution counts for each obsm slot
        for slot in obsm_slots:
            counts = contribution_counts[slot]
            obs_group.create_dataset(f'n_files_{slot}', data=counts)
        
        # Create obsm (merged centroids)
        obsm_group = f_out.create_group('obsm')
        for slot_name, centroid_data in centroids.items():
            # Store without compression for fast access
            obsm_group.create_dataset(slot_name, data=centroid_data, chunks=None)
        
        # Create minimal var (required by AnnData spec)
        var_group = f_out.create_group('var')
        var_group.create_dataset('_index', data=np.array(['dummy'], dtype='S'))
        
        # Create minimal uns with merge metadata
        uns_group = f_out.create_group('uns')
        uns_group.attrs['merged_centroids'] = True
        uns_group.attrs['n_input_files'] = len([f for f in os.listdir(input_dir) if f.endswith('.h5ad')])
        uns_group.attrs['n_output_groups'] = n_groups
        uns_group.attrs['group_columns'] = ','.join(group_columns)
        uns_group.attrs['obsm_slots'] = ','.join(obsm_slots)
        uns_group.attrs['input_directory'] = input_dir
        
        # Add detailed contribution statistics
        for slot in obsm_slots:
            counts = contribution_counts[slot]
            uns_group.attrs[f'{slot}_contribution_mean'] = float(np.mean(counts))
            uns_group.attrs[f'{slot}_contribution_min'] = int(np.min(counts))
            uns_group.attrs[f'{slot}_contribution_max'] = int(np.max(counts))
    
    if logger:
        logger.info(f"Merged H5AD file created successfully")
        logger.info(f"  Groups: {n_groups}")
        logger.info(f"  Embedding slots: {obsm_slots}")
        
        # Report file size
        file_size_mb = os.path.getsize(output_file) / (1024**2)
        logger.info(f"  Output file size: {file_size_mb:.1f} MB")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    try:
        # Find input files
        file_paths = find_centroid_files(args.input_dir, logger)
        
        # Merge centroids
        group_keys, centroids, contribution_counts = merge_centroids_across_files(file_paths, logger)
        
        # Create output file
        create_merged_h5ad(
            group_keys, 
            centroids, 
            contribution_counts,
            args.output_file, 
            args.input_dir,
            logger
        )
        
        # Final memory report
        if logger:
            final_memory_gb = psutil.Process().memory_info().rss / 1024**3
            logger.info(f"Peak memory usage: {final_memory_gb:.1f} GB")
        
    except Exception as e:
        logger.error(f"Failed to merge files: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
    
    logger.info("Merging completed successfully!")


if __name__ == "__main__":
    main()