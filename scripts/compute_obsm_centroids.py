#!/usr/bin/env python3
"""
Compute centroids for .obsm slots based on categorical grouping columns.

Takes a .h5ad file with multiple .obsm slots and produces a new .h5ad file
with centroids computed for each group based on specified categorical columns.
Uses h5py for direct HDF5 access to avoid AnnData memory overhead.
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
        help="Categorical columns in .obs to group by (e.g., cell_type cytokine donor)"
    )
    
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help="Chunk size for processing large files (default: auto-detect from HDF5 chunks)"
    )
    
    parser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=200.0,
        help="Memory limit in GB (used for warnings only)"
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


def setup_logging(verbose=False):
    """Setup logging with timestamps."""
    level = logging.INFO if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)


def get_optimal_chunk_size(h5_file, dataset_path, target_chunk_size=1000000):
    """
    Determine optimal chunk size based on HDF5 compression chunks.
    
    Args:
        h5_file: Open h5py file handle
        dataset_path: Path to dataset in HDF5 file
        target_chunk_size: Target number of rows per processing chunk
    
    Returns:
        Optimal chunk size aligned with HDF5 chunks
    """
    if dataset_path not in h5_file:
        return target_chunk_size
    
    dataset = h5_file[dataset_path]
    if dataset.chunks:
        # Use HDF5 chunk shape for efficient decompression
        hdf5_chunk_rows = dataset.chunks[0]
        # Round to nearest multiple for efficient processing
        optimal_chunks = max(1, target_chunk_size // hdf5_chunk_rows)
        return hdf5_chunk_rows * optimal_chunks
    
    return target_chunk_size


def load_grouping_metadata(h5_file, group_columns, n_cells, logger=None):
    """
    Load only the grouping columns from .obs, handling categoricals efficiently.
    
    Args:
        h5_file: Open h5py file handle
        group_columns: List of column names to load
        n_cells: Number of cells
        logger: Logger instance
    
    Returns:
        Dictionary with metadata for grouping
    """
    metadata = {}
    obs_group = h5_file['/obs']
    
    if logger:
        logger.info(f"Loading {len(group_columns)} grouping columns...")
    
    for col in tqdm(group_columns, desc="Loading metadata", disable=not logger):
        if col not in obs_group:
            raise ValueError(f"Column '{col}' not found in .obs")
        
        # Check if it's a categorical column
        if f'{col}/codes' in obs_group:
            # Categorical column - more memory efficient
            codes = obs_group[f'{col}/codes'][:]
            categories = obs_group[f'{col}/categories'][:]
            
            # Decode categories if they're bytes
            if categories.dtype.kind == 'S':
                categories = [c.decode('utf-8') if isinstance(c, bytes) else c for c in categories]
            
            metadata[col] = {
                'codes': codes,
                'categories': np.array(categories),
                'type': 'categorical'
            }
            
            if logger:
                n_missing = np.sum(codes < 0)
                if n_missing > 0:
                    logger.warning(f"Column '{col}' has {n_missing:,} missing values")
        else:
            # Direct column
            data = obs_group[col][:]
            
            # Decode if bytes
            if data.dtype.kind == 'S':
                data = np.array([d.decode('utf-8') if isinstance(d, bytes) else d for d in data])
            
            metadata[col] = {
                'data': data,
                'type': 'direct'
            }
    
    return metadata


def get_group_key(cell_idx, group_columns, metadata):
    """
    Get the group key for a specific cell.
    
    Args:
        cell_idx: Cell index
        group_columns: List of grouping columns
        metadata: Metadata dictionary from load_grouping_metadata
    
    Returns:
        Tuple representing the group key, or None if missing values
    """
    group_key = []
    
    for col in group_columns:
        if metadata[col]['type'] == 'categorical':
            code = metadata[col]['codes'][cell_idx]
            if code < 0:  # Missing value
                return None
            value = metadata[col]['categories'][code]
        else:
            value = metadata[col]['data'][cell_idx]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                return None
        
        group_key.append(str(value))
    
    return tuple(group_key)


def accumulate_centroids_streaming(h5_file, metadata, group_columns, chunk_size, logger=None):
    """
    Stream through data with progress tracking and memory monitoring.
    
    Args:
        h5_file: Open h5py file handle
        metadata: Metadata dictionary from load_grouping_metadata
        group_columns: List of grouping columns
        chunk_size: Number of cells to process at once
        logger: Logger instance
    
    Returns:
        Dictionary with accumulated statistics for each group
    """
    n_cells = h5_file['/obs/_index'].shape[0]
    obsm_slots = list(h5_file['/obsm'].keys())
    
    if logger:
        logger.info(f"Processing {n_cells:,} cells with {len(obsm_slots)} embedding slots")
        logger.info(f"Using chunk size: {chunk_size:,}")
    
    # Initialize accumulator
    group_stats = defaultdict(lambda: defaultdict(lambda: {'sum': None, 'count': 0}))
    
    # Progress tracking
    n_chunks = (n_cells + chunk_size - 1) // chunk_size
    chunk_progress = tqdm(
        total=n_chunks, 
        desc="Processing chunks", 
        unit="chunks",
        disable=not logger
    )
    
    skipped_cells = 0
    last_memory_report = 0
    
    for chunk_start in range(0, n_cells, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_cells)
        chunk_cells = chunk_end - chunk_start
        
        # Build group keys for chunk
        chunk_groups = []
        for i in range(chunk_start, chunk_end):
            group_key = get_group_key(i, group_columns, metadata)
            if group_key is None:
                skipped_cells += 1
            chunk_groups.append(group_key)
        
        # Process each embedding slot
        for slot in obsm_slots:
            # Load chunk of embeddings (triggers decompression only for this chunk)
            embeddings = h5_file[f'/obsm/{slot}'][chunk_start:chunk_end]
            
            # Accumulate
            for i, group_key in enumerate(chunk_groups):
                if group_key is None:
                    continue
                
                embedding = embeddings[i]
                
                if group_stats[group_key][slot]['sum'] is None:
                    group_stats[group_key][slot]['sum'] = np.zeros_like(embedding, dtype=np.float64)
                
                group_stats[group_key][slot]['sum'] += embedding.astype(np.float64)
                group_stats[group_key][slot]['count'] += 1
        
        # Update progress
        chunk_progress.update(1)
        
        # Report memory periodically
        if chunk_start // chunk_size % 10 == 0 or chunk_start == 0:
            mem_gb = psutil.Process().memory_info().rss / 1024**3
            n_groups = len(group_stats)
            chunk_progress.set_postfix({
                'Memory': f'{mem_gb:.1f}GB',
                'Groups': n_groups,
                'Skipped': skipped_cells
            })
        
        # Force garbage collection periodically
        if chunk_start // chunk_size % 50 == 0:
            gc.collect()
    
    chunk_progress.close()
    
    if logger:
        logger.info(f"Found {len(group_stats)} unique groups")
        if skipped_cells > 0:
            logger.info(f"Skipped {skipped_cells:,} cells with missing values")
    
    return dict(group_stats)


def compute_centroids(group_stats, logger=None):
    """
    Compute centroids from accumulated group statistics.
    
    Args:
        group_stats: Dictionary from accumulate_centroids_streaming
        logger: Logger instance
        
    Returns:
        tuple: (group_keys, centroids_dict)
    """
    if logger:
        logger.info("Computing centroids from accumulated statistics...")
    
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
        if first_sum is None:
            continue
            
        embedding_dim = first_sum.shape[0] if first_sum.ndim == 1 else first_sum.shape
        
        # Create centroid array
        if isinstance(embedding_dim, int):
            centroids_dict[obsm_name] = np.zeros((n_groups, embedding_dim), dtype=np.float32)
        else:
            centroids_dict[obsm_name] = np.zeros((n_groups,) + embedding_dim, dtype=np.float32)
    
    # Compute centroids for each group
    for group_idx, group_key in enumerate(tqdm(group_keys, desc="Computing centroids", disable=not logger)):
        group_data = group_stats[group_key]
        
        for obsm_name in obsm_names:
            stats = group_data[obsm_name]
            if stats['count'] > 0 and stats['sum'] is not None:
                centroid = stats['sum'] / stats['count']
                centroids_dict[obsm_name][group_idx] = centroid.astype(np.float32)
    
    if logger:
        logger.info(f"Computed centroids for {n_groups} groups across {len(obsm_names)} .obsm slots")
    
    return group_keys, centroids_dict


def create_output_h5ad(group_keys, centroids_dict, group_columns, input_file, output_file, logger=None):
    """
    Create output H5AD file with centroids using h5py.
    
    Args:
        group_keys: List of group key tuples
        centroids_dict: Dictionary of centroid arrays
        group_columns: List of grouping column names
        input_file: Path to input file (for metadata)
        output_file: Path to output file
        logger: Logger instance
    """
    if logger:
        logger.info(f"Creating output file: {output_file}")
    
    n_groups = len(group_keys)
    
    with h5py.File(output_file, 'w') as f_out:
        # Create minimal X (required by AnnData spec) - empty sparse matrix
        X_group = f_out.create_group('X')
        X_group.create_dataset('data', data=np.array([], dtype=np.float32))
        X_group.create_dataset('indices', data=np.array([], dtype=np.int32))
        X_group.create_dataset('indptr', data=np.zeros(n_groups + 1, dtype=np.int32))
        
        # Create obs (group metadata)
        obs_group = f_out.create_group('obs')
        obs_index = [f'group_{i}' for i in range(n_groups)]
        obs_group.create_dataset('_index', data=np.array(obs_index, dtype='S'))
        
        # Add grouping columns
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
        
        # Create obsm (centroids)
        obsm_group = f_out.create_group('obsm')
        for slot_name, centroids in centroids_dict.items():
            # Store without compression for fast access
            obsm_group.create_dataset(slot_name, data=centroids, chunks=None)
        
        # Create minimal var (required by AnnData spec)
        var_group = f_out.create_group('var')
        var_group.create_dataset('_index', data=np.array(['dummy'], dtype='S'))
        
        # Create minimal uns
        uns_group = f_out.create_group('uns')
        
        # Add metadata about the computation
        uns_group.attrs['centroid_computation'] = True
        uns_group.attrs['n_input_cells'] = len(group_keys)  # Will be updated if we have access
        uns_group.attrs['n_output_groups'] = n_groups
        uns_group.attrs['group_columns'] = ','.join(group_columns)
        uns_group.attrs['obsm_slots'] = ','.join(list(centroids_dict.keys()))
        
        # Try to get input cell count
        try:
            with h5py.File(input_file, 'r') as f_in:
                uns_group.attrs['n_input_cells'] = f_in['/obs/_index'].shape[0]
        except:
            pass
    
    if logger:
        logger.info(f"Output file created successfully")
        logger.info(f"  Groups: {n_groups}")
        logger.info(f"  Embedding slots: {list(centroids_dict.keys())}")


def main():
    """Main function."""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.verbose)
    
    # Check if input file exists
    if not os.path.exists(args.input_file):
        logger.error(f"Input file does not exist: {args.input_file}")
        sys.exit(1)
    
    # Create output directory if needed
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
    
    # Check file size
    file_size_gb = get_file_size_gb(args.input_file)
    logger.info(f"Input file size: {file_size_gb:.2f} GB")
    
    # Get available memory
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    logger.info(f"Available memory: {available_memory_gb:.2f} GB")
    
    try:
        # Open file with h5py
        logger.info(f"Opening {args.input_file} with h5py...")
        
        with h5py.File(args.input_file, 'r') as h5_file:
            # Get basic info
            n_cells = h5_file['/obs/_index'].shape[0]
            obsm_slots = list(h5_file['/obsm'].keys())
            
            logger.info(f"File info:")
            logger.info(f"  Cells: {n_cells:,}")
            logger.info(f"  .obsm slots: {obsm_slots}")
            
            # Determine chunk size
            if args.chunk_size is None:
                if obsm_slots:
                    # Use first obsm slot to determine HDF5 chunk alignment
                    chunk_size = get_optimal_chunk_size(
                        h5_file, 
                        f'/obsm/{obsm_slots[0]}',
                        target_chunk_size=1000000
                    )
                else:
                    chunk_size = 1000000
            else:
                chunk_size = args.chunk_size
            
            logger.info(f"Using chunk size: {chunk_size:,}")
            
            # Check compression
            if obsm_slots and logger:
                first_obsm = h5_file[f'/obsm/{obsm_slots[0]}']
                if first_obsm.compression:
                    logger.info(f"Note: Data is compressed with {first_obsm.compression}")
            
            # Load metadata
            metadata = load_grouping_metadata(
                h5_file, 
                args.group_by, 
                n_cells,
                logger
            )
            
            # Process data
            group_stats = accumulate_centroids_streaming(
                h5_file,
                metadata,
                args.group_by,
                chunk_size,
                logger
            )
            
            # Check if any groups were found
            if len(group_stats) == 0:
                logger.error("No valid groups found in the data")
                sys.exit(1)
        
        # Compute centroids
        group_keys, centroids_dict = compute_centroids(group_stats, logger)
        
        # Create output file
        create_output_h5ad(
            group_keys,
            centroids_dict,
            args.group_by,
            args.input_file,
            args.output_file,
            logger
        )
        
        # Report output size
        output_size_gb = get_file_size_gb(args.output_file)
        logger.info(f"Output file size: {output_size_gb:.3f} GB")
        
        # Final memory report
        final_memory_gb = psutil.Process().memory_info().rss / 1024**3
        logger.info(f"Peak memory usage: {final_memory_gb:.1f} GB")
        
    except Exception as e:
        logger.error(f"Failed to process file: {str(e)}")
        import traceback
        if args.verbose:
            traceback.print_exc()
        sys.exit(1)
    
    logger.info("Processing completed successfully!")


if __name__ == "__main__":
    main()