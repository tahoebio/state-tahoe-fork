#!/usr/bin/env python3
"""
Memory-efficient random splitting of H5AD files into N parts.

This script randomly splits a single H5AD file into N output files using
pure h5py for maximum memory efficiency. It can handle arbitrarily large
files (60-100GB+) without loading them fully into memory.

Example usage:
    # Split into 2 equal parts
    python split_h5ad_random.py input.h5ad output_dir/ --n-splits 2
    
    # Split into 3 parts with custom weights  
    python split_h5ad_random.py input.h5ad output_dir/ --n-splits 3 --split-weights 0.5 0.3 0.2
    
    # Large file with custom chunk size
    python split_h5ad_random.py large.h5ad output_dir/ --n-splits 2 --chunk-size 500000
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import h5py
import numpy as np
from tqdm import tqdm
import psutil

# === Logging Setup ===
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024


def generate_random_assignments(n_cells: int, n_splits: int, weights: Optional[List[float]] = None, 
                              seed: int = 42) -> np.ndarray:
    """
    Generate random assignments for each cell to one of N splits.
    
    Args:
        n_cells: Total number of cells
        n_splits: Number of output splits
        weights: Optional weights for splits (must sum to 1.0)
        seed: Random seed for reproducibility
        
    Returns:
        Array of assignments (0 to n_splits-1) for each cell
    """
    log.info(f"Generating random assignments for {n_cells:,} cells into {n_splits} splits")
    
    # Set random seed
    np.random.seed(seed)
    
    if weights is None:
        # Equal splits
        weights = [1.0 / n_splits] * n_splits
        log.info(f"Using equal splits: {weights}")
    else:
        # Validate weights
        if len(weights) != n_splits:
            raise ValueError(f"Number of weights ({len(weights)}) must match n_splits ({n_splits})")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {sum(weights)}")
        log.info(f"Using custom weights: {weights}")
    
    # Generate assignments
    assignments = np.random.choice(n_splits, size=n_cells, p=weights)
    
    # Log split sizes
    unique, counts = np.unique(assignments, return_counts=True)
    for split_idx, count in zip(unique, counts):
        percentage = count / n_cells * 100
        log.info(f"Split {split_idx}: {count:,} cells ({percentage:.1f}%)")
    
    return assignments


def decode_categorical_column(obs_group: h5py.Group, column_name: str, 
                            start_idx: int, end_idx: int) -> np.ndarray:
    """
    Decode a categorical column from H5AD obs group.
    Handles both string and categorical (codes/categories) formats.
    """
    column_item = obs_group[column_name]
    
    if hasattr(column_item, 'dtype'):
        # Direct string/numeric column
        values = column_item[start_idx:end_idx]
        if column_item.dtype.kind in ['S', 'U']:
            return values.astype(str)
        else:
            return values
    else:
        # Categorical column with codes/categories structure
        codes = column_item['codes'][start_idx:end_idx]
        categories = column_item['categories'][:]
        decoded_categories = np.array([
            s.decode('utf-8') if isinstance(s, bytes) else str(s) 
            for s in categories
        ])
        return decoded_categories[codes]


def copy_data_to_splits(input_data: h5py.Dataset, output_datasets: List[h5py.Dataset],
                       assignments: np.ndarray, data_name: str = "data",
                       chunk_size: int = 1000000) -> None:
    """
    Copy data from input to multiple output splits based on assignments.
    
    Args:
        input_data: Source h5py dataset
        output_datasets: List of target h5py datasets (one per split)
        assignments: Array indicating which split each cell belongs to
        data_name: Name for progress tracking
        chunk_size: Size of chunks to process
    """
    n_cells = input_data.shape[0]
    n_splits = len(output_datasets)
    
    # Initialize output positions for each split
    output_positions = [0] * n_splits
    
    log.info(f"Copying {data_name} using chunked processing...")
    
    with tqdm(total=n_cells, desc=f"Copying {data_name}", unit="cells") as pbar:
        for chunk_start in range(0, n_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_cells)
            chunk_size_actual = chunk_end - chunk_start
            
            # Read chunk
            if len(input_data.shape) == 2:
                chunk_data = input_data[chunk_start:chunk_end, :]
            else:
                chunk_data = input_data[chunk_start:chunk_end]
            
            # Get assignments for this chunk
            chunk_assignments = assignments[chunk_start:chunk_end]
            
            # Split data and write to appropriate outputs
            for split_idx in range(n_splits):
                split_mask = chunk_assignments == split_idx
                if np.any(split_mask):
                    split_data = chunk_data[split_mask]
                    n_cells_in_split = len(split_data)
                    
                    # Write to output
                    output_pos = output_positions[split_idx]
                    next_pos = output_pos + n_cells_in_split
                    
                    if len(input_data.shape) == 2:
                        output_datasets[split_idx][output_pos:next_pos, :] = split_data
                    else:
                        output_datasets[split_idx][output_pos:next_pos] = split_data
                    
                    output_positions[split_idx] = next_pos
            
            pbar.update(chunk_size_actual)
    
    # Verify all cells were copied
    total_copied = sum(output_positions)
    log.info(f"{data_name} copying completed: {total_copied:,} cells total")
    if total_copied != n_cells:
        log.warning(f"Cell count mismatch in {data_name}: copied {total_copied} but expected {n_cells}")


def copy_sparse_matrix_to_splits(input_x_group: h5py.Group, output_x_groups: List[h5py.Group],
                                assignments: np.ndarray, n_features: int,
                                chunk_size: int = 1000000) -> None:
    """
    Copy sparse matrix from input to multiple output splits based on assignments.
    
    Args:
        input_x_group: Source sparse matrix group with data/indices/indptr
        output_x_groups: List of target sparse matrix groups
        assignments: Array indicating which split each cell belongs to
        n_features: Number of features in the matrix
        chunk_size: Size of chunks to process
    """
    n_cells = len(input_x_group['indptr']) - 1
    n_splits = len(output_x_groups)
    
    # Read input sparse matrix components
    input_data = input_x_group['data']
    input_indices = input_x_group['indices']
    input_indptr = input_x_group['indptr']
    
    log.info(f"Copying sparse X matrix with {len(input_data)} non-zero values...")
    
    # Initialize output lists for each split
    split_data = [[] for _ in range(n_splits)]
    split_indices = [[] for _ in range(n_splits)]
    split_indptr = [[0] for _ in range(n_splits)]  # Start with 0
    
    with tqdm(total=n_cells, desc="Copying sparse X", unit="cells") as pbar:
        for chunk_start in range(0, n_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_cells)
            
            for cell_idx in range(chunk_start, chunk_end):
                split_idx = assignments[cell_idx]
                
                # Get the sparse row for this cell
                row_start = input_indptr[cell_idx]
                row_end = input_indptr[cell_idx + 1]
                
                if row_end > row_start:  # Non-empty row
                    row_data = input_data[row_start:row_end]
                    row_indices = input_indices[row_start:row_end]
                    
                    split_data[split_idx].extend(row_data)
                    split_indices[split_idx].extend(row_indices)
                
                # Update indptr for this split
                split_indptr[split_idx].append(len(split_data[split_idx]))
            
            pbar.update(chunk_end - chunk_start)
    
    # Create datasets in output files
    for split_idx, output_group in enumerate(output_x_groups):
        # Convert lists to arrays
        split_data_array = np.array(split_data[split_idx], dtype=input_data.dtype)
        split_indices_array = np.array(split_indices[split_idx], dtype=input_indices.dtype)
        split_indptr_array = np.array(split_indptr[split_idx], dtype=input_indptr.dtype)
        
        # Create datasets
        output_group.create_dataset('data', data=split_data_array)
        output_group.create_dataset('indices', data=split_indices_array)
        output_group.create_dataset('indptr', data=split_indptr_array)
        
        # Add shape attribute
        split_n_cells = len(split_indptr_array) - 1
        output_group.attrs['shape'] = (split_n_cells, n_features)
        
        log.info(f"Split {split_idx}: {len(split_data_array)} non-zero values, "
                f"{split_n_cells} cells, {n_features} features")


def create_empty_sparse_x(output_x_groups: List[h5py.Group], 
                         split_sizes: List[int], 
                         n_features: int) -> None:
    """
    Create empty sparse CSR matrices for X data in output splits.
    
    Args:
        output_x_groups: List of X groups to populate with empty sparse structure
        split_sizes: Number of cells in each split
        n_features: Number of features (genes)
    """
    log.info(f"Creating empty sparse X matrices with {n_features} features...")
    
    for split_idx, (x_group, n_cells) in enumerate(zip(output_x_groups, split_sizes)):
        # Create empty CSR sparse matrix components
        # data: empty float32 array
        empty_data = np.array([], dtype=np.float32)
        x_group.create_dataset('data', data=empty_data)
        
        # indices: empty int32 array
        empty_indices = np.array([], dtype=np.int32) 
        x_group.create_dataset('indices', data=empty_indices)
        
        # indptr: array of zeros with length n_cells + 1
        empty_indptr = np.zeros(n_cells + 1, dtype=np.int32)
        x_group.create_dataset('indptr', data=empty_indptr)
        
        # Set shape attribute
        x_group.attrs['shape'] = (n_cells, n_features)
        
        log.info(f"Split {split_idx}: Created empty sparse matrix ({n_cells} cells, {n_features} features)")


def copy_1d_data_to_splits(input_data: h5py.Dataset, assignments: np.ndarray,
                          data_name: str = "data", chunk_size: int = 1000000) -> List[np.ndarray]:
    """
    Copy 1D data to splits, returning numpy arrays.
    
    Args:
        input_data: Source h5py dataset
        assignments: Array indicating which split each cell belongs to
        data_name: Name for progress tracking
        chunk_size: Size of chunks to process
        
    Returns:
        List of numpy arrays, one per split
    """
    n_cells = len(input_data)
    n_splits = int(np.max(assignments)) + 1
    
    # Initialize lists to collect data for each split
    split_data_lists = [[] for _ in range(n_splits)]
    
    with tqdm(total=n_cells, desc=f"Copying {data_name}", unit="cells", leave=False) as pbar:
        for chunk_start in range(0, n_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_cells)
            
            # Read chunk
            chunk_data = input_data[chunk_start:chunk_end]
            chunk_assignments = assignments[chunk_start:chunk_end]
            
            # Split data
            for split_idx in range(n_splits):
                split_mask = chunk_assignments == split_idx
                if np.any(split_mask):
                    split_data_lists[split_idx].extend(chunk_data[split_mask])
            
            pbar.update(chunk_end - chunk_start)
    
    # Convert to numpy arrays
    return [np.array(data_list, dtype=input_data.dtype) for data_list in split_data_lists]


def create_output_files(input_file: Path, output_dir: Path, n_splits: int, 
                       assignments: np.ndarray, prefix: str = "split", skip_x: bool = False) -> Tuple[List[h5py.File], int, bool]:
    """
    Create output H5AD files with proper structure and pre-allocated datasets.
    
    Args:
        input_file: Path to input H5AD file
        output_dir: Directory for output files
        n_splits: Number of splits
        assignments: Cell assignments to determine sizes
        prefix: Prefix for output filenames
        skip_x: If True, always create sparse X structure (for empty sparse matrices)
        
    Returns:
        Tuple of (List of open h5py.File objects, n_features, is_sparse)
    """
    log.info(f"Creating {n_splits} output files in {output_dir}")
    
    # Calculate split sizes
    unique, counts = np.unique(assignments, return_counts=True)
    split_sizes = [0] * n_splits
    for split_idx, count in zip(unique, counts):
        split_sizes[split_idx] = count
    
    output_files = []
    
    # Open input to get metadata
    with h5py.File(input_file, 'r') as input_f:
        # Get dimensions
        input_x = input_f['X']
        if hasattr(input_x, 'shape'):
            # Dense matrix
            n_features = input_x.shape[1] if len(input_x.shape) == 2 else 1
            is_sparse = False
        else:
            # Sparse matrix - get n_features from indices max + 1 or shape if available
            if 'shape' in input_x.attrs:
                n_features = input_x.attrs['shape'][1]
            elif 'indices' in input_x and len(input_x['indices']) > 0:
                n_features = int(input_x['indices'][:].max()) + 1
            else:
                log.error("Cannot determine number of features from sparse X matrix")
                sys.exit(1)
            is_sparse = True
        
        # Override sparse setting if skip_x is enabled
        if skip_x:
            is_sparse = True
            log.info("Skip X mode enabled - will create sparse X structure for empty matrices")
        
        # Create output files
        for split_idx in range(n_splits):
            output_path = output_dir / f"{prefix}_{split_idx:02d}.h5ad"
            log.info(f"Creating {output_path} with {split_sizes[split_idx]:,} cells")
            
            output_f = h5py.File(output_path, 'w')
            output_files.append(output_f)
            
            # Create X dataset/group
            if is_sparse:
                # For sparse matrices, create a group with sparse structure
                x_group = output_f.create_group('X')
                # We'll populate the sparse structure during data copying
            else:
                # Dense matrix
                if len(input_x.shape) == 2:
                    output_f.create_dataset(
                        'X',
                        shape=(split_sizes[split_idx], n_features),
                        dtype=input_x.dtype
                    )
                else:
                    output_f.create_dataset(
                        'X',
                        shape=(split_sizes[split_idx],),
                        dtype=input_x.dtype
                    )
            
            # Create obs group with attributes from input
            obs_out = output_f.create_group('obs')
            if 'obs' in input_f:
                # Copy obs group attributes (encoding metadata)
                for attr_key, attr_val in input_f['obs'].attrs.items():
                    obs_out.attrs[attr_key] = attr_val
            
            # Create obsm group if it exists in input
            if 'obsm' in input_f:
                output_f.create_group('obsm')
                obsm_in = input_f['obsm']
                obsm_out = output_f['obsm']
                
                for key in obsm_in.keys():
                    embedding_data = obsm_in[key]
                    if len(embedding_data.shape) == 2:
                        n_dims = embedding_data.shape[1]
                        obsm_out.create_dataset(
                            key,
                            shape=(split_sizes[split_idx], n_dims),
                            dtype=embedding_data.dtype
                        )
            
            # Create layers group if it exists in input
            if 'layers' in input_f:
                output_f.create_group('layers')
                layers_in = input_f['layers']
                layers_out = output_f['layers']
                
                for key in layers_in.keys():
                    layer_data = layers_in[key]
                    if len(layer_data.shape) == 2:
                        n_layer_features = layer_data.shape[1]
                        layers_out.create_dataset(
                            key,
                            shape=(split_sizes[split_idx], n_layer_features),
                            dtype=layer_data.dtype
                        )
    
    return output_files, n_features, is_sparse


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient random splitting of H5AD files into N parts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Split into 2 equal parts
    python split_h5ad_random.py input.h5ad output_dir/ --n-splits 2
    
    # Split into 3 parts with custom weights
    python split_h5ad_random.py input.h5ad output_dir/ --n-splits 3 --split-weights 0.5 0.3 0.2
    
    # Large file with custom chunk size
    python split_h5ad_random.py large.h5ad output_dir/ --n-splits 2 --chunk-size 500000

The script uses pure h5py for maximum memory efficiency and can handle
arbitrarily large H5AD files without loading them fully into memory.
        """
    )
    
    parser.add_argument('input_file', 
                       help='Path to input H5AD file')
    parser.add_argument('output_dir', 
                       help='Output directory for split files')
    parser.add_argument('--n-splits', 
                       type=int, 
                       required=True,
                       help='Number of output files to create')
    parser.add_argument('--split-weights',
                       nargs='+',
                       type=float,
                       help='Optional weights for splits (must sum to 1.0)')
    parser.add_argument('--output-prefix',
                       default='split',
                       help='Prefix for output filenames (default: split)')
    parser.add_argument('--chunk-size', 
                       type=int, 
                       default=1000000,
                       help='Chunk size for processing data (default: 1M)')
    parser.add_argument('--seed', 
                       type=int, 
                       default=42,
                       help='Random seed for reproducible splitting (default: 42)')
    parser.add_argument('--skip-x',
                       action='store_true',
                       help='Skip copying X data and create empty sparse matrices instead')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Validate and create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate n_splits
    if args.n_splits < 2:
        log.error("n_splits must be at least 2")
        sys.exit(1)
    
    try:
        start_time = time.time()
        
        # Phase 1: Get input metadata and generate assignments
        log.info("Phase 1: Reading input metadata and generating random assignments...")
        
        with h5py.File(input_path, 'r') as f:
            if 'X' not in f:
                log.error("Input file does not contain 'X' dataset")
                sys.exit(1)
            
            x_item = f['X']
            if hasattr(x_item, 'shape'):
                # Dense matrix
                n_cells = x_item.shape[0]
            else:
                # Sparse matrix - use indptr length - 1
                if 'indptr' in x_item:
                    n_cells = len(x_item['indptr']) - 1
                else:
                    log.error("Cannot determine number of cells from X structure")
                    sys.exit(1)
            
            log.info(f"Input file contains {n_cells:,} cells")
        
        # Generate random assignments
        assignments = generate_random_assignments(
            n_cells, args.n_splits, args.split_weights, args.seed
        )
        
        # Phase 2: Create output files
        log.info("Phase 2: Creating output files with proper structure...")
        output_files, n_features, is_sparse = create_output_files(
            input_path, output_dir, args.n_splits, assignments, args.output_prefix, args.skip_x
        )
        
        # Phase 3: Copy data
        log.info("Phase 3: Copying data to split files...")
        
        with h5py.File(input_path, 'r') as input_f:
            # Copy or create empty X data
            if args.skip_x:
                log.info("Skipping X data copying - creating empty sparse matrices...")
                # Calculate split sizes for empty sparse matrices
                unique, counts = np.unique(assignments, return_counts=True)
                split_sizes = [0] * args.n_splits
                for split_idx, count in zip(unique, counts):
                    split_sizes[split_idx] = count
                
                output_x_groups = [f['X'] for f in output_files]
                create_empty_sparse_x(output_x_groups, split_sizes, n_features)
            else:
                log.info("Copying X data...")
                input_x = input_f['X']
                
                if is_sparse:
                    # Sparse matrix
                    output_x_groups = [f['X'] for f in output_files]
                    copy_sparse_matrix_to_splits(
                        input_x, output_x_groups, assignments, n_features, args.chunk_size
                    )
                else:
                    # Dense matrix
                    output_x_datasets = [f['X'] for f in output_files]
                    copy_data_to_splits(
                        input_x, output_x_datasets, assignments, "X data", args.chunk_size
                    )
            
            # Copy obs data
            log.info("Copying obs data...")
            if 'obs' in input_f:
                obs_in = input_f['obs']
                
                for key in tqdm(obs_in.keys(), desc="Processing obs columns"):
                    item = obs_in[key]
                    
                    if key == '_index':
                        # Handle index specially
                        if hasattr(item, 'dtype'):
                            split_data = copy_1d_data_to_splits(
                                item, assignments, f"obs.{key}", args.chunk_size
                            )
                            for split_idx, data in enumerate(split_data):
                                output_files[split_idx]['obs'].create_dataset(key, data=data)
                        else:
                            log.warning(f"Skipping complex _index structure")
                            continue
                    else:
                        if hasattr(item, 'dtype'):
                            # Simple array
                            split_data = copy_1d_data_to_splits(
                                item, assignments, f"obs.{key}", args.chunk_size
                            )
                            for split_idx, data in enumerate(split_data):
                                output_files[split_idx]['obs'].create_dataset(key, data=data)
                        else:
                            # Categorical - preserve categories and codes structure
                            categories = item['categories'][:]
                            
                            # Get split codes
                            split_codes = copy_1d_data_to_splits(
                                item['codes'], assignments, f"obs.{key}.codes", args.chunk_size
                            )
                            
                            # Create categorical structure for each split
                            for split_idx, codes in enumerate(split_codes):
                                # Create group for categorical column
                                cat_group = output_files[split_idx]['obs'].create_group(key)
                                
                                # Copy group-level attributes (encoding metadata)
                                for attr_key, attr_val in item.attrs.items():
                                    cat_group.attrs[attr_key] = attr_val
                                
                                # Copy categories (same for all splits) with attributes
                                cat_dataset = cat_group.create_dataset('categories', data=categories)
                                for attr_key, attr_val in item['categories'].attrs.items():
                                    cat_dataset.attrs[attr_key] = attr_val
                                
                                # Copy codes for this split with attributes
                                codes_dataset = cat_group.create_dataset('codes', data=codes)
                                for attr_key, attr_val in item['codes'].attrs.items():
                                    codes_dataset.attrs[attr_key] = attr_val
            
            # Copy obsm data
            if 'obsm' in input_f:
                log.info("Copying obsm data...")
                obsm_in = input_f['obsm']
                
                for key in tqdm(obsm_in.keys(), desc="Processing obsm embeddings"):
                    embedding_data = obsm_in[key]
                    
                    if len(embedding_data.shape) == 2:
                        # 2D embeddings
                        output_embeddings = [f['obsm'][key] for f in output_files]
                        copy_data_to_splits(
                            embedding_data, output_embeddings, assignments, 
                            f"obsm.{key}", args.chunk_size
                        )
                    else:
                        # 1D or unexpected shape
                        log.warning(f"Unexpected obsm shape for {key}: {embedding_data.shape}")
                        split_data = copy_1d_data_to_splits(
                            embedding_data, assignments, f"obsm.{key}", args.chunk_size
                        )
                        for split_idx, data in enumerate(split_data):
                            output_files[split_idx]['obsm'].create_dataset(key, data=data)
            
            # Copy var data (unchanged - duplicated to all splits)
            log.info("Copying var data...")
            if 'var' in input_f:
                for output_f in output_files:
                    input_f.copy('var', output_f)
            
            # Copy layers data
            if 'layers' in input_f:
                log.info("Copying layers data...")
                layers_in = input_f['layers']
                
                for key in tqdm(layers_in.keys(), desc="Processing layers"):
                    layer_data = layers_in[key]
                    
                    if len(layer_data.shape) == 2:
                        # 2D layer data
                        output_layers = [f['layers'][key] for f in output_files]
                        copy_data_to_splits(
                            layer_data, output_layers, assignments,
                            f"layers.{key}", args.chunk_size
                        )
                    else:
                        # 1D layer data
                        split_data = copy_1d_data_to_splits(
                            layer_data, assignments, f"layers.{key}", args.chunk_size
                        )
                        for split_idx, data in enumerate(split_data):
                            output_files[split_idx]['layers'].create_dataset(key, data=data)
            
            # Copy remaining groups unchanged (uns, varm, varp, obsp)
            for group_name in ['uns', 'varm', 'varp', 'obsp']:
                if group_name in input_f:
                    log.info(f"Copying {group_name} data...")
                    for output_f in output_files:
                        input_f.copy(group_name, output_f)
        
        # Close output files
        log.info("Finalizing output files...")
        output_paths = []
        for i, output_f in enumerate(output_files):
            output_path = Path(output_f.filename)
            output_paths.append(output_path)
            output_f.close()
            log.info(f"Closed {output_path}")
        
        elapsed = time.time() - start_time
        
        # Final statistics
        log.info(f"✅ Random splitting completed in {elapsed:.1f} seconds")
        log.info(f"Input: {input_path}")
        log.info(f"Output directory: {output_dir}")
        log.info(f"Created {len(output_paths)} split files:")
        for path in output_paths:
            log.info(f"  {path}")
        
    except Exception as e:
        log.error(f"Error during splitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()