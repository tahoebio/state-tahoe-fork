#!/usr/bin/env python3
"""
Memory-efficient categorical splitting of H5AD files.

This script takes a .h5ad file as input and the name of a categorical column in .obs.
It creates a subfolder named 'by_{category_name}' and creates one separate .h5ad file
for each category value. The filenames are sanitized to be compatible with filesystems.

Example usage:
    # Split by cell line
    python split_h5ad_by_category.py input.h5ad cell_line_id
    
    # Split with custom output directory
    python split_h5ad_by_category.py input.h5ad drug_dose --output-dir results/
    
    # Large file with custom chunk size
    python split_h5ad_by_category.py large.h5ad perturbation --chunk-size 500000

The script uses pure h5py for maximum memory efficiency and can handle
arbitrarily large H5AD files without loading them fully into memory.
"""

import argparse
import logging
import sys
import time
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict

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


def sanitize_filename(name: str) -> str:
    """
    Sanitize category names for valid filenames.
    
    Converts spaces, slashes, and other problematic characters to underscores
    while preserving basic alphanumeric characters, hyphens, underscores, and dots.
    
    Args:
        name: Original category name
        
    Returns:
        Sanitized filename-safe string
    """
    # Replace problematic characters with underscores
    sanitized = re.sub(r'[^\w\-_.]', '_', str(name))
    
    # Replace multiple consecutive underscores with single underscore
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    # Ensure we have a non-empty result
    if not sanitized:
        sanitized = "unnamed_category"
    
    return sanitized


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


class CategoryPerformanceTracker:
    """Performance tracking for category discovery and processing."""
    
    def __init__(self, total_cells: int, category_column: str):
        self.total_cells = total_cells
        self.category_column = category_column
        self.timings = {
            'io_loading': 0.0,
            'categorical_decoding': 0.0,
            'cell_assignment': 0.0,
            'memory_tracking': 0.0
        }
        self.processed_cells = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_cells = 0
        
        log.info(f"🚀 Starting category discovery for column '{category_column}' ({total_cells:,} cells)")
    
    def update_timing(self, component: str, duration: float, cells_processed: int = 0):
        """Update timing for a specific component."""
        self.timings[component] += duration
        self.processed_cells += cells_processed
        
        # Log progress every 100K cells
        if self.processed_cells > 0 and self.processed_cells % 100000 == 0:
            self._log_progress()
    
    def _log_progress(self):
        """Log detailed progress information."""
        current_time = time.time()
        elapsed_total = current_time - self.start_time
        elapsed_recent = current_time - self.last_update_time
        
        # Calculate rates
        overall_rate = self.processed_cells / elapsed_total if elapsed_total > 0 else 0
        recent_cells = self.processed_cells - self.last_update_cells
        recent_rate = recent_cells / elapsed_recent if elapsed_recent > 0 else 0
        
        # Calculate ETA
        remaining_cells = self.total_cells - self.processed_cells
        eta_hours = (remaining_cells / recent_rate / 3600) if recent_rate > 0 else 0
        
        # Calculate component percentages
        total_time = sum(self.timings.values())
        percentages = {k: (v/total_time)*100 for k, v in self.timings.items()} if total_time > 0 else {}
        
        # Memory usage
        mem_gb = get_memory_usage()
        
        log.info(f"📊 Category Discovery Progress: {self.processed_cells:,}/{self.total_cells:,} cells ({self.processed_cells/self.total_cells*100:.1f}%)")
        if percentages:
            log.info(f"⚡ Performance: I/O={percentages['io_loading']:.1f}%, Decode={percentages['categorical_decoding']:.1f}%, Assignment={percentages['cell_assignment']:.1f}%")
        log.info(f"🏃 Rate: {overall_rate:.0f} cells/sec (recent: {recent_rate:.0f}), ETA: {eta_hours:.1f} hours, Memory: {mem_gb:.1f}GB")
        
        self.last_update_time = current_time
        self.last_update_cells = self.processed_cells
    
    def finalize(self) -> Dict[str, float]:
        """Log final statistics and return timing breakdown."""
        elapsed_total = time.time() - self.start_time
        overall_rate = self.processed_cells / elapsed_total if elapsed_total > 0 else 0
        
        total_time = sum(self.timings.values())
        percentages = {k: (v/total_time)*100 for k, v in self.timings.items()} if total_time > 0 else {}
        
        log.info(f"✅ Category discovery completed: {self.processed_cells:,} cells in {elapsed_total:.1f}s ({overall_rate:.0f} cells/sec)")
        if percentages:
            log.info(f"📈 Final breakdown: I/O={percentages['io_loading']:.1f}%, Decode={percentages['categorical_decoding']:.1f}%, Assignment={percentages['cell_assignment']:.1f}%")
        
        return percentages


def discover_categories_and_assignments(h5ad_file: Path, category_column: str, 
                                      chunk_size: int = 1000000) -> Tuple[Dict[str, List[int]], int]:
    """
    Stream through obs data to discover categories and assign cell indices.
    
    Returns:
        category_assignments: Dict mapping category names to lists of cell indices
        total_cells: Total number of cells processed
    """
    log.info(f"Discovering categories in column '{category_column}' from {h5ad_file}")
    log.info(f"Using chunk size: {chunk_size:,} cells")
    
    category_assignments: Dict[str, List[int]] = defaultdict(list)
    total_processed = 0
    
    with h5py.File(h5ad_file, 'r') as f:
        obs_group = f['obs']
        
        # Validate column exists
        if category_column not in obs_group:
            available_columns = list(obs_group.keys())
            raise ValueError(f"Column '{category_column}' not found in obs. Available columns: {available_columns}")
        
        # Get total number of cells
        column_item = obs_group[category_column]
        if hasattr(column_item, 'dtype'):
            n_cells = len(column_item)
        else:
            n_cells = len(column_item['codes'])
        
        log.info(f"Total cells in dataset: {n_cells:,}")
        
        # Initialize performance tracker
        tracker = CategoryPerformanceTracker(n_cells, category_column)
        
        # Stream through data in chunks
        with tqdm(total=n_cells, desc="Discovering categories", unit="cells") as pbar:
            for start_idx in range(0, n_cells, chunk_size):
                chunk_start_time = time.time()
                end_idx = min(start_idx + chunk_size, n_cells)
                chunk_size_actual = end_idx - start_idx
                
                # Load categorical data for this chunk
                io_start = time.time()
                decode_start = time.time()
                chunk_categories = decode_categorical_column(obs_group, category_column, start_idx, end_idx)
                decode_time = time.time() - decode_start
                io_time = time.time() - io_start
                
                # Assign cells to categories
                assignment_start = time.time()
                for local_idx, category_value in enumerate(chunk_categories):
                    global_idx = start_idx + local_idx
                    category_str = str(category_value)
                    category_assignments[category_str].append(global_idx)
                assignment_time = time.time() - assignment_start
                
                # Update tracking
                tracker.update_timing('io_loading', io_time, chunk_size_actual)
                tracker.update_timing('categorical_decoding', decode_time)
                tracker.update_timing('cell_assignment', assignment_time)
                
                total_processed += chunk_size_actual
                pbar.update(chunk_size_actual)
                pbar.set_postfix({
                    'categories': len(category_assignments),
                    'mem_GB': f"{get_memory_usage():.1f}"
                })
        
        # Finalize tracking
        tracker.finalize()
    
    # Convert to regular dict and sort indices for each category
    final_assignments = {}
    for category, indices in category_assignments.items():
        final_assignments[category] = sorted(indices)
    
    # Log category statistics
    log.info(f"Discovered {len(final_assignments)} categories:")
    total_assigned = 0
    for category, indices in sorted(final_assignments.items()):
        sanitized_name = sanitize_filename(category)
        log.info(f"  '{category}' -> '{sanitized_name}': {len(indices):,} cells")
        total_assigned += len(indices)
    
    log.info(f"Total cells assigned: {total_assigned:,} (should match {n_cells:,})")
    if total_assigned != n_cells:
        log.warning(f"Cell count mismatch! Assigned {total_assigned} but expected {n_cells}")
    
    return final_assignments, n_cells


def copy_data_by_mask(input_data: h5py.Dataset, output_data: h5py.Dataset, 
                     selected_mask: np.ndarray, total_input_cells: int,
                     n_selected_cells: int, data_name: str = "data",
                     chunk_size: int = 1000000) -> None:
    """
    Copy data using boolean mask with chunked processing.
    
    Args:
        input_data: Source h5py dataset
        output_data: Target h5py dataset (must be pre-allocated)
        selected_mask: Boolean array indicating which cells to select
        total_input_cells: Total number of cells in input dataset
        n_selected_cells: Expected number of selected cells
        data_name: Name for progress tracking
        chunk_size: Size of chunks to process
    """
    output_idx = 0
    
    with tqdm(total=total_input_cells, desc=f"Copying {data_name}", unit="cells", leave=False) as pbar:
        for chunk_start in range(0, total_input_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_input_cells)
            
            # Sequential read (efficient!)
            if len(input_data.shape) == 2:
                chunk_data = input_data[chunk_start:chunk_end, :]
            else:
                chunk_data = input_data[chunk_start:chunk_end]
            
            # Filter to selected cells
            chunk_mask = selected_mask[chunk_start:chunk_end]
            if np.any(chunk_mask):
                selected_chunk_data = chunk_data[chunk_mask]
                
                # Write to output
                next_output_idx = output_idx + len(selected_chunk_data)
                if len(output_data.shape) == 2:
                    output_data[output_idx:next_output_idx, :] = selected_chunk_data
                else:
                    output_data[output_idx:next_output_idx] = selected_chunk_data
                
                output_idx = next_output_idx
            
            pbar.update(chunk_end - chunk_start)
            pbar.set_postfix({
                'copied': f"{output_idx:,}/{n_selected_cells:,}"
            })
    
    log.info(f"{data_name} copying completed: {output_idx:,} cells")
    
    # Verify we copied the expected number of cells
    if output_idx != n_selected_cells:
        log.warning(f"Mismatch in {data_name}: copied {output_idx} cells but expected {n_selected_cells}")


def copy_1d_data_by_mask(input_data: h5py.Dataset, selected_mask: np.ndarray, 
                        total_input_cells: int, data_name: str = "data",
                        chunk_size: int = 1000000) -> np.ndarray:
    """
    Copy 1D data using boolean mask, returning the result as numpy array.
    
    Args:
        input_data: Source h5py dataset
        selected_mask: Boolean array indicating which cells to select
        total_input_cells: Total number of cells in input dataset
        data_name: Name for progress tracking
        chunk_size: Size of chunks to process
        
    Returns:
        numpy array with selected data
    """
    selected_data_list = []
    
    with tqdm(total=total_input_cells, desc=f"Copying {data_name}", unit="cells", leave=False) as pbar:
        for chunk_start in range(0, total_input_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_input_cells)
            
            # Sequential read
            chunk_data = input_data[chunk_start:chunk_end]
            
            # Filter to selected cells
            chunk_mask = selected_mask[chunk_start:chunk_end]
            if np.any(chunk_mask):
                selected_data_list.extend(chunk_data[chunk_mask])
            
            pbar.update(chunk_end - chunk_start)
    
    return np.array(selected_data_list, dtype=input_data.dtype)


def get_matrix_dimensions(h5_file: h5py.File) -> Tuple[int, int]:
    """
    Get matrix dimensions from H5AD file, handling both dense and sparse formats.
    
    Args:
        h5_file: Open h5py file handle
        
    Returns:
        Tuple of (n_cells, n_features)
    """
    x_item = h5_file['X']
    
    if hasattr(x_item, 'shape'):
        # Dense matrix
        if len(x_item.shape) == 2:
            return x_item.shape[0], x_item.shape[1]
        else:
            return x_item.shape[0], 1
    elif hasattr(x_item, 'keys'):
        # Sparse matrix (CSR format)
        if 'indptr' in x_item:
            n_cells = len(x_item['indptr']) - 1  # indptr has n_cells + 1 elements
            
            # For n_features, we need to find max index + 1, or check var
            if 'var' in h5_file:
                # Try to get n_features from var group
                var_group = h5_file['var']
                if '_index' in var_group:
                    n_features = len(var_group['_index'])
                else:
                    # Fall back to finding max index
                    indices = x_item['indices']
                    n_features = int(np.max(indices[:])) + 1
            else:
                # Fall back to finding max index
                indices = x_item['indices']  
                n_features = int(np.max(indices[:])) + 1
                
            return n_cells, n_features
        else:
            raise ValueError("Unsupported sparse matrix format - no indptr found")
    else:
        raise ValueError("Unsupported X format - neither dataset nor sparse group")


def create_category_output_files(input_file: Path, output_dir: Path, 
                                category_assignments: Dict[str, List[int]],
                                category_column: str) -> Dict[str, h5py.File]:
    """
    Create output H5AD files for each category with proper structure and pre-allocated datasets.
    
    Args:
        input_file: Path to input H5AD file
        output_dir: Directory for output files
        category_assignments: Dict mapping categories to cell indices
        category_column: Name of the category column
        
    Returns:
        Dict mapping categories to open h5py.File objects
    """
    log.info(f"Creating output files in {output_dir}")
    
    output_files = {}
    
    # Open input to get metadata
    with h5py.File(input_file, 'r') as input_f:
        # Get dimensions (handles both dense and sparse)
        n_cells, n_features = get_matrix_dimensions(input_f)
        
        log.info(f"Input matrix dimensions: {n_cells:,} cells x {n_features:,} features")
        
        # Check if input X is sparse or dense
        input_x = input_f['X']
        is_sparse = hasattr(input_x, 'keys') and 'indptr' in input_x
        
        # Create output files for each category
        for category, indices in category_assignments.items():
            sanitized_category = sanitize_filename(category)
            output_path = output_dir / f"{sanitized_category}.h5ad"
            n_category_cells = len(indices)
            
            log.info(f"Creating {output_path} with {n_category_cells:,} cells for category '{category}'")
            
            output_f = h5py.File(output_path, 'w')
            output_files[category] = output_f
            
            # Create X structure (sparse or dense)
            if is_sparse:
                # Create sparse matrix group structure
                x_group = output_f.create_group('X')
                # We'll determine the exact sizes during the copying phase
                # For now, just create the group structure
            else:
                # Create dense matrix dataset
                if len(input_x.shape) == 2:
                    output_f.create_dataset(
                        'X',
                        shape=(n_category_cells, n_features),
                        dtype=input_x.dtype
                    )
                else:
                    output_f.create_dataset(
                        'X',
                        shape=(n_category_cells,),
                        dtype=input_x.dtype
                    )
            
            # Create obs group
            output_f.create_group('obs')
            
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
                            shape=(n_cells, n_dims),
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
                            shape=(n_cells, n_layer_features),
                            dtype=layer_data.dtype
                        )
    
    return output_files


def analyze_sparse_matrix_for_categories(input_x_group: h5py.Group, 
                                       category_assignments: Dict[str, List[int]],
                                       total_cells: int, chunk_size: int = 100000) -> Dict[str, Tuple[int, int]]:
    """
    Pre-analyze sparse matrix to determine sizes needed for each category.
    
    Args:
        input_x_group: H5 group containing sparse matrix (data, indices, indptr)
        category_assignments: Dict mapping categories to cell indices
        total_cells: Total number of cells in dataset
        chunk_size: Chunk size for processing
        
    Returns:
        Dict mapping category names to (n_cells, n_nonzeros) tuples
    """
    log.info("Analyzing sparse matrix structure for categories...")
    
    # Create cell-to-category mapping for efficient lookup
    cell_to_category = {}
    for category, cell_indices in category_assignments.items():
        for cell_idx in cell_indices:
            cell_to_category[cell_idx] = category
    
    # Initialize counters for each category
    category_stats = {}
    for category in category_assignments:
        category_stats[category] = {'n_cells': len(category_assignments[category]), 'n_nonzeros': 0}
    
    # Stream through indptr to count non-zeros per category
    indptr_dataset = input_x_group['indptr']
    
    with tqdm(total=total_cells, desc="Analyzing sparse structure", unit="cells") as pbar:
        for chunk_start in range(0, total_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_cells)
            
            # Read indptr chunk (need +1 for the ranges)
            chunk_indptr = indptr_dataset[chunk_start:chunk_end + 1]
            
            # Count non-zeros for each cell in this chunk
            for local_idx in range(chunk_end - chunk_start):
                global_cell_idx = chunk_start + local_idx
                
                # Get non-zero count for this cell
                start_ptr = chunk_indptr[local_idx]
                end_ptr = chunk_indptr[local_idx + 1]
                n_nonzeros_this_cell = end_ptr - start_ptr
                
                # Add to appropriate category counter
                if global_cell_idx in cell_to_category:
                    category = cell_to_category[global_cell_idx]
                    category_stats[category]['n_nonzeros'] += n_nonzeros_this_cell
            
            pbar.update(chunk_end - chunk_start)
    
    # Log statistics
    log.info("Sparse matrix analysis results:")
    total_nonzeros = 0
    for category in sorted(category_stats.keys()):
        stats = category_stats[category]
        avg_nonzeros = stats['n_nonzeros'] / stats['n_cells'] if stats['n_cells'] > 0 else 0
        log.info(f"  '{category}': {stats['n_cells']:,} cells, {stats['n_nonzeros']:,} non-zeros, avg {avg_nonzeros:.1f}/cell")
        total_nonzeros += stats['n_nonzeros']
    
    log.info(f"Total non-zeros to copy: {total_nonzeros:,}")
    
    # Return simplified format
    return {category: (stats['n_cells'], stats['n_nonzeros']) for category, stats in category_stats.items()}


def copy_sparse_matrix_by_category(input_x_group: h5py.Group, output_files: Dict[str, h5py.File],
                                  category_assignments: Dict[str, List[int]], 
                                  category_sizes: Dict[str, Tuple[int, int]],
                                  total_cells: int, chunk_size: int = 100000) -> None:
    """
    Copy sparse matrix data to category-specific files.
    
    Args:
        input_x_group: Input sparse matrix group
        output_files: Dict mapping categories to output h5py files
        category_assignments: Dict mapping categories to cell indices
        category_sizes: Dict mapping categories to (n_cells, n_nonzeros)
        total_cells: Total number of cells
        chunk_size: Chunk size for processing
    """
    log.info("Copying sparse matrix data to category files...")
    
    # Pre-allocate sparse matrix datasets for each category
    log.info("Pre-allocating sparse matrix datasets...")
    category_datasets = {}
    category_positions = {}
    
    for category, (n_cells, n_nonzeros) in category_sizes.items():
        if n_cells == 0 or n_nonzeros == 0:
            log.warning(f"Skipping empty category '{category}' (0 cells or 0 non-zeros)")
            continue
            
        output_f = output_files[category]
        x_group = output_f['X']  # Group was already created
        
        # Create sparse datasets
        data_ds = x_group.create_dataset('data', shape=(n_nonzeros,), dtype=input_x_group['data'].dtype)
        indices_ds = x_group.create_dataset('indices', shape=(n_nonzeros,), dtype=input_x_group['indices'].dtype)  
        indptr_ds = x_group.create_dataset('indptr', shape=(n_cells + 1,), dtype=input_x_group['indptr'].dtype)
        
        # Initialize indptr[0] = 0
        indptr_ds[0] = 0
        
        category_datasets[category] = {
            'data': data_ds,
            'indices': indices_ds, 
            'indptr': indptr_ds
        }
        
        # Track current positions for writing
        category_positions[category] = {
            'data_pos': 0,
            'indptr_pos': 1  # Start at 1 since indptr[0] = 0
        }
        
        log.info(f"Allocated sparse arrays for '{category}': {n_nonzeros:,} data entries, {n_cells + 1} indptr entries")
    
    # Create cell-to-category mapping for efficient lookup
    cell_to_category = {}
    cell_local_indices = {}  # Maps (category, global_idx) -> local_idx_in_category
    
    for category, cell_indices in category_assignments.items():
        for local_idx, global_idx in enumerate(cell_indices):
            cell_to_category[global_idx] = category
            cell_local_indices[(category, global_idx)] = local_idx
    
    # Stream through sparse data and copy to categories
    input_data = input_x_group['data']
    input_indices = input_x_group['indices']
    input_indptr = input_x_group['indptr']
    
    with tqdm(total=total_cells, desc="Copying sparse data", unit="cells") as pbar:
        for chunk_start in range(0, total_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, total_cells)
            
            # Read indptr for this chunk (need +1 for ranges)
            chunk_indptr = input_indptr[chunk_start:chunk_end + 1]
            
            # Process each cell in chunk
            for local_idx in range(chunk_end - chunk_start):
                global_cell_idx = chunk_start + local_idx
                
                # Skip if this cell doesn't belong to any category
                if global_cell_idx not in cell_to_category:
                    continue
                
                category = cell_to_category[global_cell_idx]
                
                # Skip if we don't have datasets for this category (empty category)
                if category not in category_datasets:
                    continue
                
                # Get sparse data range for this cell
                start_ptr = chunk_indptr[local_idx]
                end_ptr = chunk_indptr[local_idx + 1]
                n_entries = end_ptr - start_ptr
                
                if n_entries > 0:
                    # Read sparse data for this cell
                    cell_data = input_data[start_ptr:end_ptr]
                    cell_indices = input_indices[start_ptr:end_ptr]
                    
                    # Write to category's sparse arrays
                    datasets = category_datasets[category]
                    positions = category_positions[category]
                    
                    # Copy data and indices
                    data_start = positions['data_pos']
                    data_end = data_start + n_entries
                    datasets['data'][data_start:data_end] = cell_data
                    datasets['indices'][data_start:data_end] = cell_indices
                    
                    # Update indptr
                    datasets['indptr'][positions['indptr_pos']] = data_end
                    
                    # Update positions
                    positions['data_pos'] = data_end
                    positions['indptr_pos'] += 1
                else:
                    # Empty row - just update indptr (same value as previous)
                    datasets = category_datasets[category]
                    positions = category_positions[category]
                    prev_val = datasets['indptr'][positions['indptr_pos'] - 1]
                    datasets['indptr'][positions['indptr_pos']] = prev_val
                    positions['indptr_pos'] += 1
            
            pbar.update(chunk_end - chunk_start)
    
    # Verify all indptr arrays are complete
    log.info("Verifying sparse matrix integrity...")
    for category, datasets in category_datasets.items():
        n_cells, n_nonzeros = category_sizes[category]
        final_data_pos = category_positions[category]['data_pos']
        final_indptr_pos = category_positions[category]['indptr_pos']
        
        if final_data_pos != n_nonzeros:
            log.error(f"Data mismatch for '{category}': copied {final_data_pos} entries but expected {n_nonzeros}")
        if final_indptr_pos != n_cells + 1:
            log.error(f"Indptr mismatch for '{category}': filled {final_indptr_pos} entries but expected {n_cells + 1}")
        else:
            log.info(f"✅ Sparse matrix for '{category}' completed successfully")


def write_category_splits(input_file: Path, output_files: Dict[str, h5py.File],
                         category_assignments: Dict[str, List[int]], total_cells: int,
                         chunk_size: int = 1000000) -> None:
    """
    Write data to category-specific H5AD files using efficient boolean masking.
    
    Args:
        input_file: Path to input H5AD file
        output_files: Dict mapping categories to open h5py.File objects
        category_assignments: Dict mapping categories to cell indices
        total_cells: Total number of cells in input
        chunk_size: Chunk size for processing
    """
    log.info("Writing data to category-specific files...")
    
    with h5py.File(input_file, 'r') as input_f:
        
        # Check if X is sparse or dense
        input_x = input_f['X']
        is_sparse = hasattr(input_x, 'keys') and 'indptr' in input_x
        
        if is_sparse:
            log.info("✅ Sparse matrix detected - using specialized sparse matrix splitting")
            
            # Phase 3a: Analyze sparse matrix structure
            log.info("Phase 3a: Analyzing sparse matrix structure for categories...")
            category_sizes = analyze_sparse_matrix_for_categories(
                input_x, category_assignments, total_cells, chunk_size
            )
            
            # Phase 3b: Copy sparse matrix data
            log.info("Phase 3b: Copying sparse matrix data to category files...")
            copy_sparse_matrix_by_category(
                input_x, output_files, category_assignments, category_sizes, total_cells, chunk_size
            )
            
        else:
            log.info("✅ Dense matrix detected - using dense matrix splitting")
            
            # Dense matrix processing - process each category separately
            for category, indices in category_assignments.items():
                log.info(f"Processing dense matrix for category '{category}' ({len(indices):,} cells)")
                output_f = output_files[category]
                
                # Create boolean mask for this category
                selected_mask = np.zeros(total_cells, dtype=bool)
                selected_mask[indices] = True
                n_selected_cells = len(indices)
                
                # Copy X data for this category
                log.info(f"  Copying dense X data for '{category}'...")
                copy_data_by_mask(
                    input_data=input_f['X'],
                    output_data=output_f['X'],
                    selected_mask=selected_mask,
                    total_input_cells=total_cells,
                    n_selected_cells=n_selected_cells,
                    data_name=f"X data ({category})",
                    chunk_size=chunk_size
                )
        
        # Copy metadata (obs, obsm, var, etc.) - common for both sparse and dense
        log.info("Copying metadata and other data groups...")
        for category, indices in category_assignments.items():
            log.info(f"Processing metadata for category '{category}' ({len(indices):,} cells)")
            output_f = output_files[category]
            
            # Create boolean mask for this category
            selected_mask = np.zeros(total_cells, dtype=bool)
            selected_mask[indices] = True
            n_selected_cells = len(indices)
            
            # Copy obs data
            log.info(f"  Copying obs data for '{category}'...")
            if 'obs' in input_f:
                obs_in = input_f['obs']
                obs_out = output_f['obs']
                
                for key in obs_in.keys():
                    item = obs_in[key]
                    
                    if key == '_index':
                        # Handle index specially
                        if hasattr(item, 'dtype'):
                            selected_data = copy_1d_data_by_mask(
                                item, selected_mask, total_cells, f"obs.{key}", chunk_size
                            )
                            obs_out.create_dataset(key, data=selected_data)
                        else:
                            log.warning(f"    Skipping complex _index structure")
                            continue
                    else:
                        if hasattr(item, 'dtype'):
                            # Simple array
                            selected_data = copy_1d_data_by_mask(
                                item, selected_mask, total_cells, f"obs.{key}", chunk_size
                            )
                            obs_out.create_dataset(key, data=selected_data)
                        else:
                            # Categorical - convert to strings for scanpy compatibility
                            categories = item['categories'][:]
                            decoded_categories = np.array([
                                s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                                for s in categories
                            ])
                            
                            # Get selected codes
                            selected_codes = copy_1d_data_by_mask(
                                item['codes'], selected_mask, total_cells, f"obs.{key}.codes", chunk_size
                            )
                            
                            # Convert codes to strings
                            selected_strings = decoded_categories[selected_codes]
                            selected_bytes = [s.encode('utf-8') for s in selected_strings]
                            obs_out.create_dataset(
                                key, 
                                data=selected_bytes,
                                dtype=h5py.string_dtype(encoding='utf-8')
                            )
            
            # Copy obsm data
            if 'obsm' in input_f:
                log.info(f"  Copying obsm data for '{category}'...")
                obsm_in = input_f['obsm']
                obsm_out = output_f['obsm']
                
                for key in obsm_in.keys():
                    embedding_data = obsm_in[key]
                    
                    if len(embedding_data.shape) == 2:
                        # 2D embeddings
                        copy_data_by_mask(
                            input_data=embedding_data,
                            output_data=obsm_out[key],
                            selected_mask=selected_mask,
                            total_input_cells=total_cells,
                            n_selected_cells=n_selected_cells,
                            data_name=f"obsm.{key} ({category})",
                            chunk_size=chunk_size
                        )
                    else:
                        # 1D or unexpected shape
                        log.warning(f"    Unexpected obsm shape for {key}: {embedding_data.shape}")
                        selected_data = copy_1d_data_by_mask(
                            embedding_data, selected_mask, total_cells, f"obsm.{key}", chunk_size
                        )
                        obsm_out.create_dataset(key, data=selected_data)
            
            # Copy var data (unchanged - duplicated to all categories)
            log.info(f"  Copying var data for '{category}'...")
            if 'var' in input_f:
                input_f.copy('var', output_f)
            
            # Copy layers data
            if 'layers' in input_f:
                log.info(f"  Copying layers data for '{category}'...")
                layers_in = input_f['layers']
                layers_out = output_f['layers']
                
                for key in layers_in.keys():
                    layer_data = layers_in[key]
                    
                    if len(layer_data.shape) == 2:
                        # 2D layer data
                        copy_data_by_mask(
                            input_data=layer_data,
                            output_data=layers_out[key],
                            selected_mask=selected_mask,
                            total_input_cells=total_cells,
                            n_selected_cells=n_selected_cells,
                            data_name=f"layers.{key} ({category})",
                            chunk_size=chunk_size
                        )
                    else:
                        # 1D layer data
                        selected_data = copy_1d_data_by_mask(
                            layer_data, selected_mask, total_cells, f"layers.{key}", chunk_size
                        )
                        layers_out.create_dataset(key, data=selected_data)
            
            # Copy remaining groups unchanged (uns, varm, varp, obsp)
            for group_name in ['uns', 'varm', 'varp', 'obsp']:
                if group_name in input_f:
                    log.info(f"  Copying {group_name} data for '{category}'...")
                    input_f.copy(group_name, output_f)
            
            log.info(f"✅ Completed category '{category}'")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient categorical splitting of H5AD files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Split by cell line
    python split_h5ad_by_category.py input.h5ad cell_line_id
    
    # Split with custom output directory
    python split_h5ad_by_category.py input.h5ad drug_dose --output-dir results/
    
    # Large file with custom chunk size
    python split_h5ad_by_category.py large.h5ad perturbation --chunk-size 500000

The script uses pure h5py for maximum memory efficiency and can handle
arbitrarily large H5AD files without loading them fully into memory.

Output files are created in a subdirectory named 'by_{category_column}'
with sanitized category names as filenames.
        """
    )
    
    parser.add_argument('input_file', 
                       help='Path to input H5AD file')
    parser.add_argument('category_column', 
                       help='Name of categorical column in obs to split by')
    parser.add_argument('--output-dir',
                       help='Output directory (default: same directory as input file)')
    parser.add_argument('--chunk-size', 
                       type=int, 
                       default=1000000,
                       help='Chunk size for processing data (default: 1M)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output directory
    if args.output_dir:
        base_output_dir = Path(args.output_dir)
    else:
        base_output_dir = input_path.parent
    
    output_dir = base_output_dir / f"by_{args.category_column}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Validate chunk size
    if args.chunk_size < 1000:
        log.error("Chunk size must be at least 1000")
        sys.exit(1)
    
    try:
        start_time = time.time()
        
        # Phase 1: Discover categories and assign cells
        log.info("Phase 1: Discovering categories and assigning cells...")
        category_assignments, total_cells = discover_categories_and_assignments(
            input_path, args.category_column, args.chunk_size
        )
        
        if not category_assignments:
            log.error("No categories found - check column name and data")
            sys.exit(1)
        
        # Phase 2: Create output files
        log.info("Phase 2: Creating output files with proper structure...")
        output_files = create_category_output_files(
            input_path, output_dir, category_assignments, args.category_column
        )
        
        # Phase 3: Write data to category files
        log.info("Phase 3: Writing data to category-specific files...")
        write_category_splits(
            input_path, output_files, category_assignments, total_cells, args.chunk_size
        )
        
        # Close output files
        log.info("Finalizing output files...")
        output_paths = []
        for category, output_f in output_files.items():
            output_path = Path(output_f.filename)
            output_paths.append((category, output_path))
            output_f.close()
        
        elapsed = time.time() - start_time
        
        # Final statistics
        log.info(f"✅ Categorical splitting completed in {elapsed:.1f} seconds")
        log.info(f"Input: {input_path}")
        log.info(f"Output directory: {output_dir}")
        log.info(f"Split column: {args.category_column}")
        log.info(f"Created {len(output_paths)} category files:")
        
        total_output_cells = 0
        for category, path in sorted(output_paths):
            sanitized_name = sanitize_filename(category)
            cell_count = len(category_assignments[category])
            total_output_cells += cell_count
            log.info(f"  '{category}' -> {path.name}: {cell_count:,} cells")
        
        log.info(f"Total output cells: {total_output_cells:,} (input: {total_cells:,})")
        
    except Exception as e:
        log.error(f"Error during splitting: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()