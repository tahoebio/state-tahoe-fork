#!/usr/bin/env python3
"""
Memory-efficient subsampling of H5AD files by groups using reservoir sampling.

This script uses pure h5py for maximum performance, avoiding the need to load
large AnnData objects into memory. It streams through the obs metadata to 
identify groups, applies reservoir sampling per group, then extracts the 
selected cells while preserving all H5AD structure.

Example usage:
    # Subsample by cell line and drug dose, max 1000 cells per combination
    python subsample_h5ad_by_groups.py input.h5ad output.h5ad \
        --group-columns cell_line_id drug_dose --max-cells-per-group 1000

    # Subsample by single column
    python subsample_h5ad_by_groups.py input.h5ad output.h5ad \
        --group-columns cell_line_id --max-cells-per-group 500
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil
import random

# === Logging Setup ===
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


class ReservoirSampler:
    """
    Reservoir sampling implementation for maintaining max N samples per group.
    """
    def __init__(self, max_size: int = 1000, seed: int = 42):
        self.max_size = max_size
        self.samples = []
        self.count = 0
        self.rng = random.Random(seed)
    
    def add_sample(self, item: Any) -> None:
        """Add a sample using reservoir sampling algorithm."""
        self.count += 1
        
        if len(self.samples) < self.max_size:
            # Reservoir not full, just add the item
            self.samples.append(item)
        else:
            # Reservoir full, decide whether to replace existing item
            # Random index in range [0, count-1]
            j = self.rng.randint(0, self.count - 1)
            if j < self.max_size:
                # Replace item at position j
                self.samples[j] = item
    
    def get_samples(self) -> List[Any]:
        """Get all samples in the reservoir."""
        return self.samples.copy()
    
    def size(self) -> int:
        """Get current number of samples."""
        return len(self.samples)
    
    def is_complete(self) -> bool:
        """Check if reservoir has reached maximum size."""
        return len(self.samples) >= self.max_size


def get_memory_usage() -> float:
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024


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


def stream_obs_for_group_selection(h5ad_file: Path, group_columns: List[str], 
                                 max_cells_per_group: int, chunk_size: int = 1000000) -> Tuple[List[int], Dict[Tuple, int]]:
    """
    Stream through obs data and apply reservoir sampling per group.
    
    Returns:
        selected_indices: List of cell indices to keep
        group_counts: Dict mapping group tuples to final cell counts
    """
    log.info(f"Streaming obs data from {h5ad_file} with chunk size {chunk_size:,}")
    log.info(f"Group columns: {group_columns}")
    log.info(f"Max cells per group: {max_cells_per_group}")
    
    # Initialize reservoir samplers per group
    group_reservoirs: Dict[Tuple, ReservoirSampler] = defaultdict(
        lambda: ReservoirSampler(max_cells_per_group)
    )
    
    total_processed = 0
    
    with h5py.File(h5ad_file, 'r') as f:
        obs_group = f['obs']
        
        # Get total rows
        first_column = group_columns[0]
        if first_column not in obs_group:
            raise ValueError(f"Column '{first_column}' not found in obs. Available columns: {list(obs_group.keys())}")
        
        first_item = obs_group[first_column]
        if hasattr(first_item, 'dtype'):
            n_obs = len(first_item)
        else:
            n_obs = len(first_item['codes'])
        
        log.info(f"Total observations: {n_obs:,}")
        
        # Initialize timing accumulators
        timing_stats = {
            'io_loading': 0.0,
            'categorical_decoding': 0.0,
            'python_loop': 0.0,
            'reservoir_sampling': 0.0,
            'tuple_creation': 0.0
        }
        
        # Stream through data in chunks
        for start_idx in tqdm(range(0, n_obs, chunk_size), desc="Processing chunks"):
            chunk_start_time = time.time()
            end_idx = min(start_idx + chunk_size, n_obs)
            chunk_size_actual = end_idx - start_idx
            
            # Load all group columns for this chunk
            io_start = time.time()
            chunk_data = {}
            for col in group_columns:
                if col not in obs_group:
                    raise ValueError(f"Column '{col}' not found in obs")
                
                decode_start = time.time()
                chunk_data[col] = decode_categorical_column(obs_group, col, start_idx, end_idx)
                timing_stats['categorical_decoding'] += time.time() - decode_start
            
            timing_stats['io_loading'] += time.time() - io_start
            
            # Process each row in chunk
            loop_start = time.time()
            for local_idx in range(chunk_size_actual):
                global_idx = start_idx + local_idx
                
                # Create group tuple
                tuple_start = time.time()
                group_values = tuple(chunk_data[col][local_idx] for col in group_columns)
                timing_stats['tuple_creation'] += time.time() - tuple_start
                
                # Add to reservoir sampler for this group
                reservoir_start = time.time()
                group_reservoirs[group_values].add_sample(global_idx)
                timing_stats['reservoir_sampling'] += time.time() - reservoir_start
                
                total_processed += 1
            
            timing_stats['python_loop'] += time.time() - loop_start
            
            # Memory and timing check every 1M processed
            if total_processed % 1000000 == 0:
                mem_gb = get_memory_usage()
                chunk_elapsed = time.time() - chunk_start_time
                
                # Calculate percentages
                total_time = sum(timing_stats.values())
                if total_time > 0:
                    percentages = {k: (v/total_time)*100 for k, v in timing_stats.items()}
                    
                    log.info(f"Processed {total_processed:,} cells, memory: {mem_gb:.1f}GB")
                    log.info(f"Performance breakdown:")
                    log.info(f"  I/O Loading: {percentages['io_loading']:.1f}% ({timing_stats['io_loading']:.1f}s)")
                    log.info(f"  Categorical Decoding: {percentages['categorical_decoding']:.1f}% ({timing_stats['categorical_decoding']:.1f}s)")
                    log.info(f"  Python Loop: {percentages['python_loop']:.1f}% ({timing_stats['python_loop']:.1f}s)")
                    log.info(f"  Tuple Creation: {percentages['tuple_creation']:.1f}% ({timing_stats['tuple_creation']:.1f}s)")
                    log.info(f"  Reservoir Sampling: {percentages['reservoir_sampling']:.1f}% ({timing_stats['reservoir_sampling']:.1f}s)")
                    log.info(f"  Recent chunk: {chunk_elapsed:.1f}s ({chunk_size_actual/chunk_elapsed:.0f} cells/sec)")
                else:
                    log.info(f"Processed {total_processed:,} cells, memory: {mem_gb:.1f}GB")
    
    # Final timing summary
    total_time = sum(timing_stats.values())
    if total_time > 0:
        percentages = {k: (v/total_time)*100 for k, v in timing_stats.items()}
        log.info(f"Final performance breakdown:")
        log.info(f"  I/O Loading: {percentages['io_loading']:.1f}% ({timing_stats['io_loading']:.1f}s)")
        log.info(f"  Categorical Decoding: {percentages['categorical_decoding']:.1f}% ({timing_stats['categorical_decoding']:.1f}s)")
        log.info(f"  Python Loop: {percentages['python_loop']:.1f}% ({timing_stats['python_loop']:.1f}s)")
        log.info(f"  Tuple Creation: {percentages['tuple_creation']:.1f}% ({timing_stats['tuple_creation']:.1f}s)")
        log.info(f"  Reservoir Sampling: {percentages['reservoir_sampling']:.1f}% ({timing_stats['reservoir_sampling']:.1f}s)")
        log.info(f"Total time: {total_time:.1f}s, Rate: {total_processed/total_time:.0f} cells/sec")
    
    # Collect all selected indices
    selected_indices = []
    group_counts = {}
    
    for group_values, reservoir in group_reservoirs.items():
        indices = reservoir.get_samples()
        selected_indices.extend(indices)
        group_counts[group_values] = len(indices)
    
    # Sort indices for efficient H5 access
    selected_indices.sort()
    
    log.info(f"Selected {len(selected_indices):,} cells from {len(group_reservoirs)} groups")
    
    # Print group statistics
    log.info("Group statistics:")
    for group_values, count in sorted(group_counts.items()):
        log.info(f"  {group_values}: {count} cells")
    
    return selected_indices, group_counts


def copy_data_sequentially(input_data: h5py.Dataset, output_data: h5py.Dataset, 
                          selected_mask: np.ndarray, total_input_cells: int,
                          n_selected_cells: int, data_name: str = "data",
                          sequential_chunk_size: int = 1000000) -> None:
    """
    Helper function to copy data using sequential pass with boolean mask filtering.
    
    Args:
        input_data: Source h5py dataset to read from
        output_data: Target h5py dataset to write to (must be pre-allocated)
        selected_mask: Boolean array indicating which cells to select
        total_input_cells: Total number of cells in input dataset
        n_selected_cells: Expected number of selected cells
        data_name: Name for progress tracking
        sequential_chunk_size: Size of sequential chunks to process
    """
    output_idx = 0
    total_bytes_processed = 0
    
    # Calculate bytes per cell for progress tracking
    if len(input_data.shape) == 2:
        bytes_per_cell = input_data.shape[1] * input_data.dtype.itemsize
    else:
        bytes_per_cell = input_data.dtype.itemsize
    
    with tqdm(total=total_input_cells, desc=f"Copying {data_name}", unit="cells") as pbar:
        for chunk_start in range(0, total_input_cells, sequential_chunk_size):
            chunk_end = min(chunk_start + sequential_chunk_size, total_input_cells)
            
            # Sequential read (fast!)
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
                total_bytes_processed += len(selected_chunk_data) * bytes_per_cell
            
            # Update progress
            chunk_size_actual = chunk_end - chunk_start
            pbar.update(chunk_size_actual)
            pbar.set_postfix({
                'copied': f"{output_idx:,}/{n_selected_cells:,}",
                'GB_out': f"{total_bytes_processed / 1024**3:.1f}"
            })
    
    log.info(f"{data_name} copying completed: {output_idx:,} cells, {total_bytes_processed / 1024**3:.1f} GB")
    
    # Verify we copied the expected number of cells
    if output_idx != n_selected_cells:
        log.warning(f"Mismatch in {data_name}: copied {output_idx} cells but expected {n_selected_cells}")


def copy_1d_data_sequentially(input_data: h5py.Dataset, selected_mask: np.ndarray, 
                             total_input_cells: int, data_name: str = "data",
                             sequential_chunk_size: int = 1000000) -> np.ndarray:
    """
    Helper function to copy 1D data using sequential pass, returning the result as numpy array.
    
    Args:
        input_data: Source h5py dataset to read from
        selected_mask: Boolean array indicating which cells to select
        total_input_cells: Total number of cells in input dataset
        data_name: Name for progress tracking
        sequential_chunk_size: Size of sequential chunks to process
        
    Returns:
        numpy array with selected data
    """
    selected_data_list = []
    
    with tqdm(total=total_input_cells, desc=f"Copying {data_name}", unit="cells", leave=False) as pbar:
        for chunk_start in range(0, total_input_cells, sequential_chunk_size):
            chunk_end = min(chunk_start + sequential_chunk_size, total_input_cells)
            
            # Sequential read
            chunk_data = input_data[chunk_start:chunk_end]
            
            # Filter to selected cells
            chunk_mask = selected_mask[chunk_start:chunk_end]
            if np.any(chunk_mask):
                selected_data_list.extend(chunk_data[chunk_mask])
            
            pbar.update(chunk_end - chunk_start)
    
    return np.array(selected_data_list)


def write_subsampled_h5ad(input_file: Path, output_file: Path, selected_indices: List[int]) -> None:
    """
    Write subsampled H5AD file using pure h5py.
    
    This preserves the full H5AD structure while extracting only selected cells.
    """
    log.info(f"Writing subsampled H5AD from {len(selected_indices):,} selected cells")
    
    # Convert indices to numpy array for fancy indexing
    indices_array = np.array(selected_indices, dtype=np.int64)
    
    with h5py.File(input_file, 'r') as input_f, h5py.File(output_file, 'w') as output_f:
        
        # Create boolean mask for sequential copying (reused across all data types)
        log.info("Creating boolean mask for efficient sequential copying...")
        # Get total cells from X data dimensions
        input_x = input_f['X']
        total_input_cells = input_x.shape[0]
        selected_mask = np.zeros(total_input_cells, dtype=bool)
        selected_mask[indices_array] = True
        n_selected_cells = len(indices_array)
        
        # Copy X data (main expression matrix) with chunked processing
        log.info("Copying X data with sequential processing...")
        
        if len(input_x.shape) == 2:
            n_genes = input_x.shape[1]
            
            # Pre-allocate output dataset
            output_x = output_f.create_dataset(
                'X', 
                shape=(n_selected_cells, n_genes),
                dtype=input_x.dtype
            )
            
            # Use helper function for sequential copying
            log.info(f"Processing {n_selected_cells:,} selected cells from {total_input_cells:,} total cells")
            copy_data_sequentially(
                input_data=input_x,
                output_data=output_x,
                selected_mask=selected_mask,
                total_input_cells=total_input_cells,
                n_selected_cells=n_selected_cells,
                data_name="X data"
            )
            
            # Periodic flush for incremental writes
            output_f.flush()
            
        else:
            # Handle non-2D matrices (sparse or other formats)
            log.warning("Non-2D matrix detected - using sequential approach")
            n_features = input_x.shape[1] if len(input_x.shape) > 1 else 1
            
            output_x = output_f.create_dataset(
                'X',
                shape=(n_selected_cells, n_features) if len(input_x.shape) > 1 else (n_selected_cells,),
                dtype=input_x.dtype
            )
            
            # Use helper function for sequential copying
            copy_data_sequentially(
                input_data=input_x,
                output_data=output_x,
                selected_mask=selected_mask,
                total_input_cells=total_input_cells,
                n_selected_cells=n_selected_cells,
                data_name="X data (non-2D)"
            )
            
            output_f.flush()
        
        # Copy var data (unchanged - gene/feature info)
        log.info("Copying var data...")
        if 'var' in input_f:
            input_f.copy('var', output_f)
        
        # Copy obs data (subsetted to selected cells) using sequential approach
        log.info("Copying obs data with sequential processing...")
        if 'obs' in input_f:
            obs_group_out = output_f.create_group('obs')
            obs_group_in = input_f['obs']
            
            # Process each obs column using helper functions
            with tqdm(obs_group_in.keys(), desc="Processing obs columns") as pbar:
                for key in pbar:
                    pbar.set_description(f"Processing obs column: {key}")
                    
                    if key == '_index':
                        # Handle index specially
                        if hasattr(obs_group_in[key], 'dtype'):
                            selected_data = copy_1d_data_sequentially(
                                input_data=obs_group_in[key],
                                selected_mask=selected_mask,
                                total_input_cells=total_input_cells,
                                data_name=f"obs.{key}"
                            )
                            obs_group_out.create_dataset(key, data=selected_data)
                        else:
                            log.warning(f"Skipping complex _index structure")
                            continue
                    else:
                        item = obs_group_in[key]
                        if hasattr(item, 'dtype'):
                            # Simple array - use helper function
                            selected_data = copy_1d_data_sequentially(
                                input_data=item,
                                selected_mask=selected_mask,
                                total_input_cells=total_input_cells,
                                data_name=f"obs.{key}"
                            )
                            obs_group_out.create_dataset(key, data=selected_data)
                        else:
                            # Categorical - copy structure and subset codes
                            cat_group = obs_group_out.create_group(key)
                            # Copy categories unchanged
                            input_f.copy(f'obs/{key}/categories', cat_group, 'categories')
                            
                            # Subset codes using helper function
                            selected_codes = copy_1d_data_sequentially(
                                input_data=item['codes'],
                                selected_mask=selected_mask,
                                total_input_cells=total_input_cells,
                                data_name=f"obs.{key}.codes"
                            )
                            cat_group.create_dataset('codes', data=selected_codes)
        
        # Copy obsm data (cell embeddings - subsetted) using sequential approach
        log.info("Copying obsm data with sequential processing...")
        if 'obsm' in input_f:
            obsm_group_out = output_f.create_group('obsm')
            obsm_group_in = input_f['obsm']
            
            with tqdm(obsm_group_in.keys(), desc="Processing obsm embeddings") as pbar:
                for key in pbar:
                    pbar.set_description(f"Processing obsm: {key}")
                    embedding_data = obsm_group_in[key]
                    
                    if len(embedding_data.shape) == 2:
                        # 2D embeddings - use helper function
                        n_dims = embedding_data.shape[1]
                        output_embedding = obsm_group_out.create_dataset(
                            key,
                            shape=(n_selected_cells, n_dims),
                            dtype=embedding_data.dtype
                        )
                        
                        copy_data_sequentially(
                            input_data=embedding_data,
                            output_data=output_embedding,
                            selected_mask=selected_mask,
                            total_input_cells=total_input_cells,
                            n_selected_cells=n_selected_cells,
                            data_name=f"obsm.{key}"
                        )
                    else:
                        # 1D or unexpected shape - use helper function
                        log.warning(f"Unexpected obsm shape for {key}: {embedding_data.shape}")
                        selected_data = copy_1d_data_sequentially(
                            input_data=embedding_data,
                            selected_mask=selected_mask,
                            total_input_cells=total_input_cells,
                            data_name=f"obsm.{key}"
                        )
                        obsm_group_out.create_dataset(key, data=selected_data)
        
        # Copy remaining groups unchanged (uns, var, varm, varp, obsp)
        for group_name in ['uns', 'varm', 'varp', 'obsp']:
            if group_name in input_f:
                log.info(f"Copying {group_name} data...")
                input_f.copy(group_name, output_f)
        
        # Copy layers if present using sequential approach
        if 'layers' in input_f:
            log.info("Copying layers data with sequential processing...")
            layers_group_out = output_f.create_group('layers')
            layers_group_in = input_f['layers']
            
            with tqdm(layers_group_in.keys(), desc="Processing layers") as pbar:
                for key in pbar:
                    pbar.set_description(f"Processing layer: {key}")
                    layer_data = layers_group_in[key]
                    
                    if len(layer_data.shape) == 2:
                        # 2D layer data - use helper function
                        n_features = layer_data.shape[1]
                        output_layer = layers_group_out.create_dataset(
                            key,
                            shape=(n_selected_cells, n_features),
                            dtype=layer_data.dtype
                        )
                        
                        copy_data_sequentially(
                            input_data=layer_data,
                            output_data=output_layer,
                            selected_mask=selected_mask,
                            total_input_cells=total_input_cells,
                            n_selected_cells=n_selected_cells,
                            data_name=f"layers.{key}"
                        )
                    else:
                        # 1D layer data - use helper function
                        selected_data = copy_1d_data_sequentially(
                            input_data=layer_data,
                            selected_mask=selected_mask,
                            total_input_cells=total_input_cells,
                            data_name=f"layers.{key}"
                        )
                        layers_group_out.create_dataset(key, data=selected_data)
    
    log.info(f"Successfully wrote subsampled H5AD to {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Memory-efficient subsampling of H5AD files by groups using reservoir sampling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Subsample by cell line and drug dose
    python subsample_h5ad_by_groups.py input.h5ad output.h5ad \\
        --group-columns cell_line_id drug_dose --max-cells-per-group 1000
        
    # Subsample by single column with custom chunk size
    python subsample_h5ad_by_groups.py input.h5ad output.h5ad \\
        --group-columns cell_line_id --max-cells-per-group 500 --chunk-size 2000000
        
    # Quick test with small max per group
    python subsample_h5ad_by_groups.py input.h5ad output.h5ad \\
        --group-columns drug_dose --max-cells-per-group 100

The script uses pure h5py for maximum memory efficiency and can handle
arbitrarily large H5AD files without loading them fully into memory.
        """
    )
    
    parser.add_argument('input_file', 
                       help='Path to input H5AD file')
    parser.add_argument('output_file', 
                       help='Path to output H5AD file')
    parser.add_argument('--group-columns', 
                       nargs='+', 
                       required=True,
                       help='Column names in obs to define groups (e.g., cell_line_id drug_dose)')
    parser.add_argument('--max-cells-per-group', 
                       type=int, 
                       default=1000,
                       help='Maximum number of cells to keep per group (default: 1000)')
    parser.add_argument('--chunk-size', 
                       type=int, 
                       default=1000000,
                       help='Chunk size for streaming obs data (default: 1M)')
    parser.add_argument('--seed', 
                       type=int, 
                       default=42,
                       help='Random seed for reproducible sampling (default: 42)')
    
    args = parser.parse_args()
    
    # Validate input file
    input_path = Path(args.input_file)
    if not input_path.exists():
        log.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    output_path = Path(args.output_file)
    if output_path.exists():
        log.warning(f"Output file exists and will be overwritten: {output_path}")
    
    # Create output directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Phase 1: Stream obs and select cells
        log.info("Phase 1: Streaming obs data for group selection...")
        selected_indices, group_counts = stream_obs_for_group_selection(
            input_path, 
            args.group_columns, 
            args.max_cells_per_group,
            args.chunk_size
        )
        
        if not selected_indices:
            log.error("No cells selected - check group column names and data")
            sys.exit(1)
        
        # Phase 2: Write subsampled H5AD
        log.info("Phase 2: Writing subsampled H5AD file...")
        write_subsampled_h5ad(input_path, output_path, selected_indices)
        
        elapsed = time.time() - start_time
        
        # Final statistics
        log.info(f"✅ Subsampling completed in {elapsed:.1f} seconds")
        log.info(f"Input: {input_path}")
        log.info(f"Output: {output_path}")
        log.info(f"Selected {len(selected_indices):,} cells from {len(group_counts)} groups")
        log.info(f"Average cells per group: {len(selected_indices)/len(group_counts):.1f}")
        
    except Exception as e:
        log.error(f"Error during subsampling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()