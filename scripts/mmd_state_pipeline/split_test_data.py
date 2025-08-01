#!/usr/bin/env python3
"""
Ultra-optimized split test data - single pass extraction for 1TB+ files.
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import h5py
import gc
import psutil
from tqdm import tqdm
import warnings
import random
import scipy.sparse as sp
import anndata as ad
warnings.filterwarnings('ignore')

class ReservoirSampler:
    """
    Reservoir sampling implementation for maintaining max N samples per combination.
    """
    def __init__(self, max_size=1500, seed=42):
        self.max_size = max_size
        self.samples = []
        self.count = 0
        self.rng = random.Random(seed)
    
    def add_sample(self, item):
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
    
    def get_samples(self):
        """Get all samples in the reservoir."""
        return self.samples.copy()
    
    def size(self):
        """Get current number of samples."""
        return len(self.samples)
    
    def is_complete(self):
        """Check if reservoir has reached maximum size."""
        return len(self.samples) >= self.max_size

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024 / 1024

def auto_detect_dmso_condition(h5ad_file, chunk_size=1000000):
    """
    Auto-detect DMSO condition from the data by sampling drug_dose values.
    """
    print("Auto-detecting DMSO condition...")
    dmso_candidates = set()
    
    with h5py.File(h5ad_file, 'r') as f:
        obs_group = f['obs']
        drug_item = obs_group['drug_dose']
        
        # Get total rows
        if hasattr(drug_item, 'dtype'):
            n_obs = len(drug_item)
        else:
            n_obs = len(drug_item['codes'])
        
        # Sample first chunk to find DMSO condition
        end_idx = min(chunk_size, n_obs)
        
        if hasattr(drug_item, 'dtype'):
            drug_values = drug_item[:end_idx]
            if drug_item.dtype.kind in ['S', 'U']:
                drug_chunk = drug_values.astype(str)
            else:
                drug_chunk = drug_values
        else:
            codes = drug_item['codes'][:end_idx]
            categories = drug_item['categories'][:]
            decoded_categories = np.array([
                s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                for s in categories
            ])
            drug_chunk = decoded_categories[codes]
        
        # Look for DMSO-like conditions
        for drug in drug_chunk:
            drug_str = str(drug)
            if 'DMSO' in drug_str.upper():
                dmso_candidates.add(drug_str)
    
    if len(dmso_candidates) == 1:
        dmso_condition = list(dmso_candidates)[0]
        print(f"Auto-detected DMSO condition: '{dmso_condition}'")
        return dmso_condition
    elif len(dmso_candidates) > 1:
        # If multiple DMSO conditions, pick the most common one
        print(f"Multiple DMSO candidates found: {dmso_candidates}")
        dmso_condition = list(dmso_candidates)[0]  # Pick first one
        print(f"Using first DMSO condition: '{dmso_condition}'")
        return dmso_condition
    else:
        raise ValueError(f"No DMSO condition found in data. Please specify --dmso-condition.")

def streaming_filter_with_reservoir_sampling(h5ad_file, test_combinations, test_cell_lines, dmso_condition, chunk_size=50000000, max_per_combination=1500):
    """
    Stream through obs data with reservoir sampling - eliminates separate subsampling step.
    Returns subsampled indices directly using reservoir sampling algorithm.
    """
    print(f"Streaming with reservoir sampling, {chunk_size:,} row chunks, max {max_per_combination} per combination")
    
    # Create lookup sets for fast filtering - include DMSO combinations for test cell lines
    test_lookup = set(zip(test_combinations['drug_dose'], test_combinations['cell_line']))
    
    # Add DMSO combinations for test cell lines to test_lookup
    dmso_test_combinations = [(dmso_condition, cell_line) for cell_line in test_cell_lines]
    test_lookup.update(dmso_test_combinations)
    
    test_cell_lines_set = set(test_cell_lines)
    
    print(f"Added {len(dmso_test_combinations)} DMSO control combinations to test lookup")
    
    # Debug: Print first few test combinations
    print(f"Looking for {len(test_lookup)} test combinations")
    sample_combinations = list(test_lookup)[:3]
    for i, (drug, cell) in enumerate(sample_combinations):
        print(f"  {i+1}. '{drug}' + '{cell}'")
    
    # Initialize reservoir samplers
    test_reservoirs = {}  # combination -> ReservoirSampler
    dmso_reservoirs = {}  # cell_line -> ReservoirSampler
    
    # Pre-initialize reservoirs for all expected combinations/cell lines
    for drug_dose, cell_line in test_lookup:
        test_reservoirs[(drug_dose, cell_line)] = ReservoirSampler(max_per_combination)
    
    for cell_line in test_cell_lines:
        dmso_reservoirs[cell_line] = ReservoirSampler(max_per_combination)
    
    print(f"Initialized {len(test_reservoirs)} test reservoirs (including {len(dmso_test_combinations)} DMSO controls)")
    print(f"Initialized {len(dmso_reservoirs)} DMSO reservoirs")
    
    # Essential columns to load (optimization)
    essential_columns = {'drug_dose', 'cell_line'}
    
    total_processed = 0
    early_termination = False
    
    with h5py.File(h5ad_file, 'r') as f:
        obs_group = f['obs']
        
        # Get total rows
        first_key = [k for k in obs_group.keys() if k != '_index'][0]
        n_obs = len(obs_group[first_key])
        print(f"Total observations: {n_obs:,}")
        
        # Process in very large chunks
        for start_idx in tqdm(range(0, n_obs, chunk_size), desc="Streaming with reservoir sampling"):
            end_idx = min(start_idx + chunk_size, n_obs)
            
            # Pre-filter by cell lines to reduce data volume
            # First, load just cell_line data for pre-filtering
            cell_item = obs_group['cell_line']
            if hasattr(cell_item, 'dtype'):
                cell_values = cell_item[start_idx:end_idx]
                if cell_item.dtype.kind in ['S', 'U']:
                    cell_chunk = cell_values.astype(str)
                else:
                    cell_chunk = cell_values
            else:
                codes = cell_item['codes'][start_idx:end_idx]
                categories = cell_item['categories'][:]
                decoded_categories = np.array([
                    s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                    for s in categories
                ])
                cell_chunk = decoded_categories[codes]
            
            # Filter to only test cell lines (major optimization)
            cell_mask = np.array([cell in test_cell_lines_set for cell in cell_chunk])
            relevant_indices = np.where(cell_mask)[0]
            
            if len(relevant_indices) == 0:
                continue  # Skip chunks with no relevant cell lines
            
            # Load drug_dose data only for relevant indices
            drug_item = obs_group['drug_dose']
            if hasattr(drug_item, 'dtype'):
                drug_values = drug_item[start_idx:end_idx]
                if drug_item.dtype.kind in ['S', 'U']:
                    drug_chunk = drug_values.astype(str)
                else:
                    drug_chunk = drug_values
            else:
                codes = drug_item['codes'][start_idx:end_idx]
                categories = drug_item['categories'][:]
                decoded_categories = np.array([
                    s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                    for s in categories
                ])
                drug_chunk = decoded_categories[codes]
            
            # Debug: Print first few samples from first chunk
            if start_idx == 0:
                print(f"Sample data from chunk (pre-filtered by cell lines):")
                for i in range(min(3, len(relevant_indices))):
                    idx = relevant_indices[i]
                    drug = drug_chunk[idx]
                    cell = cell_chunk[idx]
                    print(f"  {i+1}. '{drug}' + '{cell}'")
            
            # Process only relevant indices
            for local_idx in relevant_indices:
                global_idx = start_idx + local_idx
                drug_dose = drug_chunk[local_idx]
                cell_line = cell_chunk[local_idx]
                
                # Test combinations (includes both drug combinations and DMSO controls)
                combination = (drug_dose, cell_line)
                if combination in test_lookup:
                    test_reservoirs[combination].add_sample(global_idx)
                
                # DMSO controls for test cell lines (also goes to DMSO reservoirs)
                if drug_dose == dmso_condition and cell_line in test_cell_lines_set:
                    dmso_reservoirs[cell_line].add_sample(global_idx)
            
            total_processed += len(relevant_indices)
            
            # Early termination check
            if total_processed % 1000000 == 0:  # Check every 1M processed records
                test_complete = all(reservoir.is_complete() for reservoir in test_reservoirs.values())
                dmso_complete = all(reservoir.is_complete() for reservoir in dmso_reservoirs.values())
                
                if test_complete and dmso_complete:
                    print(f"Early termination: All combinations complete after processing {total_processed:,} records")
                    early_termination = True
                    break
            
            # Memory cleanup
            gc.collect()
    
    # Collect final results
    test_indices = []
    dmso_indices = []
    
    for combination, reservoir in test_reservoirs.items():
        samples = reservoir.get_samples()
        test_indices.extend(samples)
        print(f"Test combination {combination}: {len(samples)} samples (from {reservoir.count} total)")
    
    for cell_line, reservoir in dmso_reservoirs.items():
        samples = reservoir.get_samples()
        dmso_indices.extend(samples)
        print(f"DMSO {cell_line}: {len(samples)} samples (from {reservoir.count} total)")
    
    # Sort indices for better HDF5 access
    test_indices.sort()
    dmso_indices.sort()
    
    print(f"Final results: {len(test_indices):,} test indices, {len(dmso_indices):,} DMSO indices")
    if early_termination:
        print("✓ Early termination achieved - significant time savings!")
    
    return test_indices, dmso_indices



def extract_both_datasets_sequential(h5ad_file, test_indices, dmso_indices, output_dir):
    """
    Extract both test and DMSO datasets using sequential filtering (much faster than random access).
    """
    print(f"Sequential extraction: {len(test_indices):,} test + {len(dmso_indices):,} DMSO")
    
    # Fast lookup sets
    test_set = set(test_indices)
    dmso_set = set(dmso_indices)
    all_target_indices = test_set | dmso_set
    
    # Essential columns to extract (skip unnecessary metadata)
    essential_obs_columns = {
        'drug_dose', 'cell_line', 'batch', '_index', 
        'canonical_smiles', 'moa-fine', 'drug', 'cell_line_id'
    }
    
    print(f"Scanning {len(all_target_indices):,} target indices from dataset...")
    
    # Pre-allocate result arrays
    test_data = {}
    dmso_data = {}
    test_found = 0
    dmso_found = 0
    
    with h5py.File(h5ad_file, 'r') as source_f:
        # Get dimensions - since X is dummy, just use minimal vars
        try:
            if 'var' in source_f and '_index' in source_f['var']:
                n_vars = len(source_f['var']['_index'])
            else:
                n_vars = 1  # Minimal dummy dimension since X is not used
        except Exception as e:
            print(f"Warning: Could not determine var dimensions, using dummy: {e}")
            n_vars = 1  # Minimal dummy dimension
        
        obs_group = source_f['obs']
        
        # Get total observations for chunking
        first_key = [k for k in obs_group.keys() if k != '_index'][0]
        n_obs = len(obs_group[first_key])
        
        # Sequential chunking with large chunks for better performance
        chunk_size = 2000000  # 2M rows per chunk
        
        print(f"Processing {n_obs:,} observations in chunks of {chunk_size:,}")
        
        # Process in sequential chunks
        for start_idx in tqdm(range(0, n_obs, chunk_size), desc="Sequential processing"):
            end_idx = min(start_idx + chunk_size, n_obs)
            chunk_indices = list(range(start_idx, end_idx))
            
            # Find which indices in this chunk are targets
            target_mask = [(start_idx + i) in all_target_indices for i in range(end_idx - start_idx)]
            target_positions = [i for i, is_target in enumerate(target_mask) if is_target]
            
            if not target_positions:
                continue  # Skip chunks with no target data
            
            # Extract data for all essential obs columns in this chunk
            chunk_data = {}
            for key in obs_group.keys():
                if key not in essential_obs_columns:
                    continue  # Skip non-essential columns
                    
                item = obs_group[key]
                if hasattr(item, 'dtype'):
                    # Regular dataset
                    chunk_values = item[start_idx:end_idx]
                    chunk_data[key] = chunk_values[target_positions]
                elif hasattr(item, 'keys') and 'codes' in item:
                    # Categorical data
                    codes = item['codes'][start_idx:end_idx]
                    categories = item['categories'][:]
                    # Decode categories
                    decoded_categories = np.array([
                        s.decode('utf-8') if isinstance(s, bytes) else str(s) 
                        for s in categories
                    ])
                    chunk_values = decoded_categories[codes]
                    chunk_data[key] = chunk_values[target_positions]
            
            # Process obsm data for this chunk
            chunk_obsm = {}
            if 'obsm' in source_f:
                obsm_group = source_f['obsm']
                for obsm_key in obsm_group.keys():
                    obsm_values = obsm_group[obsm_key][start_idx:end_idx]
                    chunk_obsm[obsm_key] = obsm_values[target_positions]
            
            # Split chunk data into test and DMSO
            for i, pos in enumerate(target_positions):
                global_idx = start_idx + pos
                
                if global_idx in test_set:
                    # Add to test data
                    if test_found == 0:
                        # Initialize test arrays
                        for key, values in chunk_data.items():
                            test_data[key] = []
                        for key, values in chunk_obsm.items():
                            test_data[f'obsm_{key}'] = []
                    
                    for key, values in chunk_data.items():
                        test_data[key].append(values[i])
                    for key, values in chunk_obsm.items():
                        test_data[f'obsm_{key}'].append(values[i])
                    test_found += 1
                    
                if global_idx in dmso_set:
                    # Add to DMSO data
                    if dmso_found == 0:
                        # Initialize DMSO arrays
                        for key, values in chunk_data.items():
                            dmso_data[key] = []
                        for key, values in chunk_obsm.items():
                            dmso_data[f'obsm_{key}'] = []
                    
                    for key, values in chunk_data.items():
                        dmso_data[key].append(values[i])
                    for key, values in chunk_obsm.items():
                        dmso_data[f'obsm_{key}'].append(values[i])
                    dmso_found += 1
        
        print(f"Found {test_found:,} test observations and {dmso_found:,} DMSO observations")
        
        # Convert lists to numpy arrays with proper HDF5-compatible dtypes
        print("Converting to numpy arrays...")
        for key in test_data:
            if key.startswith('obsm_'):
                test_data[key] = np.array(test_data[key])
            else:
                # Handle string data properly for HDF5
                arr = np.array(test_data[key])
                if arr.dtype.kind in ['U', 'S']:
                    # Convert to UTF-8 encoded fixed-length string to handle Unicode properly
                    max_len = max(len(str(item).encode('utf-8')) for item in arr) if len(arr) > 0 else 1
                    # Use UTF-8 encoding to handle Unicode characters
                    utf8_arr = np.array([str(item).encode('utf-8') for item in arr])
                    test_data[key] = utf8_arr.astype(f'S{max_len}')
                else:
                    test_data[key] = arr
        
        for key in dmso_data:
            if key.startswith('obsm_'):
                dmso_data[key] = np.array(dmso_data[key])
            else:
                # Handle string data properly for HDF5
                arr = np.array(dmso_data[key])
                if arr.dtype.kind in ['U', 'S']:
                    # Convert to UTF-8 encoded fixed-length string to handle Unicode properly
                    max_len = max(len(str(item).encode('utf-8')) for item in arr) if len(arr) > 0 else 1
                    # Use UTF-8 encoding to handle Unicode characters
                    utf8_arr = np.array([str(item).encode('utf-8') for item in arr])
                    dmso_data[key] = utf8_arr.astype(f'S{max_len}')
                else:
                    dmso_data[key] = arr
        
        # Create output files
        test_output_path = output_dir / "test_subset.h5ad"
        dmso_output_path = output_dir / "dmso_controls.h5ad"
        
        # Create proper AnnData objects
        print("Creating proper AnnData objects...")
        
        # Load var data from source for proper structure
        var_df = None
        uns_dict = {}
        with h5py.File(h5ad_file, 'r') as f:
            try:
                if 'var' in f:
                    var_data = {}
                    for key in f['var'].keys():
                        # Skip reserved AnnData column names
                        if key == '_index':
                            continue
                        item = f['var'][key]
                        if hasattr(item, 'dtype'):
                            if item.dtype.kind in ['S', 'U']:
                                var_data[key] = item[:].astype(str)
                            else:
                                var_data[key] = item[:]
                    var_df = pd.DataFrame(var_data)
            except Exception as e:
                print(f"Warning: Could not load var data: {e}")
                var_df = None
            
            # Load uns data if present
            if 'uns' in f:
                try:
                    for key in f['uns'].keys():
                        item = f['uns'][key]
                        if hasattr(item, 'dtype'):
                            uns_dict[key] = item[:]
                except:
                    pass  # Skip if uns data is complex
        
        # Create dummy var DataFrame if needed (since X is dummy anyway)
        if var_df is None or len(var_df) != n_vars:
            print(f"Creating dummy var DataFrame with {n_vars} variables...")
            var_df = pd.DataFrame(index=range(n_vars))
        
        # Create test AnnData object
        print(f"Creating test AnnData object ({test_found:,} x {n_vars:,})...")
        
        # Prepare obs data (exclude reserved AnnData column names)
        test_obs_data = {}
        test_obsm_data = {}
        
        for key, values in test_data.items():
            if key.startswith('obsm_'):
                obsm_key = key[5:]
                test_obsm_data[obsm_key] = values
            elif key != '_index':  # Skip reserved AnnData column
                test_obs_data[key] = values
        
        test_obs_df = pd.DataFrame(test_obs_data)
        
        # Create empty sparse X matrix
        test_X = sp.csr_matrix((test_found, n_vars))
        
        # Create AnnData object
        test_adata = ad.AnnData(
            X=test_X,
            obs=test_obs_df,
            obsm=test_obsm_data,
            var=var_df,
            uns=uns_dict
        )
        
        # Save test file
        print(f"Saving test file: {test_output_path}")
        test_adata.write(test_output_path)
        
        # Create DMSO AnnData object
        print(f"Creating DMSO AnnData object ({dmso_found:,} x {n_vars:,})...")
        
        # Prepare obs data (exclude reserved AnnData column names)
        dmso_obs_data = {}
        dmso_obsm_data = {}
        
        for key, values in dmso_data.items():
            if key.startswith('obsm_'):
                obsm_key = key[5:]
                dmso_obsm_data[obsm_key] = values
            elif key != '_index':  # Skip reserved AnnData column
                dmso_obs_data[key] = values
        
        dmso_obs_df = pd.DataFrame(dmso_obs_data)
        
        # Debug output
        print(f"Debug: dmso_obsm_data keys: {list(dmso_obsm_data.keys())}")
        print(f"Debug: dmso_data keys: {[k for k in dmso_data.keys() if k.startswith('obsm_')]}")
        print(f"Debug: Total dmso_data keys: {list(dmso_data.keys())}")
        
        # Create empty sparse X matrix
        dmso_X = sp.csr_matrix((dmso_found, n_vars))
        
        # Create AnnData object
        dmso_adata = ad.AnnData(
            X=dmso_X,
            obs=dmso_obs_df,
            obsm=dmso_obsm_data,
            var=var_df,
            uns=uns_dict
        )
        
        # Save DMSO file
        print(f"Saving DMSO file: {dmso_output_path}")
        dmso_adata.write(dmso_output_path)
    
    print(f"Saved test subset: {test_output_path}")
    print(f"Saved DMSO controls: {dmso_output_path}")
    print("Files are now properly AnnData-compatible!")




def main():
    parser = argparse.ArgumentParser(description="Ultra-optimized single-pass split extraction")
    parser.add_argument("h5ad_file", help="Path to input h5ad file")
    parser.add_argument("--split-file", 
                        default="/tahoe/drive_3/ANALYSIS/analysis_190/Data/20250618.tahoe_embeddings_70M_DC_split_assignments.parquet",
                        help="Path to split assignments file")
    parser.add_argument("--dmso-condition",
                        default=None,
                        help="DMSO condition name (default: auto-detect from data)")
    parser.add_argument("--output-dir",
                        default=None,
                        help="Custom output directory (default: {input_stem}_split_outputs)")
    
    args = parser.parse_args()
    
    # Create output directory
    input_path = Path(args.h5ad_file)
    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = input_path.parent / f"{input_path.stem}_split_outputs"
    output_dir.mkdir(exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load split assignments
    print("Loading split assignments...")
    split_df = pd.read_parquet(args.split_file)
    test_combinations = split_df[split_df['split_assignment'] == 'test'].copy()
    test_cell_lines = test_combinations['cell_line'].unique()
    
    print(f"Found {len(test_combinations):,} test combinations")
    print(f"Found {len(test_cell_lines):,} test cell lines")
    
    # Auto-detect or use specified DMSO condition
    if args.dmso_condition is None:
        dmso_condition = auto_detect_dmso_condition(args.h5ad_file)
    else:
        dmso_condition = args.dmso_condition
        print(f"Using specified DMSO condition: '{dmso_condition}'")
    
    # Stream through data with reservoir sampling (combines steps 1 and 2)
    print("Step 1: Streaming with reservoir sampling...")
    test_indices, dmso_indices = streaming_filter_with_reservoir_sampling(
        args.h5ad_file, test_combinations, test_cell_lines, dmso_condition
    )
    
    print(f"Reservoir sampling complete: {len(test_indices):,} test, {len(dmso_indices):,} DMSO")
    
    # Sequential extraction
    print("Step 2: Sequential extraction...")
    extract_both_datasets_sequential(
        args.h5ad_file, test_indices, dmso_indices, output_dir
    )
    
    print("Ultra-optimized extraction completed!")
    print(f"Final memory usage: {get_memory_usage():.2f} GB")

if __name__ == "__main__":
    main()