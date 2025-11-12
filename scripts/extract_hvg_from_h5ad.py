#!/usr/bin/env python3
"""
Extract HVG (Highly Variable Genes) data from existing .h5ad files.

This script takes .h5ad files as input and extracts HVG expression data by:
1. Loading the HVG gene mapping from token2hvg.parquet
2. Extracting gene expressions from the .X slot
3. Mapping gene names to HVG indices and handling missing genes (set to 0)
4. Normalizing by library size and multiplying by 1872
5. Optionally applying log10(x+1) transformation
6. Storing dense HVG data in .obsm['X_hvg']
7. Preserving all metadata in .obs and .obsm
8. Replacing .X with empty sparse matrix to save space
9. Writing output to hvg/ subdirectory

Performance is tracked with detailed timing breakdowns for reading, processing, and writing operations.

Example usage:
    python extract_hvg_from_h5ad.py input.h5ad --token2hvg-path /path/to/token2hvg.parquet
    python extract_hvg_from_h5ad.py *.h5ad --token2hvg-path /path/to/token2hvg.parquet --apply-log
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil
from scipy import sparse

# Suppress pandas warnings about string dtype inference
warnings.filterwarnings('ignore', category=FutureWarning, module='pandas')

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


class PerformanceTracker:
    """Detailed performance tracking for HVG extraction with component-level timing."""
    
    def __init__(self, total_cells: int, filename: str):
        self.total_cells = total_cells
        self.filename = filename
        self.processed_cells = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_cells = 0
        
        # Detailed timing components
        self.timings = {
            # Data Reading
            'reading_X': 0.0,
            'reading_var': 0.0, 
            'reading_obs': 0.0,
            'reading_obsm': 0.0,
            
            # Processing Steps
            'gene_mapping': 0.0,
            'libsize_calc': 0.0,
            'normalization': 0.0,
            'log_transform': 0.0,
            'dense_construction': 0.0,
            
            # Data Writing
            'writing_X_hvg': 0.0,
            'writing_obs': 0.0,
            'writing_obsm': 0.0,
            'writing_sparse_X': 0.0,
            'flush_operations': 0.0
        }
        
        log.info(f"🚀 Starting HVG extraction for {filename} ({total_cells:,} cells)")
    
    def update_timing(self, component: str, duration: float, cells_processed: int = 0):
        """Update timing for a specific component."""
        if component not in self.timings:
            log.warning(f"Unknown timing component: {component}")
            return
            
        self.timings[component] += duration
        self.processed_cells += cells_processed
        
        # Log progress every 50K cells
        if self.processed_cells > 0 and self.processed_cells % 50000 == 0:
            self._log_progress()
    
    def _log_progress(self):
        """Log detailed progress information with component breakdown."""
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
        if total_time == 0:
            return
            
        # Group timings
        reading_time = self.timings['reading_X'] + self.timings['reading_var'] + \
                      self.timings['reading_obs'] + self.timings['reading_obsm']
        processing_time = self.timings['gene_mapping'] + self.timings['libsize_calc'] + \
                         self.timings['normalization'] + self.timings['log_transform'] + \
                         self.timings['dense_construction']
        writing_time = self.timings['writing_X_hvg'] + self.timings['writing_obs'] + \
                      self.timings['writing_obsm'] + self.timings['writing_sparse_X'] + \
                      self.timings['flush_operations']
        
        reading_pct = (reading_time / total_time) * 100
        processing_pct = (processing_time / total_time) * 100
        writing_pct = (writing_time / total_time) * 100
        
        # Get memory usage
        memory_gb = get_memory_usage()
        
        # Progress report
        progress_pct = (self.processed_cells / self.total_cells) * 100
        log.info(f"📊 {self.filename} Progress: {self.processed_cells:,}/{self.total_cells:,} cells ({progress_pct:.1f}%)")
        
        # Detailed timing breakdown
        if reading_time > 0:
            log.info(f"⚡ Time Breakdown:")
            log.info(f"   Reading: {reading_pct:.1f}% (X={self.timings['reading_X']/total_time*100:.1f}%, "
                    f"var={self.timings['reading_var']/total_time*100:.1f}%, "
                    f"obs={self.timings['reading_obs']/total_time*100:.1f}%, "
                    f"obsm={self.timings['reading_obsm']/total_time*100:.1f}%)")
            
            log.info(f"   Processing: {processing_pct:.1f}% (mapping={self.timings['gene_mapping']/total_time*100:.1f}%, "
                    f"libsize={self.timings['libsize_calc']/total_time*100:.1f}%, "
                    f"norm={self.timings['normalization']/total_time*100:.1f}%, "
                    f"log={self.timings['log_transform']/total_time*100:.1f}%, "
                    f"dense={self.timings['dense_construction']/total_time*100:.1f}%)")
            
            if writing_time > 0:
                log.info(f"   Writing: {writing_pct:.1f}% (X_hvg={self.timings['writing_X_hvg']/total_time*100:.1f}%, "
                        f"obs={self.timings['writing_obs']/total_time*100:.1f}%, "
                        f"obsm={self.timings['writing_obsm']/total_time*100:.1f}%, "
                        f"flush={self.timings['flush_operations']/total_time*100:.1f}%)")
        
        log.info(f"🏃 Rate: {overall_rate:.0f} cells/sec (recent: {recent_rate:.0f}), ETA: {eta_hours:.1f} hours")
        log.info(f"💾 Memory: {memory_gb:.1f} GB")
        
        # Performance warnings
        if reading_pct > 60:
            log.warning(f"⚠️  High reading overhead ({reading_pct:.1f}%) - consider larger chunks")
        if any([self.timings[k]/total_time > 0.5 for k in ['gene_mapping', 'normalization']]):
            log.warning(f"⚠️  Slow processing step detected")
        if recent_rate < 100 and self.processed_cells > 100000:
            log.warning(f"⚠️  Low processing rate ({recent_rate:.0f} cells/sec)")
        
        # Update tracking variables
        self.last_update_time = current_time
        self.last_update_cells = self.processed_cells
    
    def final_summary(self):
        """Print final performance summary."""
        total_time = time.time() - self.start_time
        average_rate = self.processed_cells / total_time if total_time > 0 else 0
        
        # Component time summary
        component_time = sum(self.timings.values())
        reading_time = self.timings['reading_X'] + self.timings['reading_var'] + \
                      self.timings['reading_obs'] + self.timings['reading_obsm']
        processing_time = self.timings['gene_mapping'] + self.timings['libsize_calc'] + \
                         self.timings['normalization'] + self.timings['log_transform'] + \
                         self.timings['dense_construction']
        writing_time = self.timings['writing_X_hvg'] + self.timings['writing_obs'] + \
                      self.timings['writing_obsm'] + self.timings['writing_sparse_X'] + \
                      self.timings['flush_operations']
        
        log.info(f"\n✓ Completed {self.filename}: {self.processed_cells:,} cells in {total_time/60:.1f} minutes")
        log.info(f"📈 Average rate: {average_rate:.0f} cells/sec")
        log.info(f"⏱️  Time breakdown: Reading={reading_time/component_time*100:.1f}%, "
                f"Processing={processing_time/component_time*100:.1f}%, "
                f"Writing={writing_time/component_time*100:.1f}%")


def load_hvg_mapping(token2hvg_path: str) -> Tuple[Dict[str, int], List[str]]:
    """Load HVG token to column index mapping and gene names."""
    log.info(f"Loading HVG mapping from: {token2hvg_path}")
    
    try:
        df = pd.read_parquet(token2hvg_path)
        df = df.sort_values('token_id').reset_index(drop=True)
        
        # Create mapping from gene symbol to HVG column index
        gene_to_col_idx = {gene: i for i, gene in enumerate(df['gene_symbol'])}
        gene_names = df['gene_symbol'].tolist()
        
        log.info(f"Loaded {len(gene_to_col_idx)} HVG genes")
        return gene_to_col_idx, gene_names
        
    except Exception as e:
        log.error(f"Failed to load HVG mapping: {e}")
        sys.exit(1)


def should_skip_file(input_path: Path, output_path: Path, force: bool = False) -> bool:
    """Check if output file already exists and is complete."""
    if not output_path.exists():
        return False
    
    if force:
        log.info(f"🔄 Force mode: overwriting existing {output_path.name}")
        return False
    
    try:
        # Quick validation that output file is readable and has X_hvg
        with h5py.File(output_path, 'r') as f:
            if '/obsm/X_hvg' not in f:
                log.warning(f"⚠️  Output file exists but missing X_hvg: {output_path.name}")
                return False
                
            n_cells, n_hvg = f['/obsm/X_hvg'].shape
            log.info(f"✓ {output_path.name} already exists with {n_cells:,} cells, {n_hvg} HVGs - skipping")
            return True
            
    except Exception as e:
        log.warning(f"⚠️  Output file corrupted, will regenerate: {e}")
        return False


def extract_hvg_from_h5ad(input_path: Path, output_path: Path, 
                         gene_to_col_idx: Dict[str, int], hvg_gene_names: List[str],
                         apply_log: bool = False, chunk_size: int = 100000):
    """
    Extract HVG data from an h5ad file and write to output with detailed performance tracking.
    
    Args:
        input_path: Path to input .h5ad file
        output_path: Path to output .h5ad file
        gene_to_col_idx: Mapping from gene symbol to HVG column index
        hvg_gene_names: List of HVG gene names in order
        apply_log: Whether to apply log10(x+1) transformation
        chunk_size: Number of cells to process at once
    """
    log.info(f"🔧 Processing {input_path.name} → {output_path.name}")
    
    # Open input file to get basic info
    with h5py.File(input_path, 'r') as input_f:
        if '/X' not in input_f:
            raise ValueError(f"No /X matrix found in {input_path}")
        if '/var' not in input_f or '_index' not in input_f['/var']:
            raise ValueError(f"No gene names found in /var/_index in {input_path}")
        
        # Get shape - handle both dense and sparse matrices
        X_item = input_f['/X']
        if hasattr(X_item, 'shape'):
            # Dense matrix
            n_cells, n_genes = X_item.shape
        elif hasattr(X_item, 'attrs') and 'shape' in X_item.attrs:
            # Sparse matrix with shape in attributes
            n_cells, n_genes = X_item.attrs['shape']
        elif 'indptr' in X_item:
            # CSR sparse matrix - infer shape from indptr and number of genes
            n_cells = len(X_item['indptr']) - 1
            n_genes = len(input_f['/var/_index'])
        else:
            raise ValueError(f"Cannot determine shape of /X matrix in {input_path}")
        
    # Initialize performance tracker
    tracker = PerformanceTracker(n_cells, input_path.name)
    
    log.info(f"📊 Input shape: {n_cells:,} cells × {n_genes:,} genes")
    log.info(f"📦 Chunk size: {chunk_size:,} cells")
    log.info(f"📋 Log transformation: {'enabled' if apply_log else 'disabled'}")
    
    # Create output file
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with h5py.File(input_path, 'r') as input_f, h5py.File(output_path, 'w') as output_f:
        # Read gene names
        start_time = time.time()
        gene_names = []
        var_index = input_f['/var/_index']
        for i in range(len(var_index)):
            gene_name = var_index[i]
            if isinstance(gene_name, bytes):
                gene_name = gene_name.decode('utf-8')
            gene_names.append(str(gene_name))
        tracker.update_timing('reading_var', time.time() - start_time)
        
        # Create gene name to X matrix column mapping
        start_time = time.time()
        gene_name_to_x_col = {gene: i for i, gene in enumerate(gene_names)}
        
        # Map HVG genes to their column indices in X matrix
        hvg_to_x_mapping = {}  # HVG index -> X matrix column index
        missing_genes = []
        
        for hvg_idx, hvg_gene in enumerate(hvg_gene_names):
            if hvg_gene in gene_name_to_x_col:
                hvg_to_x_mapping[hvg_idx] = gene_name_to_x_col[hvg_gene]
            else:
                missing_genes.append(hvg_gene)
        
        if missing_genes:
            log.warning(f"⚠️  {len(missing_genes)} HVG genes missing from input (will be set to 0)")
            if len(missing_genes) <= 10:
                log.warning(f"   Missing: {', '.join(missing_genes[:10])}")
            else:
                log.warning(f"   Missing: {', '.join(missing_genes[:5])}... and {len(missing_genes)-5} more")
        
        log.info(f"📈 Mapped {len(hvg_to_x_mapping)}/{len(hvg_gene_names)} HVG genes to input")
        tracker.update_timing('gene_mapping', time.time() - start_time)
        
        # Create output datasets
        start_time = time.time()
        
        # Create empty sparse matrix as placeholder for X
        empty_csr = sparse.csr_matrix((n_cells, 1), dtype=np.float32)
        sparse_group = output_f.create_group('X')
        sparse_group.create_dataset('data', data=empty_csr.data)
        sparse_group.create_dataset('indices', data=empty_csr.indices)  
        sparse_group.create_dataset('indptr', data=empty_csr.indptr)
        sparse_group.attrs['shape'] = np.array([n_cells, 1])
        sparse_group.attrs['encoding-type'] = 'csr_matrix'
        sparse_group.attrs['encoding-version'] = '0.1.0'
        
        # Create X_hvg dataset
        obsm_group = output_f.create_group('obsm')
        hvg_dataset = obsm_group.create_dataset('X_hvg', shape=(n_cells, len(hvg_gene_names)), 
                                               dtype=np.float32, chunks=True, compression='gzip')
        
        tracker.update_timing('writing_sparse_X', time.time() - start_time)
        
        # Copy obs data
        start_time = time.time()
        if '/obs' in input_f:
            output_f.copy(input_f['/obs'], output_f, name='/obs')
        else:
            obs_group = output_f.create_group('obs')
            obs_group.create_dataset('_index', data=[f"cell_{i}" for i in range(n_cells)])
        tracker.update_timing('writing_obs', time.time() - start_time)
        
        # Copy other obsm data
        start_time = time.time()
        if '/obsm' in input_f:
            for key in input_f['/obsm'].keys():
                if key != 'X_hvg':  # Don't copy existing X_hvg
                    try:
                        input_f.copy(input_f[f'/obsm/{key}'], obsm_group, name=key)
                    except Exception as e:
                        log.warning(f"⚠️  Failed to copy obsm/{key}: {e}")
        tracker.update_timing('writing_obsm', time.time() - start_time)
        
        # Copy var data with HVG gene names
        if '/var' in input_f:
            var_group = output_f.create_group('var')
            var_group.create_dataset('_index', data=[name.encode('utf-8') for name in hvg_gene_names])
            # Copy other var columns if they exist
            for key in input_f['/var'].keys():
                if key != '_index':
                    # Only copy if the source var has same length as our HVG list
                    try:
                        source_data = input_f[f'/var/{key}']
                        if len(source_data) == len(hvg_gene_names):
                            input_f.copy(source_data, var_group, name=key)
                    except:
                        pass
        else:
            var_group = output_f.create_group('var')
            var_group.create_dataset('_index', data=[name.encode('utf-8') for name in hvg_gene_names])
        
        # Copy other top-level groups
        for key in input_f.keys():
            if key not in ['X', 'obs', 'obsm', 'var']:
                try:
                    input_f.copy(input_f[key], output_f, name=key)
                except Exception as e:
                    log.warning(f"⚠️  Failed to copy top-level group {key}: {e}")
        
        # Process cells in chunks
        X_dataset = input_f['/X']
        # Check if X is sparse by looking for sparse matrix components
        is_sparse = ('data' in X_dataset and 'indices' in X_dataset and 'indptr' in X_dataset)
        
        log.info(f"📊 X matrix format: {'sparse CSR' if is_sparse else 'dense'}")
        
        for chunk_start in tqdm(range(0, n_cells, chunk_size), desc="Processing chunks"):
            chunk_end = min(chunk_start + chunk_size, n_cells)
            chunk_cells = chunk_end - chunk_start
            
            # Read X matrix chunk
            start_time = time.time()
            if is_sparse:
                # Handle sparse matrix
                indptr = X_dataset['indptr'][chunk_start:chunk_end+1]
                data_start = indptr[0]
                data_end = indptr[-1]
                
                chunk_data = X_dataset['data'][data_start:data_end]
                chunk_indices = X_dataset['indices'][data_start:data_end]
                chunk_indptr = indptr - data_start
                
                # Convert to dense for processing
                chunk_matrix = sparse.csr_matrix((chunk_data, chunk_indices, chunk_indptr),
                                               shape=(chunk_cells, n_genes)).toarray()
            else:
                chunk_matrix = X_dataset[chunk_start:chunk_end, :]
                if hasattr(chunk_matrix, 'toarray'):
                    chunk_matrix = chunk_matrix.toarray()
            
            tracker.update_timing('reading_X', time.time() - start_time)
            
            # Calculate library sizes
            start_time = time.time()
            library_sizes = np.sum(chunk_matrix, axis=1)
            tracker.update_timing('libsize_calc', time.time() - start_time)
            
            # Create HVG matrix for this chunk
            start_time = time.time()
            hvg_chunk = np.zeros((chunk_cells, len(hvg_gene_names)), dtype=np.float32)
            
            # Map HVG genes from X matrix
            for hvg_idx, x_col in hvg_to_x_mapping.items():
                hvg_chunk[:, hvg_idx] = chunk_matrix[:, x_col]
            
            tracker.update_timing('dense_construction', time.time() - start_time)
            
            # Normalize by library size and multiply by 1872
            start_time = time.time()
            for i in range(chunk_cells):
                if library_sizes[i] > 0:
                    hvg_chunk[i] = (hvg_chunk[i] / library_sizes[i]) * 1872
            tracker.update_timing('normalization', time.time() - start_time)
            
            # Apply log transformation if requested
            if apply_log:
                start_time = time.time()
                hvg_chunk = np.log10(hvg_chunk + 1)
                tracker.update_timing('log_transform', time.time() - start_time)
            
            # Write HVG chunk
            start_time = time.time()
            hvg_dataset[chunk_start:chunk_end, :] = hvg_chunk
            tracker.update_timing('writing_X_hvg', time.time() - start_time)
            
            # Update progress
            tracker.update_timing('flush_operations', 0, chunk_cells)
        
        # Final flush
        start_time = time.time()
        output_f.flush()
        tracker.update_timing('flush_operations', time.time() - start_time)
    
    tracker.final_summary()
    log.info(f"✅ Successfully created {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract HVG data from .h5ad files with normalization and optional log transformation"
    )
    parser.add_argument(
        "input_files",
        nargs='+',
        help="Path(s) to input .h5ad file(s)"
    )
    parser.add_argument(
        "--token2hvg-path",
        required=True,
        help="Path to token2hvg.parquet file containing HVG mapping"
    )
    parser.add_argument(
        "--apply-log",
        action='store_true',
        help="Apply log10(x+1) transformation to normalized data (default: False)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Number of cells to process at once (default: 100,000)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path.cwd(),
        help="Base output directory (default: current directory, creates hvg/ subdirectory)"
    )
    parser.add_argument(
        "--force",
        action='store_true',
        help="Overwrite existing output files (default: skip existing)"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not Path(args.token2hvg_path).exists():
        log.error(f"Token2HVG file not found: {args.token2hvg_path}")
        sys.exit(1)
    
    input_files = [Path(f) for f in args.input_files]
    for input_file in input_files:
        if not input_file.exists():
            log.error(f"Input file not found: {input_file}")
            sys.exit(1)
    
    # Create output directory
    output_dir = args.output_dir / "hvg"
    output_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"📁 Output directory: {output_dir}")
    
    # Load HVG mapping
    gene_to_col_idx, hvg_gene_names = load_hvg_mapping(args.token2hvg_path)
    
    # Process each file
    log.info(f"🎯 Processing {len(input_files)} file(s)")
    successful_files = 0
    skipped_files = 0
    
    for input_path in input_files:
        output_path = output_dir / input_path.name
        
        try:
            if should_skip_file(input_path, output_path, args.force):
                skipped_files += 1
                continue
            
            extract_hvg_from_h5ad(
                input_path, output_path, gene_to_col_idx, hvg_gene_names,
                apply_log=args.apply_log, chunk_size=args.chunk_size
            )
            successful_files += 1
            
        except Exception as e:
            log.error(f"❌ Failed to process {input_path.name}: {e}")
            continue
    
    total_completed = successful_files + skipped_files
    log.info(f"\n🎉 Completed: {successful_files}/{len(input_files)} files processed successfully")
    if skipped_files > 0:
        log.info(f"⏭️  Skipped: {skipped_files} files (already exist)")
    if total_completed < len(input_files):
        sys.exit(1)


if __name__ == "__main__":
    main()