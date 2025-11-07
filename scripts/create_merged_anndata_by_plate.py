"""
Creates plate-based AnnData files from Tahoe-100M parquet data with integrated drug dose information.

CRITICAL DATA ALIGNMENT ASSUMPTION:
This script assumes that both the state embeddings and MosaicFM embeddings parquet datasets 
maintain consistent ordering by BARCODE_SUB_LIB_ID within each plate. When filtering both
datasets by the same criteria (plate/samples), we iterate through them positionally without
explicit sorting, relying on this consistent ordering for proper data alignment.

If the source datasets do not maintain this ordering consistency, the script will produce
corrupted results where cells are matched with incorrect embeddings.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from polars import StringCache
from omegaconf import OmegaConf as om, DictConfig
from scipy import sparse
from tqdm import tqdm
from datasets import load_dataset
import pyarrow.parquet as pq
import re

# === Logging Setup ===
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


# === CLI Argument Parsing ===
def parse_args():
    """Parse command line arguments for plate processing."""
    parser = argparse.ArgumentParser(
        description='Process Tahoe-100M data by plate with real-time performance monitoring',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  # Process all plates (default behavior)
  python create_merged_anndata_by_plate.py config.yaml
  
  # List all available plates
  python create_merged_anndata_by_plate.py config.yaml --list-plates
  
  # Process a specific plate
  python create_merged_anndata_by_plate.py config.yaml --plate plate11
  
  # Parallel processing across machines
  python create_merged_anndata_by_plate.py config.yaml --plate plate11 &  # Machine 1
  python create_merged_anndata_by_plate.py config.yaml --plate plate12 &  # Machine 2
        '''
    )
    
    parser.add_argument('config_file', 
                       help='Path to YAML configuration file')
    parser.add_argument('--plate', 
                       type=str, 
                       help='Process only the specified plate (e.g., "plate11"). Enables parallel processing across machines.')
    parser.add_argument('--list-plates', 
                       action='store_true', 
                       help='List all available plates and exit without processing')
    
    return parser.parse_args()


# === Real-Time Performance Tracking ===
class PlatePerformanceTracker:
    """Real-time performance monitoring for long-running plate processing."""
    
    def __init__(self, total_cells: int, plate_name: str):
        self.total_cells = total_cells
        self.plate_name = plate_name
        self.timings = {
            'data_loading': 0.0,
            'gene_processing': 0.0,
            'matrix_assignment': 0.0,
            'obs_building': 0.0
        }
        self.processed_cells = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.last_update_cells = 0
        
        log.info(f"🚀 Starting performance tracking for plate {plate_name} ({total_cells:,} cells)")
    
    def update_timing(self, component: str, duration: float, cells_processed: int = 0):
        """Update timing for a specific component."""
        self.timings[component] += duration
        self.processed_cells += cells_processed
        
        # Log progress every 50K cells
        if self.processed_cells > 0 and self.processed_cells % 50000 == 0:
            self.log_current_stats()
    
    def log_current_stats(self):
        """Log current performance statistics and projections."""
        current_time = time.time()
        total_duration = current_time - self.start_time
        total_component_time = sum(self.timings.values())
        
        if total_component_time > 0 and total_duration > 0:
            # Calculate percentages
            percentages = {k: v/total_component_time*100 for k, v in self.timings.items()}
            
            # Calculate rates and ETA
            overall_rate = self.processed_cells / total_duration
            remaining_cells = self.total_cells - self.processed_cells
            eta_seconds = remaining_cells / overall_rate if overall_rate > 0 else 0
            eta_hours = eta_seconds / 3600
            
            # Calculate recent rate (since last update)
            recent_cells = self.processed_cells - self.last_update_cells
            recent_duration = current_time - self.last_update_time
            recent_rate = recent_cells / recent_duration if recent_duration > 0 else 0
            
            # Progress info
            progress_pct = self.processed_cells / self.total_cells * 100
            log.info(f"📊 Plate {self.plate_name} Progress: {self.processed_cells:,}/{self.total_cells:,} cells ({progress_pct:.1f}%)")
            
            # Performance breakdown
            log.info(f"⚡ Performance: Gene={percentages['gene_processing']:.1f}%, "
                    f"Data={percentages['data_loading']:.1f}%, "
                    f"Matrix={percentages['matrix_assignment']:.1f}%, "
                    f"Obs={percentages['obs_building']:.1f}%")
            
            # Rate and ETA info
            log.info(f"🏃 Rate: {overall_rate:.0f} cells/sec (recent: {recent_rate:.0f}), ETA: {eta_hours:.1f} hours")
            
            # Performance warnings
            self.check_performance_alerts(percentages, overall_rate)
            
            # Update tracking
            self.last_update_time = current_time
            self.last_update_cells = self.processed_cells
    
    def check_performance_alerts(self, percentages: Dict[str, float], rate: float):
        """Check for performance issues and log warnings."""
        # Alert if gene processing is taking >70% (indicates inefficiency)
        if percentages['gene_processing'] > 70:
            log.warning("⚠️  Gene processing taking >70% of time - optimization needed!")
        
        # Alert for very slow processing
        if rate < 500:
            log.warning(f"⚠️  Processing rate ({rate:.0f} cells/sec) is very slow - check system resources!")
        
        # Alert for extremely long ETA
        remaining_cells = self.total_cells - self.processed_cells
        eta_hours = (remaining_cells / rate / 3600) if rate > 0 else float('inf')
        if eta_hours > 24:
            log.warning(f"⚠️  ETA ({eta_hours:.1f} hours) exceeds 24 hours - consider optimization!")
    
    def finalize(self) -> Dict[str, float]:
        """Finalize tracking and return summary statistics."""
        total_duration = time.time() - self.start_time
        total_component_time = sum(self.timings.values())
        
        if total_component_time > 0:
            percentages = {k: v/total_component_time*100 for k, v in self.timings.items()}
            overall_rate = self.processed_cells / total_duration
            
            log.info(f"✅ Plate {self.plate_name} completed: {self.processed_cells:,} cells in {total_duration/3600:.2f} hours")
            log.info(f"📈 Final breakdown: Gene={percentages['gene_processing']:.1f}%, "
                    f"Data={percentages['data_loading']:.1f}%, "
                    f"Matrix={percentages['matrix_assignment']:.1f}%, "
                    f"Obs={percentages['obs_building']:.1f}%")
            log.info(f"🎯 Average rate: {overall_rate:.0f} cells/sec")
            
            return {
                'total_duration_hours': total_duration / 3600,
                'cells_per_second': overall_rate,
                **percentages
            }
        
        return {}


def load_hvg_mapping(token2hvg_path: str) -> Tuple[Dict[int, int], list]:
    """Load HVG token to column index mapping and gene names."""
    log.info(f"Loading HVG mapping from: {token2hvg_path}")
    df = pd.read_parquet(token2hvg_path)
    df = df.sort_values('token_id').reset_index(drop=True)
    token_to_col_idx = {tid: i for i, tid in enumerate(df['token_id'])}
    gene_names = df['gene_symbol'].tolist()
    
    log.info(f"Loaded {len(token_to_col_idx)} HVG genes")
    return token_to_col_idx, gene_names


def load_drug_dose_mapping() -> Dict[str, str]:
    """Load sample to drug dose mapping from HuggingFace."""
    log.info("Loading sample metadata from Hugging Face...")
    sample_ds = load_dataset("tahoebio/Tahoe-100M", "sample_metadata", split="train").to_pandas()
    sample_to_dose = dict(zip(sample_ds["sample"], sample_ds["drugname_drugconc"]))
    log.info(f"Loaded {len(sample_to_dose)} sample→dose mappings")
    return sample_to_dose


def should_skip_plate(out_dir: Path, plate_value: str) -> bool:
    """Check if plate output file already exists."""
    output_path = out_dir / f"plate_{plate_value}.h5ad"
    if output_path.exists():
        try:
            # Quick validation that file is readable
            adata = ad.read_h5ad(output_path, backed='r')
            log.info(f"✓ Plate {plate_value} already exists with {adata.n_obs:,} cells - skipping")
            return True
        except Exception as e:
            log.warning(f"Existing file for plate {plate_value} is corrupted ({e}) - will reprocess")
            return False
    return False


def detect_embedding_dimensions(parquet_path: str, column_name: str) -> int:
    """
    Detect embedding dimensions from parquet schema without loading data.
    
    Args:
        parquet_path: Path to parquet file(s) (can include wildcards)
        column_name: Name of the embedding column
        
    Returns:
        int: Number of dimensions in the embedding array
        
    Raises:
        ValueError: If column not found or not a fixed-size array
    """
    import glob
    
    # Handle wildcard paths
    if '*' in parquet_path:
        files = glob.glob(parquet_path)
        if not files:
            raise ValueError(f"No files found matching pattern: {parquet_path}")
        test_file = files[0]
    else:
        test_file = parquet_path
    
    try:
        # Read schema without loading data
        pf = pq.ParquetFile(test_file)
        schema = pf.schema_arrow
        
        # Find the embedding column
        for field in schema:
            if field.name == column_name:
                # Extract dimension from fixed_size_list type
                type_str = str(field.type)
                if 'fixed_size_list' in type_str:
                    # Parse dimension from string like "fixed_size_list<element: float>[2560]"
                    match = re.search(r'\[(\d+)\]', type_str)
                    if match:
                        return int(match.group(1))
                    else:
                        raise ValueError(f"Could not parse dimensions from type: {type_str}")
                else:
                    raise ValueError(f"Column '{column_name}' is not a fixed-size array: {type_str}")
        
        raise ValueError(f"Column '{column_name}' not found in schema. Available columns: {[f.name for f in schema]}")
        
    except Exception as e:
        raise ValueError(f"Failed to read schema from {test_file}: {e}")


def discover_plates(state_path: str) -> list:
    """Discover unique plate values from the state parquet files."""
    log.info("Discovering unique plates from state data...")
    with StringCache():
        state_lf = pl.scan_parquet(state_path)
        plates = state_lf.select(pl.col('plate')).unique().collect()['plate'].to_list()
    plates.sort()  # Process in consistent order
    log.info(f"Found {len(plates)} unique plates: {plates}")
    return plates


def process_chunk_with_monitoring(state_data, mosaic_data, token_to_col_idx: Dict[int, int], 
                                 gene_names: list, sample_to_dose: Dict[str, str], target_sum: float,
                                 tracker: PlatePerformanceTracker, mosaicfm_column: str, 
                                 mosaicfm_dim: int, state_dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Simple, fast processing with real-time performance monitoring (based on original fast script)."""
    n_hvg_genes = len(gene_names)
    chunk_size = len(state_data)
    
    # Pre-allocate output arrays
    hvg_matrix = np.zeros((chunk_size, n_hvg_genes), dtype=np.float32)
    mosaicfm_matrix = np.zeros((chunk_size, mosaicfm_dim), dtype=np.float32)
    state_matrix = np.zeros((chunk_size, state_dim), dtype=np.float32)
    obs_data = []
    
    # Simple, fast processing with detailed timing (like original script)
    for i, (s_row, m_row) in enumerate(zip(
        state_data.iter_rows(named=True), mosaic_data.iter_rows(named=True)
    )):
        # Data loading timing
        t0 = time.time()
        assert s_row['BARCODE_SUB_LIB_ID'] == m_row['BARCODE_SUB_LIB_ID'], \
            f"Mismatch: {s_row['BARCODE_SUB_LIB_ID']} != {m_row['BARCODE_SUB_LIB_ID']}"
        
        genes = s_row['genes']
        exprs = s_row['expressions']
        if len(exprs) > 0 and exprs[0] < 0:
            genes, exprs = genes[1:], exprs[1:]
        
        tracker.update_timing('data_loading', time.time() - t0, 0)
        
        # Gene processing timing (identical to fast original approach)
        t1 = time.time()
        lib_size = sum(exprs) if len(exprs) > 0 else 0
        hvg_vec = np.zeros(n_hvg_genes, dtype=np.float32)
        
        # Simple, fast gene processing (identical to original)
        for gene, expr in zip(genes, exprs):
            if gene in token_to_col_idx:
                hvg_vec[token_to_col_idx[gene]] = expr
        
        if lib_size > 0:
            hvg_vec *= target_sum / lib_size
        
        tracker.update_timing('gene_processing', time.time() - t1, 1)
        
        # Matrix assignment timing
        t2 = time.time()
        hvg_matrix[i] = hvg_vec
        mosaicfm_matrix[i] = np.array(m_row[mosaicfm_column], dtype=np.float32)
        state_matrix[i] = np.array(s_row['state_embeddings'], dtype=np.float32)
        tracker.update_timing('matrix_assignment', time.time() - t2, 0)
        
        # Obs building timing
        t3 = time.time()
        obs_row = {k: s_row[k] for k in s_row.keys() if k not in ['genes', 'expressions', 'state_embeddings']}
        obs_row['library_size'] = lib_size
        obs_row['drugname_drugconc'] = sample_to_dose.get(s_row.get('sample'), None)
        obs_data.append(obs_row)
        tracker.update_timing('obs_building', time.time() - t3, 0)
    
    return hvg_matrix, mosaicfm_matrix, state_matrix, obs_data




def estimate_plate_memory(state_path: str, mosaicfm_path: str, plate_value: str, n_hvg_genes: int, 
                         mosaicfm_dim: int, state_dim: int) -> Tuple[int, float]:
    """Estimate memory usage for a specific plate."""
    with StringCache():
        state_lf = pl.scan_parquet(state_path)
        plate_rows = state_lf.filter(pl.col('plate') == plate_value).select(pl.len()).collect().item()
        
    # Estimate memory: rows × (state_dim + mosaicfm_dim + actual HVG genes) × 4 bytes/float32
    # Plus overhead for obs data
    memory_gb = plate_rows * (state_dim + mosaicfm_dim + n_hvg_genes) * 4 / (1024**3)
    memory_gb *= 1.5  # Add 50% overhead for processing
    
    return plate_rows, memory_gb


def process_plate_chunked(state_lf, mosaicfm_lf, plate_value: str, token_to_col_idx: Dict[int, int], 
                         gene_names: list, sample_to_dose: Dict[str, str], target_sum: float,
                         sample_plate_dict: Dict[str, str], mosaicfm_column: str, 
                         mosaicfm_dim: int, state_dim: int, chunk_size: int = 1000000) -> ad.AnnData:
    """Process a large plate in chunks with real-time performance monitoring."""
    n_hvg_genes = len(gene_names)
    
    # Get filtered data for this plate once (more efficient)
    log.info(f"Filtering data for plate {plate_value}")
    plate_state_lf = state_lf.filter(pl.col('plate') == plate_value)
    
    # Filter MosaicFM data by samples that belong to this plate
    plate_samples = [sample for sample, plate in sample_plate_dict.items() if plate == plate_value]
    plate_mosaic_lf = mosaicfm_lf.filter(pl.col('sample').is_in(plate_samples))
    
    # Get total rows for this plate
    plate_rows = plate_state_lf.select(pl.len()).collect().item()
    log.info(f"Processing plate {plate_value} in chunks ({plate_rows:,} cells)")
    
    # Initialize performance tracker
    tracker = PlatePerformanceTracker(plate_rows, plate_value)
    
    # Pre-allocate final arrays (avoids memory accumulation)
    log.info(f"Pre-allocating arrays for {plate_rows:,} cells")
    final_hvg = np.zeros((plate_rows, n_hvg_genes), dtype=np.float32)
    final_mosaicfm = np.zeros((plate_rows, mosaicfm_dim), dtype=np.float32)
    final_state = np.zeros((plate_rows, state_dim), dtype=np.float32)
    all_obs_data = [None] * plate_rows  # Pre-sized list
    
    # Process in chunks with position tracking
    current_row = 0
    chunk_pbar = tqdm(total=plate_rows, desc=f"Plate {plate_value}", unit="cells", leave=False)
    
    while current_row < plate_rows:
        chunk_end = min(current_row + chunk_size, plate_rows)
        actual_chunk_size = chunk_end - current_row
        
        chunk_start_time = time.time()
        
        # Get data for this chunk (slice the already-filtered data)
        state_batch = plate_state_lf.slice(current_row, actual_chunk_size).collect()
        mosaic_batch = plate_mosaic_lf.slice(current_row, actual_chunk_size).collect()
        
        # Validate chunk sizes match
        if len(state_batch) != actual_chunk_size or len(mosaic_batch) != actual_chunk_size:
            raise ValueError(f"Chunk size mismatch: expected {actual_chunk_size}, got state={len(state_batch)}, mosaic={len(mosaic_batch)}")
        if len(state_batch) != len(mosaic_batch):
            raise ValueError(f"Batch size mismatch: state={len(state_batch)}, mosaic={len(mosaic_batch)}")
        
        # Process chunk using simple, fast monitored function
        hvg_chunk, mosaicfm_chunk, state_chunk, obs_chunk = process_chunk_with_monitoring(
            state_batch, mosaic_batch, token_to_col_idx, gene_names, sample_to_dose, target_sum, tracker, mosaicfm_column, mosaicfm_dim, state_dim
        )
        
        # Validate chunk processing results
        if hvg_chunk.shape[0] != actual_chunk_size:
            raise ValueError(f"HVG chunk size mismatch: expected {actual_chunk_size}, got {hvg_chunk.shape[0]}")
        
        # Copy chunk data directly to final arrays (no accumulation)
        final_hvg[current_row:chunk_end] = hvg_chunk
        final_mosaicfm[current_row:chunk_end] = mosaicfm_chunk
        final_state[current_row:chunk_end] = state_chunk
        
        # Copy obs data to pre-sized list
        for obs_idx, obs_row in enumerate(obs_chunk):
            all_obs_data[current_row + obs_idx] = obs_row
        
        current_row = chunk_end
        chunk_pbar.update(actual_chunk_size)
        
        chunk_duration = time.time() - chunk_start_time
        chunk_rate = actual_chunk_size / chunk_duration
        log.info(f"Chunk completed: {chunk_rate:.0f} cells/sec")
    
    chunk_pbar.close()
    
    # Finalize performance tracking
    performance_stats = tracker.finalize()
    log.info(f"Completed chunked processing for plate {plate_value}")
    
    # Create AnnData (no expensive concatenation needed)
    X_dummy = sparse.csr_matrix((plate_rows, n_hvg_genes))
    adata = ad.AnnData(X=X_dummy, obs=pd.DataFrame(all_obs_data))
    adata.var_names = gene_names
    adata.var['gene_symbol'] = gene_names
    adata.obsm['X_hvg'] = final_hvg
    adata.obsm['mosaicfm-70m-merged'] = final_mosaicfm
    adata.obsm['state-SE-600M'] = final_state
    
    return adata


def process_plate_whole(state_lf, mosaicfm_lf, plate_value: str, token_to_col_idx: Dict[int, int],
                       gene_names: list, sample_to_dose: Dict[str, str], target_sum: float,
                       sample_plate_dict: Dict[str, str], mosaicfm_column: str, 
                       mosaicfm_dim: int, state_dim: int) -> ad.AnnData:
    """Process entire plate in memory with real-time performance monitoring."""
    n_hvg_genes = len(gene_names)
    
    log.info(f"Loading entire plate {plate_value} into memory")
    state_batch = state_lf.filter(pl.col('plate') == plate_value).collect()
    
    # Filter MosaicFM data by samples that belong to this plate
    plate_samples = [sample for sample, plate in sample_plate_dict.items() if plate == plate_value]
    mosaic_batch = mosaicfm_lf.filter(pl.col('sample').is_in(plate_samples)).collect()
    
    plate_rows = len(state_batch)
    log.info(f"Processing plate {plate_value} ({plate_rows:,} cells)")
    
    # Validate data sizes match
    if len(state_batch) != len(mosaic_batch):
        raise ValueError(f"Batch size mismatch: state={len(state_batch)}, mosaic={len(mosaic_batch)}")
    
    # Initialize performance tracker
    tracker = PlatePerformanceTracker(plate_rows, plate_value)
    
    # Process entire plate using simple, fast monitored function
    hvg_matrix, mosaicfm_matrix, state_matrix, obs_data = process_chunk_with_monitoring(
        state_batch, mosaic_batch, token_to_col_idx, gene_names, sample_to_dose, target_sum, tracker, mosaicfm_column, mosaicfm_dim, state_dim
    )
    
    # Finalize performance tracking
    performance_stats = tracker.finalize()
    log.info(f"Completed whole-plate processing for plate {plate_value}")
    
    # Create AnnData
    X_dummy = sparse.csr_matrix((plate_rows, n_hvg_genes))
    adata = ad.AnnData(X=X_dummy, obs=pd.DataFrame(obs_data))
    adata.var_names = gene_names
    adata.var['gene_symbol'] = gene_names
    adata.obsm['X_hvg'] = hvg_matrix
    adata.obsm['mosaicfm-70m-merged'] = mosaicfm_matrix
    adata.obsm['state-SE-600M'] = state_matrix
    
    return adata


def main(cfg: DictConfig, target_plate: Optional[str] = None, list_plates: bool = False):
    """
    Main processing function with support for single plate processing and parallel execution.
    
    Args:
        cfg: Configuration from YAML file
        target_plate: Optional specific plate name to process (e.g., "plate11")
        list_plates: If True, list available plates and exit
    """
    # Set default MosaicFM column name if not specified in config
    mosaicfm_column = cfg.get('mosaicfm_column_name', 'mosaicfm-70m-merged')
    log.info(f"Using MosaicFM column: {mosaicfm_column}")
    
    # Auto-detect embedding dimensions from parquet schemas
    log.info("Auto-detecting embedding dimensions from parquet schemas...")
    try:
        mosaicfm_dim = detect_embedding_dimensions(cfg.mosaicfm_path, mosaicfm_column)
        state_dim = detect_embedding_dimensions(cfg.state_path, 'state_embeddings')
        log.info(f"Detected dimensions - MosaicFM: {mosaicfm_dim}, State: {state_dim}")
    except Exception as e:
        log.error(f"Failed to auto-detect dimensions: {e}")
        log.info("Falling back to hardcoded dimensions: MosaicFM=512, State=2048")
        mosaicfm_dim = 512
        state_dim = 2048
    # Setup output directory
    out_dir = Path(cfg.out_dir) / "by_plate"
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {out_dir}")
    
    # Load mappings
    log.info("Loading HVG and drug dose mappings...")
    token_to_col_idx, gene_names = load_hvg_mapping(cfg.token2hvg_path)
    sample_to_dose = load_drug_dose_mapping()
    
    # Discover plates
    plates = discover_plates(cfg.state_path)
    
    # Handle plate listing
    if list_plates:
        print("\n🧬 Available plates for processing:")
        print("=" * 50)
        for i, plate in enumerate(plates, 1):
            # Quick check if already processed
            output_path = out_dir / f"plate_{plate}.h5ad"
            status = "✅ Completed" if output_path.exists() else "⏳ Pending"
            print(f"  {i:2d}. {plate:<12} {status}")
        print("=" * 50)
        print(f"Total: {len(plates)} plates")
        return
    
    # Handle single plate processing
    if target_plate:
        if target_plate not in plates:
            log.error(f"❌ Plate '{target_plate}' not found!")
            log.error(f"Available plates: {', '.join(plates)}")
            sys.exit(1)
        
        plates = [target_plate]  # Process only the specified plate
        log.info(f"🎯 Single plate mode: Processing only '{target_plate}'")
        log.info(f"💡 This enables parallel processing across multiple machines")
    else:
        log.info(f"🔄 Processing all {len(plates)} plates sequentially")
    
    # Setup lazy frames
    log.info("Setting up lazy parquet readers...")
    with StringCache():
        state_lf = pl.scan_parquet(cfg.state_path)
        
        # Create sample-to-plate mapping since MosaicFM data doesn't have 'plate' column
        log.info("Creating sample-to-plate mapping...")
        sample_to_plate = state_lf.select(["sample", "plate"]).unique().collect()
        sample_plate_dict = dict(zip(sample_to_plate["sample"], sample_to_plate["plate"]))
        log.info(f"Created mapping for {len(sample_plate_dict)} samples across {len(plates)} plates")
        
        mosaicfm_lf = pl.scan_parquet(cfg.mosaicfm_path).select([
            "BARCODE_SUB_LIB_ID", "sample", mosaicfm_column
        ])
        
        # Process each plate
        overall_pbar = tqdm(total=len(plates), desc="Overall Progress", unit="plates")
        
        for plate_idx, plate_value in enumerate(plates):
            log.info(f"Checking plate {plate_idx + 1}/{len(plates)}: {plate_value}")
            
            # Check if plate already exists FIRST (before any expensive operations)
            if should_skip_plate(out_dir, plate_value):
                overall_pbar.update(1)
                continue
            
            log.info(f"Processing plate {plate_value}...")
            
            # Only estimate memory requirements if we need to process the plate
            plate_rows, memory_gb = estimate_plate_memory(cfg.state_path, cfg.mosaicfm_path, plate_value, len(gene_names), mosaicfm_dim, state_dim)
            log.info(f"Plate {plate_value}: {plate_rows:,} cells, estimated {memory_gb:.1f} GB memory")
            
            # Choose processing strategy based on memory
            if memory_gb > 200:  # Use chunked processing for large plates
                log.info(f"Using chunked processing (memory: {memory_gb:.1f} GB)")
                adata = process_plate_chunked(
                    state_lf, mosaicfm_lf, plate_value, token_to_col_idx, 
                    gene_names, sample_to_dose, cfg.target_sum, sample_plate_dict, mosaicfm_column, 
                    mosaicfm_dim, state_dim, cfg.get('chunk_size', 1000000)
                )
            else:
                log.info(f"Using whole-plate processing (memory: {memory_gb:.1f} GB)")
                adata = process_plate_whole(
                    state_lf, mosaicfm_lf, plate_value, token_to_col_idx,
                    gene_names, sample_to_dose, cfg.target_sum, sample_plate_dict, mosaicfm_column, 
                    mosaicfm_dim, state_dim
                )
            
            # Save plate
            output_path = out_dir / f"plate_{plate_value}.h5ad"
            log.info(f"Saving plate {plate_value} to {output_path}")
            adata.write_h5ad(output_path)
            
            overall_pbar.update(1)
            log.info(f"✓ Completed plate {plate_value} ({plate_idx + 1}/{len(plates)})")
        
        overall_pbar.close()
    
    log.info(f"All {len(plates)} plates processed successfully!")


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Load configuration
    cfg = om.load(args.config_file)
    om.resolve(cfg)
    
    # Log startup information
    if args.plate:
        log.info(f"🚀 Starting single plate processing: {args.plate}")
    elif args.list_plates:
        log.info("📋 Listing available plates")
    else:
        log.info("🚀 Starting full pipeline processing")
    
    # Run main processing
    main(cfg, target_plate=args.plate, list_plates=args.list_plates)
    
    # Log completion
    if not args.list_plates:
        completion_msg = f"✅ Single plate '{args.plate}' processing completed!" if args.plate else "✅ Full pipeline processing completed!"
        log.info(completion_msg)