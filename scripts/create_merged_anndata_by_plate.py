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
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Tuple

import anndata as ad
import numpy as np
import pandas as pd
import polars as pl
from polars import StringCache
from omegaconf import OmegaConf as om, DictConfig
from scipy import sparse
from tqdm import tqdm
from datasets import load_dataset

# === Logging Setup ===
log = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s: [%(process)d][%(threadName)s]: %(levelname)s: %(name)s: %(message)s",
    level=logging.INFO,
)


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


def discover_plates(state_path: str) -> list:
    """Discover unique plate values from the state parquet files."""
    log.info("Discovering unique plates from state data...")
    with StringCache():
        state_lf = pl.scan_parquet(state_path)
        plates = state_lf.select(pl.col('plate')).unique().collect()['plate'].to_list()
    plates.sort()  # Process in consistent order
    log.info(f"Found {len(plates)} unique plates: {plates}")
    return plates


def process_chunk_vectorized(state_data, mosaic_data, token_to_col_idx: Dict[int, int], 
                            gene_names: list, sample_to_dose: Dict[str, str], target_sum: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Vectorized processing of chunk data for efficient HVG matrix construction."""
    n_hvg_genes = len(gene_names)
    chunk_size = len(state_data)
    
    # Pre-allocate output arrays
    hvg_matrix = np.zeros((chunk_size, n_hvg_genes), dtype=np.float32)
    mosaicfm_matrix = np.zeros((chunk_size, 512), dtype=np.float32)
    state_matrix = np.zeros((chunk_size, 2048), dtype=np.float32)
    obs_data = []
    
    # Extract data in bulk
    for i, (s_row, m_row) in enumerate(zip(
        state_data.iter_rows(named=True), mosaic_data.iter_rows(named=True)
    )):
        # Data alignment validation
        assert s_row['BARCODE_SUB_LIB_ID'] == m_row['BARCODE_SUB_LIB_ID'], \
            f"Mismatch: {s_row['BARCODE_SUB_LIB_ID']} != {m_row['BARCODE_SUB_LIB_ID']}"
        
        # Process gene expression - vectorized approach
        genes = np.array(s_row['genes'])
        exprs = np.array(s_row['expressions'])
        
        # Handle negative first values
        if len(exprs) > 0 and exprs[0] < 0:
            genes, exprs = genes[1:], exprs[1:]
        
        lib_size = exprs.sum() if len(exprs) > 0 else 0
        
        # Vectorized gene mapping and assignment
        if len(genes) > 0 and lib_size > 0:
            # Find valid genes that exist in HVG mapping
            valid_genes = []
            valid_exprs = []
            valid_indices = []
            
            for j, gene in enumerate(genes):
                if gene in token_to_col_idx:
                    col_idx = token_to_col_idx[gene]
                    # SAFETY CHECK: Ensure index is within bounds
                    if col_idx < n_hvg_genes:
                        valid_genes.append(gene)
                        valid_exprs.append(exprs[j])
                        valid_indices.append(col_idx)
                    else:
                        log.warning(f"Gene {gene} index {col_idx} exceeds HVG matrix bounds {n_hvg_genes}")
            
            if valid_indices:
                # Assign expressions to HVG matrix
                valid_exprs = np.array(valid_exprs, dtype=np.float32)
                hvg_matrix[i, valid_indices] = valid_exprs * (target_sum / lib_size)
        
        # Build obs row with drug dose info
        obs_row = {k: s_row[k] for k in s_row.keys() if k not in ['genes', 'expressions', 'state_embeddings']}
        obs_row['library_size'] = lib_size
        
        # Add drug dose mapping with logging for missing samples
        sample_id = s_row.get('sample')
        drug_dose = sample_to_dose.get(sample_id)
        if drug_dose is None and sample_id is not None:
            log.debug(f"No drug dose mapping found for sample: {sample_id}")
        obs_row['drugname_drugconc'] = drug_dose
        
        # Assign to matrices
        mosaicfm_matrix[i] = np.array(m_row['mosaicfm-70m-merged'], dtype=np.float32)
        state_matrix[i] = np.array(s_row['state_embeddings'], dtype=np.float32)
        obs_data.append(obs_row)
    
    return hvg_matrix, mosaicfm_matrix, state_matrix, obs_data


def estimate_plate_memory(state_path: str, mosaicfm_path: str, plate_value: str, n_hvg_genes: int) -> Tuple[int, float]:
    """Estimate memory usage for a specific plate."""
    with StringCache():
        state_lf = pl.scan_parquet(state_path)
        plate_rows = state_lf.filter(pl.col('plate') == plate_value).select(pl.len()).collect().item()
        
    # Estimate memory: rows × (2048 + 512 + actual HVG genes) × 4 bytes/float32
    # Plus overhead for obs data
    memory_gb = plate_rows * (2048 + 512 + n_hvg_genes) * 4 / (1024**3)
    memory_gb *= 1.5  # Add 50% overhead for processing
    
    return plate_rows, memory_gb


def process_plate_chunked(state_lf, mosaicfm_lf, plate_value: str, token_to_col_idx: Dict[int, int], 
                         gene_names: list, sample_to_dose: Dict[str, str], target_sum: float,
                         sample_plate_dict: Dict[str, str], chunk_size: int = 1000000) -> ad.AnnData:
    """Process a large plate in chunks with pre-allocated arrays to manage memory efficiently."""
    n_hvg_genes = len(gene_names)
    
    # Get filtered data for this plate once (more efficient)
    log.info(f"Filtering data for plate {plate_value}")
    plate_state_lf = state_lf.filter(pl.col('plate') == plate_value)
    
    # Filter MosaicFM data by samples that belong to this plate
    plate_samples = [sample for sample, plate in sample_plate_dict.items() if plate == plate_value]
    plate_mosaic_lf = mosaicfm_lf.filter(pl.col('sample').is_in(plate_samples))
    
    # ASSUMPTION: Both parquet datasets have consistent ordering by BARCODE_SUB_LIB_ID
    # This allows us to iterate through filtered data positionally without explicit sorting
    
    # Get total rows for this plate
    plate_rows = plate_state_lf.select(pl.len()).collect().item()
    log.info(f"Processing plate {plate_value} in chunks ({plate_rows:,} cells)")
    
    # Pre-allocate final arrays (avoids memory accumulation)
    log.info(f"Pre-allocating arrays for {plate_rows:,} cells")
    final_hvg = np.zeros((plate_rows, n_hvg_genes), dtype=np.float32)
    final_mosaicfm = np.zeros((plate_rows, 512), dtype=np.float32)
    final_state = np.zeros((plate_rows, 2048), dtype=np.float32)
    all_obs_data = [None] * plate_rows  # Pre-sized list
    
    # Process in chunks with position tracking
    current_row = 0
    chunk_pbar = tqdm(total=plate_rows, desc=f"Plate {plate_value}", unit="cells", leave=False)
    
    while current_row < plate_rows:
        chunk_end = min(current_row + chunk_size, plate_rows)
        actual_chunk_size = chunk_end - current_row
        
        # Get data for this chunk (slice the already-filtered data)
        state_batch = plate_state_lf.slice(current_row, actual_chunk_size).collect()
        mosaic_batch = plate_mosaic_lf.slice(current_row, actual_chunk_size).collect()
        
        # Validate chunk sizes match
        if len(state_batch) != actual_chunk_size or len(mosaic_batch) != actual_chunk_size:
            raise ValueError(f"Chunk size mismatch: expected {actual_chunk_size}, got state={len(state_batch)}, mosaic={len(mosaic_batch)}")
        if len(state_batch) != len(mosaic_batch):
            raise ValueError(f"Batch size mismatch: state={len(state_batch)}, mosaic={len(mosaic_batch)}")
        
        # Process chunk using vectorized function
        hvg_chunk, mosaicfm_chunk, state_chunk, obs_chunk = process_chunk_vectorized(
            state_batch, mosaic_batch, token_to_col_idx, gene_names, sample_to_dose, target_sum
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
    
    chunk_pbar.close()
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
                       sample_plate_dict: Dict[str, str]) -> ad.AnnData:
    """Process entire plate in memory (for smaller plates)."""
    n_hvg_genes = len(gene_names)
    
    log.info(f"Loading entire plate {plate_value} into memory")
    # ASSUMPTION: Both parquet datasets have consistent ordering by BARCODE_SUB_LIB_ID
    # This allows us to iterate through filtered data positionally without explicit sorting
    state_batch = state_lf.filter(pl.col('plate') == plate_value).collect()
    
    # Filter MosaicFM data by samples that belong to this plate
    plate_samples = [sample for sample, plate in sample_plate_dict.items() if plate == plate_value]
    mosaic_batch = mosaicfm_lf.filter(pl.col('sample').is_in(plate_samples)).collect()
    
    plate_rows = len(state_batch)
    log.info(f"Processing plate {plate_value} ({plate_rows:,} cells)")
    
    # Validate data sizes match
    if len(state_batch) != len(mosaic_batch):
        raise ValueError(f"Batch size mismatch: state={len(state_batch)}, mosaic={len(mosaic_batch)}")
    
    # Process entire plate using vectorized function
    hvg_matrix, mosaicfm_matrix, state_matrix, obs_data = process_chunk_vectorized(
        state_batch, mosaic_batch, token_to_col_idx, gene_names, sample_to_dose, target_sum
    )
    
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


def main(cfg: DictConfig):
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
            "BARCODE_SUB_LIB_ID", "sample", "mosaicfm-70m-merged"
        ])
        
        # Process each plate
        overall_pbar = tqdm(total=len(plates), desc="Overall Progress", unit="plates")
        
        for plate_idx, plate_value in enumerate(plates):
            log.info(f"Processing plate {plate_idx + 1}/{len(plates)}: {plate_value}")
            
            # Check if plate already exists (skip if complete)
            if should_skip_plate(out_dir, plate_value):
                overall_pbar.update(1)
                continue
            
            # Estimate memory requirements
            plate_rows, memory_gb = estimate_plate_memory(cfg.state_path, cfg.mosaicfm_path, plate_value, len(gene_names))
            log.info(f"Plate {plate_value}: {plate_rows:,} cells, estimated {memory_gb:.1f} GB memory")
            
            # Choose processing strategy based on memory
            if memory_gb > 200:  # Use chunked processing for large plates
                log.info(f"Using chunked processing (memory: {memory_gb:.1f} GB)")
                adata = process_plate_chunked(
                    state_lf, mosaicfm_lf, plate_value, token_to_col_idx, 
                    gene_names, sample_to_dose, cfg.target_sum, sample_plate_dict, cfg.get('chunk_size', 1000000)
                )
            else:
                log.info(f"Using whole-plate processing (memory: {memory_gb:.1f} GB)")
                adata = process_plate_whole(
                    state_lf, mosaicfm_lf, plate_value, token_to_col_idx,
                    gene_names, sample_to_dose, cfg.target_sum, sample_plate_dict
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
    yaml_path = sys.argv[1]
    cfg = om.load(yaml_path)
    om.resolve(cfg)
    
    main(cfg)
    log.info("Script execution completed.")