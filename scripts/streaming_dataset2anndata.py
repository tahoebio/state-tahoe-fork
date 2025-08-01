#!/usr/bin/env python3
"""
Convert streaming HuggingFace Tahoe-100M dataset to h5ad format with gene filtering.

This script converts the streaming vevotx/Tahoe-100M dataset to AnnData h5ad format,
filtering to only the 2000 highly variable genes and applying log transformation.
"""

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import anndata as ad
import numpy as np
import pandas as pd
from datasets import load_dataset
from scipy import sparse
from tqdm import tqdm

# Silence anndata warnings that break progress bar display
warnings.filterwarnings("ignore", category=ad.ImplicitModificationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_hvg_mapping(token2hvg_path: str) -> Tuple[Dict[int, int], List[str]]:
    """Load HVG token mapping and return token_id to column index mapping and gene names."""
    logger.info(f"Loading HVG mapping from {token2hvg_path}")
    
    hvg_df = pd.read_parquet(token2hvg_path)
    logger.info(f"Loaded {len(hvg_df)} HVG mappings")
    
    # Sort by token_id to ensure consistent ordering
    hvg_df = hvg_df.sort_values('token_id').reset_index(drop=True)
    
    # Create mapping from token_id to column index (0-1999)
    token_to_col_idx = {token_id: idx for idx, token_id in enumerate(hvg_df['token_id'])}
    gene_names = hvg_df['gene_symbol'].tolist()
    
    logger.info(f"Created mapping for {len(token_to_col_idx)} genes")
    return token_to_col_idx, gene_names


def process_cell_batch(batch_data: List[dict], token_to_col_idx: Dict[int, int], 
                      n_hvg_genes: int, apply_log: bool = True) -> Tuple[np.ndarray, List[float], List[dict]]:
    """Process a batch of cells and return HVG expression matrix, library sizes, and obs data."""
    batch_size = len(batch_data)
    hvg_matrix = np.zeros((batch_size, n_hvg_genes), dtype=np.float32)
    library_sizes = []
    obs_data = []
    
    for i, cell in enumerate(batch_data):
        genes = cell['genes']
        expressions = cell['expressions']
        
        # Handle negative expressions[0] values (skip first gene/expression if negative)
        if expressions[0] < 0:
            genes = genes[1:]
            expressions = expressions[1:]
        
        # Calculate library size (sum of all expressions for this cell)
        library_size = sum(expressions)
        library_sizes.append(library_size)
        
        # Filter to only HVG genes and populate matrix
        for gene_token, expr in zip(genes, expressions):
            if gene_token in token_to_col_idx:
                col_idx = token_to_col_idx[gene_token]
                hvg_matrix[i, col_idx] = expr
        
        # Apply normalization and optional log transformation
        if library_size > 0:
            hvg_matrix[i] = 1e4 * hvg_matrix[i] / library_size
            if apply_log:
                hvg_matrix[i] = np.log10(hvg_matrix[i] + 1)
        
        # Extract observation metadata (exclude genes and expressions)
        obs_entry = {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
        obs_entry['library_size'] = library_size
        obs_data.append(obs_entry)
    
    return hvg_matrix, library_sizes, obs_data


def convert_streaming_to_anndata(dataset_name: str = "vevotx/Tahoe-100M",
                                token2hvg_path: str = "/tahoe/drive_3/ANALYSIS/analysis_190/Data/token2hvg.parquet",
                                batch_size: int = 1000,
                                max_cells: Optional[int] = None,
                                apply_log: bool = True) -> ad.AnnData:
    """Convert streaming HuggingFace dataset to AnnData format."""
    logger.info(f"Starting conversion of {dataset_name}")
    
    # Load HVG mapping
    token_to_col_idx, gene_names = load_hvg_mapping(token2hvg_path)
    n_hvg_genes = len(gene_names)
    logger.info(f"Will filter to {n_hvg_genes} HVG genes")
    
    # Load streaming dataset with local cache
    cache_dir = "/tahoe/drive_3/ANALYSIS/analysis_216/Data/hf_cache/"
    logger.info(f"Loading streaming dataset from cache: {cache_dir}")
    streaming_dataset = load_dataset(
        dataset_name, 
        streaming=True, 
        split='train',
        cache_dir=cache_dir
    )
    
    # Process in batches
    all_hvg_matrices = []
    all_obs_data = []
    total_cells_processed = 0
    
    batch_data = []
    
    logger.info("Processing cells in batches...")
    for cell in tqdm(streaming_dataset, desc="Processing cells"):
        batch_data.append(cell)
        
        # Process batch when it reaches batch_size
        if len(batch_data) >= batch_size:
            hvg_matrix, library_sizes, obs_data = process_cell_batch(
                batch_data, token_to_col_idx, n_hvg_genes, apply_log
            )
            
            all_hvg_matrices.append(hvg_matrix)
            all_obs_data.extend(obs_data)
            total_cells_processed += len(batch_data)
            
            logger.info(f"Processed {total_cells_processed} cells")
            
            # Clear batch data for memory efficiency
            batch_data = []
            gc.collect()
            
            # Check if we've reached the maximum number of cells
            if max_cells is not None and total_cells_processed >= max_cells:
                logger.info(f"Reached maximum cells limit: {max_cells}")
                break
    
    # Process remaining cells in final batch
    if batch_data:
        hvg_matrix, library_sizes, obs_data = process_cell_batch(
            batch_data, token_to_col_idx, n_hvg_genes, apply_log
        )
        all_hvg_matrices.append(hvg_matrix)
        all_obs_data.extend(obs_data)
        total_cells_processed += len(batch_data)
    
    logger.info(f"Total cells processed: {total_cells_processed}")
    
    # Concatenate all batches
    logger.info("Concatenating all batches...")
    final_hvg_matrix = np.vstack(all_hvg_matrices)
    
    # Create observation DataFrame
    obs_df = pd.DataFrame(all_obs_data)
    obs_df['batch'] = '1'  # Add dummy batch column for compatibility
    
    # Create AnnData object
    logger.info("Creating AnnData object...")
    
    # Create dummy sparse matrix for .X (same dimensions as HVG matrix)
    X_sparse = sparse.csr_matrix((total_cells_processed, n_hvg_genes))
    
    # Create AnnData object
    adata = ad.AnnData(X=X_sparse, obs=obs_df)
    
    # Store HVG expression data in .obsm
    adata.obsm['X_hvg'] = final_hvg_matrix
    
    # Set gene names for var (even though X is dummy)
    adata.var_names = gene_names
    adata.var['gene_symbol'] = gene_names
    
    logger.info(f"AnnData created with shape: {adata.shape}")
    logger.info(f"X_hvg shape: {adata.obsm['X_hvg'].shape}")
    logger.info(f"Observation columns: {list(adata.obs.columns)}")
    logger.info(f"X matrix type: {type(adata.X)}")
    
    return adata


def save_anndata(adata: ad.AnnData, output_path: str):
    """Save AnnData object to h5ad file."""
    logger.info(f"Saving AnnData to {output_path}")
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    adata.write_h5ad(output_path)
    logger.info(f"Successfully saved h5ad file to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert streaming HuggingFace Tahoe-100M dataset to h5ad format with HVG filtering"
    )
    parser.add_argument(
        "output_path",
        help="Output path for h5ad file"
    )
    parser.add_argument(
        "--dataset-name",
        default="vevotx/Tahoe-100M",
        help="HuggingFace dataset name (default: vevotx/Tahoe-100M)"
    )
    parser.add_argument(
        "--token2hvg-path",
        default="/tahoe/drive_3/ANALYSIS/analysis_190/Data/token2hvg.parquet",
        help="Path to token2hvg.parquet file"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Process cells in batches of this size (default: 1000)"
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        help="Maximum number of cells to process (useful for testing)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Skip log transformation, store raw normalized values (1e4 * X / library_size)"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate token2hvg path
    if not Path(args.token2hvg_path).exists():
        logger.error(f"Token2HVG path does not exist: {args.token2hvg_path}")
        sys.exit(1)
    
    try:
        # Convert streaming dataset to AnnData
        adata = convert_streaming_to_anndata(
            dataset_name=args.dataset_name,
            token2hvg_path=args.token2hvg_path,
            batch_size=args.batch_size,
            max_cells=args.max_cells,
            apply_log=not args.no_log
        )
        
        # Save to h5ad
        save_anndata(adata, args.output_path)
        
        logger.info("Conversion completed successfully!")
        
        # Print summary statistics
        logger.info(f"Final dataset shape: {adata.shape}")
        logger.info(f"HVG matrix shape: {adata.obsm['X_hvg'].shape}")
        logger.info(f"Library size statistics:")
        logger.info(f"  Mean: {adata.obs['library_size'].mean():.2f}")
        logger.info(f"  Median: {adata.obs['library_size'].median():.2f}")
        logger.info(f"  Min: {adata.obs['library_size'].min()}")
        logger.info(f"  Max: {adata.obs['library_size'].max()}")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()