#!/usr/bin/env python3
"""
Reverse log10(x+1) transformation on X_hvg in .h5ad file in-place.

This script modifies the X_hvg data in an .h5ad file by applying the transformation:
    x = (10^x) - 1

The transformation is done in chunks to handle very large files efficiently.
"""
import argparse
import h5py
import numpy as np
from pathlib import Path
from tqdm import tqdm
import logging
import sys

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)


def unlog_h5ad_inplace(h5ad_path: str, chunk_size: int = 6_000_000):
    """
    Apply (10^x - 1) transformation to X_hvg data in-place.
    
    Args:
        h5ad_path: Path to .h5ad file to modify
        chunk_size: Number of rows to process at once (default 6M = ~46GB)
    """
    h5ad_path = Path(h5ad_path)
    
    if not h5ad_path.exists():
        raise FileNotFoundError(f"File not found: {h5ad_path}")
    
    logger.info(f"Opening file: {h5ad_path}")
    logger.info(f"Chunk size: {chunk_size:,} rows")
    
    # Open file in read-write mode
    with h5py.File(h5ad_path, 'r+') as f:
        # Access X_hvg dataset
        if '/obsm/X_hvg' not in f:
            raise KeyError("X_hvg not found in /obsm/X_hvg")
        
        dataset = f['/obsm/X_hvg']
        total_rows, n_features = dataset.shape
        
        logger.info(f"X_hvg shape: {total_rows:,} × {n_features:,}")
        logger.info(f"Data type: {dataset.dtype}")
        
        # Sample first few values to show transformation
        logger.info("\nSampling first 5 values before transformation:")
        sample_before = dataset[0, :5]
        logger.info(f"  Before: {sample_before}")
        logger.info(f"  After:  {(10 ** sample_before) - 1}")
        
        # Process in chunks
        n_chunks = (total_rows + chunk_size - 1) // chunk_size
        logger.info(f"\nProcessing {n_chunks} chunks...")
        
        for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, total_rows)
            chunk_rows = end - start
            
            # Read chunk
            chunk = dataset[start:end, :]
            
            # Apply transformation: reverse log10(x + 1)
            chunk = (10 ** chunk) - 1
            
            # Handle any potential numerical issues
            chunk = np.nan_to_num(chunk, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Write back to same location
            dataset[start:end, :] = chunk
            
            # Log progress every 10 chunks
            if chunk_idx % 10 == 0:
                logger.info(f"  Processed {end:,} / {total_rows:,} rows ({100*end/total_rows:.1f}%)")
        
        # Verify transformation on same sample
        logger.info("\nVerifying transformation on first 5 values:")
        sample_after = dataset[0, :5]
        logger.info(f"  Final values: {sample_after}")
        
        # Flush changes
        f.flush()
    
    logger.info(f"\n✓ Successfully transformed {total_rows:,} rows in {h5ad_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Reverse log10(x+1) transformation on X_hvg in .h5ad file"
    )
    parser.add_argument(
        "h5ad_path",
        help="Path to .h5ad file to modify in-place"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=6_000_000,
        help="Number of rows to process at once (default: 6000000, ~46GB per chunk)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without modifying the file"
    )
    
    args = parser.parse_args()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - File will not be modified")
        h5ad_path = Path(args.h5ad_path)
        if not h5ad_path.exists():
            logger.error(f"File not found: {h5ad_path}")
            sys.exit(1)
            
        with h5py.File(h5ad_path, 'r') as f:
            if '/obsm/X_hvg' not in f:
                logger.error("X_hvg not found in /obsm/X_hvg")
                sys.exit(1)
            
            dataset = f['/obsm/X_hvg']
            total_rows, n_features = dataset.shape
            n_chunks = (total_rows + args.chunk_size - 1) // args.chunk_size
            
            logger.info(f"Would process file: {h5ad_path}")
            logger.info(f"X_hvg shape: {total_rows:,} × {n_features:,}")
            logger.info(f"Chunk size: {args.chunk_size:,} rows")
            logger.info(f"Number of chunks: {n_chunks}")
            logger.info(f"Memory per chunk: ~{args.chunk_size * n_features * 4 / 1e9:.1f} GB")
            
            sample = dataset[0, :5]
            logger.info(f"\nSample transformation (first 5 values):")
            logger.info(f"  Current: {sample}")
            logger.info(f"  Would become: {(10 ** sample) - 1}")
    else:
        try:
            unlog_h5ad_inplace(args.h5ad_path, args.chunk_size)
        except Exception as e:
            logger.error(f"Error: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()