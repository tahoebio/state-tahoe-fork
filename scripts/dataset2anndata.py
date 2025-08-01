#!/usr/bin/env python3
"""
Convert HuggingFace dataset to h5ad format for compatibility with state transition model.

This script converts the HuggingFace dataset format used by Barotaxis to AnnData h5ad format
that can be used with the state transition model, enabling direct comparison between models.
"""

import argparse
import gc
import logging
import random
import sys
import warnings
from pathlib import Path
from typing import Optional

import anndata as ad
import numpy as np
import pandas as pd
from datasets import load_from_disk
from scipy import sparse
from tqdm import tqdm

# Silence anndata warnings that break progress bar display
warnings.filterwarnings("ignore", category=ad.ImplicitModificationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_hf_dataset(dataset_path: str, limit: Optional[int] = None, seed: int = 42):
    """Load HuggingFace dataset from disk."""
    logger.info(f"Loading HuggingFace dataset from {dataset_path}")
    
    dataset = load_from_disk(dataset_path)
    
    if limit is not None:
        total_samples = len(dataset)
        limit = min(limit, total_samples)
        logger.info(f"Randomly subsampling {limit} samples from {total_samples} total samples (seed={seed})")
        
        # Generate random indices
        random.seed(seed)
        indices = random.sample(range(total_samples), limit)
        indices.sort()  # Sort for more efficient access
        
        dataset = dataset.select(indices)
    
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    return dataset


def convert_to_anndata(hf_dataset, embedding_key: str = "mosaicfm-70m-merged", chunk_size: int = 100000) -> ad.AnnData:
    """Convert HuggingFace dataset to AnnData format using chunked processing."""
    logger.info("Converting to AnnData format using chunked processing")
    
    total_samples = len(hf_dataset)
    logger.info(f"Total samples: {total_samples}")
    logger.info(f"Chunk size: {chunk_size}")
    
    # Calculate number of chunks needed
    num_chunks = (total_samples + chunk_size - 1) // chunk_size
    logger.info(f"Processing {num_chunks} chunks")
    
    # Get embedding dimension from first sample
    first_embedding = hf_dataset[0][embedding_key]
    embedding_dim = len(first_embedding)
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Process chunks and collect AnnData objects
    chunk_adatas = []
    
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_samples)
        
        logger.info(f"Processing chunk {chunk_idx + 1}/{num_chunks}: samples {start_idx}-{end_idx}")
        
        # Select chunk indices
        chunk_indices = list(range(start_idx, end_idx))
        chunk_data = hf_dataset.select(chunk_indices)
        
        # Extract embeddings for this chunk
        chunk_embeddings = np.array(chunk_data[embedding_key])
        
        # Prepare observation metadata for .obs
        obs_data = {}
        
        # Copy all columns except the embedding column
        for col_name in chunk_data.column_names:
            if col_name != embedding_key:
                obs_data[col_name] = chunk_data[col_name]
        
        # Create observation dataframe
        obs_df = pd.DataFrame(obs_data)
        obs_df['batch'] = '1'  # Add a dummy batch column for compatibility with state model
        
        # Create AnnData object for this chunk
        # Use sparse matrix for X to save memory, real embeddings go in .obsm
        chunk_size_actual = len(chunk_indices)
        X_sparse = sparse.csr_matrix((chunk_size_actual, embedding_dim))
        
        chunk_adata = ad.AnnData(X=X_sparse, obs=obs_df)
        
        # Store embeddings in .obsm for state model compatibility
        chunk_adata.obsm[embedding_key] = chunk_embeddings
        
        chunk_adatas.append(chunk_adata)
        
        # Explicit memory cleanup
        del chunk_data, chunk_embeddings, obs_data, obs_df
        gc.collect()
    
    # Concatenate all chunks
    logger.info("Concatenating all chunks")
    adata = ad.concat(chunk_adatas, axis=0)
    
    # Final memory cleanup
    del chunk_adatas
    gc.collect()
    
    logger.info(f"AnnData created with shape: {adata.shape}")
    logger.info(f"Observation columns: {list(adata.obs.columns)}")
    logger.info(f"Observation matrices: {list(adata.obsm.keys())}")
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
        description="Convert HuggingFace dataset to h5ad format for state transition model compatibility"
    )
    parser.add_argument(
        "input_path",
        help="Path to HuggingFace dataset directory"
    )
    parser.add_argument(
        "output_path",
        help="Output path for h5ad file"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Randomly subsample N samples from the dataset (useful for testing)"
    )
    parser.add_argument(
        "--embedding-key",
        default="mosaicfm-70m-merged",
        help="Key for embedding vectors in dataset (default: mosaicfm-70m-merged)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Process dataset in chunks of this size to manage memory usage (default: 100000)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate input path
    if not Path(args.input_path).exists():
        logger.error(f"Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    try:
        # Load HuggingFace dataset
        hf_dataset = load_hf_dataset(args.input_path, args.limit, seed=42)
        
        # Convert to AnnData
        adata = convert_to_anndata(hf_dataset, args.embedding_key, args.chunk_size)
        
        # Save to h5ad
        save_anndata(adata, args.output_path)
        
        logger.info("Conversion completed successfully!")
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()