#!/usr/bin/env python3
"""
Convert H5AD (AnnData) format to HuggingFace dataset format.

This script converts AnnData h5ad files to HuggingFace dataset format,
focusing on .obs metadata and .obsm embeddings. The .X expression matrix
is skipped by default for memory efficiency.
"""

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, List, Dict

import anndata as ad
import numpy as np
import pandas as pd
from datasets import Dataset, Features, Value, Sequence
from tqdm import tqdm

# Silence anndata warnings
warnings.filterwarnings("ignore", category=ad.ImplicitModificationWarning)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_h5ad_info(h5ad_path: str) -> Dict:
    """Load H5AD file metadata without loading full data into memory."""
    logger.info(f"Loading H5AD metadata from {h5ad_path}")

    adata = ad.read_h5ad(h5ad_path, backed='r')

    info = {
        'n_obs': adata.n_obs,
        'n_vars': adata.n_vars,
        'obs_columns': list(adata.obs.columns),
        'obsm_keys': list(adata.obsm.keys()),
        'var_names': list(adata.var_names) if adata.n_vars < 10000 else None,  # Skip if too many genes
    }

    logger.info(f"H5AD info: {info['n_obs']} cells, {info['n_vars']} variables")
    logger.info(f"Observation columns: {info['obs_columns']}")
    logger.info(f"Observation matrices: {info['obsm_keys']}")

    adata.file.close()

    return info


def infer_feature_schema(adata: ad.AnnData, obsm_keys: List[str]) -> Features:
    """Infer HuggingFace Features schema from AnnData object."""
    features = {}

    # Add .obs columns
    for col in adata.obs.columns:
        dtype = adata.obs[col].dtype

        if pd.api.types.is_string_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            features[col] = Value('string')
        elif pd.api.types.is_integer_dtype(dtype):
            features[col] = Value('int64')
        elif pd.api.types.is_float_dtype(dtype):
            features[col] = Value('float32')
        elif pd.api.types.is_bool_dtype(dtype):
            features[col] = Value('bool')
        else:
            # Fallback to string for categorical/unknown types
            features[col] = Value('string')
            logger.warning(f"Unknown dtype for column {col}: {dtype}, using string")

    # Add .obsm embeddings
    for key in obsm_keys:
        if key in adata.obsm:
            embedding_dim = adata.obsm[key].shape[1]
            features[key] = Sequence(Value('float32'), length=embedding_dim)
            logger.info(f"Added .obsm['{key}'] with dimension {embedding_dim}")

    return Features(features)


def process_chunk(adata: ad.AnnData, start_idx: int, end_idx: int, obsm_keys: List[str]) -> List[Dict]:
    """Process a chunk of cells and return list of records."""
    chunk_records = []

    # Get obs data for this chunk
    obs_chunk = adata.obs.iloc[start_idx:end_idx]

    # Process each cell
    for i in range(len(obs_chunk)):
        global_idx = start_idx + i

        # Start with obs data
        record = {}
        for col in obs_chunk.columns:
            value = obs_chunk.iloc[i][col]
            # Convert to native Python types
            if pd.isna(value):
                record[col] = None
            elif isinstance(value, (np.integer, np.floating)):
                record[col] = value.item()
            elif isinstance(value, np.bool_):
                record[col] = bool(value)
            else:
                record[col] = str(value)

        # Add obsm embeddings
        for key in obsm_keys:
            if key in adata.obsm:
                embedding = adata.obsm[key][global_idx]
                record[key] = embedding.tolist()

        chunk_records.append(record)

    return chunk_records


def convert_h5ad_to_hf_dataset(h5ad_path: str,
                               output_path: str,
                               chunk_size: int = 100000,
                               obsm_keys: Optional[List[str]] = None,
                               save_var: bool = True):
    """Convert H5AD file to HuggingFace dataset format."""
    logger.info(f"Starting conversion of {h5ad_path}")

    # Load H5AD file
    logger.info("Loading H5AD file...")
    adata = ad.read_h5ad(h5ad_path)

    total_cells = adata.n_obs
    logger.info(f"Total cells: {total_cells}")

    # Determine which obsm keys to include
    if obsm_keys is None:
        obsm_keys = list(adata.obsm.keys())
        logger.info(f"Using all .obsm keys: {obsm_keys}")
    else:
        # Validate requested keys exist
        missing_keys = [k for k in obsm_keys if k not in adata.obsm]
        if missing_keys:
            logger.warning(f"Requested .obsm keys not found: {missing_keys}")
        obsm_keys = [k for k in obsm_keys if k in adata.obsm]
        logger.info(f"Using specified .obsm keys: {obsm_keys}")

    # Infer features schema
    logger.info("Inferring HuggingFace Features schema...")
    features = infer_feature_schema(adata, obsm_keys)
    logger.info(f"Schema contains {len(features)} fields")

    # Process in chunks
    all_records = []
    num_chunks = (total_cells + chunk_size - 1) // chunk_size

    logger.info(f"Processing {num_chunks} chunks...")
    for chunk_idx in tqdm(range(num_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_cells)

        chunk_records = process_chunk(adata, start_idx, end_idx, obsm_keys)
        all_records.extend(chunk_records)

        # Memory cleanup
        if chunk_idx % 10 == 0:
            gc.collect()

    logger.info(f"Processed {len(all_records)} total records")

    # Create HuggingFace dataset
    logger.info("Creating HuggingFace dataset...")
    hf_dataset = Dataset.from_list(all_records, features=features)

    # Save dataset
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving HuggingFace dataset to {output_path}")
    hf_dataset.save_to_disk(str(output_path))

    # Optionally save .var information
    if save_var and adata.n_vars < 100000:  # Only save if reasonable size
        var_path = output_path / "var_info.parquet"
        logger.info(f"Saving .var information to {var_path}")
        adata.var.to_parquet(var_path)

    logger.info(f"Conversion completed successfully!")
    logger.info(f"Dataset shape: {hf_dataset.shape}")
    logger.info(f"Dataset columns: {hf_dataset.column_names}")

    return hf_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Convert H5AD (AnnData) format to HuggingFace dataset format"
    )
    parser.add_argument(
        "input_h5ad",
        help="Path to input H5AD file"
    )
    parser.add_argument(
        "output_dir",
        help="Output directory for HuggingFace dataset"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Process cells in chunks of this size (default: 100000)"
    )
    parser.add_argument(
        "--obsm-keys",
        nargs='+',
        help="Specific .obsm keys to include (default: all)"
    )
    parser.add_argument(
        "--no-save-var",
        action="store_true",
        help="Do not save .var information as separate parquet file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--info-only",
        action="store_true",
        help="Only print H5AD info without converting"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Validate input path
    if not Path(args.input_h5ad).exists():
        logger.error(f"Input H5AD file does not exist: {args.input_h5ad}")
        sys.exit(1)

    try:
        # Info-only mode
        if args.info_only:
            info = load_h5ad_info(args.input_h5ad)
            logger.info("H5AD file information:")
            for key, value in info.items():
                if key != 'var_names':
                    logger.info(f"  {key}: {value}")
            return

        # Full conversion
        hf_dataset = convert_h5ad_to_hf_dataset(
            h5ad_path=args.input_h5ad,
            output_path=args.output_dir,
            chunk_size=args.chunk_size,
            obsm_keys=args.obsm_keys,
            save_var=not args.no_save_var
        )

        # Print summary
        logger.info("\n=== Conversion Summary ===")
        logger.info(f"Input: {args.input_h5ad}")
        logger.info(f"Output: {args.output_dir}")
        logger.info(f"Cells: {len(hf_dataset)}")
        logger.info(f"Columns: {len(hf_dataset.column_names)}")
        logger.info(f"Column names: {hf_dataset.column_names}")

    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()