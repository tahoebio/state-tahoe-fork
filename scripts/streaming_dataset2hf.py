#!/usr/bin/env python3
"""
Convert streaming HuggingFace Tahoe-100M dataset to HuggingFace dataset format with gene filtering.

This script converts the streaming vevotx/Tahoe-100M dataset to HuggingFace dataset format,
filtering to only the 2000 highly variable genes and applying log transformation.
"""

import argparse
import gc
import logging
import sys
import warnings
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset, Features, Value, Sequence
from tqdm import tqdm

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


def process_single_cell(cell: dict, token_to_col_idx: Dict[int, int], 
                       n_hvg_genes: int) -> dict:
    """Process a single cell and return HF dataset record."""
    genes = cell['genes']
    expressions = cell['expressions']
    
    # Handle negative expressions[0] values (skip first gene/expression if negative)
    if expressions[0] < 0:
        genes = genes[1:]
        expressions = expressions[1:]
    
    # Calculate library size (sum of all expressions for this cell)
    library_size = sum(expressions)
    
    # Initialize HVG expression array
    hvg_array = np.zeros(n_hvg_genes, dtype=np.float32)
    
    # Filter to only HVG genes and populate array
    for gene_token, expr in zip(genes, expressions):
        if gene_token in token_to_col_idx:
            col_idx = token_to_col_idx[gene_token]
            hvg_array[col_idx] = expr
    
    # Apply log transformation: log10(1e4 * X / library_size + 1)
    if library_size > 0:
        hvg_array = np.log10(1e4 * hvg_array / library_size + 1)
    
    # Create record with metadata and HVG expressions
    record = {k: v for k, v in cell.items() if k not in ['genes', 'expressions']}
    record['library_size'] = library_size
    record['X_hvg'] = hvg_array.tolist()  # Convert to list for HF dataset
    
    return record


def save_chunk_to_parquet(chunk_records: List[dict], chunk_idx: int, output_dir: str):
    """Save a chunk of processed records to parquet file."""
    import os
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    chunk_file = output_path / f"chunk_{chunk_idx:06d}.parquet"
    
    logger.info(f"Saving chunk {chunk_idx} with {len(chunk_records)} records to {chunk_file}")
    
    # Convert to pandas and save
    chunk_df = pd.DataFrame(chunk_records)
    chunk_df.to_parquet(chunk_file, compression='snappy')
    
    return chunk_file


def combine_chunks_to_hf_dataset(chunks_dir: str, output_path: str, n_hvg_genes: int):
    """Combine parquet chunks into final HuggingFace dataset."""
    logger.info(f"Combining chunks from {chunks_dir} into HF dataset at {output_path}")
    
    # Find all chunk files
    chunks_dir = Path(chunks_dir)
    chunk_files = sorted(chunks_dir.glob("chunk_*.parquet"))
    
    logger.info(f"Found {len(chunk_files)} chunk files to combine")
    
    if not chunk_files:
        raise ValueError("No chunk files found to combine")
    
    # Create features schema dynamically from chunks
    features = create_features_schema(n_hvg_genes, chunks_dir)
    
    # Load chunks one by one and create HF dataset incrementally
    logger.info("Loading and combining chunks...")
    
    all_records = []
    total_records = 0
    
    for i, chunk_file in enumerate(chunk_files):
        logger.info(f"Loading chunk {i+1}/{len(chunk_files)}: {chunk_file}")
        chunk_df = pd.read_parquet(chunk_file)
        chunk_records = chunk_df.to_dict('records')
        all_records.extend(chunk_records)
        total_records += len(chunk_df)
        
        # Log progress every 5 chunks
        if (i + 1) % 5 == 0 or (i + 1) == len(chunk_files):
            logger.info(f"Loaded {i+1} chunks, {total_records} total records")
    
    logger.info(f"Creating HuggingFace dataset from {total_records} records...")
    hf_dataset = Dataset.from_list(all_records, features=features)
    
    # Save HuggingFace dataset
    logger.info(f"Saving HuggingFace dataset to {output_path}")
    hf_dataset.save_to_disk(output_path)
    
    logger.info(f"Dataset saved successfully. Shape: {hf_dataset.shape}")
    return hf_dataset


def create_features_schema(n_hvg_genes: int, chunks_dir: str = None) -> Features:
    """Create HuggingFace Features schema dynamically from actual data."""
    features = {}
    
    if chunks_dir and Path(chunks_dir).exists():
        # Read schema from first chunk
        chunk_files = sorted(Path(chunks_dir).glob("chunk_*.parquet"))
        if chunk_files:
            logger.info(f"Dynamically creating schema from {chunk_files[0]}")
            
            # Read just the schema without loading data
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(chunk_files[0])
            
            # Read one row to infer types using PyArrow
            table_sample = pq.read_table(chunk_files[0]).slice(0, 1)
            sample_df = table_sample.to_pandas()
            
            # Get actual column names from the pandas dataframe (this handles nested structures correctly)
            column_names = list(sample_df.columns)
            logger.info(f"Found columns in chunk: {column_names}")
            
            for col_name in column_names:
                if col_name == 'X_hvg':
                    features[col_name] = Sequence(Value('float32'), length=n_hvg_genes)
                elif col_name == 'library_size':
                    features[col_name] = Value('float32')
                else:
                    # Infer type from sample data
                    col_value = sample_df[col_name].iloc[0]
                    if isinstance(col_value, str):
                        features[col_name] = Value('string')
                    elif isinstance(col_value, (int, np.integer)):
                        features[col_name] = Value('int64')
                    elif isinstance(col_value, (float, np.floating)):
                        features[col_name] = Value('float32')
                    else:
                        # Fallback to string for unknown types
                        features[col_name] = Value('string')
                        logger.warning(f"Unknown type for column {col_name}, using string")
    
    # Fallback to hardcoded schema if chunks not available
    if not features:
        logger.warning("Using fallback hardcoded schema")
        features = {
            # Core expression data
            'X_hvg': Sequence(Value('float32'), length=n_hvg_genes),
            'library_size': Value('float32'),
            
            # Common metadata fields (may not all be present)
            'drug': Value('string'),
            'sample': Value('string'),
            'cell_line': Value('string'),
            'plate': Value('string'),
        }
    
    logger.info(f"Created features schema with {len(features)} fields, X_hvg length: {n_hvg_genes}")
    return Features(features)


def process_cell_batch(batch_data: List[dict], token_to_col_idx: Dict[int, int], 
                      n_hvg_genes: int) -> List[dict]:
    """Process a batch of cells and return HF dataset records."""
    batch_records = []
    
    for cell in batch_data:
        record = process_single_cell(cell, token_to_col_idx, n_hvg_genes)
        batch_records.append(record)
    
    return batch_records


def convert_streaming_to_hf_dataset(dataset_name: str = "vevotx/Tahoe-100M",
                                   token2hvg_path: str = "/tahoe/drive_3/ANALYSIS/analysis_190/Data/token2hvg.parquet",
                                   batch_size: int = 50000,
                                   max_cells: Optional[int] = None,
                                   skip_cells: int = 0,
                                   chunk_size: int = 1000000) -> Tuple[str, int]:
    """Convert streaming HuggingFace dataset to HF dataset format."""
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
    
    # Process in batches with chunked saving
    all_records = []
    total_cells_processed = 0
    chunk_idx = 0
    chunks_dir = "temp_chunks"
    
    batch_data = []
    cells_skipped = 0
    
    logger.info(f"Processing cells in {batch_size}-cell batches (skipping first {skip_cells} cells, chunk every {chunk_size} cells)...")
    for cell in tqdm(streaming_dataset, desc="Processing cells"):
        # Skip cells if resuming from checkpoint
        if cells_skipped < skip_cells:
            cells_skipped += 1
            continue
        batch_data.append(cell)
        
        # Process batch when it reaches batch_size
        if len(batch_data) >= batch_size:
            batch_records = process_cell_batch(batch_data, token_to_col_idx, n_hvg_genes)
            all_records.extend(batch_records)
            total_cells_processed += len(batch_data)
            
            logger.info(f"Processed {total_cells_processed} cells")
            
            # Save chunk when we reach chunk_size
            if len(all_records) >= chunk_size:
                save_chunk_to_parquet(all_records, chunk_idx, chunks_dir)
                chunk_idx += 1
                all_records = []  # Clear memory
                gc.collect()
            
            # Clear batch data for memory efficiency
            batch_data = []
            gc.collect()
            
            # Check if we've reached the maximum number of cells
            if max_cells is not None and total_cells_processed >= max_cells:
                logger.info(f"Reached maximum cells limit: {max_cells}")
                break
    
    # Process remaining cells in final batch
    if batch_data:
        batch_records = process_cell_batch(batch_data, token_to_col_idx, n_hvg_genes)
        all_records.extend(batch_records)
        total_cells_processed += len(batch_data)
    
    # Save final chunk if there are remaining records
    if all_records:
        save_chunk_to_parquet(all_records, chunk_idx, chunks_dir)
        chunk_idx += 1
    
    logger.info(f"Total cells processed: {total_cells_processed}")
    logger.info(f"Created {chunk_idx} chunks in {chunks_dir}")
    
    return chunks_dir, n_hvg_genes


def save_hf_dataset(hf_dataset: Dataset, output_path: str):
    """Save HuggingFace dataset to disk."""
    logger.info(f"Saving HuggingFace dataset to {output_path}")
    
    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    hf_dataset.save_to_disk(output_path)
    logger.info(f"Successfully saved HF dataset to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert streaming HuggingFace Tahoe-100M dataset to HF dataset format with HVG filtering"
    )
    parser.add_argument(
        "output_path",
        help="Output path for HF dataset directory"
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
        default=50000,
        help="Process cells in batches of this size (default: 50000)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1000000,
        help="Save chunks every N cells to manage memory (default: 1000000)"
    )
    parser.add_argument(
        "--max-cells",
        type=int,
        help="Maximum number of cells to process (useful for testing)"
    )
    parser.add_argument(
        "--skip-cells",
        type=int,
        default=0,
        help="Skip this many cells at the start (for resuming from checkpoint)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--combine-only",
        action="store_true",
        help="Skip data processing and combine existing chunks in temp_chunks/ directory"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate token2hvg path
    if not Path(args.token2hvg_path).exists():
        logger.error(f"Token2HVG path does not exist: {args.token2hvg_path}")
        sys.exit(1)
    
    try:
        if args.combine_only:
            # Skip data processing, use existing chunks
            chunks_dir = "temp_chunks"
            if not Path(chunks_dir).exists():
                logger.error(f"Chunks directory {chunks_dir} does not exist")
                sys.exit(1)
            
            # Load HVG mapping to get n_hvg_genes
            token_to_col_idx, gene_names = load_hvg_mapping(args.token2hvg_path)
            n_hvg_genes = len(gene_names)
            logger.info(f"Using existing chunks in {chunks_dir} with {n_hvg_genes} HVG genes")
        else:
            # Step 1: Convert streaming dataset to chunked parquet files
            chunks_dir, n_hvg_genes = convert_streaming_to_hf_dataset(
                dataset_name=args.dataset_name,
                token2hvg_path=args.token2hvg_path,
                batch_size=args.batch_size,
                max_cells=args.max_cells,
                skip_cells=args.skip_cells,
                chunk_size=args.chunk_size
            )
        
        # Step 2: Combine chunks into final HF dataset
        logger.info("Combining chunks into final HuggingFace dataset...")
        hf_dataset = combine_chunks_to_hf_dataset(chunks_dir, args.output_path, n_hvg_genes)
        
        logger.info("Conversion completed successfully!")
        
        # Print summary statistics
        logger.info(f"Final dataset shape: {hf_dataset.shape}")
        logger.info(f"X_hvg dimension: {len(hf_dataset[0]['X_hvg'])}")
        logger.info(f"Library size statistics:")
        library_sizes = hf_dataset['library_size']
        logger.info(f"  Mean: {np.mean(library_sizes):.2f}")
        logger.info(f"  Median: {np.median(library_sizes):.2f}")
        logger.info(f"  Min: {np.min(library_sizes)}")
        logger.info(f"  Max: {np.max(library_sizes)}")
        
        # Clean up temporary chunks
        import shutil
        logger.info(f"Cleaning up temporary chunks directory: {chunks_dir}")
        shutil.rmtree(chunks_dir)
        
    except Exception as e:
        logger.error(f"Conversion failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()