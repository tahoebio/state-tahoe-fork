#!/usr/bin/env python3
"""
Extract true perturbation deltas from H5AD files in long-form parquet format.

Creates pseudobulk averages per (context, perturbation, plate) combination,
then computes deltas (treatment - control) and expands to gene-level long-form.

Logic exactly mirrors pearson_delta_only.py for pseudobulking and delta calculation.
"""

import argparse
import gc
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl
import anndata as ad
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s:%(name)s:%(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    force=True
)
logger = logging.getLogger(__name__)


# ============================================================================
# Helper Functions (copied from pearson_delta_only.py)
# ============================================================================

def map_control_perturbation(control_pert: str) -> str:
    """Map control perturbation names to their actual values in h5ad files.

    Copied from pearson_delta_only.py lines 71-76.
    """
    if control_pert == "DMSO_TF":
        return "[('DMSO_TF', 0.0, 'uM')]"
    return control_pert


def parse_compound_key(compound_key: str) -> Tuple[str, Optional[str]]:
    """Parse a compound key back into perturbation and group components.

    Copied from pearson_delta_only.py lines 218-230.

    Args:
        compound_key: Key like "drug_X::plate1"

    Returns:
        tuple: (perturbation, group_suffix) where group_suffix is "plate1" or None
    """
    parts = compound_key.split("::")
    perturbation = parts[0]
    group_suffix = "::".join(parts[1:]) if len(parts) > 1 else None
    return perturbation, group_suffix


def _create_pseudobulks(matrix: np.ndarray, compound_keys) -> Dict[str, np.ndarray]:
    """Create pseudobulks from matrix by averaging over compound keys.

    Copied from pearson_delta_only.py lines 472-514.
    Simplified to only return pseudobulks (no cell counts needed).

    Args:
        matrix: (n_cells, n_features) array
        compound_keys: Array of compound key strings for each cell

    Returns:
        dict mapping compound_key -> mean_vector
    """
    logger.debug(f"Creating pseudobulks from {matrix.shape[0]:,} cells x {matrix.shape[1]:,} features")

    # Create DataFrame with compound keys
    n_features = matrix.shape[1]
    logger.debug(f"Building DataFrame with {n_features:,} feature columns...")
    df_data = {f"feature_{i}": matrix[:, i] for i in range(n_features)}
    df_data["compound_key"] = list(compound_keys)  # Convert to list to avoid polars type mixing error

    df = pl.DataFrame(df_data)

    # Group by compound key and take mean
    logger.debug("Grouping by compound key and computing means...")
    feature_cols = [f"feature_{i}" for i in range(n_features)]
    pseudobulks_df = df.group_by("compound_key").agg([
        pl.col(col).mean().alias(col) for col in feature_cols
    ])

    logger.debug(f"Created {len(pseudobulks_df):,} pseudobulk groups")

    # Convert back to dictionary of arrays
    logger.debug("Converting to dictionary format...")
    pseudobulks = {}
    for row in pseudobulks_df.iter_rows(named=True):
        key = row["compound_key"]
        values = np.array([row[col] for col in feature_cols], dtype=np.float32)
        pseudobulks[key] = values

    logger.debug(f"Pseudobulking completed: {len(pseudobulks):,} groups")
    return pseudobulks


# ============================================================================
# New Functions
# ============================================================================

def load_gene_names(csv_path: str) -> List[str]:
    """Load gene names from CSV file.

    Args:
        csv_path: Path to CSV with 'gene' column

    Returns:
        List of gene names in order
    """
    df = pd.read_csv(csv_path)
    if 'gene' not in df.columns:
        raise ValueError(f"CSV must have 'gene' column, found: {df.columns.tolist()}")
    gene_names = df['gene'].tolist()
    logger.info(f"Loaded {len(gene_names):,} gene names from {csv_path}")
    return gene_names


def create_compound_keys(adata, pert_col: str, group_col: str) -> np.ndarray:
    """Create compound keys combining perturbation with group column.

    Args:
        adata: AnnData object
        pert_col: Perturbation column name
        group_col: Grouping column name (e.g., 'plate')

    Returns:
        Array of compound keys like "drug_X::plate1"
    """
    perts = adata.obs[pert_col].astype(str).values
    groups = adata.obs[group_col].astype(str).values
    compound_keys = np.array([f"{p}::{g}" for p, g in zip(perts, groups)])
    return compound_keys


def process_celltype(
    adata_ct: ad.AnnData,
    celltype: str,
    gene_names: List[str],
    pert_col: str,
    group_col: str,
    actual_control: str,
    embed_key: str,
    apply_log10: bool = False
) -> List[Dict]:
    """Process a single celltype and return list of delta records.

    Mirrors the delta calculation logic from pearson_delta_only.py lines 361-386.

    Args:
        adata_ct: AnnData filtered to this celltype
        celltype: Celltype value (for context column)
        gene_names: List of gene names
        pert_col: Perturbation column name
        group_col: Grouping column name
        actual_control: Mapped control perturbation name
        embed_key: 'X' or obsm key
        apply_log10: If True, apply log10(x+1) transformation before pseudobulking

    Returns:
        List of dicts with keys: context, perturbation, plate, gene, delta
    """
    # Get data matrix
    if embed_key == 'X':
        matrix = adata_ct.X
        if hasattr(matrix, 'toarray'):
            matrix = matrix.toarray()
    else:
        if embed_key not in adata_ct.obsm:
            raise ValueError(f"Embedding key '{embed_key}' not found in .obsm. Available: {list(adata_ct.obsm.keys())}")
        matrix = adata_ct.obsm[embed_key]

    # Apply log10 transformation if requested (before pseudobulking)
    if apply_log10:
        if np.any(matrix < 0):
            raise ValueError(
                f"Cannot apply log10 transformation: found {np.sum(matrix < 0):,} "
                f"negative values in celltype '{celltype}'. Data must be non-negative."
            )
        matrix = np.log10(matrix + 1)
        logger.info(f"  Applied log10(x+1) transformation")

    # Validate dimensions
    if matrix.shape[1] != len(gene_names):
        raise ValueError(
            f"Matrix dimension ({matrix.shape[1]}) != gene names ({len(gene_names)}). "
            f"Ensure --gene-names-csv matches the embedding dimension."
        )

    # Create compound keys (pert::plate format)
    compound_keys = create_compound_keys(adata_ct, pert_col, group_col)

    # Create pseudobulks (exact copy of pearson_delta_only.py logic)
    pseudobulks = _create_pseudobulks(matrix, compound_keys)

    # Calculate deltas (exact copy of pearson_delta_only.py lines 361-386)
    records = []
    skipped_count = 0

    for key in pseudobulks.keys():
        perturbation, group_suffix = parse_compound_key(key)

        if perturbation == actual_control:
            continue  # Skip control perturbation

        # Find matching control for this group
        if group_suffix:
            control_key = f"{actual_control}::{group_suffix}"
        else:
            control_key = actual_control

        if control_key not in pseudobulks:
            logger.debug(f"Skipping {key}: no matching control {control_key}")
            skipped_count += 1
            continue

        # Calculate delta (line 385 in original)
        delta = pseudobulks[key] - pseudobulks[control_key]

        # Expand to long-form records
        plate = group_suffix if group_suffix else ""
        for gene_name, delta_val in zip(gene_names, delta):
            records.append({
                'context': celltype,
                'perturbation': perturbation,
                'plate': plate,
                'gene': gene_name,
                'delta': float(delta_val)
            })

    if skipped_count > 0:
        logger.info(f"  Skipped {skipped_count} perturbations due to missing controls")

    return records


def extract_true_deltas(
    input_path: str,
    output_path: str,
    gene_names_csv: str,
    embed_key: str,
    pert_col: str,
    celltype_col: str,
    control_pert: str,
    group_col: str,
    use_backed: bool,
    chunk_size: int,
    apply_log10: bool = False
):
    """Main extraction function.

    Processes each celltype separately (mirrors pearson_delta_only.py --celltype-col behavior).
    Streams output to parquet using PyArrow.
    """
    if apply_log10:
        logger.info("Log10(x+1) transformation will be applied before pseudobulking")
    # Load gene names
    gene_names = load_gene_names(gene_names_csv)

    # Map control perturbation
    actual_control = map_control_perturbation(control_pert)
    if actual_control != control_pert:
        logger.info(f"Control perturbation mapped: {control_pert} -> {actual_control}")

    # Open H5AD file
    logger.info(f"Opening {input_path} (backed={use_backed})")
    adata = ad.read_h5ad(input_path, backed='r' if use_backed else None)
    logger.info(f"Loaded: {adata.n_obs:,} cells, {adata.n_vars:,} vars")

    # Validate columns exist
    for col_name, col_val in [('pert_col', pert_col), ('celltype_col', celltype_col), ('group_col', group_col)]:
        if col_val not in adata.obs.columns:
            available = list(adata.obs.columns)
            raise ValueError(f"{col_name}='{col_val}' not found in .obs. Available: {available}")

    # Get unique celltypes
    unique_celltypes = adata.obs[celltype_col].unique()
    logger.info(f"Found {len(unique_celltypes):,} unique celltypes: {sorted(unique_celltypes)}")

    # Initialize parquet writer
    schema = pa.schema([
        ('context', pa.string()),
        ('perturbation', pa.string()),
        ('plate', pa.string()),
        ('gene', pa.string()),
        ('delta', pa.float32())
    ])

    writer = pq.ParquetWriter(output_path, schema, compression='zstd')

    buffer = []
    total_rows = 0

    try:
        for ct in tqdm(sorted(unique_celltypes), desc="Processing celltypes"):
            logger.info(f"Processing celltype: {ct}")

            # Filter to this celltype
            mask = adata.obs[celltype_col] == ct

            if use_backed:
                adata_ct = adata[mask, :].to_memory()
            else:
                adata_ct = adata[mask].copy()

            logger.info(f"  Celltype {ct}: {adata_ct.n_obs:,} cells")

            # Process this celltype
            records = process_celltype(
                adata_ct, ct, gene_names,
                pert_col, group_col, actual_control, embed_key,
                apply_log10=apply_log10
            )

            logger.info(f"  Generated {len(records):,} delta records")
            buffer.extend(records)

            # Flush buffer if needed
            if len(buffer) >= chunk_size:
                table = pa.Table.from_pylist(buffer, schema=schema)
                writer.write_table(table)
                total_rows += len(buffer)
                logger.info(f"  Wrote {len(buffer):,} rows (total: {total_rows:,})")
                buffer = []

            # Clean up
            del adata_ct
            gc.collect()

        # Final flush
        if buffer:
            table = pa.Table.from_pylist(buffer, schema=schema)
            writer.write_table(table)
            total_rows += len(buffer)
            logger.info(f"Wrote final {len(buffer):,} rows (total: {total_rows:,})")

    finally:
        writer.close()
        if use_backed and hasattr(adata, 'file'):
            adata.file.close()

    logger.info(f"Complete! Wrote {total_rows:,} rows to {output_path}")

    # Verify output
    result_df = pl.read_parquet(output_path)
    logger.info(f"Verification: {len(result_df):,} rows, columns: {result_df.columns}")
    logger.info(f"Unique contexts: {result_df['context'].n_unique()}")
    logger.info(f"Unique perturbations: {result_df['perturbation'].n_unique()}")
    logger.info(f"Unique plates: {result_df['plate'].n_unique()}")
    logger.info(f"Unique genes: {result_df['gene'].n_unique()}")


def main():
    parser = argparse.ArgumentParser(
        description="Extract true perturbation deltas from H5AD files in long-form parquet format"
    )

    # Required arguments
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to input H5AD file with real/ground truth data"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output parquet file path (e.g., true_deltas.parquet)"
    )

    # Data source arguments
    parser.add_argument(
        "--gene-names-csv",
        type=str,
        default="/tahoe/drive_3/ml/pod_diffusion/Data/ST-Tahoe-HVGs.csv",
        help="Path to CSV with gene names (must match embedding dimension) [default: %(default)s]"
    )
    parser.add_argument(
        "--embed-key",
        type=str,
        default="X",
        help="'X' to use .X matrix, or key in .obsm [default: %(default)s]"
    )

    # Column name arguments
    parser.add_argument(
        "--pert-col",
        type=str,
        default="drugname_drugconc",
        help="Column name for perturbation identifiers [default: %(default)s]"
    )
    parser.add_argument(
        "--celltype-col",
        type=str,
        default="cell_line_id",
        help="Column name for cell line/context identifiers [default: %(default)s]"
    )
    parser.add_argument(
        "--control-pert",
        type=str,
        default="DMSO_TF",
        help="Control perturbation name (will be mapped to actual format) [default: %(default)s]"
    )
    parser.add_argument(
        "--group-by",
        type=str,
        default="plate",
        help="Column to group by for pseudobulk computation [default: %(default)s]"
    )

    # Performance arguments
    parser.add_argument(
        "--use-backed",
        action="store_true",
        help="Use backed mode for memory-efficient processing of large files"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100000,
        help="Rows to write per parquet chunk [default: %(default)s]"
    )

    # Data transformation arguments
    parser.add_argument(
        "--apply-log10",
        action="store_true",
        help="Apply log10(x+1) transformation to data before pseudobulking. "
             "Use when input data is in linear scale (C*x/s) rather than log scale."
    )

    args = parser.parse_args()

    # Run extraction
    extract_true_deltas(
        input_path=args.input,
        output_path=args.output,
        gene_names_csv=args.gene_names_csv,
        embed_key=args.embed_key,
        pert_col=args.pert_col,
        celltype_col=args.celltype_col,
        control_pert=args.control_pert,
        group_col=args.group_by,
        use_backed=args.use_backed,
        chunk_size=args.chunk_size,
        apply_log10=args.apply_log10
    )


if __name__ == "__main__":
    main()
