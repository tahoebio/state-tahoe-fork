#!/usr/bin/env python3
"""
Rescale embeddings in AnnData H5AD files by dividing by sqrt(d_model).
This script processes multiple H5AD files and rescales specified embeddings.
"""

import os
import argparse
import numpy as np
import anndata
from tqdm import tqdm
import psutil


def get_ram_usage():
    """Get current RAM usage in GB"""
    return psutil.virtual_memory().used / (1024**3)


def main():
    parser = argparse.ArgumentParser(
        description="Rescale embeddings in AnnData H5AD files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "input_dir",
        help="Directory containing input H5AD files"
    )

    parser.add_argument(
        "output_dir",
        help="Directory to save rescaled H5AD files"
    )

    parser.add_argument(
        "--embedding-key",
        default="tahoe_x1_3b",
        help="Name of embedding in obsm to rescale"
    )

    parser.add_argument(
        "--d-model",
        type=int,
        default=2560,
        help="Model dimension (scale factor will be sqrt(d_model))"
    )

    parser.add_argument(
        "--max-norm-check",
        type=float,
        default=3.0,
        help="Maximum L2 norm threshold for validation (set to 0 to disable)"
    )

    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files that already exist in output directory"
    )

    args = parser.parse_args()

    # Validate directories
    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Get list of H5AD files
    adata_files = [f for f in sorted(os.listdir(args.input_dir)) if f.endswith('.h5ad')]

    if not adata_files:
        print(f"No H5AD files found in {args.input_dir}")
        return

    # Calculate scale factor
    scale_factor = np.sqrt(args.d_model)

    print(f"Processing {len(adata_files)} H5AD files")
    print(f"Embedding key: {args.embedding_key}")
    print(f"D-model: {args.d_model}")
    print(f"Scale factor: {scale_factor:.4f}")
    print(f"Max norm check: {args.max_norm_check if args.max_norm_check > 0 else 'disabled'}")
    print()

    pbar = tqdm(adata_files, desc="Processing files")
    for adata_path in pbar:
        # Update progress bar with current RAM usage
        ram_gb = get_ram_usage()
        pbar.set_postfix({'RAM': f'{ram_gb:.1f}GB'})

        output_path = os.path.join(args.output_dir, adata_path)

        if args.skip_existing and os.path.exists(output_path):
            tqdm.write(f"Skipping {adata_path} (already exists)")
            continue

        try:
            # Load data
            adata = anndata.read_h5ad(os.path.join(args.input_dir, adata_path))

            # Check if embedding key exists
            if args.embedding_key not in adata.obsm:
                tqdm.write(f"Warning: {args.embedding_key} not found in {adata_path}, skipping")
                continue

            # Rescale embeddings
            adata.obsm[args.embedding_key] = adata.obsm[args.embedding_key] / scale_factor

            # Optional validation
            if args.max_norm_check > 0:
                max_norm = np.max(np.linalg.norm(adata.obsm[args.embedding_key], axis=-1))
                if max_norm >= args.max_norm_check:
                    tqdm.write(f"Warning: Max norm {max_norm:.3f} exceeds threshold {args.max_norm_check} for {adata_path}")
                else:
                    tqdm.write(f"Validation passed for {adata_path} (max norm: {max_norm:.3f})")

            # Save rescaled data
            adata.write_h5ad(output_path)

        except Exception as e:
            tqdm.write(f"Error processing {adata_path}: {e}")
            continue

    print(f"\nProcessing complete. Results saved to {args.output_dir}")


if __name__ == "__main__":
    main()
  

  