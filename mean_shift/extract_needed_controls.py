"""
Extract only the needed control cells from partitioned dataset.

This script:
1. Loads real.h5ad to get unique ctrl_cell_barcode values
2. Scans through 30 partitioned control files
3. Extracts only the needed control cells
4. Saves them to a single control_cells.h5ad file

This is much more efficient than scanning partitions multiple times!
"""

import scanpy as sc
import numpy as np
from glob import glob
from tqdm import tqdm
from pathlib import Path

def extract_needed_controls(
    real_h5ad_path: str,
    partitioned_dir: str,
    output_path: str,
    ctrl_barcode_col: str = "ctrl_cell_barcode",
    embed_key: str = "X_hvg"
):
    """
    Extract needed control cells from partitioned dataset.

    Parameters
    ----------
    real_h5ad_path : str
        Path to real.h5ad (contains ctrl_cell_barcode column)
    partitioned_dir : str
        Directory with partitioned control files
    output_path : str
        Where to save the extracted control cells
    ctrl_barcode_col : str
        Column name for control barcodes
    embed_key : str
        Embedding key in .obsm
    """
    print("="*60)
    print("EXTRACTING NEEDED CONTROL CELLS")
    print("="*60)

    # Step 1: Load real.h5ad and get needed barcodes
    print(f"\nLoading {real_h5ad_path} to get control barcodes...")
    adata_real = sc.read_h5ad(real_h5ad_path)

    if ctrl_barcode_col not in adata_real.obs.columns:
        raise ValueError(f"Column '{ctrl_barcode_col}' not found in real.h5ad")

    needed_barcodes = set(adata_real.obs[ctrl_barcode_col].unique())
    print(f"Need to find {len(needed_barcodes)} unique control cells")

    # Step 2: Scan partitions and collect needed cells
    partition_files = sorted(glob(f"{partitioned_dir}/*.h5ad"))
    if not partition_files:
        raise FileNotFoundError(f"No h5ad files found in {partitioned_dir}")

    print(f"\nScanning {len(partition_files)} partition files...")

    found_barcodes = []
    found_embeddings = []
    found_obs_data = []

    for partition_file in tqdm(partition_files, desc="Processing partitions"):
        try:
            adata_partition = sc.read_h5ad(partition_file)

            # Get barcodes for this partition
            if hasattr(adata_partition.obs, 'index'):
                partition_barcodes = adata_partition.obs.index.values
            else:
                partition_barcodes = adata_partition.obs.get('barcode', adata_partition.obs.index).values

            # Find which needed barcodes are in this partition
            barcodes_in_partition = needed_barcodes.intersection(set(partition_barcodes))

            if len(barcodes_in_partition) > 0:
                # Get embeddings
                if embed_key and embed_key in adata_partition.obsm:
                    partition_embeddings = adata_partition.obsm[embed_key]
                else:
                    try:
                        partition_embeddings = adata_partition.X.toarray()
                    except:
                        partition_embeddings = adata_partition.X

                # Extract found cells
                for i, barcode in enumerate(partition_barcodes):
                    if barcode in barcodes_in_partition:
                        found_barcodes.append(barcode)
                        found_embeddings.append(partition_embeddings[i])
                        found_obs_data.append(adata_partition.obs.iloc[i])

                print(f"  Found {len(barcodes_in_partition)} cells in {Path(partition_file).name}")

            # Free memory
            del adata_partition

        except Exception as e:
            print(f"  Warning: Could not load {partition_file}: {e}")
            continue

    print(f"\n✓ Found {len(found_barcodes)} control cells total")

    if len(found_barcodes) == 0:
        raise ValueError("No control cells found in partitions!")

    # Step 3: Create AnnData object with found cells
    print(f"\nCreating control cells AnnData...")
    import pandas as pd

    # Stack embeddings
    X = np.vstack(found_embeddings)

    # Create obs DataFrame
    obs = pd.DataFrame(found_obs_data)
    obs.index = found_barcodes

    # Create var DataFrame (gene names)
    # Use the first partition to get var info
    adata_first = sc.read_h5ad(partition_files[0])
    var = adata_first.var.copy()

    # Create AnnData
    adata_controls = sc.AnnData(X=X, obs=obs, var=var)

    # Save
    print(f"\nSaving to {output_path}...")
    adata_controls.write_h5ad(output_path)

    print(f"✓ Saved {len(found_barcodes)} control cells to {output_path}")
    print(f"  File size: {Path(output_path).stat().st_size / 1024**3:.2f} GB")


if __name__ == "__main__":
    # Configuration
    REAL_H5AD_PATH = "data/real.h5ad"
    PARTITIONED_DIR = "data/control_dataset"
    OUTPUT_PATH = "data/needed_controls.h5ad"

    extract_needed_controls(
        real_h5ad_path=REAL_H5AD_PATH,
        partitioned_dir=PARTITIONED_DIR,
        output_path=OUTPUT_PATH,
        ctrl_barcode_col="ctrl_cell_barcode",
        embed_key="X_hvg"
    )

    print("\n" + "="*60)
    print("DONE! You can now use data/needed_controls.h5ad in the notebook")
    print("="*60)
