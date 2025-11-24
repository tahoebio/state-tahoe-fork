"""
Mean Shift Ablation Study for State Model

This script computes mean shifts between control and perturbed cell embeddings
for each (cell_type, perturbation) pair, then uses these shifts as a baseline
to compare against the full State transformer model.

The key question: Can a simple mean shift capture most of the perturbation effect,
or does the transformer learn something more complex?
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, Tuple, Optional
import scanpy as sc
from tqdm import tqdm
from glob import glob


class MeanShiftTable:
    """
    Computes and stores mean shifts for (cell_type, perturbation) pairs.

    Mean shift = mean(perturbed_embeddings) - mean(control_embeddings)

    This shift can be applied to individual control cells to predict their
    perturbed state.
    """

    def __init__(self):
        self.shifts = {}  # {(cell_type, perturbation): shift_vector}
        self.control_means = {}  # {cell_type: mean_control_embedding}
        self.n_samples = {}  # {(cell_type, perturbation): (n_control, n_pert)}

    def compute_from_anndata(
        self,
        adata_path: str,
        control_pert: str = "non-targeting",
        cell_type_col: str = "cell_type",
        pert_col: str = "target_gene",
        embed_key: Optional[str] = None
    ):
        """
        Compute mean shifts from AnnData file.

        Parameters
        ----------
        adata_path : str
            Path to AnnData h5ad file with embeddings
        control_pert : str
            Name of control perturbation
        cell_type_col : str
            Column name for cell types in adata.obs
        pert_col : str
            Column name for perturbations in adata.obs
        embed_key : str or None
            Key in adata.obsm for embeddings (None = use adata.X)

        Returns
        -------
        self : MeanShiftTable
            Returns self for chaining
        """
        print("Computing mean shifts from AnnData...")
        print(f"Loading data from: {adata_path}")

        # Load data
        adata = sc.read_h5ad(adata_path)
        print(f"Data shape: {adata.shape}")

        # Get embeddings
        if embed_key is not None and embed_key in adata.obsm:
            embeddings = adata.obsm[embed_key]
            print(f"Using embeddings from adata.obsm['{embed_key}']")
        else:
            try:
                embeddings = adata.X.toarray()
            except:
                embeddings = adata.X
            print(f"Using embeddings from adata.X")

        print(f"Embeddings shape: {embeddings.shape}")

        # Get metadata
        cell_types = adata.obs[cell_type_col].values
        perturbations = adata.obs[pert_col].values

        unique_cell_types = np.unique(cell_types)
        unique_perts = np.unique(perturbations)

        print(f"\nUnique cell types: {len(unique_cell_types)}")
        print(f"Unique perturbations: {len(unique_perts)}")

        # Step 1: Compute control means for each cell type
        print("\n" + "="*60)
        print("STEP 1: Computing control means per cell type")
        print("="*60)

        for cell_type in tqdm(unique_cell_types, desc="Cell types"):
            mask = (cell_types == cell_type) & (perturbations == control_pert)
            n_control = mask.sum()

            if n_control > 0:
                control_emb = embeddings[mask]
                self.control_means[cell_type] = control_emb.mean(axis=0)
                print(f"  {cell_type}: {n_control} control cells")
            else:
                print(f"  WARNING: No control cells for {cell_type}")

        # Step 2: Compute mean shifts for each (cell_type, perturbation)
        print("\n" + "="*60)
        print("STEP 2: Computing mean shifts for (cell_type, perturbation) pairs")
        print("="*60)

        n_shifts_computed = 0

        for cell_type in tqdm(unique_cell_types, desc="Cell types"):
            if cell_type not in self.control_means:
                continue

            control_mean = self.control_means[cell_type]

            for pert in unique_perts:
                if pert == control_pert:
                    continue

                # Get perturbed cells
                mask = (cell_types == cell_type) & (perturbations == pert)
                n_cells = mask.sum()

                if n_cells > 0:
                    pert_emb = embeddings[mask]
                    pert_mean = pert_emb.mean(axis=0)

                    # Compute shift
                    shift = pert_mean - control_mean

                    # Store
                    key = (cell_type, pert)
                    self.shifts[key] = shift

                    # Count samples
                    n_control = ((cell_types == cell_type) &
                                (perturbations == control_pert)).sum()
                    self.n_samples[key] = (n_control, n_cells)

                    n_shifts_computed += 1

        print(f"\nComputed {n_shifts_computed} mean shifts")
        print(f"Coverage: {n_shifts_computed}/{len(unique_cell_types) * (len(unique_perts)-1)} possible combinations")

        return self

    def save(self, filepath: str):
        """Save mean shift table to file."""
        data = {
            'shifts': self.shifts,
            'control_means': self.control_means,
            'n_samples': self.n_samples
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"\nSaved mean shift table to {filepath}")

    def load(self, filepath: str):
        """Load mean shift table from file."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.shifts = data['shifts']
        self.control_means = data['control_means']
        self.n_samples = data['n_samples']
        print(f"Loaded mean shift table from {filepath}")
        print(f"  - {len(self.shifts)} shifts")
        print(f"  - {len(self.control_means)} cell types")
        return self

    def get_statistics(self) -> pd.DataFrame:
        """Get statistics about mean shifts."""
        stats = []
        for (cell_type, pert), shift in self.shifts.items():
            magnitude = np.linalg.norm(shift)
            n_control, n_pert = self.n_samples[(cell_type, pert)]
            stats.append({
                'cell_type': cell_type,
                'perturbation': pert,
                'shift_magnitude': magnitude,
                'n_control_cells': n_control,
                'n_perturbed_cells': n_pert
            })

        df = pd.DataFrame(stats)
        return df.sort_values('shift_magnitude', ascending=False)


def create_pred_h5ad_for_mmd(
    adata_test_path: str,
    adata_control_path: str,
    shift_table: MeanShiftTable,
    output_path: str,
    control_pert: str = "non-targeting",
    cell_type_col: str = "cell_type",
    pert_col: str = "target_gene",
    embed_key: Optional[str] = None,
    pred_embed_key: str = "model_preds",
    ctrl_barcode_col: str = "ctrl_cell_barcode",
    is_partitioned: bool = False
):
    """
    Create a pred.h5ad file compatible with mmd_anndata_pair.py.

    This function applies mean shift predictions by:
    1. Looking up the specific control cell that was paired with each perturbed cell
    2. Applying the mean shift to that control cell's embedding

    This matches State's approach of using specific control cells as baseline.

    For each perturbed cell in adata_test:
    - Look up ctrl_cell_barcode from .obs
    - Find that control cell in adata_control
    - Apply: pred = control_embedding + mean_shift[(cell_type, perturbation)]

    The output h5ad file will have:
    - Same structure as input (same .obs metadata, same cells)
    - Predictions stored in .obsm[pred_embed_key]

    Parameters
    ----------
    adata_test_path : str
        Path to test dataset h5ad file (real.h5ad with perturbed cells)
    adata_control_path : str
        Path to directory with control dataset (can contain partitioned files) or single h5ad file
    shift_table : MeanShiftTable
        Pre-computed mean shift table from test data
    output_path : str
        Path to save the output pred_lms.h5ad file
    control_pert : str
        Name of control perturbation
    cell_type_col : str
        Column name for cell types
    pert_col : str
        Column name for perturbations
    embed_key : str or None
        Key in adata.obsm for input embeddings (None = use adata.X)
    pred_embed_key : str
        Key name for storing predictions in output .obsm
    ctrl_barcode_col : str
        Column name for control cell barcodes in adata_test.obs
    is_partitioned : bool
        If True, adata_control_path is a directory with partitioned h5ad files (e.g., part01.h5ad, part02.h5ad, ...)

    Returns
    -------
    None
        Saves output file to output_path
    """
    print("\n" + "="*60)
    print("CREATING PRED.H5AD FOR MMD EVALUATION (Approach 1)")
    print("="*60)
    print(f"Test data (perturbed): {adata_test_path}")
    print(f"Control data: {adata_control_path}")
    print(f"Output: {output_path}")

    # Load test data (perturbed cells from real.h5ad)
    adata_test = sc.read_h5ad(adata_test_path)
    print(f"\nTest data shape: {adata_test.shape}")

    # Check if ctrl_barcode_col exists
    if ctrl_barcode_col not in adata_test.obs.columns:
        raise ValueError(f"Column '{ctrl_barcode_col}' not found in test data. Available columns: {list(adata_test.obs.columns)}")

    # Extract unique control barcodes we need
    unique_ctrl_barcodes = set(adata_test.obs[ctrl_barcode_col].unique())
    print(f"\nFound {len(unique_ctrl_barcodes)} unique control cells needed")

    # Load control data
    control_lookup = {}

    if is_partitioned:
        print(f"\nLoading control cells from partitioned dataset: {adata_control_path}")

        # Find all partition files
        partition_files = sorted(glob(f"{adata_control_path}/*.h5ad"))
        if not partition_files:
            raise FileNotFoundError(f"No h5ad files found in {adata_control_path}")

        print(f"Found {len(partition_files)} partition files")

        # Load control cells from each partition
        for partition_file in tqdm(partition_files, desc="Loading partitions"):
            try:
                adata_partition = sc.read_h5ad(partition_file)

                # Get barcodes for this partition
                if hasattr(adata_partition.obs, 'index'):
                    partition_barcodes = adata_partition.obs.index.values
                else:
                    partition_barcodes = adata_partition.obs.get('barcode', adata_partition.obs.index).values

                # Find which needed barcodes are in this partition
                barcodes_in_partition = unique_ctrl_barcodes.intersection(set(partition_barcodes))

                if len(barcodes_in_partition) > 0:
                    # Get embeddings from this partition
                    if embed_key and embed_key in adata_partition.obsm:
                        partition_embeddings = adata_partition.obsm[embed_key]
                    else:
                        try:
                            partition_embeddings = adata_partition.X.toarray()
                        except:
                            partition_embeddings = adata_partition.X

                    # Add found control cells to lookup
                    for i, barcode in enumerate(partition_barcodes):
                        if barcode in barcodes_in_partition:
                            control_lookup[barcode] = partition_embeddings[i]

                    print(f"  Found {len(barcodes_in_partition)} needed cells in {Path(partition_file).name}")

                # Free memory
                del adata_partition

            except Exception as e:
                print(f"  Warning: Could not load {partition_file}: {e}")
                continue

        print(f"\nLoaded {len(control_lookup)} control cells total")

    else:
        # Load single control file
        print(f"Loading control data from: {adata_control_path}")
        adata_control = sc.read_h5ad(adata_control_path)
        print(f"Control data shape: {adata_control.shape}")

        # Get control embeddings
        if embed_key and embed_key in adata_control.obsm:
            control_embeddings = adata_control.obsm[embed_key]
            print(f"Using control embeddings from adata.obsm['{embed_key}']")
        else:
            try:
                control_embeddings = adata_control.X.toarray()
            except:
                control_embeddings = adata_control.X
            print(f"Using control embeddings from adata.X")

        # Create barcode lookup
        if hasattr(adata_control.obs, 'index'):
            control_barcodes = adata_control.obs.index.values
        else:
            control_barcodes = adata_control.obs.get('barcode', adata_control.obs.index).values

        control_lookup = {barcode: control_embeddings[i] for i, barcode in enumerate(control_barcodes)}
        print(f"Created lookup for {len(control_lookup)} control cells")

    # Check embedding dimension from first control cell
    first_embedding = next(iter(control_lookup.values()))
    embedding_dim = len(first_embedding)

    # Initialize predictions array
    predictions = np.zeros((len(adata_test), embedding_dim), dtype=np.float32)

    cell_types = adata_test.obs[cell_type_col].values
    perturbations = adata_test.obs[pert_col].values
    ctrl_barcodes = adata_test.obs[ctrl_barcode_col].values

    # Apply mean shifts to each cell
    n_shifted = 0
    n_no_shift = 0
    n_no_control = 0
    n_control_cells = 0

    print(f"\nApplying mean shifts to {len(adata_test)} cells...")

    for i in tqdm(range(len(adata_test)), desc="Processing cells"):
        cell_type = cell_types[i]
        pert = perturbations[i]
        ctrl_barcode = ctrl_barcodes[i]

        # For control cells in test set, just use their own embedding
        if pert == control_pert:
            if ctrl_barcode in control_lookup:
                predictions[i] = control_lookup[ctrl_barcode]
            else:
                # Fallback: use the test cell's own embedding
                if embed_key and embed_key in adata_test.obsm:
                    predictions[i] = adata_test.obsm[embed_key][i]
                else:
                    predictions[i] = adata_test.X[i].toarray() if hasattr(adata_test.X[i], 'toarray') else adata_test.X[i]
            n_control_cells += 1
            continue

        # Look up the matched control cell
        if ctrl_barcode not in control_lookup:
            print(f"Warning: Control barcode '{ctrl_barcode}' not found in control dataset")
            # Fallback: use test cell's embedding
            if embed_key and embed_key in adata_test.obsm:
                predictions[i] = adata_test.obsm[embed_key][i]
            else:
                predictions[i] = adata_test.X[i].toarray() if hasattr(adata_test.X[i], 'toarray') else adata_test.X[i]
            n_no_control += 1
            continue

        control_embedding = control_lookup[ctrl_barcode]

        # Look up the shift for this (cell_type, perturbation)
        key = (cell_type, pert)

        if key in shift_table.shifts:
            # Apply mean shift to the specific control cell
            shift = shift_table.shifts[key]
            predictions[i] = control_embedding + shift
            n_shifted += 1
        else:
            # No shift available - use control embedding as is
            predictions[i] = control_embedding
            n_no_shift += 1

    print(f"\nProcessing complete:")
    print(f"  Control cells in test set: {n_control_cells}")
    print(f"  Perturbed cells with mean shift applied: {n_shifted}")
    print(f"  Perturbed cells without available shift: {n_no_shift}")
    print(f"  Cells with missing control barcode: {n_no_control}")

    # Store predictions in .obsm
    adata_test.obsm[pred_embed_key] = predictions

    # Save output
    print(f"\nSaving predictions to: {output_path}")
    adata_test.write_h5ad(output_path)

    print(f"✓ Created {output_path}")
    print(f"  - Predictions stored in .obsm['{pred_embed_key}']")
    print(f"  - Ready for mmd_anndata_pair.py comparison")


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("MEAN SHIFT ABLATION STUDY")
    print("="*60)

    # Example: Compute mean shifts from training data
    print("\n[1/3] Computing mean shift table from training data...")
    shift_table = MeanShiftTable()

    # Uncomment and set your actual paths:
    # shift_table.compute_from_anndata(
    #     adata_path="path/to/train_data.h5ad",
    #     control_pert="non-targeting",
    #     cell_type_col="cell_type",
    #     pert_col="target_gene",
    #     embed_key="X_hvg"  # or None for adata.X
    # )

    # Save for later use
    # shift_table.save("mean_shift_table.pkl")

    # Get statistics
    # stats = shift_table.get_statistics()
    # print("\nTop 10 perturbations by shift magnitude:")
    # print(stats.head(10))
    # stats.to_csv("mean_shift_statistics.csv", index=False)

    print("\n[2/2] Creating pred.h5ad for MMD evaluation...")

    # Uncomment and set your actual paths:
    # create_pred_h5ad_for_mmd(
    #     adata_test_path="path/to/test_data.h5ad",
    #     shift_table=shift_table,
    #     output_path="pred_lms.h5ad",
    #     control_pert="non-targeting",
    #     cell_type_col="cell_type",
    #     pert_col="target_gene",
    #     embed_key="X_hvg"
    # )

    print("\n" + "="*60)
    print("Example complete! Uncomment code to run on real data.")
    print("="*60)
