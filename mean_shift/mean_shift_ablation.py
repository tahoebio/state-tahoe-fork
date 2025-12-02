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
from typing import Dict, Tuple, Optional
import scanpy as sc
from tqdm import tqdm


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
    shift_table: MeanShiftTable,
    output_path: str,
    control_pert: str = "non-targeting",
    cell_type_col: str = "cell_type",
    pert_col: str = "target_gene",
    embed_key: Optional[str] = None,
    pred_embed_key: str = "model_preds",
    ctrl_barcode_col: str = "ctrl_cell_barcode"
):
    """
    Create a pred.h5ad file compatible with mmd_anndata_pair.py using mean shift baseline.

    For each cell in adata_test:
    - If PBS control: use its own embedding
    - If perturbed: look up matched control cell by barcode, apply mean shift

    Parameters
    ----------
    adata_test_path : str
        Path to real.h5ad (contains both PBS controls and perturbed cells)
    shift_table : MeanShiftTable
        Pre-computed mean shifts from real.h5ad
    output_path : str
        Path to save predictions
    control_pert : str
        Name of control perturbation (e.g., 'PBS')
    cell_type_col : str
        Column for cell type (e.g., 'donor')
    pert_col : str
        Column for perturbation (e.g., 'cytokine')
    embed_key : str or None
        Key in .obsm for embeddings (None = use .X)
    pred_embed_key : str
        Key for storing predictions in .obsm
    ctrl_barcode_col : str
        Column containing matched control cell barcodes

    Returns
    -------
    None
        Saves predictions to output_path
    """
    print("\n" + "="*60)
    print("CREATING MEAN SHIFT PREDICTIONS")
    print("="*60)
    print(f"Data: {adata_test_path}")
    print(f"Output: {output_path}")

    # Load data
    adata = sc.read_h5ad(adata_test_path)
    print(f"\nData shape: {adata.shape}")

    # Get embeddings
    if embed_key and embed_key in adata.obsm:
        embeddings = adata.obsm[embed_key]
        print(f"Using embeddings from .obsm['{embed_key}']")
    else:
        embeddings = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        print(f"Using embeddings from .X")

    # Create barcode lookup (all cells indexed by barcode)
    control_lookup = {barcode: embeddings[i] for i, barcode in enumerate(adata.obs.index)}
    print(f"Created lookup for {len(control_lookup)} cells")

    # Initialize predictions
    predictions = np.zeros_like(embeddings, dtype=np.float32)

    cell_types = adata.obs[cell_type_col].values
    perturbations = adata.obs[pert_col].values
    ctrl_barcodes = adata.obs[ctrl_barcode_col].values

    n_control_cells = 0
    n_shifted = 0
    n_no_shift = 0
    n_missing_barcode = 0

    print(f"\nApplying mean shifts to {len(adata)} cells...")

    for i in tqdm(range(len(adata)), desc="Processing cells"):
        pert = perturbations[i]
        cell_type = cell_types[i]
        ctrl_barcode = ctrl_barcodes[i]

        # PBS controls: use their own embedding
        if pert == control_pert:
            predictions[i] = embeddings[i]
            n_control_cells += 1
            continue

        # Perturbed cells: look up control cell and apply shift
        if ctrl_barcode not in control_lookup:
            # Fallback: use cell's own embedding
            predictions[i] = embeddings[i]
            n_missing_barcode += 1
            continue

        control_embedding = control_lookup[ctrl_barcode]
        key = (cell_type, pert)

        if key in shift_table.shifts:
            # Apply mean shift
            predictions[i] = control_embedding + shift_table.shifts[key]
            n_shifted += 1
        else:
            # No shift available: use control as-is
            predictions[i] = control_embedding
            n_no_shift += 1

    print(f"\nProcessing complete:")
    print(f"  PBS control cells: {n_control_cells}")
    print(f"  Perturbed cells with shift applied: {n_shifted}")
    print(f"  Perturbed cells without available shift: {n_no_shift}")
    print(f"  Cells with missing control barcode: {n_missing_barcode}")

    # Store predictions
    adata.obsm[pred_embed_key] = predictions

    # Save
    print(f"\nSaving to: {output_path}")
    adata.write_h5ad(output_path)

    print(f"✓ Done!")
    print(f"  Predictions in .obsm['{pred_embed_key}']")


def create_control_passthrough_h5ad(
    adata_test_path: str,
    output_path: str,
    control_pert: str = "non-targeting",
    pert_col: str = "target_gene",
    embed_key: Optional[str] = None,
    pred_embed_key: str = "model_preds",
    ctrl_barcode_col: str = "ctrl_cell_barcode"
):
    """
    Create control passthrough baseline (no shift applied).

    For each cell:
    - If PBS control: use its own embedding
    - If perturbed: use matched control cell's embedding (NO shift)

    This tests if mean shift is even better than doing nothing.

    Parameters
    ----------
    adata_test_path : str
        Path to real.h5ad (contains both PBS and perturbed cells)
    output_path : str
        Path to save predictions
    control_pert : str
        Name of control perturbation (e.g., 'PBS')
    pert_col : str
        Column for perturbation (e.g., 'cytokine')
    embed_key : str or None
        Key in .obsm for embeddings (None = use .X)
    pred_embed_key : str
        Key for storing predictions in .obsm
    ctrl_barcode_col : str
        Column containing matched control cell barcodes

    Returns
    -------
    None
        Saves predictions to output_path
    """
    print("\n" + "="*60)
    print("CREATING CONTROL PASSTHROUGH BASELINE (No Shift)")
    print("="*60)
    print(f"Data: {adata_test_path}")
    print(f"Output: {output_path}")

    # Load data
    adata = sc.read_h5ad(adata_test_path)
    print(f"\nData shape: {adata.shape}")

    # Get embeddings
    if embed_key and embed_key in adata.obsm:
        embeddings = adata.obsm[embed_key]
        print(f"Using embeddings from .obsm['{embed_key}']")
    else:
        embeddings = adata.X.toarray() if hasattr(adata.X, 'toarray') else adata.X
        print(f"Using embeddings from .X")

    # Create barcode lookup
    control_lookup = {barcode: embeddings[i] for i, barcode in enumerate(adata.obs.index)}
    print(f"Created lookup for {len(control_lookup)} cells")

    # Initialize predictions
    predictions = np.zeros_like(embeddings, dtype=np.float32)

    perturbations = adata.obs[pert_col].values
    ctrl_barcodes = adata.obs[ctrl_barcode_col].values

    n_control_cells = 0
    n_copied = 0
    n_missing = 0

    print(f"\nCopying control embeddings (NO SHIFT) for {len(adata)} cells...")

    for i in tqdm(range(len(adata)), desc="Processing cells"):
        pert = perturbations[i]
        ctrl_barcode = ctrl_barcodes[i]

        # PBS controls: use their own embedding
        if pert == control_pert:
            predictions[i] = embeddings[i]
            n_control_cells += 1
            continue

        # Perturbed cells: use matched control cell (no shift)
        if ctrl_barcode in control_lookup:
            predictions[i] = control_lookup[ctrl_barcode]
            n_copied += 1
        else:
            # Fallback: use cell's own embedding
            predictions[i] = embeddings[i]
            n_missing += 1

    print(f"\nProcessing complete:")
    print(f"  PBS control cells: {n_control_cells}")
    print(f"  Control embeddings copied: {n_copied}")
    print(f"  Cells with missing control barcode: {n_missing}")

    # Store predictions
    adata.obsm[pred_embed_key] = predictions

    # Save
    print(f"\nSaving to: {output_path}")
    adata.write_h5ad(output_path)

    print(f"✓ Done!")
    print(f"  Predictions in .obsm['{pred_embed_key}']")


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
