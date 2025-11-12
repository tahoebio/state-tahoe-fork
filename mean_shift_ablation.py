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


def evaluate_mean_shift_baseline(
    adata_test_path: str,
    shift_table: MeanShiftTable,
    control_pert: str = "non-targeting",
    cell_type_col: str = "cell_type",
    pert_col: str = "target_gene",
    embed_key: Optional[str] = None
) -> Dict:
    """
    Evaluate mean shift baseline on test data.

    For each (cell_type, perturbation) combination:
    1. Get all control cells of that cell type
    2. Apply the mean shift to each control cell
    3. Compare predictions to actual perturbed cells with the same (cell_type, perturbation)

    Parameters
    ----------
    adata_test_path : str
        Path to test dataset h5ad file
    shift_table : MeanShiftTable
        Pre-computed mean shift table
    control_pert : str
        Name of control perturbation
    cell_type_col : str
        Column name for cell types
    pert_col : str
        Column name for perturbations
    embed_key : str or None
        Key in adata.obsm for embeddings

    Returns
    -------
    results : dict
        Dictionary with evaluation metrics and predictions
    """
    print("\n" + "="*60)
    print("EVALUATING MEAN SHIFT BASELINE")
    print("="*60)
    print(f"Loading test data from: {adata_test_path}")

    # Load data
    adata_test = sc.read_h5ad(adata_test_path)

    # Get embeddings
    if embed_key and embed_key in adata_test.obsm:
        embeddings = adata_test.obsm[embed_key]
        print(f"Using embeddings from adata_test.obsm['{embed_key}']")
    else:
        try:
            embeddings = adata_test.X.toarray()
        except:
            embeddings = adata_test.X
        print(f"Using embeddings from adata_test.X")

    print(f"Test data shape: {embeddings.shape}")

    cell_types = adata_test.obs[cell_type_col].values
    perturbations = adata_test.obs[pert_col].values

    all_predictions = []
    all_ground_truth = []
    all_cell_types = []
    all_perturbations = []

    # For each (cell_type, perturbation) pair
    unique_cell_types = np.unique(cell_types)
    unique_perts = np.unique(perturbations)

    print(f"\nProcessing {len(unique_cell_types)} cell types × {len(unique_perts)} perturbations")

    n_comparisons = 0
    n_skipped = 0

    for cell_type in tqdm(unique_cell_types, desc="Cell types"):
        for pert in unique_perts:
            if pert == control_pert:
                continue  # Skip control

            # Get control cells for this cell type
            control_mask = (cell_types == cell_type) & (perturbations == control_pert)
            control_cells = embeddings[control_mask]  # [n_control, D]

            # Get perturbed cells for this (cell_type, pert)
            pert_mask = (cell_types == cell_type) & (perturbations == pert)
            pert_cells = embeddings[pert_mask]  # [n_pert, D]

            if control_cells.shape[0] == 0 or pert_cells.shape[0] == 0:
                n_skipped += 1
                continue  # Skip if no data

            # Get the mean shift for this combination
            key = (cell_type, pert)
            if key not in shift_table.shifts:
                n_skipped += 1
                continue  # Skip if we didn't compute this shift

            shift = shift_table.shifts[key]  # [D]

            # Apply shift to EACH control cell
            # predictions shape: [n_control, D]
            predictions = control_cells + shift  # Broadcasting

            # Match sizes: we have n_control predictions and n_pert ground truths
            n_control = control_cells.shape[0]
            n_pert = pert_cells.shape[0]

            # Take minimum to ensure equal sizes
            n_samples = min(n_control, n_pert)
            predictions = predictions[:n_samples]
            ground_truth = pert_cells[:n_samples]

            # Store
            all_predictions.append(predictions)
            all_ground_truth.append(ground_truth)
            all_cell_types.extend([cell_type] * n_samples)
            all_perturbations.extend([pert] * n_samples)

            n_comparisons += n_samples

    if len(all_predictions) == 0:
        print("\nERROR: No predictions could be made!")
        return {}

    # Concatenate all predictions and ground truths
    all_predictions = np.vstack(all_predictions)  # [total_n, D]
    all_ground_truth = np.vstack(all_ground_truth)  # [total_n, D]

    print(f"\nTotal predictions: {all_predictions.shape[0]}")
    print(f"Skipped combinations: {n_skipped}")

    # Compute overall MSE
    overall_mse = np.mean((all_predictions - all_ground_truth) ** 2)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Overall MSE: {overall_mse:.6f}")

    # Compute per cell type MSE
    print(f"\nPer Cell Type MSE:")
    per_celltype_mse = {}
    for ct in unique_cell_types:
        ct_mask = np.array(all_cell_types) == ct
        if ct_mask.sum() > 0:
            ct_mse = np.mean((all_predictions[ct_mask] - all_ground_truth[ct_mask]) ** 2)
            per_celltype_mse[ct] = ct_mse
            print(f"  {ct}: {ct_mse:.6f} ({ct_mask.sum()} predictions)")

    # Compute per perturbation MSE
    print(f"\nTop 10 Perturbations by MSE:")
    per_pert_mse = {}
    for pert in unique_perts:
        if pert == control_pert:
            continue
        pert_mask = np.array(all_perturbations) == pert
        if pert_mask.sum() > 0:
            pert_mse = np.mean((all_predictions[pert_mask] - all_ground_truth[pert_mask]) ** 2)
            per_pert_mse[pert] = pert_mse

    # Sort and show top 10
    sorted_perts = sorted(per_pert_mse.items(), key=lambda x: x[1], reverse=True)
    for pert, mse in sorted_perts[:10]:
        n = np.sum(np.array(all_perturbations) == pert)
        print(f"  {pert}: {mse:.6f} ({n} predictions)")

    results = {
        'overall_mse': overall_mse,
        'per_celltype_mse': per_celltype_mse,
        'per_perturbation_mse': per_pert_mse,
        'n_predictions': all_predictions.shape[0],
        'predictions': all_predictions,
        'ground_truth': all_ground_truth,
        'cell_types': all_cell_types,
        'perturbations': all_perturbations
    }

    return results


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

    print("\n[2/3] Evaluating on test data...")

    # Uncomment and set your actual paths:
    # results = evaluate_mean_shift_baseline(
    #     adata_test_path="path/to/test_data.h5ad",
    #     shift_table=shift_table,
    #     control_pert="non-targeting",
    #     cell_type_col="cell_type",
    #     pert_col="target_gene",
    #     embed_key="X_hvg"
    # )

    # Save results
    # with open("mean_shift_results.pkl", "wb") as f:
    #     pickle.dump(results, f)

    print("\n" + "="*60)
    print("Example complete! Uncomment code to run on real data.")
    print("="*60)
