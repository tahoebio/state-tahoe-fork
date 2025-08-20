#!/usr/bin/env python3
"""
Compute Perturbation Mean baseline using centroids with hierarchical summaries.

This script implements the PERTURBATION MEAN baseline for drug response prediction 
using pre-computed centroid embeddings. It computes mean perturbation effects by 
averaging deltas across cell lines for each drug, then uses these drug-specific 
effects to predict responses on test data. Evaluation uses Pearson correlation 
between predicted and true deltas. All computations maintain strict plate boundaries 
to avoid batch effects.

Baseline Logic:
- For each perturbation: average response deltas across training cell lines
- Prediction: control(cell_line) + mean_delta(perturbation)

Key features:
- Loads centroid H5AD files maintaining plate boundaries
- Parses TOML splits for train/test assignments  
- Computes correlations directly without storing all deltas
- Generates hierarchical summaries: per cell-type per plate, per cell-type across plates, and overall
- Progress tracking with tqdm for all major operations

Usage:
    python centroid_perturbation_mean_baseline.py \
        --toml-file path/to/splits.toml \
        --centroids-dir path/to/centroids/directory \
        --output-dir results/
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from tqdm import tqdm
import toml
import warnings
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compute Perturbation Mean baseline using centroids",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        '--toml-file', 
        required=True, 
        help='Path to TOML split file (e.g., generalization_converted_cell_lines.toml)'
    )
    parser.add_argument(
        '--centroids-dir', 
        required=True, 
        help='Directory containing plate centroid H5AD files (e.g., by_plate_centroids/)'
    )
    parser.add_argument(
        '--cell-col', 
        default='cell_line_id', 
        help='Column name for cell type/line identifiers'
    )
    parser.add_argument(
        '--pert-col', 
        default='drugname_drugconc', 
        help='Column name for perturbation identifiers'
    )
    parser.add_argument(
        '--control-pert', 
        default='DMSO_TF', 
        help='Control condition name (will be mapped to actual format)'
    )
    parser.add_argument(
        '--embed-key', 
        default='X_hvg', 
        help='Key for embeddings in .obsm'
    )
    parser.add_argument(
        '--output-dir', 
        default='./pearson_baseline_results', 
        help='Output directory for results'
    )
    parser.add_argument(
        '--ignore-plate-boundaries',
        action='store_true',
        help='Compute global mean effects across all plates, ignoring plate boundaries (experimental)'
    )
    
    return parser.parse_args()


def load_plate_centroids(centroids_dir):
    """Load plate files with progress tracking."""
    plate_files = list(Path(centroids_dir).glob("plate_*.h5ad"))
    
    if not plate_files:
        raise ValueError(f"No plate_*.h5ad files found in {centroids_dir}")
    
    print(f"Found {len(plate_files)} plate files")
    
    plate_data = {}
    for plate_file in tqdm(plate_files, desc="Loading plate centroids"):
        plate_name = plate_file.stem.replace('plate_', '')  # e.g., "plate1" from "plate_plate1"
        adata = sc.read_h5ad(plate_file)
        plate_data[plate_name] = adata
        print(f"  Loaded {plate_name}: {adata.shape[0]} observations, {adata.shape[1]} variables")
    
    return plate_data


def parse_toml_splits(toml_file):
    """Parse TOML with progress tracking.
    
    Logic:
    1. If cell lines are NOT mentioned in TOML → all perturbations are training data
    2. If cell lines ARE mentioned in TOML → only val/test perturbations are held out,
       all other perturbations for those cell lines are training data
    """
    print(f"Parsing TOML file: {toml_file}")
    
    if not os.path.exists(toml_file):
        raise ValueError(f"TOML file not found: {toml_file}")
    
    data = toml.load(toml_file)
    
    explicit_splits = {}  # Only stores val/test combinations
    holdout_cells = set()  # Cell lines with explicit splits
    
    if 'fewshot' in data:
        cell_lines = list(data['fewshot'].keys())
        for cell_line_key in tqdm(cell_lines, desc="Parsing TOML splits"):
            cell_line = cell_line_key.split('.')[-1]  # Extract CVCL_XXXX from "tahoe.CVCL_XXXX"
            holdout_cells.add(cell_line)
            assignments = data['fewshot'][cell_line_key]
            
            # Only store val and test - everything else is implicitly training
            for split_type in ['val', 'test']:
                if split_type in assignments:
                    for pert in assignments[split_type]:
                        explicit_splits[(cell_line, pert)] = split_type
    else:
        raise ValueError("TOML file missing 'fewshot' section")
    
    # Print split statistics
    split_counts = {}
    for split in explicit_splits.values():
        split_counts[split] = split_counts.get(split, 0) + 1
    
    print(f"Explicit holdout combinations:")
    for split_type, count in split_counts.items():
        print(f"  {split_type}: {count} combinations")
    
    print(f"Holdout cell lines: {len(holdout_cells)} ({sorted(list(holdout_cells))})")
    print(f"NOTE: All other cell lines and unlisted perturbations are training data")
    
    return explicit_splits, holdout_cells


def evaluate_plate(plate_adata, plate_name, explicit_splits, holdout_cells, control_pert, pert_col, 
                   cell_col, embed_key):
    """Process one plate with progress tracking."""
    
    # Map control string if needed  
    actual_control = f"[('{control_pert}', 0.0, 'uM')]" if control_pert == "DMSO_TF" else control_pert
    
    # Get controls for this plate
    controls = {}
    control_mask = plate_adata.obs[pert_col] == actual_control
    
    if not control_mask.any():
        print(f"  WARNING: No control perturbation '{actual_control}' found in plate {plate_name}")
        return []
    
    control_indices = plate_adata.obs[control_mask].index
    for i, idx in enumerate(tqdm(control_indices, 
                                desc=f"  Loading controls for {plate_name}", 
                                leave=False)):
        cell_line = plate_adata.obs.loc[idx, cell_col]
        # Get the integer position for the obsm array indexing
        int_idx = plate_adata.obs.index.get_loc(idx)
        controls[cell_line] = plate_adata.obsm[embed_key][int_idx]
    
    print(f"  Found controls for {len(controls)} cell lines")
    
    # Get unique perturbations (excluding control)
    perturbations = [p for p in plate_adata.obs[pert_col].unique() if p != actual_control]
    
    # Compute mean perturbation effects from training data
    pert_effects = {}
    for pert in tqdm(perturbations, 
                     desc=f"  Computing training effects for {plate_name}", 
                     leave=False):
        training_deltas = []
        for idx in plate_adata.obs[plate_adata.obs[pert_col] == pert].index:
            cell_line = plate_adata.obs.loc[idx, cell_col]
            # Check if this is training data:
            # 1. Cell line not in holdout_cells (implicit training) OR
            # 2. Cell line in holdout_cells but this perturbation not in explicit_splits (implicit training)
            is_training = (cell_line not in holdout_cells or 
                          explicit_splits.get((cell_line, pert)) is None)
            
            if is_training and cell_line in controls:
                # Get the integer position for the obsm array indexing
                int_idx = plate_adata.obs.index.get_loc(idx)
                delta = plate_adata.obsm[embed_key][int_idx] - controls[cell_line]
                training_deltas.append(delta)
        
        if training_deltas:
            pert_effects[pert] = np.mean(training_deltas, axis=0)
    
    print(f"  Computed {len(pert_effects)} perturbation effects from training data")
    
    # Evaluate on test set
    test_indices = []
    for idx in plate_adata.obs.index:
        cell_line = plate_adata.obs.loc[idx, cell_col]
        pert = plate_adata.obs.loc[idx, pert_col]
        # Check if this is test data: explicitly marked as 'test' in explicit_splits
        is_test = explicit_splits.get((cell_line, pert)) == 'test'
        
        if (is_test and 
            pert in pert_effects and 
            cell_line in controls):
            test_indices.append(idx)
    
    print(f"  Found {len(test_indices)} test combinations to evaluate")
    
    correlations = []
    for idx in tqdm(test_indices, 
                    desc=f"  Computing correlations for {plate_name}", 
                    leave=False):
        cell_line = plate_adata.obs.loc[idx, cell_col]
        pert = plate_adata.obs.loc[idx, pert_col]
        
        # True delta
        # Get the integer position for the obsm array indexing  
        int_idx = plate_adata.obs.index.get_loc(idx)
        true_delta = plate_adata.obsm[embed_key][int_idx] - controls[cell_line]
        
        # Predicted delta (mean effect from training)
        pred_delta = pert_effects[pert]
        
        # Compute Pearson correlation
        if len(true_delta) > 1 and len(pred_delta) > 1:
            corr, p_value = pearsonr(true_delta, pred_delta)
            
            # Handle NaN correlations (can occur with constant vectors)
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        correlations.append({
            'plate': plate_name,
            'cell_line': cell_line,
            'perturbation': pert,
            'pearson_correlation': corr
        })
    
    print(f"  Computed {len(correlations)} test correlations for {plate_name}")
    return correlations


def evaluate_global(plate_data, explicit_splits, holdout_cells, control_pert, pert_col, 
                    cell_col, embed_key):
    """Evaluate using global mean effects across all plates (ignoring plate boundaries)."""
    
    print("Computing global mean effects across all plates...")
    
    # Map control string if needed  
    actual_control = f"[('{control_pert}', 0.0, 'uM')]" if control_pert == "DMSO_TF" else control_pert
    
    # Combine all plate data while preserving plate information
    all_data = []
    for plate_name, plate_adata in plate_data.items():
        # Add plate name to a copy of obs for tracking
        obs_copy = plate_adata.obs.copy()
        obs_copy['original_plate'] = plate_name
        
        # Create temporary combined data structure
        for idx in plate_adata.obs.index:
            int_idx = plate_adata.obs.index.get_loc(idx)
            all_data.append({
                'plate': plate_name,
                'original_index': idx,
                'int_idx': int_idx,
                'cell_line': plate_adata.obs.loc[idx, cell_col],
                'perturbation': plate_adata.obs.loc[idx, pert_col],
                'embedding': plate_adata.obsm[embed_key][int_idx],
                'adata_ref': plate_adata  # Keep reference for later access
            })
    
    print(f"  Combined {len(all_data)} observations from {len(plate_data)} plates")
    
    # Build global controls dictionary
    global_controls = {}
    control_count = 0
    for data_point in tqdm(all_data, desc="  Loading global controls", leave=False):
        if data_point['perturbation'] == actual_control:
            cell_line = data_point['cell_line']
            if cell_line not in global_controls:
                global_controls[cell_line] = []
            global_controls[cell_line].append(data_point['embedding'])
            control_count += 1
    
    # Average controls per cell line across all plates
    for cell_line in global_controls:
        global_controls[cell_line] = np.mean(global_controls[cell_line], axis=0)
    
    print(f"  Found global controls for {len(global_controls)} cell lines ({control_count} total control observations)")
    
    # Get all unique perturbations (excluding control)
    all_perturbations = set()
    for data_point in all_data:
        if data_point['perturbation'] != actual_control:
            all_perturbations.add(data_point['perturbation'])
    
    all_perturbations = list(all_perturbations)
    print(f"  Found {len(all_perturbations)} unique perturbations")
    
    # Compute global mean perturbation effects from ALL training data
    global_pert_effects = {}
    for pert in tqdm(all_perturbations, desc="  Computing global training effects", leave=False):
        training_deltas = []
        for data_point in all_data:
            if data_point['perturbation'] == pert:
                cell_line = data_point['cell_line']
                # Check if this is training data using same logic as plate-specific version
                is_training = (cell_line not in holdout_cells or 
                              explicit_splits.get((cell_line, pert)) is None)
                
                if is_training and cell_line in global_controls:
                    delta = data_point['embedding'] - global_controls[cell_line]
                    training_deltas.append(delta)
        
        if training_deltas:
            global_pert_effects[pert] = np.mean(training_deltas, axis=0)
    
    print(f"  Computed {len(global_pert_effects)} global perturbation effects from training data")
    
    # Evaluate on test set using global effects
    test_data_points = []
    for data_point in all_data:
        cell_line = data_point['cell_line']
        pert = data_point['perturbation']
        # Check if this is test data: explicitly marked as 'test' in explicit_splits
        is_test = explicit_splits.get((cell_line, pert)) == 'test'
        
        if (is_test and 
            pert in global_pert_effects and 
            cell_line in global_controls):
            test_data_points.append(data_point)
    
    print(f"  Found {len(test_data_points)} test combinations to evaluate with global effects")
    
    # Compute correlations using global effects
    correlations = []
    for data_point in tqdm(test_data_points, desc="  Computing global correlations", leave=False):
        cell_line = data_point['cell_line']
        pert = data_point['perturbation']
        plate_name = data_point['plate']
        
        # True delta (same as before)
        true_delta = data_point['embedding'] - global_controls[cell_line]
        
        # Predicted delta (now using GLOBAL effect)
        pred_delta = global_pert_effects[pert]
        
        # Compute Pearson correlation
        if len(true_delta) > 1 and len(pred_delta) > 1:
            corr, p_value = pearsonr(true_delta, pred_delta)
            
            # Handle NaN correlations (can occur with constant vectors)
            if not np.isfinite(corr):
                corr = 0.0
        else:
            corr = 0.0
        
        correlations.append({
            'plate': plate_name,  # Still track plate for analysis
            'cell_line': cell_line,
            'perturbation': pert,
            'pearson_correlation': corr
        })
    
    print(f"  Computed {len(correlations)} test correlations using global effects")
    return correlations


def compute_hierarchical_summaries(all_correlations):
    """Compute summaries with progress tracking."""
    print("Computing hierarchical summaries...")
    df = pd.DataFrame(all_correlations)
    
    summaries = {}
    
    with tqdm(total=4, desc="Computing summaries") as pbar:
        # 1. Per cell-type per plate
        cell_plate_summary = df.groupby(['cell_line', 'plate'])['pearson_correlation'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).reset_index()
        summaries['per_cell_per_plate'] = cell_plate_summary.to_dict('records')
        pbar.set_postfix({'level': 'cell_per_plate'})
        pbar.update(1)
        
        # 2. Per cell-type across plates
        cell_summary = df.groupby('cell_line')['pearson_correlation'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).reset_index()
        summaries['per_cell_across_plates'] = cell_summary.to_dict('records')
        pbar.set_postfix({'level': 'cell_across_plates'})
        pbar.update(1)
        
        # 3. Per plate across cell types
        plate_summary = df.groupby('plate')['pearson_correlation'].agg([
            'mean', 'std', 'count', 'min', 'max'  
        ]).reset_index()
        summaries['per_plate'] = plate_summary.to_dict('records')
        pbar.set_postfix({'level': 'per_plate'})
        pbar.update(1)
        
        # 4. Overall summary
        summaries['overall'] = {
            'mean': float(df['pearson_correlation'].mean()),
            'std': float(df['pearson_correlation'].std()),
            'count': int(len(df)),
            'min': float(df['pearson_correlation'].min()),
            'max': float(df['pearson_correlation'].max()),
            'median': float(df['pearson_correlation'].median()),
            'q25': float(df['pearson_correlation'].quantile(0.25)),
            'q75': float(df['pearson_correlation'].quantile(0.75))
        }
        pbar.set_postfix({'level': 'overall'})
        pbar.update(1)
    
    return summaries, df


def main():
    """Main execution flow with overall progress tracking."""
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print("\n" + "="*60)
    print("STEP 1: Loading plate centroids")
    print("="*60)
    plate_data = load_plate_centroids(args.centroids_dir)
    
    print("\n" + "="*60)
    print("STEP 2: Parsing TOML splits")
    print("="*60)
    explicit_splits, holdout_cells = parse_toml_splits(args.toml_file)
    
    # Process plates (either plate-aware or global)
    print("\n" + "="*60)
    if args.ignore_plate_boundaries:
        print("STEP 3: Computing global effects (ignoring plate boundaries)")
        print("WARNING: Experimental mode - computing mean effects across all plates")
    else:
        print("STEP 3: Processing plates (plate-aware)")
    print("="*60)
    
    if args.ignore_plate_boundaries:
        # Global evaluation: compute mean effects across all plates
        all_correlations = evaluate_global(
            plate_data, explicit_splits, holdout_cells,
            args.control_pert, args.pert_col, 
            args.cell_col, args.embed_key
        )
    else:
        # Plate-specific evaluation: process each plate separately (original behavior)
        all_correlations = []
        for plate_name, plate_adata in tqdm(plate_data.items(), desc="Processing plates"):
            print(f"\nProcessing plate {plate_name} ({plate_adata.shape[0]} observations)...")
            correlations = evaluate_plate(
                plate_adata, plate_name, explicit_splits, holdout_cells,
                args.control_pert, args.pert_col, 
                args.cell_col, args.embed_key
            )
            all_correlations.extend(correlations)
    
    if not all_correlations:
        print("ERROR: No test correlations computed! Check your data and parameters.")
        return 1
    
    # Compute hierarchical summaries
    print("\n" + "="*60)
    print("STEP 4: Computing summary statistics")
    print("="*60)
    summaries, full_df = compute_hierarchical_summaries(all_correlations)
    
    # Save results
    print("\n" + "="*60)
    print("STEP 5: Saving results")
    print("="*60)
    
    # Save detailed correlations
    output_file = f"{args.output_dir}/detailed_correlations.csv"
    full_df.to_csv(output_file, index=False)
    print(f"Saved detailed correlations to: {output_file}")
    
    # Save summaries as JSON
    summary_file = f"{args.output_dir}/hierarchical_summaries.json"
    with open(summary_file, 'w') as f:
        json.dump(summaries, f, indent=2)
    print(f"Saved hierarchical summaries to: {summary_file}")
    
    # Print key statistics
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Overall Statistics:")
    print(f"  Mean correlation: {summaries['overall']['mean']:.3f}")
    print(f"  Std deviation: {summaries['overall']['std']:.3f}")
    print(f"  Median: {summaries['overall']['median']:.3f}")
    print(f"  Q25-Q75: [{summaries['overall']['q25']:.3f}, {summaries['overall']['q75']:.3f}]")
    print(f"  Range: [{summaries['overall']['min']:.3f}, {summaries['overall']['max']:.3f}]")
    print(f"  N combinations: {summaries['overall']['count']}")
    
    # Print top performing cell lines
    cell_summary = pd.DataFrame(summaries['per_cell_across_plates'])
    if not cell_summary.empty:
        top_cells = cell_summary.nlargest(5, 'mean')[['cell_line', 'mean', 'count']]
        print("\nTop 5 cell lines by mean correlation:")
        for _, row in top_cells.iterrows():
            print(f"  {row['cell_line']}: {row['mean']:.3f} (n={row['count']})")
    
    print("\n" + "="*60)
    print("COMPLETE!")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())