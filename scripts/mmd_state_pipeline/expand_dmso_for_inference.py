#!/usr/bin/env python3
"""
Expand DMSO control dataset for state tx inference on held-out perturbations.

This script:
1. Loads dmso_controls.h5ad from specified directory
2. Gets held-out test perturbations from split assignments
3. Filters to cell lines that have test perturbations
4. For each cell line: expands DMSO controls to (test perturbations + DMSO control)
5. Saves expanded dataset for inference in the same directory

The output includes both test drug predictions AND DMSO controls for proper evaluation.

Usage:
    python expand_dmso_for_inference.py /path/to/split_outputs_directory/
    python expand_dmso_for_inference.py  # Uses current directory
    python expand_dmso_for_inference.py --embedding-key X_hvg  # Specify embedding key
    python expand_dmso_for_inference.py --dmso-condition DMSO_TF_00  # Specify DMSO condition
"""

import pandas as pd
import scanpy as sc
import numpy as np
from tqdm import tqdm
import os
import argparse
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Expand DMSO controls for state transport inference")
    parser.add_argument('directory', nargs='?', default='.', 
                        help='Directory containing dmso_controls.h5ad (default: current directory)')
    parser.add_argument('--split-file', 
                        default='/tahoe/drive_3/ANALYSIS/analysis_190/Data/20250618.tahoe_embeddings_70M_DC_split_assignments.parquet',
                        help='Path to split assignments file')
    parser.add_argument('--embedding-key', 
                        default=None,
                        help='Embedding key in obsm to use (default: auto-detect from X_hvg, embedding, etc.)')
    parser.add_argument('--dmso-condition', 
                        default=None,
                        help='DMSO condition name (default: auto-detect from dmso_controls.h5ad)')
    
    args = parser.parse_args()
    
    # Validate directory and input file
    input_dir = Path(args.directory)
    if not input_dir.exists():
        raise FileNotFoundError(f"Directory not found: {input_dir}")
    
    input_file = input_dir / 'dmso_controls.h5ad'
    if not input_file.exists():
        raise FileNotFoundError(f"Required file not found: {input_file}")
    
    output_file = input_dir / 'dmso_controls_expanded_for_inference.h5ad'
    
    print(f"Input directory: {input_dir}")
    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    
    # Load DMSO controls
    print("\nLoading DMSO control dataset...")
    dmso_adata = sc.read_h5ad(input_file)
    print(f"Original dataset shape: {dmso_adata.shape}")
    print(f"Original obs columns: {dmso_adata.obs.columns.tolist()}")
    print(f"Available obsm keys: {list(dmso_adata.obsm.keys())}")
    
    # Find the embedding key (explicit or auto-detect)
    embedding_key = args.embedding_key
    
    if embedding_key is None:
        # Auto-detect embedding key (flexible to handle different naming)
        print("Auto-detecting embedding key...")
        for key in dmso_adata.obsm.keys():
            if 'embedding' in key.lower() or 'hvg' in key.lower() or 'X_' in key:
                embedding_key = key
                break
        
        if embedding_key is None:
            raise ValueError(f"No embedding found in obsm. Available keys: {list(dmso_adata.obsm.keys())}")
        print(f"Auto-detected embedding key: '{embedding_key}'")
    else:
        # Validate that the specified embedding key exists
        if embedding_key not in dmso_adata.obsm.keys():
            raise ValueError(f"Specified embedding key '{embedding_key}' not found in obsm. Available keys: {list(dmso_adata.obsm.keys())}")
        print(f"Using specified embedding key: '{embedding_key}'")
    
    # Find the DMSO condition (explicit or auto-detect)
    dmso_condition = args.dmso_condition
    
    if dmso_condition is None:
        # Auto-detect DMSO condition from the dmso_controls data
        print("Auto-detecting DMSO condition...")
        dmso_conditions = dmso_adata.obs['drug_dose'].unique()
        if len(dmso_conditions) == 1:
            dmso_condition = dmso_conditions[0]
            print(f"Auto-detected DMSO condition: '{dmso_condition}'")
        else:
            raise ValueError(f"Multiple drug_dose values found in DMSO controls: {dmso_conditions}. Please specify --dmso-condition.")
    else:
        print(f"Using specified DMSO condition: '{dmso_condition}'")
        # Validate that the DMSO condition exists in the data
        if dmso_condition not in dmso_adata.obs['drug_dose'].unique():
            available = dmso_adata.obs['drug_dose'].unique()
            raise ValueError(f"Specified DMSO condition '{dmso_condition}' not found in dmso_controls. Available: {available}")
    
    # Get held-out test perturbations and filter by cell lines
    print("\nLoading held-out test perturbations...")
    split_df = pd.read_parquet(args.split_file)
    test_df = split_df[split_df['split_assignment'] == 'test']
    
    # Get cell lines that have test perturbations
    available_cell_lines = set(dmso_adata.obs['cell_line'].unique())
    test_cell_lines = set(test_df['cell_line'].unique())
    relevant_cell_lines = available_cell_lines.intersection(test_cell_lines)
    
    print(f"Cell lines in DMSO controls: {len(available_cell_lines)}")
    print(f"Cell lines with test perturbations: {len(test_cell_lines)}")
    print(f"Cell lines to process (intersection): {len(relevant_cell_lines)}")
    
    if len(relevant_cell_lines) == 0:
        raise ValueError("No overlap between cell lines in DMSO controls and test perturbations")
    
    # Filter DMSO data to only relevant cell lines
    dmso_adata = dmso_adata[dmso_adata.obs['cell_line'].isin(relevant_cell_lines)].copy()
    print(f"Filtered DMSO dataset to {dmso_adata.n_obs} cells from {len(relevant_cell_lines)} cell lines")
    
    # Create expanded dataset with cell-line-specific perturbations
    print("\nCreating cell-line-specific expanded dataset...")
    
    obs_data = []
    all_embeddings = []
    row_idx = 0
    
    for cell_line in tqdm(sorted(relevant_cell_lines), desc="Processing cell lines"):
        # Get test perturbations for this specific cell line
        cell_line_test_df = test_df[test_df['cell_line'] == cell_line]
        cell_line_test_perturbations = sorted(cell_line_test_df['drug_dose'].unique())
        
        # Add DMSO condition to perturbations for this cell line
        all_perturbations_for_cell_line = cell_line_test_perturbations + [dmso_condition]
        
        print(f"  {cell_line}: {len(cell_line_test_perturbations)} test perturbations + 1 DMSO = {len(all_perturbations_for_cell_line)} total")
        
        # Get DMSO cells for this cell line
        cell_line_dmso_mask = dmso_adata.obs['cell_line'] == cell_line
        cell_line_dmso_indices = np.where(cell_line_dmso_mask)[0]
        
        # Expand each DMSO cell from this cell line to all perturbations
        for cell_idx in cell_line_dmso_indices:
            cell_obs = dmso_adata.obs.iloc[cell_idx]
            cell_embedding = dmso_adata.obsm[embedding_key][cell_idx]
            
            # Create rows for all perturbations for this cell
            for pert_idx, perturbation in enumerate(all_perturbations_for_cell_line):
                obs_row = {
                    'cell_line': cell_obs['cell_line'],
                    'drug_dose': perturbation,
                    'drug_dose_id': row_idx,
                    'sample_idx': row_idx,
                    'batch': cell_obs['batch'],
                    'original_cell_idx': cell_idx,
                    'perturbation_idx': pert_idx,
                    'is_dmso_control': perturbation == dmso_condition
                }
                # Add any other available columns from the original data
                for col in cell_obs.index:
                    if col not in obs_row and col != 'drug_dose':
                        obs_row[col] = cell_obs[col]
                
                obs_data.append(obs_row)
                all_embeddings.append(cell_embedding)
                row_idx += 1
    
    total_rows = len(obs_data)
    print(f"\nTotal expanded rows: {total_rows}")
    
    # Create DataFrame from obs data
    print("Creating obs DataFrame...")
    expanded_obs_df = pd.DataFrame(obs_data)
    
    # Convert embeddings list to array
    print("Creating embeddings array...")
    expanded_embeddings_array = np.array(all_embeddings)
    
    # Create new AnnData object
    expanded_adata = sc.AnnData(
        X=np.zeros((total_rows, dmso_adata.n_vars)),  # Empty X matrix
        obs=expanded_obs_df,
        var=dmso_adata.var.copy(),
        obsm={embedding_key: expanded_embeddings_array}
    )
    
    print(f"Expanded dataset shape: {expanded_adata.shape}")
    print(f"Expanded obs columns: {expanded_adata.obs.columns.tolist()}")
    print(f"Expanded obsm keys: {list(expanded_adata.obsm.keys())}")
    
    # Verify expanded dataset
    print("\nVerifying expanded dataset:")
    print(f"Unique perturbations: {len(expanded_adata.obs['drug_dose'].unique())}")
    print(f"Unique cell lines: {len(expanded_adata.obs['cell_line'].unique())}")
    print(f"Unique original cells: {len(expanded_adata.obs['original_cell_idx'].unique())}")
    print(f"DMSO control rows: {expanded_adata.obs['is_dmso_control'].sum()}")
    print(f"Test perturbation rows: {(~expanded_adata.obs['is_dmso_control']).sum()}")
    print(f"Total rows: {expanded_adata.n_obs}")
    
    # Verify each cell line has DMSO controls
    dmso_by_cell_line = expanded_adata.obs[expanded_adata.obs['is_dmso_control']]['cell_line'].unique()
    print(f"Cell lines with DMSO controls: {len(dmso_by_cell_line)} (should equal {len(relevant_cell_lines)})")
    
    # Save expanded dataset
    print(f"\nSaving expanded dataset to {output_file}...")
    expanded_adata.write(output_file)
    
    print("Done!")
    print(f"Expanded dataset saved: {output_file}")
    print(f"Ready for state tx inference on {len(relevant_cell_lines)} cell lines with test perturbations + DMSO controls")
    print(f"Dataset includes both test predictions and DMSO baselines for proper evaluation")

if __name__ == "__main__":
    main()