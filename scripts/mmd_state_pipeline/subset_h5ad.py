#!/usr/bin/env python3
"""
Subset large .h5ad files to extract specific drug-dose and cell-line combinations.
Creates a filtered copy in 'single_condition/' subdirectory with all .obsm fields preserved.
"""

import argparse
import anndata as ad
import pandas as pd
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

def subset_h5ad(input_file, target_drug_doses=['DMSO_TF_00', 'Adenine_50'], target_cell_line='CVCL_0334'):
    """
    Subset h5ad file to retain only specified drug_dose and cell_line combinations.
    
    Args:
        input_file (str): Path to input .h5ad file
        target_drug_doses (list): List of drug_dose values to retain
        target_cell_line (str): Cell line to retain
    
    Returns:
        str: Path to output file
    """
    input_path = Path(input_file)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    
    print(f"Loading h5ad file: {input_file}")
    
    # Load the full dataset
    adata = ad.read_h5ad(input_file)
    
    print(f"Original dataset shape: {adata.shape}")
    print(f"Available drug_dose values: {sorted(adata.obs['drug_dose'].unique())}")
    print(f"Available cell_line values: {sorted(adata.obs['cell_line'].unique())}")
    
    # Create boolean filters
    drug_filter = adata.obs['drug_dose'].isin(target_drug_doses)
    cell_filter = adata.obs['cell_line'] == target_cell_line
    combined_filter = drug_filter & cell_filter
    
    print(f"Cells matching drug_dose filter ({target_drug_doses}): {drug_filter.sum()}")
    print(f"Cells matching cell_line filter ({target_cell_line}): {cell_filter.sum()}")
    print(f"Cells matching both filters: {combined_filter.sum()}")
    
    if combined_filter.sum() == 0:
        print("WARNING: No cells match the filtering criteria!")
        return None
    
    # Apply filter - this preserves all .obsm, .var, .uns data
    adata_subset = adata[combined_filter, :].copy()
    
    print(f"Filtered dataset shape: {adata_subset.shape}")
    print(f"Preserved .obsm keys: {list(adata_subset.obsm.keys())}")
    
    # Create output directory and file path
    output_dir = input_path.parent / "single_condition"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / input_path.name
    
    print(f"Saving subset to: {output_file}")
    
    # Save the subset
    adata_subset.write(output_file)
    
    print(f"Successfully created subset with {adata_subset.shape[0]} cells and {adata_subset.shape[1]} genes")
    
    return str(output_file)

def main():
    parser = argparse.ArgumentParser(
        description="Subset .h5ad file to extract specific drug-dose and cell-line combinations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage with default filters
    python subset_h5ad.py /path/to/data.h5ad
    
    # Custom drug doses
    python subset_h5ad.py /path/to/data.h5ad --drug-doses DMSO_TF_00 Adenine_100
    
    # Custom cell line
    python subset_h5ad.py /path/to/data.h5ad --cell-line CVCL_1234

Default filters:
    drug_dose: DMSO_TF_00, Adenine_50
    cell_line: CVCL_0334
        """
    )
    
    parser.add_argument("input_file", help="Path to input .h5ad file")
    parser.add_argument("--drug-doses", nargs='+', default=['DMSO_TF_00', 'Adenine_50'],
                        help="Drug dose conditions to retain (default: DMSO_TF_00 Adenine_50)")
    parser.add_argument("--cell-line", default='CVCL_0334',
                        help="Cell line to retain (default: CVCL_0334)")
    
    args = parser.parse_args()
    
    try:
        output_file = subset_h5ad(
            args.input_file, 
            target_drug_doses=args.drug_doses,
            target_cell_line=args.cell_line
        )
        
        if output_file:
            print(f"\n✅ Success! Subset saved to: {output_file}")
        else:
            print("\n❌ No output created - check filtering criteria")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()