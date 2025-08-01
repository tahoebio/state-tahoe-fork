#!/usr/bin/env python3
"""
Convert generalization_converted.toml to parquet split assignments.

This script creates a parquet file similar to the existing DC split assignments
but reflecting the splits defined in the generalization_converted.toml file.

The key insight is to use the full universe of drug-dose × cell-line combinations
from the existing split assignments, then apply the TOML splits:
- Combinations marked as 'val' or 'test' in TOML get those assignments
- All other combinations from the universe get 'train'
"""

import toml
import polars as pl
from pathlib import Path
import sys


def load_toml_splits(toml_path):
    """
    Load the TOML file and extract split assignments.
    
    Returns:
        dict: {(cell_line, drug_dose): split_assignment}
    """
    print(f"Loading TOML file: {toml_path}")
    with open(toml_path, 'r') as f:
        data = toml.load(f)
    
    split_assignments = {}
    
    # Process fewshot sections
    if 'fewshot' not in data:
        print("Warning: No fewshot section found in TOML file")
        return split_assignments
    
    for key, splits in data['fewshot'].items():
        # Extract cell line from key like "tahoe.CVCL_1097"
        if '.' in key:
            dataset, cell_line = key.split('.', 1)
            print(f"Processing cell line: {cell_line}")
            
            # Process val and test arrays
            for split_type, drug_list in splits.items():
                if split_type in ['val', 'test'] and isinstance(drug_list, list):
                    print(f"  {split_type}: {len(drug_list)} drugs")
                    for drug_dose in drug_list:
                        split_assignments[(cell_line, drug_dose)] = split_type
        else:
            print(f"Warning: Unexpected key format: {key}")
    
    print(f"Total split assignments extracted: {len(split_assignments)}")
    return split_assignments


def load_universe_combinations(universe_path):
    """
    Load the full universe of drug-dose × cell-line combinations.
    
    Returns:
        pl.DataFrame: with columns ['drug_dose', 'cell_line', 'split_assignment']
    """
    print(f"Loading universe combinations from: {universe_path}")
    
    # Load the existing split assignments to get the full universe
    df = pl.read_parquet(universe_path)
    print(f"Loaded {len(df)} combinations from universe")
    print(f"Columns: {df.columns}")
    print(f"Unique drug_dose combinations: {df['drug_dose'].n_unique()}")
    print(f"Unique cell_lines: {df['cell_line'].n_unique()}")
    
    return df


def apply_toml_splits(universe_df, toml_splits):
    """
    Apply TOML split assignments to the universe of combinations.
    
    Args:
        universe_df: DataFrame with all drug-dose × cell-line combinations
        toml_splits: Dictionary of {(cell_line, drug_dose): split_assignment}
    
    Returns:
        pl.DataFrame: with updated split assignments
    """
    print("Applying TOML split assignments...")
    
    # Create a lookup for TOML assignments
    toml_df = pl.DataFrame([
        {
            'cell_line': cell_line,
            'drug_dose': drug_dose,
            'toml_split': split_type
        }
        for (cell_line, drug_dose), split_type in toml_splits.items()
    ])
    
    print(f"TOML assignments to apply: {len(toml_df)}")
    if len(toml_df) > 0:
        print("Sample TOML assignments:")
        print(toml_df.head())
    
    # Join with universe to apply TOML splits
    result_df = universe_df.join(
        toml_df,
        on=['cell_line', 'drug_dose'],
        how='left'
    )
    
    # Apply the split logic:
    # - If toml_split is not null, use it
    # - Otherwise, use 'train'
    result_df = result_df.with_columns([
        pl.when(pl.col('toml_split').is_not_null())
        .then(pl.col('toml_split'))
        .otherwise(pl.lit('train'))
        .alias('split_assignment')
    ]).select(['drug_dose', 'cell_line', 'split_assignment'])  # Keep only the 3 required columns
    
    # Check for missing TOML entries
    matched_toml = result_df.filter(pl.col('split_assignment') != 'train').shape[0]
    print(f"TOML entries matched in universe: {matched_toml} / {len(toml_splits)}")
    if matched_toml < len(toml_splits):
        missing_count = len(toml_splits) - matched_toml
        print(f"Warning: {missing_count} TOML entries not found in universe")
    
    # Count the splits
    split_counts = result_df.group_by('split_assignment').len().sort('split_assignment')
    print("Split distribution:")
    for row in split_counts.iter_rows():
        split_type, count = row
        print(f"  {split_type}: {count:,}")
    
    return result_df


def main():
    # File paths
    toml_path = "/tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/tahoe_5_holdout/generalization_converted.toml"
    universe_path = "/tahoe/drive_3/ANALYSIS/analysis_190/Data/tahoe_embeddings_70M_drug_dose_DC_split_assignments.parquet"
    output_path = "/tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/tahoe_5_holdout_generalization_split_assignments.parquet"
    
    print("=" * 60)
    print("Converting TOML to Parquet Split Assignments")
    print("=" * 60)
    
    # Check input files exist
    if not Path(toml_path).exists():
        print(f"Error: TOML file not found: {toml_path}")
        sys.exit(1)
    
    if not Path(universe_path).exists():
        print(f"Error: Universe parquet file not found: {universe_path}")
        sys.exit(1)
    
    print(f"Input TOML: {toml_path}")
    print(f"Universe parquet: {universe_path}")
    print(f"Output parquet: {output_path}")
    print()
    
    # Step 1: Load TOML splits
    toml_splits = load_toml_splits(toml_path)
    print()
    
    # Step 2: Load universe combinations
    universe_df = load_universe_combinations(universe_path)
    print()
    
    # Step 3: Apply TOML splits
    result_df = apply_toml_splits(universe_df, toml_splits)
    print()
    
    # Step 4: Save result
    print(f"Saving results to: {output_path}")
    result_df.write_parquet(output_path)
    
    # Validation
    print()
    print("=" * 60)
    print("Validation")
    print("=" * 60)
    
    # Reload and verify
    saved_df = pl.read_parquet(output_path)
    print(f"Saved file contains {len(saved_df)} rows")
    print(f"Columns: {saved_df.columns}")
    
    final_counts = saved_df.group_by('split_assignment').len().sort('split_assignment')
    print("Final split distribution:")
    for row in final_counts.iter_rows():
        split_type, count = row
        print(f"  {split_type}: {count:,}")
    
    # Check for any TOML assignments that didn't match
    toml_cell_lines = set(cell_line for (cell_line, drug_dose) in toml_splits.keys())
    universe_cell_lines = set(universe_df['cell_line'].unique().to_list())
    
    print(f"\nTOML cell lines: {sorted(toml_cell_lines)}")
    print(f"Universe cell lines: {len(universe_cell_lines)} total")
    print(f"TOML cell lines in universe: {toml_cell_lines.intersection(universe_cell_lines)}")
    
    missing_toml_cells = toml_cell_lines - universe_cell_lines
    if missing_toml_cells:
        print(f"Warning: TOML cell lines not found in universe: {missing_toml_cells}")
    
    print()
    print("Conversion completed successfully!")


if __name__ == "__main__":
    main()