#!/usr/bin/env python3
"""
Script to convert generalization.toml file to use correct nomenclatures:
1. Convert drug_dosage from [('Drug name', concentration, 'uM')] format to 'Drug_name_dose' format
2. Convert cell line names from human-readable names to CVCL IDs
"""

import re
import ast
import toml
import polars as pl
from pathlib import Path


def convert_drug_dosage_to_drug_dose(drug_dosage_str):
    """
    Convert from format like "[('Tubulin inhibitor 6', 0.5, 'uM')]" 
    to format like "Tubulin inhibitor 6_05"
    """
    try:
        # Parse the string as a Python literal
        drug_list = ast.literal_eval(drug_dosage_str)
        
        if isinstance(drug_list, list) and len(drug_list) > 0:
            drug_name, concentration, unit = drug_list[0]
            
            # Convert concentration to string and format
            conc_str = str(float(concentration))
            # Remove decimal point: 0.5 -> 05, 5.0 -> 50, 0.05 -> 005
            dose_suffix = conc_str.replace(".", "")
            
            # Create drug_dose format
            drug_dose = f"{drug_name}_{dose_suffix}"
            return drug_dose
        else:
            return drug_dosage_str
            
    except (ValueError, SyntaxError, TypeError) as e:
        print(f"Error parsing drug dosage '{drug_dosage_str}': {e}")
        return drug_dosage_str


def load_cell_line_mapping():
    """Load cell line metadata and create mapping from cell_name to Cell_ID_Cellosaur"""
    metadata_path = "/tahoe/drive_3/ANALYSIS/analysis_190/Data/cell_line_metadata.parquet"
    df = pl.read_parquet(metadata_path)
    
    # Create mapping from cell_name to CVCL ID
    mapping = {}
    for row in df.select(['cell_name', 'Cell_ID_Cellosaur']).unique().iter_rows():
        cell_name, cvcl_id = row
        mapping[cell_name] = cvcl_id
    
    return mapping


def convert_toml_file(input_path, output_path):
    """Convert the TOML file with corrected nomenclatures"""
    # Load cell line mapping
    cell_mapping = load_cell_line_mapping()
    print(f"Loaded cell line mapping for {len(cell_mapping)} cell lines")
    
    # Load the TOML file
    with open(input_path, 'r') as f:
        data = toml.load(f)
    
    print("Converting fewshot sections...")
    
    # Process fewshot sections
    if 'fewshot' in data:
        new_fewshot = {}
        
        for key, splits in data['fewshot'].items():
            # Extract cell line name from key like "tahoe_holdout.C32"
            if '.' in key:
                dataset, cell_name = key.split('.', 1)
                
                # Convert cell name to CVCL ID
                if cell_name in cell_mapping:
                    cvcl_id = cell_mapping[cell_name]
                    new_key = f"{dataset}.{cvcl_id}"
                    print(f"Converting {key} -> {new_key}")
                else:
                    print(f"Warning: Cell line '{cell_name}' not found in mapping, keeping original")
                    new_key = key
            else:
                new_key = key
            
            # Convert drug dosages in val and test arrays
            new_splits = {}
            for split_type, drug_list in splits.items():
                if isinstance(drug_list, list):
                    converted_drugs = []
                    for drug_dosage in drug_list:
                        converted_drug = convert_drug_dosage_to_drug_dose(drug_dosage)
                        converted_drugs.append(converted_drug)
                    new_splits[split_type] = converted_drugs
                else:
                    new_splits[split_type] = drug_list
            
            new_fewshot[new_key] = new_splits
        
        # Replace the fewshot section
        data['fewshot'] = new_fewshot
    
    # Write the converted TOML file
    with open(output_path, 'w') as f:
        toml.dump(data, f)
    
    print(f"Converted TOML file saved to: {output_path}")


def main():
    input_path = "/tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/tahoe_5_holdout/generalization.toml"
    output_path = "/tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/tahoe_5_holdout/generalization_converted.toml"
    
    print("Converting generalization.toml file...")
    print(f"Input: {input_path}")
    print(f"Output: {output_path}")
    
    convert_toml_file(input_path, output_path)
    print("Conversion completed!")


if __name__ == "__main__":
    main()