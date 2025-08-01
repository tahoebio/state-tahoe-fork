#!/usr/bin/env python3
"""
Script to safely examine X_hvg data in h5ad files using h5py for lazy access
"""

import h5py
import numpy as np

def examine_file(filepath, label):
    print(f'=== {label} ===')
    try:
        with h5py.File(filepath, 'r') as f:
            print(f'Top-level keys: {list(f.keys())}')
            
            # Check obsm group
            if 'obsm' in f:
                obsm_keys = list(f['obsm'].keys())
                print(f'obsm keys: {obsm_keys}')
                
                if 'X_hvg' in f['obsm']:
                    hvg_dataset = f['obsm']['X_hvg']
                    print(f'X_hvg shape: {hvg_dataset.shape}')
                    print(f'X_hvg dtype: {hvg_dataset.dtype}')
                    
                    # Read only first 3 rows, 10 columns
                    hvg_slice = hvg_dataset[:3, :10]
                    print('X_hvg first 3 rows, first 10 columns:')
                    print(hvg_slice)
                    
                    # Read small sample to get min/max without loading all
                    sample_slice = hvg_dataset[:100, :100]  # Small sample
                    print(f'X_hvg sample range: min={sample_slice.min():.6f}, max={sample_slice.max():.6f}')
                else:
                    print('No X_hvg found in obsm')
            else:
                print('No obsm group found')
                
    except Exception as e:
        print(f'Error reading {label}: {e}')
    print()

if __name__ == "__main__":
    # Check both files
    examine_file('/tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg/20250711.tahoe.hvg.h5ad', 'LOGGED VERSION')
    examine_file('/tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_tahoe_hvg_unlogged/20250711.tahoe.hvg.h5ad', 'UNLOGGED VERSION')