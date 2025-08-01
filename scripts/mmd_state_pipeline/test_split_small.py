#!/usr/bin/env python3
"""
Test the HDF5 splitting approach on a small subset first.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from split_test_data_hdf5 import *

def test_small_subset():
    """Test with just 1000 cells to verify the approach works."""
    
    h5ad_file = "/tahoe/drive_3/ANALYSIS/analysis_190/Data/20250711.tahoe.hvg.h5ad"
    split_file = "/tahoe/drive_3/ANALYSIS/analysis_190/Data/20250618.tahoe_embeddings_70M_DC_split_assignments.parquet"
    
    print("=== TESTING SMALL SUBSET (1000 cells) ===")
    
    # Load split assignments
    test_combinations, test_cell_lines = load_split_assignments(split_file)
    
    # Load obs metadata with smaller chunks for testing
    print("Loading obs metadata (chunked)...")
    obs_df = load_obs_metadata_hdf5(h5ad_file)
    
    # Get test indices but limit to 1000 for testing
    print("Getting test indices (limited to 1000)...")
    test_indices = get_filtered_indices(obs_df, test_combinations, max_cells_per_combination=100)
    
    # Limit to first 1000 indices
    test_indices = test_indices[:1000]
    print(f"Testing with {len(test_indices)} indices")
    
    # Test extraction
    output_path = "/tmp/test_small_extraction.h5ad"
    extract_hdf5_data(h5ad_file, test_indices, output_path)
    
    # Verify output
    if os.path.exists(output_path):
        print("SUCCESS: Small test extraction completed")
        test_adata = ad.read_h5ad(output_path)
        print(f"Output shape: {test_adata.shape}")
        print(f"Output obsm keys: {list(test_adata.obsm.keys())}")
        
        # Clean up
        os.remove(output_path)
        return True
    else:
        print("FAILED: No output file created")
        return False

if __name__ == "__main__":
    success = test_small_subset()
    if success:
        print("\n✅ Small test passed - the approach should work on full dataset")
    else:
        print("\n❌ Small test failed - need to fix the approach first")