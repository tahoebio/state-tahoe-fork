#!/usr/bin/env python3
"""
Test only the extraction part with predefined indices (skip obs loading).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from split_test_data_hdf5 import extract_hdf5_data
import numpy as np

def test_extraction_only():
    """Test extraction with predefined indices."""
    
    h5ad_file = "/tahoe/drive_3/ANALYSIS/analysis_190/Data/20250711.tahoe.hvg.h5ad"
    
    print("=== TESTING EXTRACTION ONLY (1000 random indices) ===")
    
    # Use random indices for testing
    np.random.seed(42)
    test_indices = sorted(np.random.choice(95624334, size=1000, replace=False))
    
    print(f"Testing extraction with {len(test_indices)} random indices")
    print(f"Index range: {min(test_indices)} to {max(test_indices)}")
    
    # Test extraction
    output_path = "/tmp/test_extraction_only.h5ad"
    try:
        extract_hdf5_data(h5ad_file, test_indices, output_path)
        
        # Verify output
        if os.path.exists(output_path):
            print("SUCCESS: Extraction completed")
            
            # Quick verification without loading full file
            import h5py
            with h5py.File(output_path, 'r') as f:
                print(f"Output X format: {'sparse' if 'X' in f and hasattr(f['X'], 'keys') else 'dense'}")
                if 'obsm' in f:
                    print(f"Output obsm keys: {list(f['obsm'].keys())}")
                    for key in f['obsm'].keys():
                        shape = f['obsm'][key].shape
                        print(f"  {key}: {shape}")
                if 'obs' in f:
                    print(f"Output obs keys: {list(f['obs'].keys())}")
            
            # Clean up
            os.remove(output_path)
            return True
        else:
            print("FAILED: No output file created")
            return False
            
    except Exception as e:
        print(f"FAILED with error: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)
        return False

if __name__ == "__main__":
    success = test_extraction_only()
    if success:
        print("\n✅ Extraction test passed - chunked indexing works!")
    else:
        print("\n❌ Extraction test failed - need to debug further")