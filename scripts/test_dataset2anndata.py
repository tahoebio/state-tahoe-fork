#!/usr/bin/env python3
"""
Test script for dataset2anndata.py functionality with synthetic data.
"""

import os
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from datasets import Dataset

# Add the scripts directory to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset2anndata import convert_to_anndata, save_anndata


def create_test_dataset(n_samples=1000, embedding_dim=512):
    """Create a synthetic HuggingFace dataset for testing."""
    print(f"Creating test dataset with {n_samples} samples, {embedding_dim} dimensions")
    
    # Create synthetic data
    np.random.seed(42)
    
    # Generate synthetic embeddings
    embeddings = np.random.randn(n_samples, embedding_dim).astype(np.float32)
    
    # Generate synthetic metadata
    cell_lines = np.random.choice(['A549', 'HEK293', 'MCF7', 'HeLa'], n_samples)
    drugs = np.random.choice(['DMSO_TF', 'Drug_A', 'Drug_B', 'Drug_C'], n_samples)
    
    # Create dataset dictionary
    data = {
        'mosaicfm-70m-merged': embeddings.tolist(),
        'cell_line': cell_lines.tolist(),
        'drug': drugs.tolist(),
        'dose': np.random.uniform(0.1, 10.0, n_samples).tolist(),
        'time': np.random.choice([24, 48, 72], n_samples).tolist(),
    }
    
    # Create HuggingFace dataset
    dataset = Dataset.from_dict(data)
    
    return dataset


def test_dataset2anndata():
    """Test the dataset2anndata conversion."""
    print("Testing dataset2anndata conversion...")
    
    # Create test dataset
    test_dataset = create_test_dataset(n_samples=1000, embedding_dim=512)
    
    # Test with small chunk size for verification
    chunk_size = 250
    
    print(f"Converting dataset with chunk_size={chunk_size}")
    
    # Convert to AnnData
    adata = convert_to_anndata(test_dataset, chunk_size=chunk_size)
    
    # Verify results
    print(f"AnnData shape: {adata.shape}")
    print(f"Expected shape: (1000, 512)")
    print(f"Observation columns: {list(adata.obs.columns)}")
    print(f"Observation matrices: {list(adata.obsm.keys())}")
    print(f"X matrix type: {type(adata.X)}")
    print(f"obsm embedding shape: {adata.obsm['mosaicfm-70m-merged'].shape}")
    
    # Test saving
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir) / "test_output.h5ad"
        print(f"Saving to {output_path}")
        save_anndata(adata, str(output_path))
        
        # Verify file exists and has reasonable size
        if output_path.exists():
            file_size = output_path.stat().st_size
            print(f"Output file size: {file_size / (1024*1024):.2f} MB")
            print("✅ Test passed!")
        else:
            print("❌ Test failed: Output file not created")
    
    return adata


if __name__ == "__main__":
    try:
        test_dataset2anndata()
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()