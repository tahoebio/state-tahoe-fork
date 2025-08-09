# Data Processing Scripts

## create_merged_anndata_by_plate.py

**Purpose**: Creates plate-based AnnData files from Tahoe-100M parquet data with integrated drug dose information.

**Key Features**:
- **Plate-based organization**: Creates one .h5ad file per plate (14 total) instead of arbitrary chunks
- **Memory-efficient processing**: Pre-allocation approach works within 250GB RAM budget
- **Integrated drug mapping**: Adds drug dose information during creation (no post-processing needed)  
- **Adaptive strategy**: Uses whole-plate or chunked processing based on memory estimation
- **Comprehensive progress tracking**: Shows plate-level and cell-level progress
- **Resume capability**: Automatically skips completed plates to allow restarting interrupted runs
- **Vectorized HVG processing**: Eliminates Python loops for 10-100x performance improvement

**Usage**:
```bash
python create_merged_anndata_by_plate.py tahoe_100m_data_processing.yaml
```

**Input Data**:
- State embeddings: `/tahoe/mosaicfm/datasets/tahoe100m_with_state_embeddings_parquet/*`
- MosaicFM embeddings: `/tahoe/mosaicfm/datasets/barotaxis/embeddings_tahoe_100m/*`  
- HVG mapping: `/tahoe/state_tahoe/token2hvg.parquet`
- Drug metadata: `tahoebio/Tahoe-100M` HuggingFace dataset

**Output**:
- Location: `/tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input_merged/by_plate/`
- Files: `plate_{plate_name}.h5ad` (one per plate)
- Contains: HVG expressions, MosaicFM embeddings, state embeddings, drug dose info

**Memory Management**:
- Estimates memory per plate, uses chunked processing for plates >200GB
- Pre-allocated arrays eliminate memory accumulation issues
- Designed for 250GB RAM budget with safety margin
- Default chunk size: 1M cells for optimal vectorization performance

**Performance Optimizations**:
- **Vectorized HVG processing**: Replaced cell-by-cell Python loops with numpy array operations
- **Dense matrix construction**: Direct numpy array assignment instead of sparse matrix building
- **Larger chunks**: Increased from 100K to 1M cells per chunk for better vectorization
- **Resume functionality**: Skip completed plates to avoid reprocessing
- **Expected speedup**: 10-100x faster processing (~5-15 hours vs 113+ hours)

**Data Alignment Requirements**:
- **CRITICAL ASSUMPTION**: Both state and MosaicFM parquet datasets must maintain consistent ordering by BARCODE_SUB_LIB_ID
- Script uses positional iteration without explicit sorting for performance
- Violating this assumption will cause silent data corruption (cells matched with incorrect embeddings)