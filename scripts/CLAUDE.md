# Data Processing Scripts

## centroid_pearson_baseline.py

**Purpose**: Computes Pearson Delta correlation baseline for drug response prediction using pre-computed centroid embeddings.

**Key Features**:
- **Centroid-based baseline**: Uses mean perturbation effects from training data to predict test responses
- **Plate-aware processing**: Maintains strict plate boundaries to avoid batch effects
- **TOML split integration**: Handles complex holdout logic where unmentioned cell lines are training data
- **Hierarchical summaries**: Provides statistics at multiple levels (per cell-type per plate, across plates, overall)
- **Progress tracking**: Comprehensive tqdm progress bars for all major operations
- **Memory efficient**: Computes correlations directly without storing intermediate deltas

**Usage**:
```bash
python centroid_pearson_baseline.py \
    --toml-file train_state_tx/tahoe_5_holdout/generalization_converted_cell_lines.toml \
    --centroids-dir /path/to/by_plate_centroids/ \
    --output-dir results/
```

**Input Data**:
- **Centroid H5AD files**: Pre-computed centroids from `by_plate_centroids/` directory (e.g., `plate_plate1.h5ad`)
- **TOML split file**: Holdout experiment definitions with train/val/test assignments
- **Embedding key**: Typically `X_hvg` (2000-dimensional HVG embeddings)

**TOML Logic**:
1. **Implicit training data**: Cell lines NOT mentioned in TOML → all perturbations are training
2. **Explicit holdouts**: Cell lines mentioned in TOML → only val/test perturbations held out, rest are training
3. **Test evaluation**: Only combinations explicitly marked as 'test' are evaluated

**Algorithm**:
1. **Load centroids**: Read plate-based H5AD files maintaining plate boundaries
2. **Parse splits**: Extract holdout logic from TOML (5 holdout cell lines, ~3,679 test combinations)
3. **Compute mean effects**: For each perturbation, average deltas across training cell lines within each plate
4. **Predict test**: Apply plate-specific mean effect to test cell line controls: `prediction = μ_control + δ_mean`
5. **Evaluate**: Compute Pearson correlation between predicted and true deltas

**Performance**:
- **Baseline correlation**: 0.403 mean (std=0.159) across 4,233 test combinations
- **Processing time**: ~4 minutes for 14 plates (66,223 total observations)
- **Best cell line**: CVCL_1285 (0.602 correlation), Worst: CVCL_1098 (0.250 correlation)

**Output Files**:
- `detailed_correlations.csv`: Individual correlations for each (plate, cell_line, perturbation)
- `hierarchical_summaries.json`: Multi-level statistics with overall, per-cell, and per-plate summaries

**Key Implementation Details**:
- **Index handling**: Converts pandas index to integer positions for numpy array indexing
- **Control mapping**: Maps `DMSO_TF` to `[('DMSO_TF', 0.0, 'uM')]` format automatically
- **NaN handling**: Sets correlation to 0.0 for constant vectors or other edge cases
- **Validation**: Ensures controls exist for all test cell lines within each plate

**Integration with Analysis Pipeline**:
- **Upstream**: Requires centroid files from `compute_obsm_centroids.py` 
- **Downstream**: Provides baseline metrics for comparing ML model performance
- **Comparison**: 0.403 baseline sets performance target for state transformation models

## create_merged_anndata_by_plate.py

**Purpose**: Creates plate-based AnnData files from Tahoe-100M parquet data with integrated drug dose information.

**Key Features**:
- **Plate-based organization**: Creates one .h5ad file per plate (14 total) instead of arbitrary chunks
- **Real-time performance monitoring**: Live progress tracking with component timing breakdown
- **Memory-efficient processing**: Adaptive strategy uses whole-plate or chunked processing based on memory estimation
- **Integrated drug mapping**: Adds drug dose information during creation (no post-processing needed)  
- **Resume capability**: Instantly skips completed plates to allow restarting interrupted runs
- **Simple, fast processing**: Uses proven dictionary-based approach for maximum performance

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
- Default chunk size: 1M cells for processing efficiency

**Real-Time Performance Monitoring**:
- **Live progress tracking**: Updates every 50K cells with ETA and component timing
- **Performance breakdown**: Shows time spent on gene processing, data loading, matrix assignment
- **Early warning alerts**: Detects performance issues (>70% gene processing time, <500 cells/sec)
- **Example output**:
```
📊 Plate plate11 Progress: 500,000/7,062,820 cells (7.1%)
⚡ Performance: Gene=45.2%, Data=15.3%, Matrix=25.1%, Obs=14.4%
🏃 Rate: 892 cells/sec (recent: 945), ETA: 2.1 hours
```

**Data Alignment Requirements**:
- **CRITICAL ASSUMPTION**: Both state and MosaicFM parquet datasets must maintain consistent ordering by BARCODE_SUB_LIB_ID
- Script uses positional iteration without explicit sorting for performance
- Violating this assumption will cause silent data corruption (cells matched with incorrect embeddings)

## Performance Optimization History

### Original Performance Issues (August 2025)
**Problem**: Script was taking 113+ hours to process 14 plates, with hour-long hangs during processing

### Root Cause Analysis
**Issue 1: Complex Vectorization Overhead**
- Attempted coordinate-based vectorization with bulk data extraction
- `all_genes = state_data['genes'].to_list()` for 7M+ cells caused memory issues
- Complex coordinate array building was slower than simple approach

**Issue 2: Inefficient Plate Checking**
- `estimate_plate_memory()` scanned massive parquet files even for existing plates
- Each existing plate took **minutes** to check before skipping
- Wasted hours on startup when resuming interrupted runs

### Final Solution: Simple + Monitored Approach (August 2025)
**Approach**: Return to proven simple processing with real-time monitoring

**Key Changes**:
1. **Simple gene processing** (identical to fast original script):
```python
for gene, expr in zip(genes, exprs):
    if gene in token_to_col_idx:  # Fast dictionary lookup
        hvg_vec[token_to_col_idx[gene]] = expr
```

2. **Real-time performance tracking**:
- `PlatePerformanceTracker` class monitors component timing
- Live updates every 50K cells with breakdown and ETA
- Early warning system for performance issues

3. **Efficient plate checking**:
- Check file existence BEFORE expensive memory estimation
- Existing plates skip in milliseconds instead of minutes

**Performance Results**:
- **Processing speed**: Returns to fast dictionary-based approach
- **Monitoring**: Real-time feedback prevents hour-long mystery hangs  
- **Resume speed**: Instant skipping of completed plates (was taking minutes per plate)
- **Observability**: Component timing breakdown identifies bottlenecks within minutes

### Key Lessons Learned
1. **Simple approaches often win** - dictionary lookups beat complex vectorization for sparse data
2. **Real-time monitoring is critical** for long-running processes (plates take hours)
3. **Check existence before expensive operations** - file checks should happen first
4. **Vectorization isn't always faster** - especially for variable-length sparse data
5. **Observability matters** - need to identify issues within minutes, not hours