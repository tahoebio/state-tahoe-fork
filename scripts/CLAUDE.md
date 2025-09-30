# Data Processing Scripts

## h5ad_to_huggingface.py

**Purpose**: Converts H5AD (AnnData) format to HuggingFace dataset format for compatibility with dataset-based training pipelines.

**Key Features**:
- **Memory-efficient chunked processing**: Handles large H5AD files by processing in configurable chunks (default 100K cells)
- **Automatic schema inference**: Dynamically creates HuggingFace Features schema from .obs column types
- **Flexible .obsm selection**: Include all or specific embedding keys from .obsm
- **Skips .X by default**: Focuses on metadata and embeddings to minimize file size and memory usage
- **Optional .var export**: Saves gene/variable information as separate parquet file
- **Info mode**: Check H5AD structure without converting

**Usage**:
```bash
# Check H5AD file info without converting
python h5ad_to_huggingface.py input.h5ad output_dir/ --info-only

# Convert with all .obsm embeddings
python h5ad_to_huggingface.py input.h5ad output_dir/

# Convert with specific .obsm keys only
python h5ad_to_huggingface.py input.h5ad output_dir/ --obsm-keys X_hvg mosaicfm-70m-merged

# Adjust chunk size for memory management
python h5ad_to_huggingface.py input.h5ad output_dir/ --chunk-size 50000

# Skip saving .var information
python h5ad_to_huggingface.py input.h5ad output_dir/ --no-save-var
```

**Input/Output**:
- **Input**: H5AD file with .obs metadata and .obsm embeddings
- **Output**: HuggingFace dataset directory (loadable with `datasets.load_from_disk()`)
- **Optional**: var_info.parquet containing gene/variable metadata

**Data Structure**:
- **Source .obs columns** → HF dataset columns (with automatic dtype conversion)
- **Source .obsm embeddings** → HF dataset columns as Sequence(Value('float32'))
- **Source .X** → Skipped (not included in output)
- **Source .var** → Optional separate parquet file

**Schema Inference**:
- String/object columns → Value('string')
- Integer columns → Value('int64')
- Float columns → Value('float32')
- Boolean columns → Value('bool')
- Embedding arrays → Sequence(Value('float32'), length=dim)

**Integration with Analysis Pipeline**:
- **Reverse operation**: Use `dataset2anndata.py` to convert HF dataset back to H5AD
- **Use case**: Convert H5AD files for training pipelines that require HuggingFace dataset format
- **Complementary**: Works with existing streaming_dataset2hf.py for different data sources

## decode_embeddings_to_gene_expression.py

**Purpose**: Decodes embeddings to gene expression using a trained State TX model's gene decoder for evaluation with pearson_delta_only.py.

**Key Features**:
- Comprehensive performance tracking with detailed timing breakdown
- Live progress monitoring with bottleneck detection
- Memory-efficient batch processing with adaptive batch sizing
- Uses AnnData instead of h5py for robust H5AD file handling
- Supports UV arc-state package for StateTransitionPerturbationModel

**Usage**:
```bash
uv run --with psutil --python /home/valentine/.local/share/uv/tools/arc-state/bin/python \
/tahoe/drive_3/ANALYSIS/analysis_190/Code/state-tahoe-fork/scripts/decode_embeddings_to_gene_expression.py \
--input adata_real.h5ad \
--checkpoint ../checkpoints/final.ckpt \
--embedding-key X_state \
--output adata_decoded.h5ad
```

**Current Limitation**:
- **IMMEDIATE REFACTOR NEEDED**: Script currently saves predictions to `.obsm['X_hvg']` but pearson_delta_only.py expects predictions in `.X`
- **Next Steps**: Modify save_results_with_timing() to save decoded gene expression predictions to `.X` instead of `.obsm['X_hvg']` for direct compatibility with evaluation workflow

**Integration Issues**:
- pearson_delta_only.py compares real gene expression (`.X`) with predictions but current script saves to `.obsm['X_hvg']`
- Requires either: (1) refactoring decoder to save to `.X`, or (2) creating wrapper to copy predictions from `.obsm['X_hvg']` to `.X`

**Input/Output**:
- **Input**: H5AD with embeddings in `.obsm[embedding_key]` (e.g., `X_state`)
- **Current Output**: Predictions in `.obsm['X_hvg']` (incompatible)
- **Needed Output**: Predictions in `.X` (compatible with evaluation)

## Baseline Scripts

### centroid_perturbation_mean_baseline.py

**Purpose**: Computes PERTURBATION MEAN baseline for drug response prediction using pre-computed centroid embeddings.

**Baseline Logic**:
- **For each perturbation**: Average response deltas across training cell lines
- **Prediction**: `control(cell_line) + mean_delta(perturbation)`

**Key Features**:
- **Plate-aware processing**: Maintains strict plate boundaries to avoid batch effects
- **TOML split integration**: Handles complex holdout logic where unmentioned cell lines are training data
- **Hierarchical summaries**: Provides statistics at multiple levels (per cell-type per plate, across plates, overall)
- **Progress tracking**: Comprehensive tqdm progress bars for all major operations
- **Memory efficient**: Computes correlations directly without storing intermediate deltas

**Usage**:
```bash
python centroid_perturbation_mean_baseline.py \
    --toml-file train_state_tx/tahoe_5_holdout/generalization_converted_cell_lines.toml \
    --centroids-dir /path/to/by_plate_centroids/ \
    --output-dir results/
```

**Output Files**:
- `detailed_correlations.csv`: Individual correlations for each (plate, cell_line, perturbation)
- `hierarchical_summaries.json`: Multi-level statistics with overall, per-cell, and per-plate summaries

### centroid_context_mean_baseline.py

**Purpose**: Computes CONTEXT MEAN baseline for drug response prediction using pre-computed centroid embeddings.

**Baseline Logic**:
- **For each cell line**: Average response deltas across training perturbations (excluding controls)
- **Prediction**: `control(cell_line) + mean_delta(cell_line)`

**Key Features**:
- Same as perturbation mean baseline but with inverted averaging strategy
- Uses cell line context instead of perturbation context for predictions

**Usage**:
```bash
python centroid_context_mean_baseline.py \
    --toml-file train_state_tx/tahoe_5_holdout/generalization_converted_cell_lines.toml \
    --centroids-dir /path/to/by_plate_centroids/ \
    --output-dir results/
```

**Output Files**:
- `context_mean_detailed_correlations.csv`: Individual correlations for each (plate, cell_line, perturbation)
- `context_mean_hierarchical_summaries.json`: Multi-level statistics with overall, per-cell, and per-plate summaries

### centroid_global_mean_baseline.py

**Purpose**: Computes GLOBAL MEAN baseline for drug response prediction using pre-computed centroid embeddings.

**Baseline Logic**:
- **For each plate**: Average ALL response deltas from training data (ignoring cell line and perturbation identity)
- **Prediction**: `control(cell_line) + global_mean_delta(plate)`

**Key Features**:
- Simplest possible baseline - single average delta per plate
- Ignores both drug and cell line context for predictions
- Useful as lower bound for model performance comparison

**Usage**:
```bash
python centroid_global_mean_baseline.py \
    --toml-file train_state_tx/tahoe_5_holdout/generalization_converted_cell_lines.toml \
    --centroids-dir /path/to/by_plate_centroids/ \
    --output-dir results/
```

**Output Files**:
- `global_mean_detailed_correlations.csv`: Individual correlations for each (plate, cell_line, perturbation)
- `global_mean_hierarchical_summaries.json`: Multi-level statistics with overall, per-cell, and per-plate summaries

### Shared Implementation Details

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
3. **Compute mean effects**: Either per perturbation or per cell line depending on baseline type
4. **Predict test**: Apply plate-specific mean effect to test combinations
5. **Evaluate**: Compute Pearson correlation between predicted and true deltas

**Key Implementation Details**:
- **Index handling**: Converts pandas index to integer positions for numpy array indexing
- **Control mapping**: Maps `DMSO_TF` to `[('DMSO_TF', 0.0, 'uM')]` format automatically
- **NaN handling**: Sets correlation to 0.0 for constant vectors or other edge cases
- **Validation**: Ensures controls exist for all test cell lines within each plate

**Global Mode**:
All baseline scripts support `--ignore-plate-boundaries` flag to compute effects across all plates:
```bash
# Example: Perturbation mean baseline ignoring plate boundaries
python centroid_perturbation_mean_baseline.py \
    --toml-file train_state_tx/tahoe_5_holdout/generalization_converted_cell_lines.toml \
    --centroids-dir /path/to/by_plate_centroids/ \
    --output-dir results/ \
    --ignore-plate-boundaries
```

**Performance Comparison Results**:

| Baseline Type | With Plate Boundaries | Without Plate Boundaries | 
|---|---|---|
| **Context Mean** | 0.651 | 0.381 |
| **Perturbation Mean** | 0.403 | 0.405 |  
| **Global Mean** | 0.278 | 0.168 |

**Key Insights**:
- **Context mean performs best** when plate boundaries are maintained (0.651 correlation)
- **Perturbation mean is plate-independent** (consistent ~0.40 correlation)
- **Cell line effects are plate-dependent** while drug effects generalize across plates
- **Plate boundaries contain significant biological signal** - all baselines suffer when ignored

**Integration with Analysis Pipeline**:
- **Upstream**: Requires centroid files from `compute_obsm_centroids.py` 
- **Downstream**: Provides baseline metrics for comparing ML model performance
- **Recommendation**: Use context mean baseline with plate boundaries as primary performance target

## split_h5ad_random.py

**Purpose**: Memory-efficient random splitting of large H5AD files into N smaller parts without loading entire files into memory.

**Key Features**:
- **Pure h5py processing**: Handles arbitrarily large files (60-100GB+) using chunked processing
- **Sparse and dense matrix support**: Correctly preserves CSR sparse structure and dense matrices
- **Skip X mode**: `--skip-x` flag creates empty sparse matrices instead of copying X data (reduces IO burden)
- **Categorical data preservation**: Maintains codes/categories structure for obs columns
- **Reproducible splitting**: Configurable random seed ensures consistent splits
- **Flexible weighting**: Support for custom split proportions

**Usage**:
```bash
# Standard random splitting (copies all data including X)
python split_h5ad_random.py input.h5ad output_dir/ --n-splits 4

# Skip X data for IO efficiency (creates empty sparse X matrices)
python split_h5ad_random.py input.h5ad output_dir/ --n-splits 4 --skip-x

# Custom split weights
python split_h5ad_random.py input.h5ad output_dir/ --n-splits 3 --split-weights 0.5 0.3 0.2
```

**Output Files**:
- `split_00.h5ad`, `split_01.h5ad`, etc.: Randomly split H5AD files with preserved AnnData structure
- Each file contains subset of cells with identical var/uns/varm/varp data

**Integration with Analysis Pipeline**:
- **Use case**: Split large training datasets to reduce IO burden during model training
- **Skip X mode**: Essential when models use embeddings/obs data but not raw expression (X)
- **Tool compatibility**: Output files maintain proper H5AD structure expected by all AnnData tools

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