# CLAUDE.md - MMD State Pipeline

This file provides guidance to Claude Code when working with the MMD state embeddings pipeline.

## Overview

This directory contains scripts for evaluating drug response prediction models using Maximum Mean Discrepancy (MMD) and related metrics. The pipeline processes cell line embeddings and compares model predictions against ground truth drug-treated cells.

## Key Scripts and Their Purpose

### Data Preparation
- **`split_test_data.py`**: Memory-optimized script for extracting test datasets from large h5ad files (1TB+)
- **`expand_dmso_for_inference.py`**: Creates inference input by expanding DMSO controls to cell-line-specific test drug combinations + DMSO controls

### Model Inference  
- **`run_state_tx_inference.sh`**: Core script for running state transport inference
- **`run_state_tx_inference_tmux.sh`**: Tmux wrapper for background execution with monitoring

### Evaluation Scripts
- **`evaluate_transport_mmd_h5ad_test.py`**: **PRIMARY EVALUATION SCRIPT** - Dual-kernel MMD with technical duplicates
- **`evaluate_transport_mmd_h5ad.py`**: HuggingFace dataset version (legacy)
- **`pearson_delta_evaluation.py`**: Gene expression correlation metric

## Critical Implementation Details

### MMD Evaluation with Technical Duplicates

The **`evaluate_transport_mmd_h5ad_test.py`** script implements three types of MMD comparisons:

1. **Baseline MMD**: DMSO controls vs observed drug-treated cells
   - Measures biological effect size of drug treatment
   - Higher values indicate stronger drug effects

2. **Transport MMD**: Model predictions vs observed drug-treated cells  
   - Measures prediction quality
   - Lower values indicate better model performance

3. **Technical Duplicate MMD**: Rep1 vs Rep2 of same drug-treated condition
   - Establishes "noise floor" for ideal performance
   - Provides interpretable reference for transport MMD values

### Embedding Key Usage (CRITICAL)

**⚠️ Always use the correct embedding keys:**
- **Test data**: `--test-embedding-key X_hvg` (observed drug-treated cells)
- **DMSO controls**: `--dmso-embedding-key X_hvg` (control cells)
- **Predictions**: `--transported-embedding-key model_preds` (actual model predictions)

**DO NOT use `X_hvg` for predictions** - this contains the original DMSO input, not the model output.

### Memory Optimization Features

The `split_test_data.py` script includes comprehensive optimizations for handling very large datasets:
- **Reservoir sampling**: Eliminates subsampling bottlenecks
- **Sequential filtering**: Uses optimal HDF5 access patterns  
- **Pre-filtering**: Reduces data volume by filtering to test cell lines
- **Early termination**: Stops when sample limits are reached

## Common Workflow Patterns

### Standard Pipeline Execution
```bash
# 1. Extract test data and controls (creates dmso_controls_expanded_for_inference.h5ad automatically)
python split_test_data.py input.h5ad --split-file splits.parquet

# 2. OR expand DMSO controls separately (includes DMSO controls + test drugs per cell line)
python expand_dmso_for_inference.py /path/to/split_outputs_directory/

# 3. Run state transport inference  
./run_state_tx_inference_tmux.sh dmso_controls_expanded_for_inference.h5ad predictions.h5ad

# 4. Evaluate with technical duplicates (RECOMMENDED)
python evaluate_transport_mmd_h5ad_test.py \
  --test-dataset test_subset.h5ad \
  --transported-data predictions.h5ad \
  --dmso-controls dmso_controls.h5ad \
  --transported-embedding-key model_preds
```

### Key Parameters for Evaluation
- `--min-cells 50`: Minimum cells for reliable MMD computation
- `--min-tech-dup-cells 100`: Minimum cells for technical duplicate analysis (~50 per replicate)
- `--max-cells 1500`: Subsample large combinations for efficiency
- `--device cuda`: Use GPU acceleration when available

## Result Interpretation

### MMD Value Ranges
- **Transport MMD ≈ Technical Duplicate MMD**: Excellent prediction quality
- **Transport MMD << Baseline MMD**: Good improvement over DMSO baseline  
- **Transport MMD >> Technical Duplicate MMD**: Significant room for improvement

### Coverage Statistics
- Technical duplicate coverage depends on `--min-tech-dup-cells` threshold
- Higher thresholds provide more reliable technical duplicates but lower coverage
- Typical coverage: 60-80% with 100-cell minimum

## Data Structure Requirements

### Input Files Must Have:
- `.obs['drug_dose']`: Drug-dose combination identifiers
- `.obs['cell_line']`: Cell line identifiers (CVCL format recommended)
- `.obsm['X_hvg']`: Cell embeddings (test data and DMSO controls)
- `.obsm['model_preds']`: Model predictions (predictions file only)

### Split Assignment Format:
- `drug_dose`: Drug-dose combination
- `cell_line`: Cell line identifier  
- `split`: 'train' or 'test' assignment

## Performance Considerations

### Memory Usage
- `split_test_data.py`: Handles 1TB+ files with <1GB peak memory
- MMD evaluation: Scales with `--max-cells` parameter
- Use `--max-combinations` for testing on subsets

### Computational Efficiency
- Energy kernel: Generally faster than RBF for large datasets
- GPU acceleration: Use `--device cuda` when available
- Parallel processing: Scripts handle multiple drug-cell combinations efficiently

## Common Issues and Solutions

### Missing Predictions
- **Symptom**: `model_preds` key not found in predictions file
- **Solution**: Check state transport inference completed successfully
- **Verification**: `h5py` to inspect `.obsm` keys in predictions file

### Cell Count Warnings
- **Symptom**: Many combinations below minimum cell thresholds
- **Solution**: Reduce `--min-cells` or `--min-tech-dup-cells` parameters
- **Trade-off**: Lower thresholds vs. statistical reliability

### Memory Issues
- **Symptom**: Out of memory errors during evaluation
- **Solution**: Reduce `--max-cells` or `--max-combinations` parameters
- **Alternative**: Use `--device cpu` if GPU memory is limited

## File Naming Conventions

### Standard Output Names
- `test_subset.h5ad`: Ground truth drug-treated cells
- `dmso_controls.h5ad`: DMSO control cells for test cell lines
- `dmso_controls_expanded_for_inference.h5ad`: Inference input dataset
- `predictions.h5ad`: Model predictions with both input and output embeddings

### Results Directory Structure
```
mmd_evaluation_results/
├── mmd_results.parquet      # Detailed per-combination results
└── mmd_summary.json         # Overall statistics and coverage
```

## Integration with Analysis Pipeline

### Upstream Dependencies
- Split assignments from Barotaxis framework experiments
- Test datasets from analysis_190 data processing
- State transport models from model training pipeline

### Downstream Analysis
- Results feed into model comparison notebooks
- MMD metrics used for hyperparameter optimization
- Technical duplicate analysis informs model architecture decisions

## Development Guidelines

### Adding New Metrics
- Follow the pattern in `compute_mmd_for_combination()` function
- Update results collection in `evaluate_all_combinations()`
- Add summary statistics in `analyze_results()`
- Update help text and documentation

### Error Handling
- Use try-catch blocks around MMD computations
- Provide clear error messages for missing files or keys
- Continue processing other combinations when individual computations fail

### Testing
- Use `--max-combinations 10` for quick testing
- Verify embedding key usage with small datasets
- Check technical duplicate coverage with different thresholds

## Related Documentation

- **README.md**: Complete pipeline documentation with usage examples
- **Barotaxis/CLAUDE.md**: Vector field learning framework documentation  
- **analysis_190 notebooks**: Results analysis and visualization examples

## Future Improvements

### Potential Enhancements
- **Multi-GPU support**: Parallel MMD computation across combinations
- **Streaming evaluation**: Process combinations without loading full datasets
- **Additional kernels**: Implement Wasserstein distance or other metrics
- **Batch processing**: Handle multiple prediction files simultaneously

### Performance Optimizations
- **Caching**: Store computed embeddings for repeated evaluations
- **Approximation methods**: Use random projections for very high-dimensional data
- **Hybrid approaches**: Combine MMD with other evaluation metrics