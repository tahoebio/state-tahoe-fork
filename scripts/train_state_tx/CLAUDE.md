# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

This repository contains State Transition model training for drug response prediction using the State framework. The project focuses on training transformer-based models to predict cellular state transitions in response to drug perturbations, utilizing various embedding representations including MosaicFM embeddings and HVG (Highly Variable Genes) features.

## Project Structure

### Key Directories
- **`experiments/`**: Training experiments and results (see `experiments/CLAUDE.md`)
- **`pretrained/ST-Tahoe/`**: Pre-trained State Transition checkpoint with full documentation
- **`tahoe_5_holdout/`**: Configuration files for drug-cell holdout validation splits

### Configuration Files
- **`st_tahoe_config.yaml`**: Hydra-based configuration template
- **`tahoe_fewshot.toml`**: TOML configuration defining dataset paths and train/test splits
- **`tahoe_5_holdout/generalization_converted.toml`**: Alternative configuration for holdout scenarios

### Training Scripts
- **`train_tahoe_state_tx.sh`**: Main training script for MosaicFM embeddings (100k steps)
- **`train_tahoe_state_tx_20250723.sh`**: Alternative training script for HVG features (300k steps)

## Core Architecture

### Model: PertSets (Transformer-based)
- **Architecture**: Transformer with LLaMA backbone
- **Input Dimensions**: 512 (MosaicFM) or 2000 (HVG genes)
- **Hidden Dimensions**: 1488
- **Transformer Layers**: 6 hidden layers, 12 attention heads
- **Cell Set Length**: 256 (max cells per perturbation batch)
- **Loss Function**: MSE with energy-based distributional loss

### Data Pipeline
- **Input Format**: H5AD files in `/tahoe/drive_3/ANALYSIS/analysis_190/Data/state_input*/`
- **Embedding Keys**: `mosaicfm-70m-merged` or `X_hvg`
- **Split Strategy**: Drug-Cell (DC) holdout - 20% drugs × 20% cells for testing
- **Control Condition**: `DMSO_TF_00`

## Common Development Tasks

### Training New Models

#### MosaicFM Embeddings (Full Dataset)
```bash
# Train with MosaicFM embeddings (100k steps)
./train_tahoe_state_tx.sh
```

#### HVG Features (Extended Training)
```bash  
# Train with HVG features (300k steps)
./train_tahoe_state_tx_20250723.sh
```

### Model Inference
```bash
# Run inference with trained model
state tx infer \
    --adata /path/to/input.h5ad \
    --model_dir ./experiments/tahoe_state_tx_TIMESTAMP \
    --embed_key mosaicfm-70m-merged \
    --pert_col drug_dose \
    --output /path/to/output.h5ad \
    --celltype_col cell_line \
    --celltypes CVCL_0179,CVCL_0320,CVCL_1094
```

### Checkpoint Management
```bash
# Clean up intermediate checkpoints (see experiments/CLAUDE.md)
cd experiments
./cleanup_checkpoints.sh  # Review first (dry run enabled)
```

### Evaluation
```bash
# Evaluate with cell-eval framework
cell-eval run \
    --adata-pred /path/to/predictions.h5ad \
    --adata-real /path/to/ground_truth.h5ad \
    --control-pert DMSO_TF_00 \
    --pert-col drug_dose \
    --celltype-col cell_line \
    --embed-key X_hvg \
    --outdir /path/to/results
```

## Key Configuration Parameters

### Training Parameters
- **`training.max_steps`**: 100k (MosaicFM) or 300k (HVG)
- **`training.batch_size`**: 64
- **`training.lr`**: 1e-4
- **`training.weight_decay`**: 0.0005
- **`training.val_freq`**: 100 steps
- **`training.ckpt_every_n_steps`**: 1k-50k depending on experiment

### Model Parameters  
- **`model.kwargs.cell_set_len`**: 256
- **`model.kwargs.hidden_dim`**: 1488
- **`model.kwargs.n_encoder_layers`**: 4
- **`model.kwargs.n_decoder_layers`**: 4
- **`model.kwargs.loss`**: energy (distributional loss)
- **`model.kwargs.predict_residual`**: true

### Data Parameters
- **`data.kwargs.embed_key`**: `mosaicfm-70m-merged` or `X_hvg`
- **`data.kwargs.pert_col`**: `drug_dose`
- **`data.kwargs.cell_type_key`**: `cell_line`
- **`data.kwargs.control_pert`**: `DMSO_TF_00`
- **`data.kwargs.batch_col`**: `batch`

## Experimental Workflow

### 1. Training Phase
- Configure TOML files for dataset paths and splits
- Run training script with appropriate parameters
- Monitor training via Weights & Biases (project: `state_tx_tahoe`)
- Training outputs saved to `experiments/tahoe_state_tx_TIMESTAMP/`

### 2. Evaluation Phase
- Run inference on held-out test data
- Evaluate predictions using multiple metrics:
  - MMD distance between predicted and real distributions
  - Pearson correlation of delta changes
  - Differential expression analysis
  - Cell-eval comprehensive evaluation suite

### 3. Results Analysis
- Training metrics: `experiments/*/version_0/metrics.csv`
- Evaluation results: Various CSV files in `eval_final.ckpt/`
- Wandb logging: Track training curves and hyperparameters

## Data Requirements

### Input Data Format
- **Format**: H5AD (AnnData) files
- **Required obsm keys**: Embedding matrices (`mosaicfm-70m-merged`, `X_hvg`)
- **Required obs columns**: `drug_dose`, `cell_line`, `batch`
- **Split definitions**: TOML files defining train/test perturbation assignments

### Pre-trained Models
- **ST-Tahoe checkpoint**: Available in `pretrained/ST-Tahoe/`
- **Model compatibility**: Requires specific gene ordering for HVG features
- **Vocabulary**: Perturbation names must match checkpoint format
- **Critical bug**: See `pretrained/ST-Tahoe/CLAUDE.md` for required inference fix

## Important Notes

### Memory Management
- **Large files**: Be extremely careful when loading H5AD files - check sizes first
- **Checkpoint storage**: Regular cleanup required (experiments can generate 100GB+ each)
- **RAM usage**: Monitor memory during training, especially with large datasets

### Known Issues
- **Inference bug**: Pre-trained ST-Tahoe has critical control perturbation bug (see docs)
- **Gene ordering**: HVG models require specific gene order matching training data
- **Wandb entity**: May need configuration for proper experiment tracking

### Best Practices
- **Use tmux**: Long training runs should use tmux sessions
- **Monitor logs**: Check `log.txt` files for training progress
- **Backup results**: Important evaluation outputs should be backed up
- **Reproducibility**: Always set seeds and save configuration files