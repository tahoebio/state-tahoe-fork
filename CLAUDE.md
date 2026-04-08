# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

State is a machine learning framework for predicting cellular responses to perturbations across diverse contexts. It consists of two main components:

1. **State Transition (ST)**: Predicts how perturbations modify cell state (gene expression/embeddings)
2. **State Embedding (SE)**: Learns dense gene representations via transformer pretraining on scRNA-seq data

## Development Environment

### Installation

```bash
# From PyPI
uv tool install arc-state

# From source (editable install for development)
git clone git@github.com:ArcInstitute/state.git
cd state
uv tool install -e .
```

### Python Requirements
- Python 3.10-3.12 (strictly <3.13)
- Uses `uv` for dependency management

### Linting and Formatting
```bash
# Run ruff (configured in ruff.toml)
ruff check .
ruff format .

# Configuration: line-length=120, auto-fix enabled, ignores E722
```

## Command Reference

### State Transition (TX) Commands

#### Training
```bash
# Basic training with TOML config
state tx train \
  data.kwargs.toml_config_path="examples/fewshot.toml" \
  data.kwargs.embed_key=X_hvg \
  data.kwargs.num_workers=12 \
  data.kwargs.batch_col=batch_var \
  data.kwargs.pert_col=target_gene \
  data.kwargs.cell_type_key=cell_type \
  data.kwargs.control_pert=TARGET1 \
  training.max_steps=40000 \
  training.val_freq=100 \
  training.ckpt_every_n_steps=100 \
  training.batch_size=8 \
  training.lr=1e-4 \
  model.kwargs.cell_set_len=64 \
  model.kwargs.hidden_dim=328 \
  model=pertsets \
  wandb.tags="[test]" \
  output_dir="$HOME/state" \
  name="test"
```

#### Prediction (on configured test sets)
```bash
state tx predict --output_dir $HOME/state/test/ --checkpoint final.ckpt
```

#### Inference (on arbitrary data)
```bash
state tx infer \
  --output $HOME/state/test/ \
  --output_dir /path/to/model/ \
  --checkpoint /path/to/model/final.ckpt \
  --adata /path/to/anndata/processed.h5 \
  --pert_col gene \
  --embed_key X_hvg
```

### State Embedding (SE) Commands

#### Training
```bash
state emb fit --conf ${CONFIG}
```

#### Inference
```bash
state emb transform \
  --model-folder "/path/to/model/" \
  --input "/path/to/input.h5ad" \
  --output "/path/to/output.h5ad"
```

## Architecture Overview

### Two-Component Design

**State Embedding (SE)** → **State Transition (TX)**

- SE learns universal gene representations via transformer pretraining
- TX uses those representations to predict perturbation effects
- Both use Hydra configuration for reproducible experiments
- Both leverage PyTorch Lightning for training infrastructure

### Key Code Structure

```
src/state/
├── __main__.py          # CLI entry point, Hydra config loading
├── _cli/                # Command implementations
│   ├── _tx/             # TX train/predict/infer
│   └── _emb/            # SE fit/transform
├── configs/             # Hydra YAML configurations
│   ├── config.yaml      # TX root config
│   ├── state-defaults.yaml  # SE root config
│   ├── data/            # Data loading configs
│   ├── model/           # Model architecture configs (pertsets, cpa, scvi, scgpt, etc.)
│   └── training/        # Training hyperparameters
├── tx/                  # State Transition implementation
│   ├── models/          # Model implementations (PerturbationModel subclasses)
│   ├── data/dataset/    # Dataset classes
│   ├── callbacks/       # PyTorch Lightning callbacks
│   └── utils/           # Model factory, loggers, checkpoint utilities
└── emb/                 # State Embedding implementation
    ├── nn/              # StateEmbeddingModel, transformer, loss functions
    ├── data/            # Data loaders for pretraining
    ├── train/           # Training loop, callbacks
    └── inference.py     # Transform/inference utilities
```

### Data Flow (TX Training)

```
TOML Config → Hydra Composition → PerturbationDataModule → Dataset Instances
    ↓
Collate & Transform → DataLoader → Model Forward Pass → Loss Computation
    ↓
Backward Pass → Checkpoint Management → Metrics Logging (CSV/WandB)
```

### Configuration System

Uses **Hydra** for hierarchical configuration:

1. Base configs in `src/state/configs/`
2. CLI overrides: `data.batch_size=32 model=cpa`
3. Hydra merges all layers into final config
4. Config saved as YAML in output_dir for reproducibility

**TX Config Groups:**
- `data/`: Dataset loading (default, perturbation)
- `model/`: Model architectures (pertsets, cpa, scvi, scgpt, embedsum, etc.)
- `training/`: Hyperparameters (batch_size, lr, max_steps, etc.)
- `wandb/`: Experiment tracking config

### Model Types (TX)

**Embedding-based Models:**
- `pertsets`: Transformer over cell sets (main State model)
- `embedsum`: Simple perturbation + basal addition
- `decoder_only`: Decoder-only architecture

**VAE-based Models:**
- `cpa`: Compositional Perturbation Autoencoder with adversarial training
- `scvi`: Variational inference model

**Pretrained Models:**
- `scgpt-genetic`: scGPT for genetic perturbations
- `scgpt-chemical`: scGPT for chemical perturbations

**Baseline Models:**
- `celltypemean`: Cell type averages (no training)
- `globalsimplesum`: Simple global mean baseline
- `cellcontextmean`: Cell context-based baseline

All models inherit from `PerturbationModel` (PyTorch Lightning module).

## TOML Configuration Files

TX experiments require TOML files that define:

1. **Dataset paths**: Map dataset names to directories containing h5ad files
2. **Training splits**: Which datasets participate in training
3. **Zeroshot evaluation**: Reserve entire cell types for val/test
4. **Fewshot evaluation**: Reserve specific perturbations within cell types for val/test

### Example TOML Structure

```toml
[datasets]
replogle = "/path/to/replogle/"

[training]
replogle = "train"  # Include in training unless overridden

[zeroshot]
"replogle.jurkat" = "test"  # Hold out entire cell type

[fewshot]
[fewshot."replogle.k562"]
val = ["AARS"]
test = ["NUP107", "RPUSD4"]
```

**Key Rules:**
- Cell types not in `[zeroshot]` automatically participate in training
- Perturbations not in `[fewshot]` go to training set
- Use format `"dataset_name.cell_type"` for cell type specifications
- Control perturbations must be available across all splits

## Testing

The repository does not currently have a comprehensive test suite. When adding tests:

```bash
# Test structure (if implemented)
pytest tests/

# Test individual modules
pytest tests/test_models.py
```

## Key Dependencies

- **torch**: PyTorch for deep learning
- **pytorch-lightning**: Training infrastructure (implied via code structure)
- **hydra-core**: Configuration management
- **anndata**: Single-cell data format
- **scanpy**: Single-cell analysis tools
- **cell-load**: ArcInstitute's data loading library (PerturbationDataModule)
- **cell-eval**: ArcInstitute's evaluation framework
- **wandb**: Experiment tracking
- **transformers**: Hugging Face transformers library

## Important Development Notes

### Model Factory Pattern
- Models are instantiated via `get_lightning_module()` in `src/state/tx/utils/__init__.py`
- Config model name routes to appropriate class
- Runtime data (var_dims, onehot mappings) injected during instantiation

### Dimension Management
- Input dimensions derived from data via `var_dims` dict
- Gene decoder dimensions depend on `output_space` setting
- Transfer learning handles dimension mismatches automatically by rebuilding decoders

### Checkpoint Handling
- Auto-resume from `last.ckpt` in checkpoints/ directory
- `init_from` parameter enables pretrain→finetune workflows
- Final checkpoint saved as `final.ckpt`

### Data Alignment (Critical for Scripts)
When working with scripts that process embeddings (see scripts/CLAUDE.md):
- **CRITICAL**: Parquet datasets must maintain consistent ordering by BARCODE_SUB_LIB_ID
- Scripts use positional iteration for performance
- Violating ordering assumptions causes silent data corruption

### Configuration Best Practices
1. Always specify control_pert matching your dataset
2. Verify cell_type_key and pert_col match your h5ad obs columns
3. Use absolute paths in TOML configs
4. Save configs with experiments for reproducibility

### Model Selection Guide
- **General use**: `pertsets` (transformer-based, most flexible)
- **VAE-based**: `cpa` (compositional, adversarial) or `scvi` (standard VAE)
- **Pretrained foundation**: `scgpt-genetic` or `scgpt-chemical`
- **Quick baselines**: `celltypemean` (instant, no training)

## Common Development Workflows

### Adding a New Model
1. Create model class inheriting from `PerturbationModel` in `src/state/tx/models/`
2. Implement `_build_networks()` and `forward()` methods
3. Add YAML config in `src/state/configs/model/`
4. Update model factory in `src/state/tx/utils/__init__.py`

### Running Experiments
1. Create/modify TOML config for dataset splits
2. Choose model config group: `model=pertsets`
3. Override hyperparameters via CLI
4. Monitor training via WandB or CSV logs
5. Evaluate with `state tx predict`

### Debugging Training Issues
- Check `output_dir/config.yaml` for final merged config
- Review `output_dir/logs/` for training metrics
- Use `--checkpoint` with predict/infer to test specific checkpoints
- Verify data dimensions match model expectations in logs
