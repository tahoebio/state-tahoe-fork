# Predicting cellular responses to perturbation across diverse contexts with State

> Train State transition models or pretrain State embedding models. See the State [paper](https://arcinstitute.org/manuscripts/State).
> 
> See the [Google Colab](https://colab.research.google.com/drive/1QKOtYP7bMpdgDJEipDxaJqOchv7oQ-_l) to train STATE for the [Virtual Cell Challenge](https://virtualcellchallenge.org/).

## Associated repositories

- Model evaluation framework: [cell-eval](https://github.com/ArcInstitute/cell-eval)
- Dataloaders and preprocessing: [cell-load](https://github.com/ArcInstitute/cell-load)

## Installation

### Installation from PyPI

This package is distributed via [`uv`](https://docs.astral.sh/uv).

```bash
uv tool install arc-state
```

### Installation from Source

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv run state
```

When making fundamental changes to State, install an editable version with the `-e` flag.

```bash
git clone git@github.com:ArcInstitute/state.git
cd state
uv tool install -e .
```

## CLI Usage

You can access the CLI help menu with:

```state --help```

Output:
```
usage: state [-h] {emb,tx} ...

positional arguments:
  {emb,tx}

options:
  -h, --help  show this help message and exit
```

## State Transition Model (ST)

To start an experiment, write a TOML file (see `examples/zeroshot.toml` or
`examples/fewshot.toml` to start). The TOML file specifies the dataset paths
(containing h5ad files) as well as the machine learning task.

Training an ST example below.

```bash
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

The cell lines and perturbations specified in the TOML should match the values appearing in the
`data.kwargs.cell_type_key` and `data.kwargs.pert_col` used above. To evaluate STATE on the specified task,
you can use the `tx predict` command:

```bash
state tx predict --output_dir $HOME/state/test/ --checkpoint final.ckpt
```

It will look in the `output_dir` above, for a `checkpoints` folder.

If you instead want to use a trained checkpoint for inference (e.g. on data not specified)
in the TOML file:


```bash
state tx infer --output $HOME/state/test/ --output_dir /path/to/model/ --checkpoint /path/to/model/final.ckpt --adata /path/to/anndata/processed.h5 --pert_col gene --embed_key X_hvg
```

Here, `/path/to/model/` is the folder downloaded from [HuggingFace](https://huggingface.co/arcinstitute).

## TOML Configuration Files

State experiments are configured using TOML files that define datasets, training splits, and evaluation scenarios. The configuration system supports both **zeroshot** (unseen cell types) and **fewshot** (limited perturbation examples) evaluation paradigms.

### Configuration Structure

#### Required Sections

**`[datasets]`** - Maps dataset names to their file system paths
```toml
[datasets]
replogle = "/path/to/replogle/dataset/"
# YOU CAN ADD MORE
```

**`[training]`** - Specifies which datasets participate in training
```toml
[training]
replogle = "train"  # Include all replogle data in training (unless overridden below)
```

#### Optional Evaluation Sections

**`[zeroshot]`** - Reserves entire cell types for validation/testing
```toml
[zeroshot]
"replogle.jurkat" = "test"     # All jurkat cells go to test set
"replogle.k562" = "val"        # All k562 cells go to validation set
```

**`[fewshot]`** - Specifies perturbation-level splits within cell types
```toml
[fewshot]
[fewshot."replogle.rpe1"]      # Configure splits for rpe1 cell type
val = ["AARS", "TUFM"]         # These perturbations go to validation
test = ["NUP107", "RPUSD4"]    # These perturbations go to test
# Note: All other perturbations in rpe1 automatically go to training

```

### Configuration Examples

#### Example 1: Pure Zeroshot Evaluation
```toml
# Evaluate generalization to completely unseen cell types
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"     # Hold out entire jurkat cell line
"replogle.rpe1" = "val"        # Hold out entire rpe1 cell line

[fewshot]
# Empty - no perturbation-level splits
```

#### Example 2: Pure Fewshot Evaluation
```toml
# Evaluate with limited examples of specific perturbations
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
# Empty - all cell types participate in training

[fewshot]
[fewshot."replogle.k562"]
val = ["AARS"]                 # Limited AARS examples for validation
test = ["NUP107", "RPUSD4"]    # Limited examples of these genes for testing

[fewshot."replogle.jurkat"]
val = ["TUFM"]
test = ["MYC", "TP53"]
```

#### Example 3: Mixed Evaluation Strategy
```toml
# Combine both zeroshot and fewshot evaluation
[datasets]
replogle = "/data/replogle/"

[training]
replogle = "train"

[zeroshot]
"replogle.jurkat" = "test"        # Zeroshot: unseen cell type

[fewshot]
[fewshot."replogle.k562"]      # Fewshot: limited perturbation examples
val = ["STAT1"]
test = ["MYC", "TP53"]
```

### Important Notes

- **Automatic training assignment**: Any cell type not mentioned in `[zeroshot]` automatically participates in training, with perturbations not listed in `[fewshot]` going to the training set
- **Overlapping splits**: Perturbations can appear in both validation and test sets within fewshot configurations
- **Dataset naming**: Use the format `"dataset_name.cell_type"` when specifying cell types in zeroshot and fewshot sections
- **Path requirements**: Dataset paths should point to directories containing h5ad files
- **Control perturbations**: Ensure your control condition (specified via `control_pert` parameter) is available across all splits

### Validation

The configuration system will validate that:
- All referenced datasets exist at the specified paths
- Cell types mentioned in zeroshot/fewshot sections exist in the datasets
- Perturbations listed in fewshot sections are present in the corresponding cell types
- No conflicts exist between zeroshot and fewshot assignments for the same cell type


## State Embedding Model (SE)

After following the same installation commands above:

```bash
state emb fit --conf ${CONFIG}
```

To run inference with a trained State checkpoint, e.g., the State trained to 4 epochs:

```bash
state emb transform \
  --model-folder "/large_storage/ctc/userspace/aadduri/SE-600M" \
  --input "/large_storage/ctc/datasets/replogle/rpe1_raw_singlecell_01.h5ad" \
  --output "/home/aadduri/vci_pretrain/test_output.h5ad"
```

## Licenses
State code is [licensed](LICENSE) under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (CC BY-NC-SA 4.0).

The model weights and output are licensed under the [Arc Research Institute State Model Non-Commercial License](MODEL_LICENSE.md) and subject to the [Arc Research Institute State Model Acceptable Use Policy](MODEL_ACCEPTABLE_USE_POLICY.md).

Any publication that uses this source code or model parameters should cite the State [paper](https://arcinstitute.org/manuscripts/State).
