#!/bin/bash

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/splits/caspian_5_holdout.toml"
OUTPUT_DIR="/nvme-shared/models/arc-state-caspian"
EXPERIMENT_NAME="arc-state-caspian-5-$(date +%Y%m%d_%H%M%S)"
CONTROL_PERT="[('DMSO_TF', 0.0, 'uM')]"

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

echo "Starting State Transition training..."
echo "Config file: $CONFIG_FILE"
echo "Output directory: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo "Timestamp: $(date)"

# Navigate to state directory for training
cd /home/umair/state

# Train the State Transition model
state tx train \
    data.kwargs.toml_config_path="$CONFIG_FILE" \
    data.kwargs.embed_key="X_hvg" \
    data.kwargs.output_space="gene" \
    data.kwargs.num_workers=12 \
    data.kwargs.pert_col="drugname_drugconc" \
    data.kwargs.cell_type_key="Cell_ID_Cellosaur" \
    "data.kwargs.control_pert=\"$CONTROL_PERT\"" \
    data.kwargs.batch_col="plate" \
    training.wandb_track=true \
    training.batch_size=64 \
    training.lr=1e-4 \
    training.max_steps=300000 \
    training.val_freq=4000 \
    training.ckpt_every_n_steps=50000 \
    model.kwargs.cell_set_len=256 \
    model.kwargs.residual_decoder=false \
    model=tahoe_llama_212693232 \
    wandb.tags="[caspian,fewshot,HVG]" \
    wandb.project="state_tx_tahoe" \
    +wandb.name="$EXPERIMENT_NAME" \
    ++wandb.entity="vevotx" \
    output_dir="$OUTPUT_DIR" \
    name="$EXPERIMENT_NAME"

echo "Training completed!"
echo "Model saved in: $OUTPUT_DIR/$EXPERIMENT_NAME"
