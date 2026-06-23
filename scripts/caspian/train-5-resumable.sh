#!/bin/bash

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/splits/caspian_5_holdout.toml"
OUTPUT_DIR="/nvme-shared/models/arc-state-caspian"
CONTROL_PERT="[('DMSO_TF', 0.0, 'uM')]"

# --- RESUMABILITY ---
# Use a FIXED experiment name (no timestamp) so that re-running this script after
# the reboot points at the SAME checkpoint dir. _train.py auto-loads
# {OUTPUT_DIR}/{name}/checkpoints/last.ckpt when it exists and resumes the
# optimizer/global-step state from it.
EXPERIMENT_NAME="arc-state-caspian-5"
RUN_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}"

# Persist a W&B run id so the run continues in the SAME W&B run after the reboot.
# wandb.init (used under the hood by Lightning's WandbLogger) honors WANDB_RUN_ID
# and WANDB_RESUME when no explicit id is passed in code.
mkdir -p "$RUN_DIR"
WANDB_ID_FILE="${RUN_DIR}/wandb_run_id.txt"
if [ ! -f "$WANDB_ID_FILE" ]; then
    # 8-char alphanumeric id, generated once and reused on every restart.
    tr -dc 'a-z0-9' < /dev/urandom | head -c 8 > "$WANDB_ID_FILE"
fi
export WANDB_RUN_ID="$(cat "$WANDB_ID_FILE")"
export WANDB_RESUME=allow

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "Starting State Transition training (resumable)..."
echo "Config file:      $CONFIG_FILE"
echo "Output directory: $RUN_DIR"
echo "W&B run id:        $WANDB_RUN_ID (resume=$WANDB_RESUME)"
if [ -f "${RUN_DIR}/checkpoints/last.ckpt" ]; then
    echo ">>> Found existing last.ckpt -- training will RESUME from it."
else
    echo ">>> No checkpoint found -- starting a fresh run."
fi
echo "Timestamp: $(date)"

# Navigate to state directory for training
cd /home/umair/state

# Train the State Transition model
# NOTE: do NOT pass overwrite=true here -- that would wipe the checkpoint dir.
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
echo "Model saved in: $RUN_DIR"
