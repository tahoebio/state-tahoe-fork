#!/bin/bash
#
# train-fold-resumable.sh -- log1p Caspian STATE training, parameterized by fold.
#
# Usage:
#   ./train-fold-resumable.sh <FOLD> [CUDA_DEVICE]
#     FOLD         one of 5 6 7 8 9 (holdout fold; selects splits/caspian_<FOLD>_holdout_log1p.toml)
#     CUDA_DEVICE  optional GPU index to pin this run to (sets CUDA_VISIBLE_DEVICES). Default: 0.
#
# Examples (parallel launch -- NOTE: this node has ONE B200, so all of these share it
# unless you spread across nodes/MIG and pass distinct device ids):
#   ./train-fold-resumable.sh 5 0 &
#   ./train-fold-resumable.sh 6 0 &
#
# This trains on LOG1P data (state_input_log1p/) and is a FRESH run distinct from the
# linear baseline.

set -e  # Exit on any error

# --- Args ---
FOLD="${1:?Usage: $0 <FOLD: 5|6|7|8|9> [CUDA_DEVICE]}"
case "$FOLD" in
    5|6|7|8|9) ;;
    *) echo "Error: FOLD must be one of 5 6 7 8 9 (got '$FOLD')"; exit 1 ;;
esac
CUDA_DEVICE="${2:-0}"
export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"

# --- Configuration ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/splits/caspian_${FOLD}_holdout_log1p.toml"
OUTPUT_DIR="/nvme-shared/models/state-arc"
CONTROL_PERT="[('DMSO_TF', 0.0, 'uM')]"

# --- RESUMABILITY ---
# Use a FIXED experiment name (no timestamp) so re-running after a reboot points at the
# SAME checkpoint dir. _train.py auto-loads {OUTPUT_DIR}/{name}/checkpoints/last.ckpt when
# it exists and resumes the optimizer/global-step state from it. The `-log1p` suffix keeps
# these runs separate from the linear-space baseline.
EXPERIMENT_NAME="fewshot-caspian${FOLD}"
RUN_DIR="${OUTPUT_DIR}/${EXPERIMENT_NAME}"

# Persist a W&B run id so the run continues in the SAME W&B run after a reboot.
mkdir -p "$RUN_DIR"
WANDB_ID_FILE="${RUN_DIR}/wandb_run_id.txt"
if [ ! -f "$WANDB_ID_FILE" ]; then
    tr -dc 'a-z0-9' < /dev/urandom | head -c 8 > "$WANDB_ID_FILE"
fi
export WANDB_RUN_ID="$(cat "$WANDB_ID_FILE")"
export WANDB_RESUME=allow

# Check if config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found at $CONFIG_FILE"
    exit 1
fi

echo "Starting State Transition training (log1p, resumable)..."
echo "Fold:             $FOLD"
echo "Config file:      $CONFIG_FILE"
echo "Output directory: $RUN_DIR"
echo "CUDA device:      $CUDA_VISIBLE_DEVICES"
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
    wandb.tags="[caspian,fewshot,HVG,log1p]" \
    wandb.project="state_tx_tahoe" \
    +wandb.name="$EXPERIMENT_NAME" \
    ++wandb.entity="vevotx" \
    output_dir="$OUTPUT_DIR" \
    name="$EXPERIMENT_NAME"

echo "Training completed!"
echo "Model saved in: $RUN_DIR"
