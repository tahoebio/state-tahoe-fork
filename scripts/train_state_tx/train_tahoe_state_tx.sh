#!/bin/bash

# Train State Transition model with MosaicFM embeddings on Tahoe data
# This script implements a fewshot scenario with drug-cell (DC) splits

set -e  # Exit on any error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/tahoe_fewshot.toml"
OUTPUT_DIR="${SCRIPT_DIR}/experiments"
EXPERIMENT_NAME="tahoe_state_tx_$(date +%Y%m%d_%H%M%S)_mfm_full"

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
cd /tahoe/drive_3/ANALYSIS/analysis_190/Code/state

# Train the State Transition model
state tx train \
    data.kwargs.toml_config_path="$CONFIG_FILE" \
    data.kwargs.embed_key="mosaicfm-70m-merged" \
    data.kwargs.output_space="latent" \
    data.kwargs.num_workers=12 \
    data.kwargs.pert_col="drug_dose" \
    data.kwargs.cell_type_key="cell_line" \
    data.kwargs.control_pert="DMSO_TF_00" \
    data.kwargs.batch_col="batch" \
    training.wandb_track=true \
    training.weight_decay=0.0005 \
    training.batch_size=64 \
    training.lr=1e-4 \
    training.max_steps=100000 \
    training.train_seed=42 \
    training.val_freq=100 \
    training.ckpt_every_n_steps=1000 \
    training.gradient_clip_val=10 \
    training.loss_fn=mse \
    model.name=PertSets \
    model.checkpoint=null \
    model.device=cuda \
    model.kwargs.cell_set_len=256 \
    model.kwargs.blur=0.05 \
    model.kwargs.hidden_dim=1488 \
    model.kwargs.loss=energy \
    +model.kwargs.confidence_head=false \
    model.kwargs.n_encoder_layers=4 \
    model.kwargs.n_decoder_layers=4 \
    model.kwargs.predict_residual=true \
    +model.kwargs.softplus=true \
    +model.kwargs.freeze_pert=false \
    +model.kwargs.transformer_decoder=false \
    model.kwargs.finetune_vci_decoder=false \
    model.kwargs.residual_decoder=false \
    model.kwargs.decoder_loss_weight=1.0 \
    model.kwargs.batch_encoder=false \
    model.kwargs.nb_decoder=false \
    model.kwargs.mask_attn=false \
    +model.kwargs.use_effect_gating_token=false \
    model.kwargs.distributional_loss=energy \
    model.kwargs.transformer_backbone_key=llama \
    model.kwargs.transformer_backbone_kwargs.max_position_embeddings=256 \
    model.kwargs.transformer_backbone_kwargs.hidden_size=1488 \
    +model.kwargs.transformer_backbone_kwargs.intermediate_size=5952 \
    +model.kwargs.transformer_backbone_kwargs.num_hidden_layers=6 \
    +model.kwargs.transformer_backbone_kwargs.num_attention_heads=12 \
    +model.kwargs.transformer_backbone_kwargs.num_key_value_heads=12 \
    +model.kwargs.transformer_backbone_kwargs.head_dim=124 \
    model.kwargs.transformer_backbone_kwargs.use_cache=false \
    +model.kwargs.transformer_backbone_kwargs.attention_dropout=0.0 \
    +model.kwargs.transformer_backbone_kwargs.hidden_dropout=0.0 \
    +model.kwargs.transformer_backbone_kwargs.layer_norm_eps=1.0e-06 \
    +model.kwargs.transformer_backbone_kwargs.pad_token_id=0 \
    +model.kwargs.transformer_backbone_kwargs.bos_token_id=1 \
    +model.kwargs.transformer_backbone_kwargs.eos_token_id=2 \
    +model.kwargs.transformer_backbone_kwargs.tie_word_embeddings=false \
    +model.kwargs.transformer_backbone_kwargs.rotary_dim=0 \
    +model.kwargs.transformer_backbone_kwargs.use_rotary_embeddings=false \
    wandb.tags="[tahoe,fewshot,drug_cell_split,mosaicfm]" \
    wandb.project="state_tx_tahoe" \
    +wandb.name="$EXPERIMENT_NAME" \
    ++wandb.entity="vevotx" \
    output_dir="$OUTPUT_DIR" \
    name="$EXPERIMENT_NAME"

echo "Training completed!"
echo "Model saved in: $OUTPUT_DIR/$EXPERIMENT_NAME"

# Run evaluation on the trained model
echo "Running evaluation..."
state tx predict --output_dir "$OUTPUT_DIR/$EXPERIMENT_NAME/" --checkpoint final.ckpt

echo "Training and evaluation complete!"
echo "Results available in: $OUTPUT_DIR/$EXPERIMENT_NAME"