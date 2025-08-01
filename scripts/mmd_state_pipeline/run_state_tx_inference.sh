#!/bin/bash

# State TX Inference Core Script
# Simple, focused script that runs state transport inference

set -e  # Exit on error

# Default values
MODEL_DIR="/tahoe/drive_3/ANALYSIS/analysis_190/Code/train_state_tx/experiments/tahoe_state_tx_20250714_203322_hvg_full"
EMBED_KEY=""  # Will be auto-detected or must be specified
PERT_COL="drug_dose"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] INPUT_FILE OUTPUT_FILE

Run state transport inference on expanded DMSO controls.

ARGUMENTS:
    INPUT_FILE    Path to expanded DMSO controls h5ad file
    OUTPUT_FILE   Path for output predictions h5ad file

OPTIONS:
    -m, --model-dir DIR     Model directory (default: $MODEL_DIR)
    -e, --embed-key KEY     Embedding key in obsm (REQUIRED - common options: X_hvg, embedding, mosaicfm-70m-merged)
    -p, --pert-col COL      Perturbation column name (default: $PERT_COL)
    -h, --help             Show this help message

EXAMPLES:
    # Basic usage with default model (embedding key required)
    $0 -e X_hvg dmso_controls_expanded_for_inference.h5ad predictions.h5ad

    # With custom model directory
    $0 -m /path/to/model -e X_hvg dmso_controls_expanded_for_inference.h5ad predictions.h5ad

    # With different embedding key
    $0 -e embedding dmso_controls_expanded_for_inference.h5ad predictions.h5ad

EOF
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model-dir)
            MODEL_DIR="$2"
            shift 2
            ;;
        -e|--embed-key)
            EMBED_KEY="$2"
            shift 2
            ;;
        -p|--pert-col)
            PERT_COL="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        -*)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
        *)
            # Positional arguments
            if [[ -z "$INPUT_FILE" ]]; then
                INPUT_FILE="$1"
            elif [[ -z "$OUTPUT_FILE" ]]; then
                OUTPUT_FILE="$1"
            else
                echo "Too many positional arguments"
                show_usage
                exit 1
            fi
            shift
            ;;
    esac
done

# Validate required arguments
if [[ -z "$INPUT_FILE" || -z "$OUTPUT_FILE" ]]; then
    echo "Error: INPUT_FILE and OUTPUT_FILE are required"
    show_usage
    exit 1
fi

# Validate embedding key is provided
if [[ -z "$EMBED_KEY" ]]; then
    echo "Error: --embed-key is required"
    echo "Common embedding keys: X_hvg, embedding, mosaicfm-70m-merged"
    echo "Use 'scanpy.read_h5ad(file).obsm.keys()' to check available keys"
    show_usage
    exit 1
fi

# Validate input file exists
if [[ ! -f "$INPUT_FILE" ]]; then
    echo "Error: Input file does not exist: $INPUT_FILE"
    exit 1
fi

# Validate model directory exists
if [[ ! -d "$MODEL_DIR" ]]; then
    echo "Error: Model directory does not exist: $MODEL_DIR"
    exit 1
fi

# Create output directory if needed
OUTPUT_DIR=$(dirname "$OUTPUT_FILE")
mkdir -p "$OUTPUT_DIR"

# Print configuration
echo "=== State TX Inference Configuration ==="
echo "Input file:       $INPUT_FILE"
echo "Output file:      $OUTPUT_FILE"
echo "Model directory:  $MODEL_DIR"
echo "Embedding key:    $EMBED_KEY"
echo "Perturbation col: $PERT_COL"
echo "========================================"

# Run the command
echo "Running state transport inference..."
state tx infer \
    --adata "$INPUT_FILE" \
    --model_dir "$MODEL_DIR" \
    --embed_key "$EMBED_KEY" \
    --pert_col "$PERT_COL" \
    --output "$OUTPUT_FILE"

echo "Inference completed successfully!"
echo "Output saved to: $OUTPUT_FILE"