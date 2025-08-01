#!/bin/bash

# State TX Inference Tmux Wrapper
# Convenience wrapper that launches core script in tmux with monitoring

set -e  # Exit on error

# Default values
TMUX_SESSION="state_inference"
CORE_SCRIPT="$(dirname "$0")/run_state_tx_inference.sh"

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [OPTIONS] INPUT_FILE OUTPUT_FILE

Run state transport inference in tmux session with monitoring.

ARGUMENTS:
    INPUT_FILE    Path to expanded DMSO controls h5ad file
    OUTPUT_FILE   Path for output predictions h5ad file

OPTIONS:
    -s, --session NAME      Tmux session name (default: $TMUX_SESSION)
    -m, --model-dir DIR     Model directory (passed to core script)
    -e, --embed-key KEY     Embedding key in obsm (passed to core script)
    -p, --pert-col COL      Perturbation column name (passed to core script)
    -h, --help             Show this help message

EXAMPLES:
    # Basic usage
    $0 dmso_controls_expanded_for_inference.h5ad predictions.h5ad

    # With custom session name
    $0 -s my_inference dmso_controls_expanded_for_inference.h5ad predictions.h5ad

    # With custom model
    $0 -m /path/to/model dmso_controls_expanded_for_inference.h5ad predictions.h5ad

MONITORING:
    # Attach to session
    tmux attach-session -t $TMUX_SESSION

    # Monitor log
    tail -f OUTPUT_FILE.inference.log

    # Check session status
    tmux list-sessions

    # Kill session
    tmux kill-session -t $TMUX_SESSION

EOF
}

# Parse command line arguments
CORE_SCRIPT_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -s|--session)
            TMUX_SESSION="$2"
            shift 2
            ;;
        -m|--model-dir|-e|--embed-key|-p|--pert-col)
            # Pass these to core script
            CORE_SCRIPT_ARGS+=("$1" "$2")
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

# Validate core script exists
if [[ ! -f "$CORE_SCRIPT" ]]; then
    echo "Error: Core script not found: $CORE_SCRIPT"
    exit 1
fi

# Set up log file
LOG_FILE="${OUTPUT_FILE%.h5ad}.inference.log"

# Check if tmux session already exists
if tmux has-session -t "$TMUX_SESSION" 2>/dev/null; then
    echo "Warning: Tmux session '$TMUX_SESSION' already exists"
    echo "Kill it first with: tmux kill-session -t $TMUX_SESSION"
    exit 1
fi

# Build the command
CORE_CMD="$CORE_SCRIPT ${CORE_SCRIPT_ARGS[*]} '$INPUT_FILE' '$OUTPUT_FILE'"
CMD_WITH_LOG="$CORE_CMD 2>&1 | tee '$LOG_FILE'"

# Print configuration
echo "=== State TX Inference Tmux Wrapper ==="
echo "Input file:       $INPUT_FILE"
echo "Output file:      $OUTPUT_FILE"
echo "Log file:         $LOG_FILE"
echo "Tmux session:     $TMUX_SESSION"
echo "Core script:      $CORE_SCRIPT"
echo "========================================"

# Start tmux session
echo "Starting tmux session: $TMUX_SESSION"
tmux new-session -d -s "$TMUX_SESSION" "$CMD_WITH_LOG"

echo ""
echo "✓ Command started in tmux session: $TMUX_SESSION"
echo ""
echo "Monitoring commands:"
echo "  tmux attach-session -t $TMUX_SESSION    # Attach to session"
echo "  tail -f $LOG_FILE                       # Monitor log"
echo "  tmux list-sessions                      # Check status"
echo ""
echo "Management commands:"
echo "  tmux kill-session -t $TMUX_SESSION      # Kill session"
echo ""
echo "The inference is now running in the background..."