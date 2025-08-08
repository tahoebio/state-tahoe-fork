#!/bin/bash

CONFIG_PATH="tahoe_100m_data_processing.yaml"  # Path to your config file
NUM_WORKERS=8
SCRIPT_NAME="create_merged_anndata.py"  # or whatever your script is named

LOG_DIR="logs"
mkdir -p $LOG_DIR

for RANK in $(seq 0 $((NUM_WORKERS - 1))); do
    LOG_FILE="$LOG_DIR/worker_rank${RANK}.log"
    echo "Launching rank $RANK, writing logs to $LOG_FILE"
    python "$SCRIPT_NAME" "$CONFIG_PATH" rank=$RANK > "$LOG_FILE" 2>&1 &
done

wait
echo "All workers completed."
