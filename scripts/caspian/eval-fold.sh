#!/bin/bash
#
# eval-fold.sh -- evaluate ONE log1p Caspian STATE fold end-to-end.
# Designed to run independently per fold, so the 5 folds can run in parallel
# across separate RunAI workspaces (one fold each).
#
# Usage:
#   ./eval-fold.sh <FOLD> [CHECKPOINT]
#     FOLD        one of 5 6 7 8 9
#     CHECKPOINT  checkpoint filename (default: best.ckpt)
#
# Two stages:
#   1) `state tx predict`  (GPU) -> writes per-cell adata_pred/adata_real for the
#      held-out TEST treatments. Default [0,14] clip is a no-op on log1p data and
#      keeps cell-eval's scale check happy, so we do NOT pass --no-clip.
#   2) eval_arc_checkpoints.py (CPU) -> pseudobulk deltas + pdex metrics. Uses
#      --pdex-log1p (canonical cell_eval: pearson_delta in LOG space, expm1 only inside
#      pdex's DE path) and NO clipping (--clip-max omitted). NOT --log1p (that would
#      expm1 the delta path too -> non-canonical, the artifact that crushed folds 6/8).

set -euo pipefail

FOLD="${1:?Usage: $0 <FOLD: 5|6|7|8|9> [CHECKPOINT]}"
case "$FOLD" in
    5|6|7|8|9) ;;
    *) echo "Error: FOLD must be one of 5 6 7 8 9 (got '$FOLD')"; exit 1 ;;
esac
CKPT="${2:-best.ckpt}"

# --- Paths ---
MODEL_DIR="/nvme-shared/models/state-arc/fewshot-caspian${FOLD}"
RESULTS_DIR="${MODEL_DIR}/eval_${CKPT}"
PRED="${RESULTS_DIR}/adata_pred.h5ad"
REAL="${RESULTS_DIR}/adata_real.h5ad"
METRICS_JSON="${RESULTS_DIR}/metrics_log1p.json"

# Caspian-branch copies (self-contained: no dependency on the 2x2 worktree).
RHAISTER_PY="/home/umair/Rhaister-caspian/.venv/bin/python"
EVAL_SCRIPT="/home/umair/Rhaister-caspian/scripts/eval_arc_checkpoints.py"
CONTROL_PERT="[('DMSO_TF', 0.0, 'uM')]"

# --- Preflight ---
command -v state >/dev/null    || { echo "ERROR: 'state' CLI not on PATH"; exit 1; }
[ -f "${MODEL_DIR}/config.yaml" ]                 || { echo "ERROR: no config.yaml in ${MODEL_DIR}"; exit 1; }
[ -f "${MODEL_DIR}/checkpoints/${CKPT}" ]         || { echo "ERROR: checkpoint ${CKPT} missing in ${MODEL_DIR}/checkpoints"; exit 1; }
[ -x "${RHAISTER_PY}" ]                           || { echo "ERROR: Rhaister venv python not found at ${RHAISTER_PY}"; exit 1; }
[ -f "${EVAL_SCRIPT}" ]                           || { echo "ERROR: eval script not found at ${EVAL_SCRIPT}"; exit 1; }

echo "=================================================="
echo "Fold ${FOLD}  |  checkpoint ${CKPT}"
echo "Model dir:   ${MODEL_DIR}"
echo "Results dir: ${RESULTS_DIR}"
echo "Timestamp:   $(date)"
echo "=================================================="

# --- Stage 1: predict (GPU) ---
echo ">>> [1/2] state tx predict ..."
state tx predict \
    --output-dir "${MODEL_DIR}" \
    --checkpoint "${CKPT}" \
    --predict-only

[ -f "${PRED}" ] && [ -f "${REAL}" ] || { echo "ERROR: predict did not produce ${PRED} / ${REAL}"; exit 1; }
echo ">>> predict done: ${PRED}"

# --- Stage 2: eval (CPU) ---
echo ">>> [2/2] eval_arc_checkpoints.py (log1p, no clip) ..."
"${RHAISTER_PY}" "${EVAL_SCRIPT}" \
    --pred "${PRED}" \
    --real "${REAL}" \
    --batch-col plate \
    --cell-col Cell_ID_Cellosaur \
    --pert-col drugname_drugconc \
    --control "${CONTROL_PERT}" \
    --pdex-log1p \
    --threads 16 \
    --output "${METRICS_JSON}"

echo ">>> Done. Metrics written to: ${METRICS_JSON}"
