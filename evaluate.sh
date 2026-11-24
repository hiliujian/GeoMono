#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"
WEIGHTS="${1:?Usage: bash evaluate.sh tmp/TAG/models/weights_EPOCH [GPU]}"
export CUDA_VISIBLE_DEVICES="${2:-0}"
export NUMPY_MADVISE_HUGEPAGE=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="$PROJECT/_shims:$PROJECT"
exec "$PROJECT/.venv_train/bin/python" -u evaluate_depth.py --data_path "${GCDID_DATA:-$PROJECT/data/kitti}" --load_weights_folder "$WEIGHTS" --eval_mono --eval_split eigen --post_process --num_workers "${WORKERS:-8}"
