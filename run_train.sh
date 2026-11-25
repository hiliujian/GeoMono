#!/usr/bin/env bash
# Usage: bash run_train.sh [tag] [GPU index|auto]
# Prepare GCDID masks offline, then train with single-frame GCIDC.
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"
export PYTHONDONTWRITEBYTECODE=1
TAG="${1:-GeoMono_GCDID_GCIDC_$(date +%Y%m%d_%H%M%S)}"
GPU="${2:-auto}"
HEIGHT="${HEIGHT:-192}"; WIDTH="${WIDTH:-640}"
CACHE="${GCDID_CACHE:-$(python3 gcdid/temporal.py "$HEIGHT" "$WIDTH")}"
CACHE="$(python3 -c 'import sys; from gcdid.temporal import output_path; print(output_path(sys.argv[1]))' "$CACHE")"
DATA="${GCDID_DATA:-$PROJECT/data/kitti}"
LOG_DIR="$(python3 -c 'import sys; from options import resolve_log_dir, validate_model_name; validate_model_name(sys.argv[2]); print(resolve_log_dir(sys.argv[1]))' "${LOG_DIR:-$PROJECT/tmp}" "$TAG")"
CANDIDATES="${GCDID_CANDIDATES:-$PROJECT/candidates/kitti_dynamic_mask}"
ROAD_MASKS="${GCDID_ROAD_MASKS:-$PROJECT/data/road_masks}"
ROAD_CACHE="${GCIDC_ROAD_CACHE:-$PROJECT/tmp/cache/geometry/GCIDC_roads_${HEIGHT}x${WIDTH}}"
ROAD_CACHE="$(python3 -c 'import sys; from gcdid.temporal import output_path; print(output_path(sys.argv[1]))' "$ROAD_CACHE")"
if [[ "$GPU" == auto ]]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | awk -F, 'NR == 1 {gsub(/[[:space:]]/, "", $1); print $1}')"
fi
PREPARE=("$PROJECT/.venv_flow/bin/python" -u gcdid/precompute.py
  --data "$DATA" --candidates "$CANDIDATES" --road-masks "$ROAD_MASKS" --split eigen_zhou
  --height "$HEIGHT" --width "$WIDTH" --cache "$CACHE" --gcidc-road-cache "$ROAD_CACHE"
  --gpu "$GPU")
read -r -a EXTRA_FLAGS <<< "${EXTRA:-}"
TRAIN=("$PROJECT/.venv_train/bin/python" -u -c 'from trainer import run_training; run_training()'
  --data_path "$DATA" --log_dir "$LOG_DIR" --model_name "$TAG"
  --split eigen_zhou --dataset kitti --height "$HEIGHT" --width "$WIDTH"
  --batch_size "${BATCH:-12}" --num_epochs "${EPOCHS:-20}" --num_workers "${WORKERS:-8}"
  --weights_init pretrained --frame_ids 0 -1 1 --scales 0 --post_process --no_eval --gcdid_cache "$CACHE"
  --gcidc_road_cache "$ROAD_CACHE"
  "${EXTRA_FLAGS[@]}")
echo "tag=$TAG GPU=$GPU cache=$CACHE"
if [[ "${DRYRUN:-0}" == 1 ]]; then
  printf '%q ' "${PREPARE[@]}"; echo
  printf '%q ' "${TRAIN[@]}"; echo
  exit 0
fi
if [[ ! -x .venv_train/bin/python || ! -x .venv_flow/bin/python ]]; then bash setup_env.sh; fi
mkdir -p "$LOG_DIR"
mkdir -p "$PROJECT/tmp/runtime"
export TMPDIR="$PROJECT/tmp/runtime" TORCH_HOME="$PROJECT/tmp/cache/torch"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Avoid costly NumPy huge-page compaction on this shared training machine.
export NUMPY_MADVISE_HUGEPAGE=0
export CUDA_VISIBLE_DEVICES="$GPU"
"${PREPARE[@]}" 2>&1 | tee "$LOG_DIR/${TAG}_prepare.log"
export PYTHONHASHSEED=3407
export PYTHONPATH="$PROJECT/_shims:$PROJECT"
"${TRAIN[@]}" 2>&1 | tee "$LOG_DIR/${TAG}_$(date +%Y%m%d_%H%M%S).log"
