#!/usr/bin/env bash
# Usage: bash run_train.sh [tag] [full] [GPU index|auto]
# Prepare GCDID masks offline, then train with the Ground Contact Inverse Depth Constraint (GCIDC).
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT"
export PYTHONDONTWRITEBYTECODE=1
TAG="${1:-GeoMono_GCDID_GCIDC_$(date +%Y%m%d_%H%M%S)}"
MODE="${2:-full}"
GPU="${3:-auto}"
HEIGHT="${HEIGHT:-192}"; WIDTH="${WIDTH:-640}"
GCIDC_WEIGHT="${GCIDC_WEIGHT:-0.1}"
CACHE="${GCDID_CACHE:-$(python3 gcdid/temporal.py "$HEIGHT" "$WIDTH")}"
DATA="${GCDID_DATA:-$PROJECT/data/kitti}"
LOG_DIR="$(python3 -c 'import sys; from options import resolve_log_dir, validate_model_name; validate_model_name(sys.argv[2]); print(resolve_log_dir(sys.argv[1]))' "${LOG_DIR:-$PROJECT/tmp}" "$TAG")"
CANDIDATES="${GCDID_CANDIDATES:-$PROJECT/candidates/kitti_dynamic_mask}"
ROAD_MASKS="${GCDID_ROAD_MASKS:-$PROJECT/data/road_masks}"
ROAD_CACHE="${GCIDC_ROAD_CACHE:-$PROJECT/cache/geometry/GCIDC_roads_${HEIGHT}x${WIDTH}}"
FLOW_CACHE="${GCIDC_FLOW_CACHE:-$(python3 gcdid/temporal.py "$HEIGHT" "$WIDTH" --flow-for "$CACHE")}"
if [[ "$GPU" == auto ]]; then
  GPU="$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | sort -t, -k2,2nr | awk -F, 'NR == 1 {gsub(/[[:space:]]/, "", $1); print $1}')"
fi
if [[ "$MODE" != full ]]; then
  echo "Use full for GCDID with GCIDC" >&2
  exit 2
fi
PREPARE=("$PROJECT/.venv_flow/bin/python" -u gcdid/precompute.py
  --data "$DATA" --candidates "$CANDIDATES" --road-masks "$ROAD_MASKS" --split eigen_zhou
  --height "$HEIGHT" --width "$WIDTH" --cache "$CACHE" --gcidc-flow-cache "$FLOW_CACHE" --gcidc-road-cache "$ROAD_CACHE"
  --gpu "$GPU")
ROADS=("$PROJECT/.venv_train/bin/python" -u tools/prepare_gcidc_roads.py
  --source "$ROAD_MASKS" --cache "$ROAD_CACHE" --height "$HEIGHT" --width "$WIDTH" --split eigen_zhou)
FLOW=("$PROJECT/.venv_flow/bin/python" -u tools/prepare_gcidc_flow.py --gcdid "$CACHE" --cache "$FLOW_CACHE" --data "$DATA")
read -r -a EXTRA_FLAGS <<< "${EXTRA:-}"
TRAIN=("$PROJECT/.venv_train/bin/python" -u train.py
  --data_path "$DATA" --log_dir "$LOG_DIR" --model_name "$TAG"
  --split eigen_zhou --dataset kitti --height "$HEIGHT" --width "$WIDTH"
  --batch_size "${BATCH:-12}" --num_epochs "${EPOCHS:-20}" --num_workers "${WORKERS:-8}"
  --weights_init pretrained --frame_ids 0 -1 1 --scales 0 --post_process --no_eval --gcdid_cache "$CACHE"
  --gcidc_weight "$GCIDC_WEIGHT" --gcidc_road_cache "$ROAD_CACHE" --gcidc_flow_cache "$FLOW_CACHE"
  "${EXTRA_FLAGS[@]}")
echo "tag=$TAG mode=$MODE GPU=$GPU cache=$CACHE"
if [[ "${DRYRUN:-0}" == 1 ]]; then
  if [[ "${SKIP_PREPARE:-0}" != 1 ]]; then
    printf '%q ' "${PREPARE[@]}"; echo
  else
    printf '%q ' "${ROADS[@]}"; echo
  fi
  if [[ "${PREPARE_ONLY:-0}" != 1 ]]; then
    printf '%q ' "${TRAIN[@]}"; echo
  elif [[ "${SKIP_PREPARE:-0}" == 1 ]]; then
    printf '%q ' "${FLOW[@]}"; echo
  fi
  exit 0
fi
if [[ ! -x .venv_train/bin/python || ! -x .venv_flow/bin/python ]]; then bash setup_env.sh; fi
mkdir -p "$LOG_DIR"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
# Avoid costly NumPy huge-page compaction on this shared training machine.
export NUMPY_MADVISE_HUGEPAGE=0
export CUDA_VISIBLE_DEVICES="$GPU"
if [[ "${SKIP_PREPARE:-0}" != 1 ]]; then
  # Build all geometry caches from the shared road and image observations.
  "${PREPARE[@]}" 2>&1 | tee "$LOG_DIR/${TAG}_prepare.log"
  export GEOMONO_CACHE_READY=1
else
  "${ROADS[@]}" 2>&1 | tee "$LOG_DIR/${TAG}_roads.log"
fi
if [[ "${PREPARE_ONLY:-0}" == 1 ]]; then
  if [[ "${SKIP_PREPARE:-0}" == 1 ]]; then
    "${FLOW[@]}" 2>&1 | tee "$LOG_DIR/${TAG}_flow.log"
  fi
  exit 0
fi
export PYTHONPATH="$PROJECT/_shims:$PROJECT"
"${TRAIN[@]}" 2>&1 | tee "$LOG_DIR/${TAG}_$(date +%Y%m%d_%H%M%S).log"
