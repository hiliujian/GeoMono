#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

prepare_environment() {
  local base_python="$1" env_dir="$2" torch_version="$3" vision_version="$4" cuda_index="$5" requirements="$6"
  if [[ ! -x "$env_dir/bin/python" ]]; then
    "$base_python" -m venv "$env_dir"
  fi
  "$env_dir/bin/python" -m pip install --index-url "https://download.pytorch.org/whl/$cuda_index" \
    "torch==$torch_version" "torchvision==$vision_version"
  "$env_dir/bin/python" -m pip install -r "$requirements"
}

prepare_environment "${GEOMONO_TRAIN_PYTHON:-python3.8}" "$PROJECT/.venv_train" \
  1.13.1 0.14.1 cu117 "$PROJECT/requirements.txt"
prepare_environment "${GEOMONO_FLOW_PYTHON:-python3.10}" "$PROJECT/.venv_flow" \
  2.5.1 0.20.1 cu118 "$PROJECT/requirements-flow.txt"
