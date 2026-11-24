#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export NUMPY_MADVISE_HUGEPAGE=0
exec "$PROJECT/.venv_train/bin/python" -m tensorboard.main --logdir "${LOG_DIR:-$PROJECT/tmp}" --host 127.0.0.1 --port "${1:-6006}"
