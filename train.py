# Copyright Niantic 2019. Patent Pending. All rights reserved.
# See LICENSE for the original Monodepth2 non-commercial licence.
"""Single-GPU GeoMono training with cached GCDID masks and the Ground Contact Inverse Depth Constraint (GCIDC)."""
import os
import sys

# Use the same GPU selection, cache preparation and defaults as the launcher.
# The launcher calls this file with explicit training arguments.
if __name__ == "__main__" and len(sys.argv) == 1:
    launcher = os.path.join(os.path.dirname(os.path.realpath(__file__)), "run_train.sh")
    os.execlp("bash", "bash", launcher)

# Set before NumPy is imported; shared-machine memory compaction otherwise
# stalls large CPU buffers during post-processing and evaluation.
os.environ.setdefault('NUMPY_MADVISE_HUGEPAGE','0')
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SHIM_ROOT = os.path.join(PROJECT_ROOT, '_shims')
if SHIM_ROOT not in sys.path:
    sys.path.insert(0, SHIM_ROOT)
import torch
import numpy as np
from options import MonodepthOptions
from trainer import Trainer


if __name__ == "__main__":
    opts = MonodepthOptions().parse()
    if opts.no_cuda or not torch.cuda.is_available():
        raise SystemExit("GCDID and GCIDC training requires a CUDA GPU")
    if os.environ.get('GEOMONO_CACHE_READY') != '1':
        # A direct invocation can still fill correspondence files for an existing mask cache.
        import subprocess
        subprocess.run([os.path.join(PROJECT_ROOT, '.venv_flow/bin/python'),
                        os.path.join(PROJECT_ROOT, 'tools/prepare_gcidc_flow.py'),
                        '--gcdid', opts.gcdid_cache, '--cache', opts.gcidc_flow_cache,
                        '--data', opts.data_path], check=True)
    seed = 3407
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    Trainer(opts).train()
