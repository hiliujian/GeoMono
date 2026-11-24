"""Compute column-wise silhouette envelopes on the active CUDA stream."""
import ctypes
from functools import lru_cache
from pathlib import Path

import torch
from gcidc_cuda import load_library


@lru_cache(maxsize=None)
def library(capability):
    module = load_library(Path(__file__).with_name("geometry_kernels.cu"), "gcdid", capability, ("--fmad=false",))
    ptr = ctypes.c_void_p
    module.envelopes.argtypes = [ptr, ctypes.c_int, ctypes.c_int, ptr, ptr, ptr, ptr]
    return module


def envelopes(bottom):
    with torch.cuda.device(bottom.device):
        major, minor = torch.cuda.get_device_capability(bottom.device)
        columns = bottom.int().contiguous()
        hull = torch.empty_like(bottom, dtype=torch.float64)
        stack = torch.empty_like(columns)
        facets = torch.empty_like(columns)
        stream = torch.cuda.current_stream(bottom.device).cuda_stream
        err = library(major*10 + minor).envelopes(columns.data_ptr(), *columns.shape,
                                                  stack.data_ptr(), hull.data_ptr(), facets.data_ptr(), stream)
        if err:
            raise RuntimeError(f"GCDID contour kernel failed ({err})")
        return hull, facets
