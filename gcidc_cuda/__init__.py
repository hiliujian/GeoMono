"""CUDA implementation of the GCIDC column completion operator."""
import ctypes
from functools import lru_cache
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import torch
import torch.nn.functional as F


def load_library(source, prefix, capability, flags=()):
    compiler = shutil.which("nvcc") or "/usr/local/cuda/bin/nvcc"
    archs = subprocess.check_output([compiler, "--list-gpu-arch"], text=True)
    arch = max(int(v.split("_")[1]) for v in archs.split() if v.startswith("compute_") and v[8:].isdigit() and int(v[8:]) <= capability)
    digest = hashlib.sha256(source.read_bytes() + " ".join(flags).encode() + subprocess.check_output([compiler,"--version"])).hexdigest()[:20]
    directory = Path(os.environ.get("XDG_CACHE_HOME",Path.home()/".cache")) / "geomono" / "cuda"
    directory.mkdir(parents=True, exist_ok=True)
    binary = directory / f"{prefix}_{digest}_{arch}.so"
    with (directory / (binary.name+".lock")).open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not binary.exists():
            tmp = binary.with_suffix(f".{os.getpid()}.so")
            try:
                subprocess.run([compiler, "-O3", "-std=c++14", "-shared", "-Xcompiler=-fPIC",
                                "-gencode", f"arch=compute_{arch},code=[sm_{arch},compute_{arch}]",
                                *flags, str(source), "-o", str(tmp)], check=True)
                tmp.replace(binary)
            finally:
                if tmp.exists(): tmp.unlink()
    return ctypes.CDLL(str(binary))


@lru_cache(maxsize=None)
def library(capability):
    lib = load_library(Path(__file__).with_name("targets.cu"), "gcidc", capability)
    ptr = ctypes.c_void_p
    lib.gcidc_complete.argtypes = [ctypes.POINTER(ptr)] + [ctypes.c_int]*6 + [ptr]
    sizes = (ctypes.c_int*6)()
    lib.gcidc_sizes(sizes)
    return lib, tuple(sizes)


def complete(depth, grid, cells, valid, batch_index, selected, road, target, mask,
             flow, intrinsics, poses, flipped):
    """Insert only previously unconstrained values into the single detached target."""
    B,H,W = depth.shape
    M,L = cells.shape
    if flow.shape != (B,4,2,H,W) or poses.shape != (B,2,4,4) or intrinsics.shape != (B,4):
        raise ValueError("GCIDC correspondence, intrinsics or pose dimensions do not match the training images")
    dev=depth.device
    if dev.type!="cuda" or depth.dtype!=torch.float32:
        raise ValueError("GCIDC expects CUDA FP32 depth")
    tensors=(grid,cells,valid,batch_index,selected,road,target,mask,flow,intrinsics,poses,flipped)
    if any(t.device!=dev for t in tensors):
        raise ValueError("All GCIDC inputs must share the depth tensor's CUDA device")
    with torch.cuda.device(dev):
        maj,minor=torch.cuda.get_device_capability(dev)
        lib,(obs,equation,ground,column,sample,block)=library(maj*10+minor)
        k=intrinsics.double().clone()
        k[:,2]=torch.where(flipped,W-1-k[:,2],k[:,2])
        pavement=road & (F.max_pool2d((grid>=0).float()[:,None],5,1,2)[:,0]==0)
        pix=cells+batch_index[:,None]*H*W
        needed=(selected & ((~mask.flatten()[pix]) & valid).any(-1)).int()
        per_frame=torch.zeros(B,dtype=torch.int32,device=dev).scatter_add_(0,batch_index,needed)
        lengths=valid.long().sum(1)
        offsets=lengths.cumsum(0)-lengths
        capacity=2*((H+3)//4)*((W+3)//4)
        stride=capacity*(equation//8+3)
        def space(size):return torch.empty(size,dtype=torch.uint8,device=dev)
        args=[flow.float(),k,poses.double(),flipped.bool(),per_frame,space(B*2*H*W*obs),
              pavement,depth,space(B*2*stride*8),torch.zeros(B*ground,dtype=torch.uint8,device=dev),
              batch_index.int(),needed,cells.long(),valid.bool(),offsets,grid.long(),
              space(M*W*column),space(2*min(B*H*W,M*L)*sample),space(M*W*2*4),space(M*W*block),
              space(M*W*8),target,mask,road]
        # All allocations, reads and writes use PyTorch's current CUDA stream.
        args=[a.contiguous() for a in args]
        pointers=(ctypes.c_void_p*len(args))(*(a.data_ptr() for a in args))
        err=lib.gcidc_complete(pointers,B,M,H,W,L,stride,torch.cuda.current_stream(dev).cuda_stream)
        if err:raise RuntimeError(f"GCIDC CUDA kernel launch failed ({err})")
