"""Build and load the shared CUDA geometry library."""
import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import subprocess


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
