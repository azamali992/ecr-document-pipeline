"""Opt-in GPU switch, shared by recognize.py and crnn_digits.py.

Set ECR_USE_GPU=1 to request CUDA inference; unset (the default), nothing
here changes and both modules behave exactly as before. The GPU app
(app_gpu/) sets this before importing anything pipeline-related; the
existing CPU app and CLI never set it, so this module is a no-op for them.

Also does the (Windows-only) work of putting the pip-installed CUDA/cuDNN
runtime DLLs on this process's DLL search path. Those come from pip
packages (nvidia-cublas-cu12, nvidia-cudnn-cu12, ...) rather than a system
CUDA Toolkit install, so nothing puts their bin/ folders on PATH the way a
Toolkit installer would -- onnxruntime's CUDA provider can't find them
without this. Runs at import time, so it's in effect before any
onnxruntime import. os.add_dll_directory is per-process and does not
inherit across multiprocessing's spawn -- but since a spawned worker
re-imports this module fresh in its own process (see pipeline_bridge.py's
module docstring for why that's true), each worker sets this up for
itself automatically.
"""
import glob
import os
import sys

GPU_ENABLED = os.environ.get("ECR_USE_GPU") == "1"

if GPU_ENABLED and sys.platform == "win32":
    _bin_dirs = glob.glob(os.path.join(sys.prefix, "Lib", "site-packages", "nvidia", "*", "bin"))
    for _bin_dir in _bin_dirs:
        os.add_dll_directory(_bin_dir)
    # add_dll_directory alone isn't enough: cudnn64_9.dll is a thin dispatcher
    # that lazily loads its sibling engine DLLs (cudnn_graph64_9.dll, etc.)
    # via a plain LoadLibraryA(name) call, which does not consult
    # AddDllDirectory-registered paths -- only PATH. Confirmed by direct
    # testing: cudnnCreate() failed to find cudnn_graph64_9.dll with only
    # add_dll_directory in effect, and succeeded once these were also on
    # PATH. Both are kept: PATH covers cudnn's lazy loads, add_dll_directory
    # covers everything else (onnxruntime's own CUDA provider DLL and its
    # direct PE-import dependencies).
    os.environ["PATH"] = os.pathsep.join(_bin_dirs) + os.pathsep + os.environ.get("PATH", "")


def verify_gpu():
    """Actually construct a CUDA session and confirm it's the provider in
    use, rather than trusting the request. onnxruntime does NOT raise when
    CUDAExecutionProvider fails to load -- it logs a warning and silently
    falls back to CPU (see CreateExecutionProviderInstance in its source).
    That's exactly wrong for this app: the GPU app deliberately runs with
    far fewer worker processes than the CPU app (see app_gpu/desktop_app.py),
    sized for GPU throughput. A silent CPU fallback wouldn't just be slower
    than expected, it would be slower than the CPU app itself, with no
    error to explain why. Call this once at startup, before spawning any
    workers, so a broken CUDA setup fails loudly and immediately instead of
    quietly crawling through a multi-hour run.

    Raises RuntimeError with the concrete reason if CUDA isn't actually
    active; returns silently if it is.
    """
    import onnxruntime as ort

    from pipeline.crnn_digits import MODEL_PATH

    session = ort.InferenceSession(MODEL_PATH, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    active = session.get_providers()
    if "CUDAExecutionProvider" not in active:
        raise RuntimeError(
            "CUDAExecutionProvider did not initialise -- onnxruntime silently fell back to "
            f"{active}. Check that .venv-gpu has the nvidia-cublas-cu12/nvidia-cudnn-cu12 "
            "packages installed (see requirements-gpu.txt) and that their versions match what "
            "onnxruntime-gpu expects; run this same check with onnxruntime's logging turned up "
            "to see the specific missing DLL."
        )
