"""Peshawar bundle extraction -- desktop app entry point (beta).

Run from the project root, using the shared .venv-gpu environment (this
beta reuses it rather than a third venv -- see PESHAWAR_NOTES.md):
    .venv-gpu\\Scripts\\python peshawar_beta\\app\\desktop_app.py

GPU-enabled, same pattern as the main archive's app_gpu/desktop_app.py:
ECR_USE_GPU must be set before anything imports pipeline.recognize or
pipeline.crnn_digits (both read it once at import time), and
multiprocessing's spawned workers each re-run this file's own top-level
code before importing further, so setting it here is what makes every
worker -- not just this main process -- pick it up too. verify_gpu() then
actually constructs a CUDA session and fails loudly at startup if it didn't
really take (onnxruntime silently falls back to CPU otherwise, which for a
GPU-sized worker count would just be slower than the CPU app with no
explanation why -- see gpu_env.verify_gpu's own docstring).

Entry point kept out of peshawar_beta/'s own root and named desktop_app.py,
not main.py -- the main archive hit a real bug from having two files named
main.py both reachable on sys.path (see app/pipeline_bridge.py's module
docstring for the full story); this avoids ever creating that ambiguity
inside the beta project itself, on top of peshawar_beta/main.py already
being this project's only main.py.
"""
import multiprocessing
import os
import sys

os.environ["ECR_USE_GPU"] = "1"

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import webview  # noqa: E402

import pipeline_bridge  # noqa: E402
from api import Api  # noqa: E402

# pipeline_bridge's own import already put peshawar_beta/ on sys.path.
from pipeline.gpu_env import verify_gpu  # noqa: E402

# Measured on this exact GPU (RTX 4050, 6GB) for the main archive app --
# see app_gpu/desktop_app.py's own comment for the throughput/VRAM curve
# that number came from (gains flatten out past 4 workers; 6 workers left
# too little VRAM headroom). Reused here rather than separately measured:
# same GPU, same OCR engine and model files underneath -- only detect.py
# (a cheap CPU-side classification step, not the GPU-heavy OCR calls) is
# actually different in this beta. Still overridable per run from the UI.
DEFAULT_GPU_WORKERS = 4
pipeline_bridge.DEFAULT_WORKERS = DEFAULT_GPU_WORKERS


def _fatal(message):
    """sys.exit's message goes to stderr -- invisible under pythonw.exe.
    A real message box is visible regardless of how this was started."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "Peshawar Bundle Extraction (beta)", 0x10)  # MB_ICONERROR
    except Exception:  # noqa: BLE001 -- message box itself failing must not hide the original error
        pass
    sys.exit(message)


def main():
    try:
        verify_gpu()
    except RuntimeError as exc:
        _fatal(f"Can't start with GPU: {exc}")

    api = Api()
    window = webview.create_window(
        "Peshawar Bundle Extraction (beta)",
        os.path.join(_APP_DIR, "frontend", "index.html"),
        js_api=api,
        width=1200,
        height=800,
        min_size=(900, 600),
    )
    api.set_window(window)
    webview.start()


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
