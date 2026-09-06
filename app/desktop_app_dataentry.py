"""Peshawar ECR review app -- data-entry / CPU entry point.

Sibling to desktop_app.py, not a replacement for it: that one requires a
working CUDA GPU (it calls verify_gpu() and fails loudly without one) and
is meant to run the batch OCR pass on this machine. This one is meant to
travel -- handed to data-entry staff on a machine with no GPU, sometimes no
Python at all -- to sort the review pile a run has already produced.

No os.environ["ECR_USE_GPU"] here, and no verify_gpu() call: unset is
already the documented default (see pipeline/gpu_env.py), so recognize.py
and crnn_digits.py fall back to CPUExecutionProvider on their own. That
CPU path IS exercised here -- review_bridge._nominations(), the local half
of the AI-assist cross-check, runs the real RapidOCR + CRNN pipeline, just
on CPU instead of CUDA. Only the batch Run tab is genuinely unreachable
(frontend_dataentry/ has no Run tab at all -- see its index.html).

Loads frontend_dataentry/ instead of frontend/ -- same api.py, same
Review/Scanned/Discarded/Duplicates behaviour, just without the Run tab
markup+JS wiring that assumes a batch-capable machine.
"""
import multiprocessing
import os
import sys

_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import webview  # noqa: E402

from api import Api  # noqa: E402


def _fatal(message):
    """sys.exit's message goes to stderr -- invisible under pythonw.exe.
    A real message box is visible regardless of how this was started."""
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, message, "Peshawar ECR Review", 0x10)  # MB_ICONERROR
    except Exception:  # noqa: BLE001 -- message box itself failing must not hide the original error
        pass
    sys.exit(message)


def main():
    if not os.path.isdir(os.path.join(os.getcwd(), "peshawar_output")):
        _fatal(
            "Can't find peshawar_output/ in the current folder.\n\n"
            "This app must be started from the project root (double-click "
            "the .bat launcher rather than running the .exe directly)."
        )

    api = Api()
    window = webview.create_window(
        "Peshawar ECR Review",
        os.path.join(_APP_DIR, "frontend_dataentry", "index.html"),
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
