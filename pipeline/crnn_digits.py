"""Fine-tuned digit-only recognizer -- a second, independent voter.

Not a replacement for RapidOCR's reread. A small CRNN+CTC model, trained from
scratch on ~20k synthetic sequences composited from real stamp-font glyphs
(extracted from this archive's own confirmed-correct crops), validated
against real crops the synthetic generator never saw. See v3/pipeline's
README section for the full account.

Its output vocabulary is 10 digits + blank -- nothing else. That makes it
architecturally decorrelated from RapidOCR's ~6625-class model in a specific,
useful way: RapidOCR's 3-zoom consensus can reach unanimous agreement on a
WRONG read when every zoom variant makes the identical mistake (measured
directly: "11444" collapsed to "1144" at 3/3 agreement, a CTC repeated-digit
undercount). A differently-trained, differently-architected model making the
identical mistake at the same time is far less likely -- and measured
directly to not happen on that exact page (see v3/pipeline's evaluation).

Weakness, also measured directly: this model saw only tight, single-field
crops in training. Fed a RapidOCR detection box that merged in extra content
(the neighbouring Urdu title, or an unrelated field), it has no "ignore this"
output the way a full-vocabulary model does, and hallucinates extra digits
instead. MAX_ASPECT below is the guard against that -- outside it, this
module simply declines to vote rather than guess.
"""
import os

import cv2
import numpy as np

from pipeline.gpu_env import GPU_ENABLED

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "crnn_digits.onnx"
)
IMG_H = 32
BLANK = 0
CLASSES = "0123456789"

# Measured on the confirmed-correct real crops (79-sample): genuine single-
# field boxes run ~2.0-3.9 wide-to-tall. The two known merged-box failures
# (a title-text merge, an adjacent-field merge) measured 6.4 and 7.2. 4.5
# sits in the gap -- permissive enough not to reject a slightly wide genuine
# 5-digit number, tight enough to decline the merged cases seen so far.
MAX_ASPECT = 4.5

_session = None


_PROVIDERS = ["CUDAExecutionProvider", "CPUExecutionProvider"] if GPU_ENABLED else ["CPUExecutionProvider"]


def _get_session():
    global _session
    if _session is None:
        import onnxruntime as ort

        _session = ort.InferenceSession(MODEL_PATH, providers=_PROVIDERS)
    return _session


def _decode_greedy(logits):
    """logits: (W, NUM_CLASSES) for one sample. Standard CTC greedy collapse."""
    ids = logits.argmax(axis=-1).tolist()
    out = []
    prev = BLANK
    for i in ids:
        if i != BLANK and i != prev:
            out.append(CLASSES[i - 1])
        prev = i
    return "".join(out)


def read_digits(tight_patch):
    """Read a TIGHT (unpadded) digit-run crop. None if the crop looks too
    wide to be a single field, or is degenerate.

    Deliberately takes the detector's raw box, not crop.py's padded
    candidate_field -- that padding is sized for RapidOCR's own reread (see
    crop.py's CANDIDATE_PAD comment) and pulls in exactly the kind of
    neighbouring content this model can't safely handle.
    """
    if tight_patch is None or tight_patch.size == 0:
        return None
    h, w = tight_patch.shape
    if h == 0 or w / h > MAX_ASPECT:
        return None

    scale = IMG_H / h
    resized = cv2.resize(tight_patch, (max(8, int(w * scale)), IMG_H), interpolation=cv2.INTER_AREA)
    img = (resized.astype(np.float32) / 127.5) - 1.0
    img = img[None, None, :, :]

    logits = _get_session().run(None, {"image": img})[0]  # (W, 1, NUM_CLASSES)
    return _decode_greedy(logits[:, 0, :]) or None
