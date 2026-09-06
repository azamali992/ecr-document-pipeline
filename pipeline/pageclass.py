"""Visual page-type classifier -- the safety net under the discard decision.

Why this exists
---------------
The marker classifier in detect.py identifies a page by finding one printed
string in the OCR output ("DELIVERYCHALLAN", "ECRLN", ...). When that string
is cropped off, faint, skewed or simply misread, the page falls through to
"other" and is DISCARDED -- no human ever sees it. A hand audit of 48 random
pages from the v1 discard pile found 10 genuine ECR/DC pages in it (~21%),
which projects to roughly 1,470 real documents silently lost across the
fiscal year. Liquid Nitrogen was 70% of those losses: its marker is the tiny
"MCL-ECRLN-015" form code at the very top edge of the page.

This model reads the whole page's LAYOUT instead of one small string, so it
survives exactly the conditions that break marker matching. It is trained on
the pipeline's own output (see the training notes in PESHAWAR_NOTES.md), and
it beats its teacher because it uses a different feature basis, not because
it saw better labels.

How it is used
--------------
Strictly as a second chance at "other", never as an override: if detect.scan
positively identifies a page, that answer stands. Only when the page was
about to be discarded is this consulted, and only a very confident
relevant-type prediction rescues it. See CONFIDENCE_THRESHOLD.

Model format
------------
A plain multinomial logistic regression exported as raw weights, so
inference is one matrix multiply in numpy -- no sklearn, torch or
onnxruntime dependency at runtime, and no GPU. Roughly 0.2 MB and well
under a millisecond per page, which matters because this runs on every
page that would otherwise be discarded.
"""
import os

import cv2
import numpy as np

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "pagecls_model.npz"
)

# Input geometry -- must match exactly what the model was trained on
# (build_dataset.py): whole page, squashed to a square, grayscale.
SIZE = 96
RENDER_WIDTH = 220.0

# Measured on a 48-page hand-labelled set drawn from the v1 discard pile
# (the adversarial set: every page in it is one the marker system got wrong
# or gave up on). Threshold sweep on that set:
#
#     thresh   genuine rescued   false alarms   projected over 7,047 discards
#     0.80         10/10            6/38         1468 rescued / 881 junk
#     0.90         10/10            4/38         1468 rescued / 587 junk
#     0.98         10/10            1/38         1468 rescued / 147 junk
#     0.999         6/10            1/38          881 rescued / 147 junk
#
# 0.98 is the knee: it still rescues every genuine page in the sample while
# admitting the least junk, and 0.999 starts costing real documents for no
# further gain. The asymmetry is deliberate and matches gate.py's: a page
# wrongly discarded is a silent permanent loss, a page wrongly promoted just
# costs one human glance in review.
CONFIDENCE_THRESHOLD = 0.98

IRRELEVANT = "irrelevant"

_model = None


def _load():
    global _model
    if _model is None:
        d = np.load(MODEL_PATH, allow_pickle=True)
        _model = (d["coef"], d["intercept"], [str(c) for c in d["classes"]])
    return _model


def available():
    return os.path.exists(MODEL_PATH)


def _features(page_img):
    """Same preprocessing as training: square thumbnail, per-image contrast
    normalisation. Per-image (not dataset-wide) normalisation is what makes
    this robust to CamScanner's very variable exposure."""
    small = cv2.resize(page_img, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
    x = small.astype(np.float32).ravel() / 255.0
    return (x - x.mean()) / (x.std() + 1e-6)


def classify(page_img):
    """(label, confidence) for a full-page grayscale image.

    `page_img` is the whole page, NOT the top fraction detect.scan works on:
    the layout signature of a job card includes its table body, and that is
    most of what separates it from an unrelated receipt.
    """
    coef, intercept, classes = _load()
    z = coef @ _features(page_img) + intercept
    z -= z.max()
    p = np.exp(z)
    p /= p.sum()
    i = int(p.argmax())
    return classes[i], float(p[i])


def rescue_type(page_img, threshold=CONFIDENCE_THRESHOLD):
    """The relevant book type this page almost certainly is, or None.

    None means "leave the discard decision alone" -- either the model agrees
    it is irrelevant, or it is not confident enough to overrule a discard.
    """
    label, conf = classify(page_img)
    if label == IRRELEVANT or conf < threshold:
        return None
    return label
