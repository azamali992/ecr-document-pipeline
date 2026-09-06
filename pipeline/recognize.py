"""Read the ECR number out of a cropped number field.

Engine choice is measured, not assumed. On the 180 anchored crops of the sample
batch (121 known numbers):

    Tesseract 5.4 --psm 7      27.3% recall, 34 false positives
    EasyOCR (best config)      74.4% recall, 32 false positives
    RapidOCR (single read)     87.6% recall, 14 false positives
    RapidOCR (6/6 consensus)   77.7% recall,  0 false positives

So: RapidOCR, and read each crop under several preprocessing variants rather
than once. The variants are not there to boost recall -- a single read already
gets 87.6% -- they are there to manufacture a *confidence signal*. Agreement
across independent preprocessings is the only trustworthy signal available
offline, because per-engine confidence scores are not usable here (the old
pipeline's manifest contains confidence-100 false positives sitting next to
confidence-8 correct reads).

That gives a tunable precision dial, which gate.py turns into accept/review.
"""

import collections
import os
import re
from dataclasses import dataclass

import cv2

from pipeline.gpu_env import GPU_ENABLED

OCR_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets",
    "rapidocr_gpu.yaml" if GPU_ENABLED else "rapidocr.yaml",
)

# ECR/DC numbers can run 1-6 digits -- confirmed directly by the client, not
# inferred from a sample (see detect.CANDIDATE_DIGITS's comment, which this
# matches exactly and for the same reason). pipeline.detect passes
# DETECTION_VALID_NUMBER instead of this VALID_NUMBER: its candidates come
# from RapidOCR's own detection box rather than a hand-calibrated offset, so
# a merged or mis-bounded box can settle on a different digit count on
# reread than the one that got it nominated in the first place.
#
# This range used to be narrower here specifically to close a real failure
# seen during testing: a merged box that started as a plausible longer
# candidate reread as a stable but WRONG short number. Digit count was never
# actually what protected against that -- full zoom-variant agreement is
# still required to trust ANY reread regardless of its length, same bar a
# wrong long-number reread always had to clear too, and detect.py's height
# filter is what separates stamp-sized print from everything else on the
# page. A narrow range here was papering over not fully trusting that gate;
# widen to the client's real, stated range instead of a guessed-safe subset
# of it, and lean on the gate to do the job it's actually there for.
VALID_NUMBER = re.compile(r"^\d{1,6}$")
DETECTION_VALID_NUMBER = re.compile(r"^\d{1,6}$")

# Three zoom levels, grayscale only. Upscaling matters because the crops are
# small, and reading each zoom independently is what produces the agreement
# signal the gate runs on.
#
# Otsu-binarised variants used to sit alongside these. An ablation over all 63
# subsets (scratchpad/ablate_variants.py) showed they were actively harmful:
# they are the noisy readers, so requiring unanimity across all six rejected
# correct reads. Dropping them took recall from 85.7% to 97.5% at zero false
# positives *and* halved the cost. Worth re-testing if the wider archive turns
# out to contain much fainter stamps than this sample, where binarisation may
# start earning its place again.
#
# These are sized for the flatbed pipeline's crops: small, tightly bound by a
# hand-calibrated per-template offset. pipeline.detect's candidate crops are a
# different regime -- several hundred px already, since they come from
# RapidOCR's own detection box rather than a fixed offset -- and blowing one of
# those up by 3.5x doesn't just waste time, it actively breaks reading: on a
# page where the true stamp was read correctly at conf 1.00 by the detector
# itself, re-reading it through these zooms got only 1/3 votes (the 2.5x and
# 3.5x variants returned nothing), while a stray, non-numeric field label
# happened to mis-read as a stable "556" across the SAME three zooms and stood
# alone as the only trusted candidate on the page -- exactly the false-positive
# failure mode this whole voting scheme exists to prevent. DETECTION_ZOOMS
# below is the same idea recalibrated for that crop size: gentle enough that a
# genuine printed stamp reads the same at every level, and (empirically, same
# page) unstable enough that the false-positive stopped reaching agreement at
# all once the zoom stopped being aggressive enough to coincidentally launder
# it into looking like digits.
ZOOMS = (1.5, 2.5, 3.5)
DETECTION_ZOOMS = (0.8, 1.0, 1.2)

_engine = None


def _get_engine():
    """Lazily construct the engine.

    Deliberately per-process: ONNX Runtime sessions are not fork-safe, so a
    multiprocessing pool must build its own rather than inherit one.

    The config file pins ONNX Runtime to one thread per process -- see the
    comment block in assets/rapidocr.yaml for why that is not a typo. OpenCV is
    pinned for the same reason: it parallelises internally by default, which
    just adds contention when the pool already owns every core.
    """
    global _engine
    if _engine is None:
        from rapidocr_onnxruntime import RapidOCR

        cv2.setNumThreads(1)
        _engine = RapidOCR(config_path=OCR_CONFIG)
    return _engine


@dataclass(frozen=True)
class Reading:
    """A consensus read of one crop."""

    text: str
    votes: int
    variants: int

    @property
    def agreement(self):
        return self.votes / self.variants


def _variants(crop, zooms=ZOOMS):
    out = []
    for zoom in zooms:
        interp = cv2.INTER_CUBIC if zoom >= 1 else cv2.INTER_AREA
        scaled = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=interp)
        out.append(cv2.createCLAHE(2.0, (8, 8)).apply(scaled))
    # A quiet margin stops the detector from clipping edge strokes.
    return [
        cv2.copyMakeBorder(v, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=255) for v in out
    ]


def _read_once(image, valid_number):
    """Best valid number in one image, by engine score. None if nothing valid."""
    result, _ = _get_engine()(cv2.cvtColor(image, cv2.COLOR_GRAY2RGB))
    best = None
    for item in result or []:
        text, score = item[1], float(item[2])
        # The D/A card suffixes its number with "(F)"; stripping non-digits
        # handles that and the occasional stray tick mark.
        digits = re.sub(r"\D", "", text)
        if valid_number.match(digits) and (best is None or score > best[1]):
            best = (digits, score)
    return best


def read_number(crop, zooms=ZOOMS, valid_number=VALID_NUMBER):
    """Read one number-field crop. Returns a Reading, or None if unreadable.

    `zooms`/`valid_number` default to the flatbed pipeline's calibration; pass
    DETECTION_ZOOMS/DETECTION_VALID_NUMBER for pipeline.detect candidate crops
    (see the comments above for why the two crop regimes need different
    values for both).
    """
    votes = collections.Counter()
    for variant in _variants(crop, zooms):
        hit = _read_once(variant, valid_number)
        if hit:
            votes[hit[0]] += 1
    if not votes:
        return None
    text, count = votes.most_common(1)[0]
    return Reading(text=text, votes=count, variants=len(zooms))
