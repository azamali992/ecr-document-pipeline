"""Cut the ECR number field out of a page, given a located anchor.

Layout of a job card's title line, left to right:

    [ جاب کارڈ برائے گیس سلنڈرز ]   [ 6915 ]   [ نمبر ]
      ^ the anchor                    ^ what we want   ^ the printed label "number"

The number's position on the *page* varies (loose sheets on a flatbed), but its
position relative to the anchor does not -- that is the whole reason anchoring
works. The offsets live on the CardTemplate in locate.py, in units of the matched
template's width/height, so they follow the anchor's scale automatically and can
differ per card type (the D/A card leaves a much wider title-to-number gap than
the cylinder card).
"""

import cv2

from .render import SCALE

MIN_CROP_PX = (40, 30)  # (width, height) below which a crop is not worth reading


def number_field(crop_img, anchor):
    """Return the de-skewed number-field crop, or None if it falls off the page.

    `crop_img` is the CROP_WIDTH render; `anchor` is in MATCH_WIDTH space.
    """
    card = anchor.card
    x0 = int((anchor.x + anchor.width * card.x_start) * SCALE)
    x1 = int((anchor.x + anchor.width * card.x_end) * SCALE)
    y0 = int((anchor.y + anchor.height * card.y_start) * SCALE)
    y1 = int((anchor.y + anchor.height * card.y_end) * SCALE)

    h, w = crop_img.shape
    x0, y0 = max(0, x0), max(0, y0)
    x1, y1 = min(w, x1), min(h, y1)
    if x1 - x0 < MIN_CROP_PX[0] or y1 - y0 < MIN_CROP_PX[1]:
        return None

    patch = crop_img[y0:y1, x0:x1]
    if anchor.angle:
        # The anchor angle is the rotation that made the template fit, so the
        # page content carries the same tilt; undo it to hand OCR level text.
        ph, pw = patch.shape
        m = cv2.getRotationMatrix2D((pw / 2, ph / 2), anchor.angle, 1.0)
        patch = cv2.warpAffine(
            patch, m, (pw, ph), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
        )
    return patch


# Padding as a fraction of the candidate box's own size, each side. Detected
# boxes hug the glyphs tightly; some margin stops the reader's own zoom/border
# step (recognize._variants) from immediately re-clipping what the detector
# just found. 0.3 was measured against 0.05-0.4 on a real false-positive case
# (a crossed-out field label misread as digits): padding barely moved the false
# positive's stability, but recognize.DETECTION_ZOOMS -- calibrated for crops
# in roughly this padded size range -- is what actually separated it from a
# genuine stamp. See recognize.py's DETECTION_ZOOMS comment for the fuller story.
CANDIDATE_PAD = 0.3


def candidate_field(crop_img, candidate, page_shape):
    """Cut a detect.Candidate out of the CROP_WIDTH render, padded.

    `candidate` is in the *page_img* coordinate space that detect.scan ran
    on (i.e. MATCH_WIDTH); `page_shape` is that same page image's (h, w),
    needed to clip padding at the page edge before scaling up to CROP_WIDTH.
    """
    ph, pw = page_shape
    bw, bh = candidate.x1 - candidate.x0, candidate.y1 - candidate.y0
    x0 = max(0, candidate.x0 - bw * CANDIDATE_PAD)
    x1 = min(pw, candidate.x1 + bw * CANDIDATE_PAD)
    y0 = max(0, candidate.y0 - bh * CANDIDATE_PAD)
    y1 = min(ph, candidate.y1 + bh * CANDIDATE_PAD)

    h, w = crop_img.shape
    x0, y0 = max(0, int(x0 * SCALE)), max(0, int(y0 * SCALE))
    x1, y1 = min(w, int(x1 * SCALE)), min(h, int(y1 * SCALE))
    if x1 - x0 < MIN_CROP_PX[0] or y1 - y0 < MIN_CROP_PX[1]:
        return None
    return crop_img[y0:y1, x0:x1]
