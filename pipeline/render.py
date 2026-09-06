"""Page rendering.

The scans are A3 landscape PDFs (1224x842 pt) wrapping a single 5100x3510 JPEG,
i.e. 300 DPI. There is no text layer, so everything downstream works off pixels.

Two resolutions are used throughout the pipeline:

  MATCH_WIDTH  a downscaled page, used to find the job-card anchor. Template
               matching is O(page area x scales x angles), so doing it at full
               resolution is pointlessly slow -- the printed title we match on
               is perfectly legible at 1500px wide.
  CROP_WIDTH   the native 5100px, used to cut the number field once we know
               where it is. The stamped digits need the stroke detail.
"""

import os

import cv2
import fitz  # PyMuPDF
import numpy as np

MATCH_WIDTH = 1500
CROP_WIDTH = 5100
SCALE = CROP_WIDTH / MATCH_WIDTH

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
SUPPORTED_EXTS = {".pdf"} | IMAGE_EXTS


def _render(page, target_width):
    zoom = target_width / page.rect.width
    pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csGRAY)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _resize_width(img, target_width):
    f = target_width / img.shape[1]
    interp = cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC
    return cv2.resize(img, None, fx=f, fy=f, interpolation=interp)


def page_pairs(path):
    """Yield (match_img, crop_img) grayscale arrays, one pair per page.

    Both arrays are the same page at different resolutions; convert a match-space
    coordinate to crop space by multiplying by SCALE.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        with fitz.open(path) as doc:
            for page in doc:
                yield _render(page, MATCH_WIDTH), _render(page, CROP_WIDTH)
    elif ext in IMAGE_EXTS:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        yield _resize_width(img, MATCH_WIDTH), _resize_width(img, CROP_WIDTH)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def render_page(path, page_index):
    """(match_img, crop_img) for a single page, for page-level parallel work.

    page_pairs() opens the file once and streams every page from that one
    handle, which is the right shape for whole-file analysis. Batch scans need
    the opposite: many pages, each dispatched to a different worker process, so
    each call here opens the file itself.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        with fitz.open(path) as doc:
            page = doc[page_index]
            return _render(page, MATCH_WIDTH), _render(page, CROP_WIDTH)
    elif ext in IMAGE_EXTS:
        if page_index != 0:
            raise ValueError(f"{path} is a single-page image; page_index must be 0")
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {path}")
        return _resize_width(img, MATCH_WIDTH), _resize_width(img, CROP_WIDTH)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def page_count(path):
    """Number of pages path will yield from page_pairs, without decoding pixels."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        with fitz.open(path) as doc:
            return doc.page_count
    return 1


def extract_page(path, page_index, dest_path, rotation=0):
    """Copy a single page out of a multi-page PDF as its own standalone PDF.

    Uses insert_pdf rather than a render round-trip, so the page keeps its
    original pixels exactly -- no re-encoding, no generation loss. `rotation`
    (any multiple of 90) corrects a page that analyse_page found rotated: it
    sets the PDF's page-rotation metadata rather than touching pixels, so
    viewers display it right-side up at full quality. For a single-page image file,
    page_index must be 0 and this just copies the file (rotation is not
    applied to bare images -- not needed by anything that calls this today).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        with fitz.open(path) as src, fitz.open() as out:
            out.insert_pdf(src, from_page=page_index, to_page=page_index)
            if rotation:
                out[0].set_rotation((out[0].rotation + rotation) % 360)
            out.save(dest_path)
    elif ext in IMAGE_EXTS:
        if page_index != 0:
            raise ValueError(f"{path} is a single-page image; page_index must be 0")
        import shutil

        shutil.copy2(path, dest_path)
    else:
        raise ValueError(f"Unsupported file type: {path}")
