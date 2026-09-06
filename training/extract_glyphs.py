"""Extract individual digit glyphs from the confirmed-correct crops in
output_v2/renamed and output_v2/dc.

For each accepted page we know the true digit string from the filename.
Segment the crop into connected components, keep it only if the component
count matches the digit count (cheap, safe validation -- ambiguous
segmentations are just dropped rather than guessed at), then save each digit
image labelled by its true class. This is the real stamp font, not a
approximation -- synthetic training data built from these glyphs should
generalise far better than a generic system font would.
"""
import csv
import os
import sys

import cv2
import numpy as np

PROJECT = r"c:\Users\mcl11\Desktop\ecr"
sys.path.insert(0, PROJECT)

from pipeline import detect
from pipeline.render import render_page, SCALE

HERE = os.path.dirname(os.path.abspath(__file__))
GLYPH_DIR = os.path.join(HERE, "glyphs")
for d in range(10):
    os.makedirs(os.path.join(GLYPH_DIR, str(d)), exist_ok=True)

PATH = os.path.join(PROJECT, "incoming", "Image_002.pdf")

rows = []
with open(os.path.join(PROJECT, "output_v2", "results.csv"), encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["status"] in ("renamed", "dc"):
            num = os.path.splitext(os.path.basename(row["destination"]))[0]
            rows.append((int(row["page"]), num))

counts = {d: 0 for d in range(10)}
used_pages = skipped_pages = 0

for page_no, true_num in rows:
    match_img, crop_img = render_page(PATH, page_no - 1)
    page_type, candidates = detect.scan(match_img)
    # the accepted candidate is whichever one's nomination equals the filed number
    cand = next((c for c in candidates if c.digits == true_num), None)
    if cand is None:
        skipped_pages += 1
        continue
    # Deliberately NOT candidate_field's padded crop: the 0.3 pad pulls in
    # part of the neighbouring Urdu word, which connected-components then
    # picks up as extra "digits". The detector's own tight box is exactly
    # the number run and nothing else.
    x0, y0, x1, y1 = (int(v * SCALE) for v in (cand.x0, cand.y0, cand.x1, cand.y1))
    patch = crop_img[y0:y1, x0:x1]
    if patch.size == 0:
        skipped_pages += 1
        continue

    _, binary = cv2.threshold(patch, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    h = patch.shape[0]
    boxes = []
    for i in range(1, n):  # skip background label 0
        x, y, w, bh, area = stats[i]
        # The stamp prints a faint stray dash/tick above each digit on some
        # cards (visible in dbg_tight.png); those run ~0.01-0.08 of the crop
        # height, real digit strokes 0.3+ -- a height filter cleanly drops
        # the former without needing to merge anything.
        if bh / h < 0.3:
            continue
        boxes.append((x, y, w, bh))
    boxes.sort(key=lambda b: b[0])  # left to right

    if len(boxes) != len(true_num):
        skipped_pages += 1
        continue

    used_pages += 1
    for (x, y, w, bh), digit_char in zip(boxes, true_num):
        pad = int(0.15 * bh)
        y0, y1 = max(0, y - pad), min(patch.shape[0], y + bh + pad)
        x0, x1 = max(0, x - pad), min(patch.shape[1], x + w + pad)
        glyph = patch[y0:y1, x0:x1]
        d = int(digit_char)
        cv2.imwrite(os.path.join(GLYPH_DIR, str(d), f"p{page_no}_{counts[d]}.png"), glyph)
        counts[d] += 1

print(f"used {used_pages} pages, skipped {skipped_pages} (segmentation didn't match digit count)")
print("glyphs per digit:", counts)
print(f"total glyphs: {sum(counts.values())}")
