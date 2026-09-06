"""Synthesize training sequences from the 158 real stamp-font glyphs.

Composites real digit glyphs (not a generic system font -- see
extract_glyphs.py) into 4-5 digit sequences with augmentation matched to what
actually varies on these scans: ink darkness/erosion (stamp wear, double
strikes), slight rotation and baseline jitter (mechanical stamp, not a
printer), elastic warp (paper isn't flat), and paper-texture/noise background
(re-used from real crop backgrounds, not synthetic noise, so the background
statistics match the real scans).
"""
import os
import random

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
GLYPH_DIR = os.path.join(HERE, "glyphs")
OUT_DIR = os.path.join(HERE, "synth")

TARGET_H = 64          # rendered glyph height before composition
MIN_LEN, MAX_LEN = 4, 5

glyphs = {d: [] for d in range(10)}
for d in range(10):
    for f in os.listdir(os.path.join(GLYPH_DIR, str(d))):
        img = cv2.imread(os.path.join(GLYPH_DIR, str(d), f), cv2.IMREAD_GRAYSCALE)
        glyphs[d].append(img)

backgrounds = []


def _register_background(patch):
    backgrounds.append(patch)


def _random_background(w, h):
    if not backgrounds:
        return np.full((h, w), 245, np.uint8) + np.random.randint(-4, 4, (h, w)).astype(np.uint8)
    bg = random.choice(backgrounds)
    bh, bw = bg.shape
    if bw < w or bh < h:
        bg = cv2.resize(bg, (max(w, bw) + 1, max(h, bh) + 1))
        bh, bw = bg.shape
    x = random.randint(0, bw - w)
    y = random.randint(0, bh - h)
    return bg[y : y + h, x : x + w].copy()


def _augment_glyph(img):
    h, w = img.shape
    scale = TARGET_H / h
    img = cv2.resize(img, (max(1, int(w * scale)), TARGET_H), interpolation=cv2.INTER_CUBIC)

    # Ink weight: erosion/dilation mimics stamp wear and double-strike bleed.
    k = random.choice([-2, -1, -1, 0, 0, 0, 1, 1, 2])
    if k != 0:
        kernel = np.ones((abs(k) + 1, abs(k) + 1), np.uint8)
        img = cv2.erode(img, kernel) if k > 0 else cv2.dilate(img, kernel)

    # Small rotation -- a mechanical stamp isn't perfectly level.
    angle = random.uniform(-6, 6)
    hh, ww = img.shape
    M = cv2.getRotationMatrix2D((ww / 2, hh / 2), angle, 1.0)
    img = cv2.warpAffine(img, M, (ww, hh), borderValue=255)

    # Mild elastic warp -- paper isn't flat under the stamp.
    dx = np.random.uniform(-1, 1, (4, 4)).astype(np.float32)
    dy = np.random.uniform(-1, 1, (4, 4)).astype(np.float32)
    dx = cv2.resize(dx, (ww, hh)) * 2
    dy = cv2.resize(dy, (ww, hh)) * 2
    xx, yy = np.meshgrid(np.arange(ww), np.arange(hh))
    map_x = (xx + dx).astype(np.float32)
    map_y = (yy + dy).astype(np.float32)
    img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR, borderValue=255)

    # Contrast/brightness jitter -- inking varies strike to strike. Only the
    # ink itself should move; naively scaling the whole rectangle also drags
    # the white margin around each glyph into a visible gray patch once
    # composited onto a shared background (caught by eye in synth_preview.png
    # before this fix -- every digit sat in its own faint rectangle).
    alpha = random.uniform(0.75, 1.15)
    beta = random.uniform(-15, 15)
    adjusted = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    img = np.where(img > 230, 255, adjusted).astype(np.uint8)
    return img


def make_sequence():
    length = random.choice([MIN_LEN] * 3 + [MAX_LEN] * 2)  # 4-digit more common in this archive
    digits = [random.randint(0, 9) for _ in range(length)]
    imgs = [_augment_glyph(random.choice(glyphs[d])) for d in digits]

    gap = random.randint(-3, 8)
    total_w = sum(im.shape[1] for im in imgs) + gap * (length - 1) + 20
    canvas_h = TARGET_H + 20
    canvas = _random_background(total_w, canvas_h)
    if canvas.shape != (canvas_h, total_w):
        canvas = cv2.resize(canvas, (total_w, canvas_h))

    x = 10
    y_base = 10
    for im in imgs:
        h, w = im.shape
        y = y_base + random.randint(-3, 3)
        y1 = min(canvas_h, y + h)
        region = canvas[y:y1, x : x + w]
        im_c = im[: region.shape[0], : region.shape[1]]
        canvas[y:y1, x : x + w] = np.minimum(region, im_c)
        x += w + gap

    canvas = canvas[:, : max(x, 10)]

    # Whole-sequence noise + occasional blur, matching a scanned/compressed page.
    noise = np.random.normal(0, random.uniform(2, 8), canvas.shape)
    canvas = np.clip(canvas.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if random.random() < 0.3:
        canvas = cv2.GaussianBlur(canvas, (3, 3), random.uniform(0.3, 0.8))

    return canvas, "".join(str(d) for d in digits)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, r"c:\Users\mcl11\Desktop\ecr")
    import glob

    import fitz

    # Real paper texture, sampled from many random regions of the actual
    # scans (below the top 45% where a real number could live, so a
    # background patch never accidentally contradicts a synthetic label) --
    # keeps noise statistics matched to the real thing instead of guessing at
    # a noise model. Some of these regions catch real printed content
    # (signature lines, phone numbers) rather than blank paper, which given
    # 60+ varied samples is a feature, not a bug -- the real crops the model
    # will see in production have exactly this kind of incidental clutter
    # near the number.
    with fitz.open(r"c:\Users\mcl11\Desktop\ecr\incoming\Image_002.pdf") as doc:
        for i in range(0, 79, 3):
            page = doc[i]
            pix = page.get_pixmap(matrix=fitz.Matrix(3, 3), colorspace=fitz.csGRAY)
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
            h, w = img.shape
            top = int(h * 0.45)
            for _ in range(2):
                bh, bw = 300, 500
                y = random.randint(top, h - bh)
                x = random.randint(0, w - bw)
                _register_background(img[y : y + bh, x : x + bw])

    os.makedirs(OUT_DIR, exist_ok=True)
    n = 20000
    labels = []
    for i in range(n):
        img, label = make_sequence()
        cv2.imwrite(os.path.join(OUT_DIR, f"{i:06d}.png"), img)
        labels.append(f"{i:06d}.png\t{label}")
        if i % 2000 == 0:
            print(i, flush=True)
    with open(os.path.join(OUT_DIR, "labels.txt"), "w") as fh:
        fh.write("\n".join(labels))
    print(f"wrote {n} synthetic sequences -> {OUT_DIR}")
