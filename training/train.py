"""Train the CRNN on synthetic data, validate on REAL crops.

Validation set is the actual candidate crops from output_v2/renamed and
output_v2/dc -- real scans, real stamp, ground truth from the filename (the
same numbers already scored against Google Vision earlier). Synthetic-only
validation would just measure how well the model learned the synthesis
process, not whether it reads real ink -- this is the only honest check
available.
"""
import csv
import os
import random
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
V3_DIR = os.path.dirname(HERE)  # v3/ -- use v3's own pipeline, not root's or another version's
PROJECT_ROOT = os.path.dirname(V3_DIR)
sys.path.insert(0, HERE)
sys.path.insert(0, V3_DIR)

from model import BLANK, CRNN, IMG_H, NUM_CLASSES, decode_greedy, encode
from pipeline import detect
from pipeline.render import SCALE, render_page

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SYNTH_DIR = os.path.join(HERE, "synth")
CKPT = os.path.join(HERE, "crnn_digits.pt")


def load_gray(path, target_h=IMG_H):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    h, w = img.shape
    scale = target_h / h
    img = cv2.resize(img, (max(8, int(w * scale)), target_h), interpolation=cv2.INTER_AREA)
    return img


class SynthDataset(Dataset):
    def __init__(self, labels_path):
        with open(labels_path, encoding="utf-8") as fh:
            self.items = [line.split("\t") for line in fh.read().splitlines() if line.strip()]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        fname, label = self.items[idx]
        img = load_gray(os.path.join(SYNTH_DIR, fname))
        return img, label


def collate(batch):
    imgs, labels = zip(*batch)
    max_w = max(im.shape[1] for im in imgs)
    batch_imgs = np.full((len(imgs), 1, IMG_H, max_w), 255, np.float32)
    for i, im in enumerate(imgs):
        batch_imgs[i, 0, :, : im.shape[1]] = im
    batch_imgs = (batch_imgs / 127.5) - 1.0
    targets = torch.cat([torch.tensor(encode(l), dtype=torch.long) for l in labels])
    target_lens = torch.tensor([len(l) for l in labels], dtype=torch.long)
    return torch.from_numpy(batch_imgs), targets, target_lens, labels


def build_real_validation():
    """Real, ground-truth-labelled crops -- tight box, same as extract_glyphs.py.

    Deliberately reads output_v2/results.csv, not v3's: v2's accepted pages
    were all confirmed without any involvement from this model, so they're
    independent ground truth. Validating against v3's own output would be
    partly circular -- some of those acceptances are this model's own votes.
    """
    path = os.path.join(PROJECT_ROOT, "incoming", "Image_002.pdf")
    items = []
    with open(os.path.join(PROJECT_ROOT, "output_v2", "results.csv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["status"] in ("renamed", "dc"):
                num = os.path.splitext(os.path.basename(row["destination"]))[0]
                items.append((int(row["page"]), num))

    val = []
    for page_no, true_num in items:
        match_img, crop_img = render_page(path, page_no - 1)
        page_type, candidates = detect.scan(match_img)
        cand = next((c for c in candidates if c.digits == true_num), None)
        if cand is None:
            continue
        x0, y0, x1, y1 = (int(v * SCALE) for v in (cand.x0, cand.y0, cand.x1, cand.y1))
        patch = crop_img[y0:y1, x0:x1]
        if patch.size == 0:
            continue
        h, w = patch.shape
        scale = IMG_H / h
        patch = cv2.resize(patch, (max(8, int(w * scale)), IMG_H), interpolation=cv2.INTER_AREA)
        val.append((patch, true_num, page_no))
    return val


def evaluate_real(model, val):
    model.eval()
    correct = 0
    wrong = []
    with torch.no_grad():
        for patch, true_num, page_no in val:
            img = (patch.astype(np.float32) / 127.5) - 1.0
            t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            logits = model(t)  # (W, 1, C)
            pred = decode_greedy(logits[:, 0, :].cpu())
            if pred == true_num:
                correct += 1
            else:
                wrong.append((page_no, true_num, pred))
    model.train()
    return correct, len(val), wrong


def main():
    print(f"device: {DEVICE}")
    train_ds = SynthDataset(os.path.join(SYNTH_DIR, "labels.txt"))
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, collate_fn=collate, num_workers=0)
    val = build_real_validation()
    print(f"train sequences: {len(train_ds)}   real validation crops: {len(val)}")

    model = CRNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=15 * len(train_loader))
    ctc = nn.CTCLoss(blank=BLANK, zero_infinity=True)

    best_acc = -1
    EPOCHS = 15
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for imgs, targets, target_lens, labels in train_loader:
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            logits = model(imgs)  # (W, B, C)
            log_probs = logits.log_softmax(2)
            input_lens = torch.full((imgs.size(0),), logits.size(0), dtype=torch.long)
            loss = ctc(log_probs, targets, input_lens, target_lens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            opt.step()
            sched.step()
            total_loss += loss.item()

        correct, total, wrong = evaluate_real(model, val)
        acc = correct / total
        print(f"epoch {epoch+1}/{EPOCHS}  loss={total_loss/len(train_loader):.4f}  "
              f"real_val_acc={correct}/{total}={acc:.3f}", flush=True)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), CKPT)
            print(f"  -> saved (best so far)", flush=True)

    print(f"\nbest real-validation accuracy: {best_acc:.3f}, checkpoint at {CKPT}")
    model.load_state_dict(torch.load(CKPT))
    correct, total, wrong = evaluate_real(model, val)
    print(f"final check: {correct}/{total}")
    for page_no, true_num, pred in wrong:
        print(f"  WRONG p{page_no}: true={true_num} pred={pred!r}")


if __name__ == "__main__":
    main()
