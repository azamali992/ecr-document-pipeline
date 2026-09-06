"""Small CRNN+CTC digit-sequence recognizer.

Deliberately narrow: 11 output classes (blank + digits 0-9), versus PP-OCR's
~6625-class vocabulary competing across Urdu/CJK/Latin for the same ink --
the earlier research flagged that as a likely source of the "None" reread
votes on faint crops (a broken digit stroke losing to some other script's
glyph in the softmax). A model that can ONLY output digits can't lose that
way. Small enough to train from scratch on ~20k synthetic sequences in
minutes on a 6GB GPU, rather than needing PaddlePaddle's fine-tuning path
(and its far less currently-maintained GPU wheel situation).
"""
import torch
import torch.nn as nn

BLANK = 0
CLASSES = "0123456789"  # index i -> digit i-1 after the blank
NUM_CLASSES = len(CLASSES) + 1
IMG_H = 32


class CRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/2
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2),  # H/4
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d((2, 1), (2, 1)),  # H/8, keep width resolution for digit-level timesteps
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(),
        )
        # IMG_H=32 -> after pools: 32/2/2/2 = 4
        self.rnn = nn.LSTM(256 * 4, 128, num_layers=2, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(256, NUM_CLASSES)

    def forward(self, x):
        # x: (B, 1, H, W)
        f = self.cnn(x)  # (B, C, H', W')
        b, c, h, w = f.shape
        f = f.permute(0, 3, 1, 2).reshape(b, w, c * h)  # (B, W, C*H) -- W is the timestep axis
        out, _ = self.rnn(f)
        out = self.fc(out)  # (B, W, NUM_CLASSES)
        return out.permute(1, 0, 2)  # (W, B, NUM_CLASSES) for CTC


def encode(label):
    return [CLASSES.index(ch) + 1 for ch in label]


def decode_greedy(logits):
    """logits: (W, NUM_CLASSES) for one sample. Standard CTC greedy collapse."""
    ids = logits.argmax(dim=-1).tolist()
    out = []
    prev = BLANK
    for i in ids:
        if i != BLANK and i != prev:
            out.append(CLASSES[i - 1])
        prev = i
    return "".join(out)
