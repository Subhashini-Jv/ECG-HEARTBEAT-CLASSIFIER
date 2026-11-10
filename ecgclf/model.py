"""1D residual CNN over the beat waveform, fused with an RR-interval branch."""

from __future__ import annotations

import torch
from torch import nn

from . import config as C


class ResBlock1d(nn.Module):
    def __init__(self, c_in: int, c_out: int, stride: int, k: int = 7, p_drop: float = 0.1):
        super().__init__()
        pad = k // 2
        self.body = nn.Sequential(
            nn.Conv1d(c_in, c_out, k, stride=stride, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
            nn.ReLU(inplace=True),
            nn.Dropout(p_drop),
            nn.Conv1d(c_out, c_out, k, padding=pad, bias=False),
            nn.BatchNorm1d(c_out),
        )
        self.skip = (nn.Identity() if stride == 1 and c_in == c_out else
                     nn.Sequential(nn.Conv1d(c_in, c_out, 1, stride=stride, bias=False), nn.BatchNorm1d(c_out)))
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.body(x) + self.skip(x))


class BeatNet(nn.Module):
    """Waveform (2 x 256: beat, beat minus patient's median beat) -> residual CNN -> [avg || max] pooling,
    RR-ratio features (5) -> small MLP, then concatenated into a classifier head.

    ~120k parameters; trains in minutes on a laptop CPU.
    """

    def __init__(self, n_classes: int = len(C.CLASSES), n_rr: int = 5, use_rr: bool = True, in_ch: int = 2,
                 widths: tuple[int, ...] = (16, 32, 48, 64), p_drop: float = 0.3, p_wave_drop: float = 0.5):
        super().__init__()
        self.use_rr = use_rr
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, widths[0], 11, padding=5, bias=False),
            nn.BatchNorm1d(widths[0]),
            nn.ReLU(inplace=True),
        )
        blocks, c = [], widths[0]
        for i, w in enumerate(widths):
            blocks.append(ResBlock1d(c, w, stride=1 if i == 0 else 2))
            c = w
        self.blocks = nn.Sequential(*blocks)
        # Heavy dropout on the waveform embedding stops the CNN from memorising the
        # morphology of the training patients and forces use of the RR branch.
        self.wave_drop = nn.Dropout(p_wave_drop)
        feat = 2 * c
        if use_rr:
            self.rr_net = nn.Sequential(
                nn.BatchNorm1d(n_rr), nn.Linear(n_rr, 32), nn.ReLU(inplace=True), nn.Linear(32, 32), nn.ReLU(inplace=True)
            )
            feat += 32
        self.head = nn.Sequential(
            nn.Dropout(p_drop), nn.Linear(feat, 128), nn.ReLU(inplace=True), nn.Dropout(p_drop), nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor, rr: torch.Tensor | None = None) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        z = self.wave_drop(torch.cat([h.mean(-1), h.amax(-1)], dim=1))
        if self.use_rr:
            z = torch.cat([z, self.rr_net(rr)], dim=1)
        return self.head(z)
