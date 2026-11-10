"""Training loop with class-weighted loss, on-the-fly augmentation and
model selection on validation macro recall."""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import config as C
from .data import Beats, class_counts, two_channel
from .metrics import report
from .model import BeatNet

log = logging.getLogger(__name__)

RR_CLIP = (0.0, 5.0)


@dataclass
class TrainConfig:
    epochs: int = 15
    batch_size: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-3
    weight_power: float = 1.0      # class weight = (N / (K * n_c)) ** power; 0 = unweighted
    label_smoothing: float = 0.0   # note: interacts badly with strong class weights in PyTorch
    augment: bool = True
    use_rr: bool = True
    patience: int = 8
    channels: str = "diff"         # "both" | "raw" | "diff" (beat minus patient median beat)
    ema_decay: float = 0.998       # exponential moving average of weights (0 disables)
    seed: int = 0
    device: str = "auto"
    extra: dict = field(default_factory=dict)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


CHANNELS = {"both": [0, 1], "raw": [0], "diff": [1]}


def to_tensors(b: Beats, device: torch.device, channels: str = "both") -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.from_numpy(np.ascontiguousarray(two_channel(b)[:, CHANNELS[channels]])).to(device)
    rr = torch.from_numpy(np.clip(b.rr, *RR_CLIP)).to(device)
    y = torch.from_numpy(b.y).to(device)
    return x, rr, y


def augment(x: torch.Tensor, rr: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Cheap, physiologically plausible perturbations applied per batch."""
    n, _, t = x.shape
    dev = x.device
    scale = torch.empty(n, 1, 1, device=dev).uniform_(0.8, 1.2)
    offset = torch.empty(n, 1, 1, device=dev).uniform_(-0.2, 0.2)
    slope = torch.empty(n, 1, 1, device=dev).uniform_(-0.3, 0.3) * torch.linspace(-1, 1, t, device=dev)
    noise = torch.randn_like(x) * torch.empty(n, 1, 1, device=dev).uniform_(0.0, 0.05)
    x = x * scale + offset + slope + noise
    # small R-peak misalignment (+-4 samples ~ 11 ms), as a real detector would produce
    shift = int(torch.randint(-4, 5, (1,)).item())
    if shift:
        x = torch.roll(x, shift, dims=-1)
    rr = rr * torch.empty(n, 1, device=dev).uniform_(0.97, 1.03)
    return x, rr


def class_weights(y: np.ndarray, power: float) -> torch.Tensor:
    counts = np.bincount(y, minlength=len(C.CLASSES)).astype(np.float64)
    counts[counts == 0] = 1
    w = (counts.sum() / (len(counts) * counts)) ** power
    return torch.tensor(w / w.mean(), dtype=torch.float32)


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, rr: torch.Tensor, bs: int = 2048) -> np.ndarray:
    """Return softmax probabilities (n, K)."""
    model.eval()
    out = [torch.softmax(model(x[i:i + bs], rr[i:i + bs]), dim=1).cpu() for i in range(0, len(x), bs)]
    return torch.cat(out).numpy() if out else np.zeros((0, len(C.CLASSES)), np.float32)


def train(train_b: Beats, val_b: Beats, cfg: TrainConfig, out_dir: Path) -> tuple[BeatNet, dict]:
    set_seed(cfg.seed)
    device = pick_device(cfg.device)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("device=%s  train=%s  val=%s", device, class_counts(train_b.y), class_counts(val_b.y))

    xt, rt, yt = to_tensors(train_b, device, cfg.channels)
    xv, rv, yv = to_tensors(val_b, device, cfg.channels)
    model = BeatNet(use_rr=cfg.use_rr, in_ch=len(CHANNELS[cfg.channels])).to(device)
    weights = class_weights(train_b.y, cfg.weight_power).to(device)
    log.info("class weights %s", dict(zip(C.CLASSES, np.round(weights.cpu().numpy(), 3))))
    loss_fn = nn.CrossEntropyLoss(weight=weights, label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps = cfg.epochs * ((len(yt) + cfg.batch_size - 1) // cfg.batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr, total_steps=steps, pct_start=0.2)

    # An EMA of the weights is what gets validated and saved: it is far more stable
    # across epochs than the raw weights, which matters with tiny validation classes.
    ema = (torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg.ema_decay), use_buffers=True)
        if cfg.ema_decay > 0 else None)
    eval_model = ema.module if ema is not None else model

    history, best, best_state, bad = [], -1.0, None, 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        perm = torch.randperm(len(yt), device=device)
        tot, n = 0.0, 0
        for i in range(0, len(perm), cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            xb, rb, yb = xt[idx], rt[idx], yt[idx]
            if cfg.augment:
                xb, rb = augment(xb, rb)
            loss = loss_fn(model(xb, rb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            tot += loss.item() * len(idx)
            n += len(idx)

        pv = predict(eval_model, xv, rv)
        val_loss = float(nn.functional.cross_entropy(torch.from_numpy(np.log(pv + 1e-9)), torch.from_numpy(val_b.y),
                                                     weight=weights.cpu()))
        r = report(val_b.y, pv.argmax(1))
        rec = {c: round(r["per_class"][c]["recall"], 4) for c in C.CLASSES}
        history.append({"epoch": epoch, "train_loss": tot / n, "val_loss": val_loss,
                        "val_macro_recall": r["macro_recall"], "val_macro_f1": r["macro_f1"],
                        "val_recall": rec, "seconds": time.time() - t0})
        log.info("epoch %2d  loss %.4f  val_loss %.4f  val macro-recall %.4f  %s  (%.1fs)",
                 epoch, tot / n, val_loss, r["macro_recall"], rec, time.time() - t0)
        if r["macro_recall"] > best:
            best, bad = r["macro_recall"], 0
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
            best_epoch = epoch
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stopping at epoch %d", epoch)
                break

    model.load_state_dict(best_state)
    ckpt = {"state_dict": best_state, "train_config": asdict(cfg), "best_epoch": best_epoch,
            "best_val_macro_recall": best, "classes": C.CLASSES,
            "window": [C.PRE_SAMPLES, C.POST_SAMPLES]}
    torch.save(ckpt, out_dir / "model.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    log.info("best epoch %d, val macro recall %.4f -> %s", best_epoch, best, out_dir / "model.pt")
    return model, {"history": history, "best_epoch": best_epoch, "best_val_macro_recall": best}


class Ensemble(nn.Module):
    """Averages member probabilities; returns log-probabilities so that
    softmax(output) == mean of member softmaxes."""

    def __init__(self, members: list[nn.Module]):
        super().__init__()
        self.members = nn.ModuleList(members)

    def forward(self, x: torch.Tensor, rr: torch.Tensor | None = None) -> torch.Tensor:
        p = torch.stack([torch.softmax(m(x, rr), dim=1) for m in self.members]).mean(0)
        return torch.log(p.clamp_min(1e-12))


def save_ensemble(member_ckpts: list[Path], out: Path) -> None:
    cks = [torch.load(p, map_location="cpu", weights_only=False) for p in member_ckpts]
    torch.save({"members": [c["state_dict"] for c in cks], "train_config": cks[0]["train_config"],
                "best_epoch": [c["best_epoch"] for c in cks],
                "best_val_macro_recall": [c["best_val_macro_recall"] for c in cks],
                "classes": C.CLASSES, "window": [C.PRE_SAMPLES, C.POST_SAMPLES]}, out)


def load_model(path: Path, device: str = "auto") -> tuple[nn.Module, dict]:
    """Load a single model or an ensemble checkpoint."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    use_rr = ckpt["train_config"]["use_rr"]
    states = ckpt["members"] if "members" in ckpt else [ckpt["state_dict"]]
    members = []
    for st in states:
        m = BeatNet(use_rr=use_rr, in_ch=len(CHANNELS[ckpt["train_config"].get("channels", "both")]))
        m.load_state_dict(st)
        members.append(m)
    model = members[0] if len(members) == 1 else Ensemble(members)
    return model.to(pick_device(device)).eval(), ckpt
