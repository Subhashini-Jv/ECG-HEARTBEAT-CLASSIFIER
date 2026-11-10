"""Figures for the README / reports (matplotlib, headless)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import config as C  # noqa: E402
from .data import Beats  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
# Fixed categorical order: class identity never changes colour between figures.
CLASS_COLORS = {"N": "#2a78d6", "S": "#eb6834", "V": "#1baf7a", "F": "#eda100"}
BLUES = LinearSegmentedColormap.from_list(
    "blues", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titlecolor": INK, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "font.size": 10, "legend.frameon": False,
})


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def confusion_matrices(cm: np.ndarray, path: Path, title: str = "Test set (DS2, unseen patients)") -> None:
    cm = np.asarray(cm)
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
    for ax, mat, sub in [(axes[0], cm, "Counts"), (axes[1], norm, "Row-normalised (recall on diagonal)")]:
        shown = norm  # colour always by row share so tiny classes stay visible
        ax.imshow(shown, cmap=BLUES, vmin=0, vmax=1)
        ax.grid(False)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                txt = f"{mat[i, j]:,}" if mat is cm else f"{mat[i, j]:.2f}"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10,
                        color="white" if shown[i, j] > 0.55 else INK)
        ax.set_xticks(range(len(C.CLASSES)), C.CLASSES)
        ax.set_yticks(range(len(C.CLASSES)), C.CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(sub, fontsize=11, fontweight="normal", color=INK_2)
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, path)


def training_curves(history: list[dict], path: Path) -> None:
    ep = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    ax = axes[0]
    ax.plot(ep, [h["train_loss"] for h in history], lw=2, color="#2a78d6", label="train")
    ax.plot(ep, [h["val_loss"] for h in history], lw=2, color="#eb6834", label="validation")
    ax.set_title("Weighted cross-entropy")
    ax.set_xlabel("Epoch")
    ax.legend()
    ax = axes[1]
    for c in C.CLASSES:
        ax.plot(ep, [h["val_recall"][c] for h in history], lw=2, color=CLASS_COLORS[c], label=c)
    ax.plot(ep, [h["val_macro_recall"] for h in history], lw=2, color=INK, ls="--", label="macro")
    best = max(history, key=lambda h: h["val_macro_recall"])["epoch"]
    ax.axvline(best, color=INK_2, lw=1, ls=":")
    ax.text(best, 0.02, " best", color=INK_2, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title("Validation recall per class")
    ax.set_xlabel("Epoch")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    _save(fig, path)


def example_beats(b: Beats, path: Path, n: int = 40, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    t = (np.arange(C.WINDOW) - C.PRE_SAMPLES) / C.FS * 1000
    fig, axes = plt.subplots(1, len(C.CLASSES), figsize=(13, 3.2), sharey=True)
    for ax, (k, c) in zip(axes, enumerate(C.CLASSES)):
        idx = np.where(b.y == k)[0]
        if len(idx):
            pick = rng.choice(idx, min(n, len(idx)), replace=False)
            for i in pick:
                ax.plot(t, b.X[i], color=CLASS_COLORS[c], lw=0.6, alpha=0.25)
            ax.plot(t, np.median(b.X[idx], 0), color=INK, lw=2)
        ax.set_title(f"{c}: {C.CLASS_NAMES[c]}", fontsize=10)
        ax.set_xlabel("ms from R-peak")
    axes[0].set_ylabel("amplitude (robust z)")
    fig.suptitle(f"{n} random beats per class, median in black", fontsize=12)
    fig.tight_layout()
    _save(fig, path)


def per_record_recall(rows: list[dict], path: Path) -> None:
    """Dot plot: recall per class for each test record (patient)."""
    fig, ax = plt.subplots(figsize=(12, 3.8))
    recs = [r["record"] for r in rows]
    x = np.arange(len(recs))
    for k, c in enumerate(C.CLASSES):
        off = (k - 1.5) * 0.15
        ys = [r.get(f"recall_{c}") for r in rows]
        xs = [xi + off for xi, yv in zip(x, ys) if yv is not None]
        ys = [yv for yv in ys if yv is not None]
        ax.scatter(xs, ys, s=28, color=CLASS_COLORS[c], edgecolor=SURFACE, linewidth=1.5, label=c, zorder=3)
    ax.set_xticks(x, recs, rotation=0, fontsize=9)
    ax.set_ylim(-0.03, 1.05)
    ax.set_ylabel("Recall")
    ax.set_xlabel("Test record (one patient each); classes with fewer than 5 beats omitted")
    ax.set_title("Recall varies a lot between patients", loc="left")
    ax.legend(ncol=4, loc="lower right", bbox_to_anchor=(1.0, 1.0), borderaxespad=0.2)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    _save(fig, path)
