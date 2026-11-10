"""Per-class metrics in the vocabulary used by the ECG literature (AAMI EC57)."""

from __future__ import annotations

import numpy as np

from . import config as C


def confusion(y_true: np.ndarray, y_pred: np.ndarray, k: int = len(C.CLASSES)) -> np.ndarray:
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def _div(a: float, b: float) -> float:
    return float(a / b) if b else float("nan")


def report(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Recall (sensitivity, Se), precision (+P), specificity (Sp), F1 per class,
    plus accuracy, macro recall (= balanced accuracy) and macro F1."""
    cm = confusion(y_true, y_pred)
    total = cm.sum()
    per_class = {}
    for i, c in enumerate(C.CLASSES):
        tp = cm[i, i]
        fn = cm[i].sum() - tp
        fp = cm[:, i].sum() - tp
        tn = total - tp - fn - fp
        rec, prec = _div(tp, tp + fn), _div(tp, tp + fp)
        f1 = _div(2 * prec * rec, prec + rec) if not (np.isnan(prec) or np.isnan(rec)) else float("nan")
        per_class[c] = {
            "support": int(cm[i].sum()), "recall": rec, "precision": prec,
            "specificity": _div(tn, tn + fp), "f1": f1,
        }
    recalls = [v["recall"] for v in per_class.values() if not np.isnan(v["recall"])]
    f1s = [0.0 if np.isnan(v["f1"]) else v["f1"] for v in per_class.values() if v["support"] > 0]
    return {
        "accuracy": _div(np.trace(cm), total),
        "macro_recall": float(np.mean(recalls)) if recalls else float("nan"),
        "macro_f1": float(np.mean(f1s)) if f1s else float("nan"),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def format_report(r: dict) -> str:
    lines = [f"{'class':<6}{'support':>9}{'recall':>9}{'precision':>11}{'spec':>8}{'F1':>8}"]
    for c, v in r["per_class"].items():
        lines.append(f"{c:<6}{v['support']:>9}{v['recall']:>9.3f}{v['precision']:>11.3f}"
                     f"{v['specificity']:>8.3f}{v['f1']:>8.3f}")
    lines.append(f"accuracy {r['accuracy']:.4f} | macro recall {r['macro_recall']:.4f} | macro F1 {r['macro_f1']:.4f}")
    return "\n".join(lines)
