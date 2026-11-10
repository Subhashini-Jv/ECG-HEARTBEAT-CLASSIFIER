"""Command-line interface.

    python -m ecgclf all                     # download -> prepare -> train -> evaluate
    python -m ecgclf prepare                 # download + cache beats
    python -m ecgclf train   [--epochs 30] [--no-rr] [--split inter|intra] [--out runs/main]
    python -m ecgclf evaluate --out runs/main
    python -m ecgclf predict  path/to/record [--model runs/main/model.pt]
    python -m ecgclf ablate                  # RR features, class weights, split leakage
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from . import config as C
from . import plots
from .data import Beats, beats_from_signal, build_dataset, class_counts, pick_lead, split
from .metrics import format_report, report
from .train import TrainConfig, load_model, predict, save_ensemble, to_tensors, train

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "mitdb"
CACHE = ROOT / "data" / "processed" / "beats.npz"
log = logging.getLogger("ecgclf")


def _dataset() -> Beats:
    return build_dataset(DATA_DIR, CACHE)


def cmd_prepare(_args) -> None:
    b = _dataset()
    print(f"{len(b):,} beats from {len(set(b.record))} records -> {CACHE}")
    for name, part in zip(("train", "val", "test"), split(b, "inter")):
        print(f"  {name:<5} records={len(set(part.record)):>2}  {class_counts(part.y)}")


def _train_cfg(args) -> TrainConfig:
    return TrainConfig(epochs=args.epochs, batch_size=args.batch_size, lr=args.lr, weight_power=args.weight_power,
                       augment=not args.no_augment, use_rr=not args.no_rr, channels=args.channels, seed=args.seed, device=args.device)


def cmd_train(args) -> None:
    b = _dataset()
    tr, va, _ = split(b, args.split, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "split.txt").write_text(args.split)
    if args.ensemble <= 1:
        _, info = train(tr, va, _train_cfg(args), out)
        plots.training_curves(info["history"], out / "figures" / "training_curves.png")
        return
    # Ensemble: same recipe, different seeds; probabilities are averaged at inference.
    members = []
    for k in range(args.ensemble):
        cfg = _train_cfg(args)
        cfg.seed = args.seed + k
        mdir = out / f"member_{k}"
        log.info("=== ensemble member %d/%d (seed %d) ===", k + 1, args.ensemble, cfg.seed)
        _, info = train(tr, va, cfg, mdir)
        members.append(mdir / "model.pt")
        if k == 0:
            plots.training_curves(info["history"], out / "figures" / "training_curves.png")
            (out / "history.json").write_text((mdir / "history.json").read_text())
    save_ensemble(members, out / "model.pt")


def evaluate_run(out: Path, device: str = "auto", make_plots: bool = True) -> dict:
    b = _dataset()
    split_mode = (out / "split.txt").read_text().strip() if (out / "split.txt").exists() else "inter"
    _, _, te = split(b, split_mode)
    model, ckpt = load_model(out / "model.pt", device)
    dev = next(model.parameters()).device
    x, rr, _ = to_tensors(te, dev, ckpt["train_config"].get("channels", "both"))
    prob = predict(model, x, rr)
    pred = prob.argmax(1)
    r = report(te.y, pred)
    r["split"] = split_mode
    r["n_models"] = len(ckpt.get("members", [0]))
    r["best_epoch"] = ckpt["best_epoch"]
    r["best_val_macro_recall"] = ckpt["best_val_macro_recall"]
    r["train_config"] = ckpt["train_config"]

    rows = []
    for rec in sorted(set(te.record)):
        m = te.record == rec
        row = {"record": rec, "beats": int(m.sum()), "accuracy": float((pred[m] == te.y[m]).mean())}
        for k, c in enumerate(C.CLASSES):
            mk = m & (te.y == k)
            row[f"n_{c}"] = int(mk.sum())
            row[f"recall_{c}"] = float((pred[mk] == k).mean()) if mk.sum() >= 5 else None
        rows.append(row)

    (out / "metrics.json").write_text(json.dumps(r, indent=2))
    (out / "report.txt").write_text(format_report(r) + "\n")
    with open(out / "per_record.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    if make_plots:
        fig = out / "figures"
        title = "Test set: DS2, 22 unseen patients" if split_mode == "inter" else "Test set: random beats (leaky)"
        plots.confusion_matrices(np.array(r["confusion_matrix"]), fig / "confusion_matrix.png", title)
        plots.example_beats(te, fig / "example_beats.png")
        if split_mode == "inter":
            plots.per_record_recall(rows, fig / "per_record_recall.png")
        hist = out / "history.json"
        if hist.exists():
            plots.training_curves(json.loads(hist.read_text()), fig / "training_curves.png")
    return r


def cmd_evaluate(args) -> None:
    r = evaluate_run(Path(args.out), args.device)
    print(format_report(r))


def cmd_all(args) -> None:
    cmd_prepare(args)
    cmd_train(args)
    cmd_evaluate(args)


def cmd_ablate(args) -> None:
    """Train variants and write a markdown table comparing them on DS2."""
    variants = [
        ("single model, default recipe", {}),
        ("+ raw beat channel (not patient-normalised)", {"channels": "both"}),
        ("no RR-ratio features", {"no_rr": True}),
        ("unweighted loss", {"weight_power": 0.0}),
        ("intra-patient split (leaky, for contrast)", {"split": "intra"}),
    ]
    base = Path(args.out)
    lines = ["| variant | acc | macro recall | N | S | V | F |", "|---|---|---|---|---|---|---|"]
    for i, (name, over) in enumerate(variants):
        a = argparse.Namespace(**{**vars(args), "ensemble": 1, **over, "out": str(base / f"ablation_{i}")})
        log.info("=== %s ===", name)
        cmd_train(a)
        r = evaluate_run(Path(a.out), args.device, make_plots=(over.get("split") == "intra"))
        pc = r["per_class"]
        lines.append(f"| {name} | {r['accuracy']:.3f} | {r['macro_recall']:.3f} | "
                     + " | ".join(f"{pc[c]['recall']:.3f}" for c in C.CLASSES) + " |")
        (base / "ablation.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def cmd_predict(args) -> None:
    import wfdb
    from wfdb import processing

    model, ckpt = load_model(Path(args.model), args.device)
    record = wfdb.rdrecord(args.record)
    if record.fs != C.FS:
        raise SystemExit(f"model expects {C.FS} Hz input, record is {record.fs} Hz (resample first)")
    sig = pick_lead(record)
    if Path(args.record + ".atr").exists() and not args.detect:
        ann = wfdb.rdann(args.record, "atr")
        peaks = np.array([s for s, sym in zip(ann.sample, ann.symbol) if sym in "NLRBAaJSVrFejnE/fQ?"])
        source = "annotation"
    else:
        peaks = processing.xqrs_detect(sig, fs=C.FS, verbose=False)
        peaks = processing.correct_peaks(sig, peaks, search_radius=int(0.05 * C.FS), smooth_window_size=20)
        source = "xqrs detector"
    beats = beats_from_signal(sig, np.asarray(peaks), np.zeros(len(peaks), dtype=np.int64), Path(args.record).name)
    x, rr, _ = to_tensors(beats, next(model.parameters()).device, ckpt["train_config"].get("channels", "both"))
    prob = predict(model, x, rr)
    out = Path(args.output or (Path(args.record).name + "_predictions.csv"))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample", "time_s", "pred"] + [f"p_{c}" for c in C.CLASSES])
        for s, p in zip(beats.sample, prob):
            w.writerow([int(s), round(s / C.FS, 3), C.CLASSES[int(p.argmax())]] + [round(float(v), 4) for v in p])
    counts = {c: int((prob.argmax(1) == k).sum()) for k, c in enumerate(C.CLASSES)}
    print(f"{len(beats)} beats (R-peaks from {source}) -> {out}\n{counts}")


def main() -> None:
    p = argparse.ArgumentParser(prog="ecgclf", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add_train_args(sp):
        sp.add_argument("--out", default=str(ROOT / "runs" / "main"))
        sp.add_argument("--split", choices=["inter", "intra"], default="inter")
        sp.add_argument("--epochs", type=int, default=15)
        sp.add_argument("--ensemble", type=int, default=3, help="number of seeds to train and average")
        sp.add_argument("--batch-size", type=int, default=256)
        sp.add_argument("--lr", type=float, default=2e-3)
        sp.add_argument("--weight-power", type=float, default=1.0)
        sp.add_argument("--no-rr", action="store_true")
        sp.add_argument("--channels", choices=["both", "raw", "diff"], default="diff")
        sp.add_argument("--no-augment", action="store_true")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--device", default="auto")

    sub.add_parser("prepare").set_defaults(fn=cmd_prepare)
    for name, fn in [("train", cmd_train), ("all", cmd_all)]:
        sp = sub.add_parser(name)
        add_train_args(sp)
        sp.set_defaults(fn=fn)
    sp = sub.add_parser("ablate")
    add_train_args(sp)
    sp.set_defaults(fn=cmd_ablate, out=str(ROOT / "runs"))
    sp = sub.add_parser("evaluate")
    sp.add_argument("--out", default=str(ROOT / "runs" / "main"))
    sp.add_argument("--device", default="auto")
    sp.set_defaults(fn=cmd_evaluate)
    sp = sub.add_parser("predict")
    sp.add_argument("record", help="WFDB record path without extension, e.g. data/mitdb/100")
    sp.add_argument("--model", default=str(ROOT / "runs" / "main" / "model.pt"))
    sp.add_argument("--output")
    sp.add_argument("--detect", action="store_true", help="ignore .atr annotations and detect R-peaks")
    sp.add_argument("--device", default="auto")
    sp.set_defaults(fn=cmd_predict)

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    args.fn(args)


if __name__ == "__main__":
    main()
