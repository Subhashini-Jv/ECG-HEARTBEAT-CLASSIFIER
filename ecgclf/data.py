"""Download MIT-BIH records, extract R-peak-centred beats and RR-interval features."""

from __future__ import annotations

import logging
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import wfdb
from scipy.signal import butter, sosfiltfilt

from . import config as C

log = logging.getLogger(__name__)

# Every annotation symbol that marks a real heartbeat (used for RR intervals,
# even when the beat itself is not one of the four AAMI classes we classify).
BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")

# Only *ratios* are used: absolute intervals depend on each patient's heart rate
# and let the network memorise training patients instead of learning prematurity.
RR_FEATURE_NAMES = [
    "pre_over_local", "post_over_local", "post_over_pre", "pre_over_record", "local_over_record",
]


# --------------------------------------------------------------------------- #
# Download
# --------------------------------------------------------------------------- #
def download(data_dir: Path, records: list[str] | None = None) -> None:
    """Fetch .hea/.dat/.atr files, trying each mirror in turn. Skips existing files."""
    data_dir.mkdir(parents=True, exist_ok=True)
    records = records or C.ALL_RECORDS
    for rec in records:
        for ext in ("hea", "dat", "atr"):
            dest = data_dir / f"{rec}.{ext}"
            if dest.exists() and dest.stat().st_size > 0:
                continue
            last_err = None
            for mirror in C.MIRRORS:
                try:
                    with urllib.request.urlopen(f"{mirror}/{rec}.{ext}", timeout=60) as r:
                        dest.write_bytes(r.read())
                    break
                except Exception as e:  # noqa: BLE001 - try next mirror
                    last_err = e
            else:
                raise RuntimeError(f"Could not download {rec}.{ext}: {last_err}")
        log.info("record %s ready", rec)


# --------------------------------------------------------------------------- #
# Signal processing
# --------------------------------------------------------------------------- #
def bandpass(sig: np.ndarray, fs: int = C.FS, lo: float = 0.5, hi: float = 40.0) -> np.ndarray:
    """Zero-phase Butterworth band-pass: removes baseline wander and mains/EMG noise."""
    sos = butter(3, [lo, hi], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, sig)


def robust_standardize(sig: np.ndarray) -> np.ndarray:
    """Record-level scaling (median / IQR). Keeps relative beat amplitudes within a
    patient intact, which per-beat z-scoring would destroy."""
    med = np.median(sig)
    q75, q25 = np.percentile(sig, [75, 25])
    scale = (q75 - q25) or 1.0
    return (sig - med) / scale


def pick_lead(record: wfdb.Record) -> np.ndarray:
    """Use the MLII lead when present (it is channel 1 in record 114)."""
    names = [n.upper() for n in record.sig_name]
    idx = names.index("MLII") if "MLII" in names else 0
    return record.p_signal[:, idx].astype(np.float64)


def rr_features(r_peaks: np.ndarray, fs: int = C.FS) -> np.ndarray:
    """RR-interval features for every R-peak except the first and last.

    Returns array of shape (len(r_peaks), 5); rows 0 and -1 are NaN.
    Ratios normalise for each patient's own heart rate, which is what makes
    S beats (premature, but morphologically normal) separable across patients.
    """
    n = len(r_peaks)
    out = np.full((n, len(RR_FEATURE_NAMES)), np.nan, dtype=np.float32)
    if n < 3:
        return out
    rr = np.diff(r_peaks) / fs  # rr[i] = interval between beat i and i+1
    record_rr = float(np.median(rr))
    for i in range(1, n - 1):
        pre, post = rr[i - 1], rr[i]
        lo = max(0, i - 1 - C.LOCAL_RR_BEATS)
        local = float(np.mean(rr[lo:i]))
        out[i] = [pre / local, post / local, post / pre, pre / record_rr, local / record_rr]
    return out


@dataclass
class Beats:
    X: np.ndarray        # (n, WINDOW) float32 waveforms
    rr: np.ndarray       # (n, 5) float32 RR features
    y: np.ndarray        # (n,) int64 class index into C.CLASSES (or -1 if unlabeled)
    record: np.ndarray   # (n,) str record id
    sample: np.ndarray   # (n,) int64 R-peak sample index

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Beats":
        return Beats(self.X[mask], self.rr[mask], self.y[mask], self.record[mask], self.sample[mask])


def beats_from_signal(sig: np.ndarray, r_peaks: np.ndarray, labels: np.ndarray, rec_id: str) -> Beats:
    """Cut fixed windows around each R-peak. `labels` is -1 for beats to skip."""
    sig = robust_standardize(bandpass(sig))
    rr = rr_features(r_peaks)
    keep = (
        (labels >= 0)
        & (r_peaks - C.PRE_SAMPLES >= 0)
        & (r_peaks + C.POST_SAMPLES <= len(sig))
        & ~np.isnan(rr).any(axis=1)
    )
    idx = np.where(keep)[0]
    X = np.stack([sig[r_peaks[i] - C.PRE_SAMPLES: r_peaks[i] + C.POST_SAMPLES] for i in idx]).astype(np.float32) \
        if len(idx) else np.zeros((0, C.WINDOW), np.float32)
    return Beats(X, rr[idx], labels[idx].astype(np.int64),
                 np.array([rec_id] * len(idx)), r_peaks[idx].astype(np.int64))


def load_record(data_dir: Path, rec: str) -> Beats:
    record = wfdb.rdrecord(str(data_dir / rec))
    ann = wfdb.rdann(str(data_dir / rec), "atr")
    sym = np.array(ann.symbol)
    samp = np.array(ann.sample)
    is_beat = np.isin(sym, list(BEAT_SYMBOLS))
    sym, samp = sym[is_beat], samp[is_beat]
    labels = np.array([C.CLASSES.index(C.SYMBOL_TO_AAMI[s]) if s in C.SYMBOL_TO_AAMI else -1 for s in sym])
    return beats_from_signal(pick_lead(record), samp, labels, rec)


def concat(parts: list[Beats]) -> Beats:
    return Beats(*(np.concatenate([getattr(p, f) for p in parts])
                   for f in ("X", "rr", "y", "record", "sample")))


def build_dataset(data_dir: Path, cache: Path, records: list[str] | None = None) -> Beats:
    """Load (or build and cache) beats for every non-paced record."""
    if cache.exists():
        d = np.load(cache, allow_pickle=False)
        return Beats(d["X"], d["rr"], d["y"], d["record"], d["sample"])
    records = records or [r for r in C.ALL_RECORDS if r not in C.PACED_RECORDS]
    download(data_dir, records)
    beats = concat([load_record(data_dir, r) for r in records])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, X=beats.X, rr=beats.rr, y=beats.y, record=beats.record, sample=beats.sample)
    return beats


def split(beats: Beats, mode: str = "inter", seed: int = 0) -> tuple[Beats, Beats, Beats]:
    """Return (train, val, test).

    inter (default): patient-wise. Train = DS1 minus VAL_RECORDS, val = VAL_RECORDS,
        test = DS2. No patient's beats appear in more than one set.
    intra: random beat-level split (70/10/20) over DS1+DS2. Included only to show
        how optimistic the leaky, commonly reported setup is.
    """
    if mode == "inter":
        train_recs = [r for r in C.DS1 if r not in C.VAL_RECORDS]
        return (beats.subset(np.isin(beats.record, train_recs)),
                beats.subset(np.isin(beats.record, C.VAL_RECORDS)),
                beats.subset(np.isin(beats.record, C.DS2)))
    if mode == "intra":
        pool = np.where(np.isin(beats.record, C.DS1 + C.DS2))[0]
        rng = np.random.default_rng(seed)
        rng.shuffle(pool)
        n = len(pool)
        a, b = int(0.7 * n), int(0.8 * n)
        mk = lambda ix: beats.subset(np.isin(np.arange(len(beats)), ix))  # noqa: E731
        return mk(pool[:a]), mk(pool[a:b]), mk(pool[b:])
    raise ValueError(f"unknown split mode {mode!r}")


def two_channel(b: Beats) -> np.ndarray:
    """(n, 2, W): the beat itself, and the beat minus its patient's median beat.

    The median over all of a recording's beats approximates that patient's own
    dominant beat shape, computed without labels, so it is available at
    inference time too. The difference channel tells the network "how unusual
    is this beat *for this person*", which generalises across patients far
    better than absolute morphology.
    """
    out = np.empty((len(b), 2, b.X.shape[1]), dtype=np.float32)
    out[:, 0] = b.X
    for rec in np.unique(b.record):
        m = b.record == rec
        out[m, 1] = b.X[m] - np.median(b.X[m], axis=0)
    return out


def class_counts(y: np.ndarray) -> dict[str, int]:
    return {c: int((y == i).sum()) for i, c in enumerate(C.CLASSES)}
