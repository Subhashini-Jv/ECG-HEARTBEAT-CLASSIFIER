"""Fast unit tests that need no download (synthetic signals only)."""

import numpy as np
import torch

from ecgclf import config as C
from ecgclf.data import Beats, beats_from_signal, rr_features, split
from ecgclf.metrics import report
from ecgclf.model import BeatNet


def test_splits_are_patient_disjoint():
    assert not set(C.DS1) & set(C.DS2)
    assert set(C.VAL_RECORDS) <= set(C.DS1)
    assert not set(C.PACED_RECORDS) & (set(C.DS1) | set(C.DS2))
    assert len(C.DS1) == len(C.DS2) == 22


def test_rr_features_flag_premature_beat():
    peaks = np.arange(0, 20 * 360, 360)            # 60 bpm
    peaks[10] -= 120                                # beat 10 arrives 0.33 s early
    f = rr_features(peaks)
    assert np.isnan(f[0]).all() and np.isnan(f[-1]).all()
    pre_over_local = f[:, 0]
    assert pre_over_local[10] < 0.75                # premature
    assert abs(pre_over_local[5] - 1.0) < 1e-6      # regular


def test_beats_from_signal_shapes_and_bounds():
    fs = C.FS
    sig = np.random.default_rng(0).normal(size=30 * fs)
    peaks = np.arange(50, 30 * fs, fs)              # first peak too close to the edge
    labels = np.zeros(len(peaks), dtype=np.int64)
    b = beats_from_signal(sig, peaks, labels, "x")
    assert b.X.shape[1] == C.WINDOW
    assert b.sample.min() - C.PRE_SAMPLES >= 0
    assert len(b) == len(peaks) - 2                 # first (edge + no pre-RR) and last (no post-RR) dropped


def test_inter_split_has_no_record_overlap():
    recs = np.array(C.DS1 + C.DS2)
    n = len(recs)
    b = Beats(np.zeros((n, C.WINDOW), np.float32), np.ones((n, 5), np.float32),
              np.zeros(n, np.int64), recs, np.arange(n))
    tr, va, te = split(b, "inter")
    sets = [set(tr.record), set(va.record), set(te.record)]
    assert not (sets[0] & sets[1]) and not (sets[0] & sets[2]) and not (sets[1] & sets[2])
    assert sets[2] == set(C.DS2)


def test_model_forward_and_metrics():
    m = BeatNet().eval()
    out = m(torch.randn(8, 2, C.WINDOW), torch.rand(8, 5))
    assert out.shape == (8, len(C.CLASSES))
    y = np.array([0, 0, 1, 2, 3, 3])
    p = np.array([0, 1, 1, 2, 3, 0])
    r = report(y, p)
    assert r["per_class"]["N"]["recall"] == 0.5
    assert r["per_class"]["F"]["recall"] == 0.5
    assert abs(r["macro_recall"] - 0.75) < 1e-9
