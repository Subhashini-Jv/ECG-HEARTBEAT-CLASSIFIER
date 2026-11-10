"""Central constants: AAMI class mapping, record splits, signal parameters."""

from __future__ import annotations

FS = 360  # MIT-BIH sampling rate (Hz)

# Window around each R-peak (samples at 360 Hz). 96 + 160 = 256 samples ~= 0.71 s.
PRE_SAMPLES = 96
POST_SAMPLES = 160
WINDOW = PRE_SAMPLES + POST_SAMPLES

# Number of preceding beats used for the "local" average RR interval.
LOCAL_RR_BEATS = 10

# AAMI EC57 heartbeat classes. Q (unclassifiable / paced fusion) is excluded,
# as is standard practice, because it has almost no examples after removing
# the paced records.
CLASSES = ["N", "S", "V", "F"]
CLASS_NAMES = {
    "N": "Normal / bundle-branch block",
    "S": "Supraventricular ectopic",
    "V": "Ventricular ectopic",
    "F": "Fusion (V + N)",
}

# MIT-BIH beat symbol -> AAMI class.
SYMBOL_TO_AAMI = {
    # N: normal, LBBB, RBBB, atrial escape, nodal escape
    "N": "N", "L": "N", "R": "N", "e": "N", "j": "N",
    # S: atrial premature, aberrated APC, nodal premature, supraventricular premature
    "A": "S", "a": "S", "J": "S", "S": "S",
    # V: premature ventricular contraction, ventricular escape
    "V": "V", "E": "V",
    # F: fusion of ventricular and normal
    "F": "F",
}

# All 48 records. The four paced records are dropped per AAMI recommendations.
ALL_RECORDS = [
    "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115", "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201", "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214", "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231", "232", "233", "234",
]
PACED_RECORDS = ["102", "104", "107", "217"]

# Inter-patient split of de Chazal et al. (2004): no patient appears in both sets.
DS1 = [
    "101", "106", "108", "109", "112", "114", "115", "116", "118", "119", "122",
    "124", "201", "203", "205", "207", "208", "209", "215", "220", "223", "230",
]
DS2 = [
    "100", "103", "105", "111", "113", "117", "121", "123", "200", "202", "210",
    "212", "213", "214", "219", "221", "222", "228", "231", "232", "233", "234",
]

# Records held out from DS1 for early stopping / model selection. Chosen so
# every class has examples in validation (see README for counts), and still
# patient-disjoint from both the training records and DS2.
VAL_RECORDS = ["106", "118", "207", "223"]

# Mirrors to try for downloading the raw records (first that works wins).
MIRRORS = [
    "https://physionet.org/files/mitdb/1.0.0",
    "https://physionet-open.s3.amazonaws.com/mitdb/1.0.0",
]
