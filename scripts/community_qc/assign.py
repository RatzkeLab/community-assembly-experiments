"""Per-read nearest-strain assignment, expected members first.

For every read the step records two facts and nothing else:

    d_member, member   distance to the nearest haplotype of an EXPECTED member
    d_other,  other    distance to the nearest haplotype of any OTHER strain in the DB,
                       searched only up to d_member (a non-member that is not at least as
                       close cannot change the call, so it is never aligned in full)

Every downstream call (member / foreign / unexplained, at whatever threshold) is derived
from those columns, so thresholds can change without re-aligning.

Why members first, and why ties go to the member: 254 of 378 reference strains share a
byte-identical 16S with another strain (20261021/README). A read from member M19 is
exactly as close to M15, its 16S twin; an open nearest-reference search would split
M19's reads between the two by tie order and report half of them as contamination.
Within a community the design keeps members >= 30 bp apart, so a tie is never between
two members; a tie with a non-member is the twin case and belongs to the member. The
cost is that contamination BY a member's 16S twin is invisible -- 16S cannot see it.

Distances are edit distances of the read's interior (READ_TRIM bp dropped at both ends)
aligned as an infix of the haplotype (edlib HW), as in 20261021's resolution
calibration: read-end trimming and reference length cannot bias the race.
"""
from multiprocessing import Pool

import edlib
import numpy as np
import pandas as pd

from . import io as cio

READ_TRIM = 25
LEN_LO, LEN_HI = 1300, 1600     # full-length amplicon after minibar trimming
D_CAP = 300                     # beyond this a read matches nothing; stored as NaN
READ_CAP = 3000                 # reads per well (random, seeded); plates 5/10 median ~2000

_HAPS = None          # strain -> [seq], set in each worker by _init
_STRAINS = None       # sorted strain names
_ANCHOR = None


def _init(haps, anchor):
    global _HAPS, _STRAINS, _ANCHOR
    _HAPS, _STRAINS, _ANCHOR = haps, sorted(haps), anchor


def _ed(q, t, k):
    return edlib.align(q, t, mode="HW", task="distance", k=k)["editDistance"]


_RC = str.maketrans("ACGT", "TGCA")


def _orient(seq):
    """Reads come unflipped from the pipeline; this is a guard, not a fix. Any bacterial
    16S aligns to any other at >=75% identity forward and ~50% reverse-complemented
    (deep_consensus.orient), so one anchor decides orientation for every read."""
    f = edlib.align(seq, _ANCHOR, mode="NW", task="distance")["editDistance"]
    rc = seq[::-1].translate(_RC)
    r = edlib.align(rc, _ANCHOR, mode="NW", task="distance", k=f)["editDistance"]
    return (rc, True) if 0 <= r < f else (seq, False)


def nearest(core, strains, k_cap):
    """(distance, strain, n_strains_tied) over `strains`, bounded by k_cap (inclusive).
    Returns (nan, None, 0) when nothing lies within k_cap."""
    best, who, ties = None, [], 0
    for s in strains:
        for h in _HAPS[s]:
            k = k_cap if best is None else min(best, k_cap)
            d = _ed(core, h, k)
            if d < 0:
                continue
            if best is None or d < best:
                best, who = d, [s]
            elif d == best and s not in who:
                who.append(s)
    if best is None:
        return np.nan, None, 0
    return float(best), who[0], len(who)


def _job(args):
    sample_id, path, members, seed, cap = args
    reads = cio.read_fastq(path)
    n_file = len(reads)
    idx = [i for i, (s, _) in enumerate(reads) if LEN_LO <= len(s) <= LEN_HI]
    n_len = len(idx)
    if len(idx) > cap:
        rng = np.random.default_rng(seed)
        idx = sorted(rng.choice(idx, cap, replace=False).tolist())
    members = [m for m in members if m in _HAPS]
    others = [s for s in _STRAINS if s not in set(members)]
    rows, n_flip = [], 0
    for i in idx:
        seq, q = reads[i]
        seq, flipped = _orient(seq)
        n_flip += flipped
        core = seq[READ_TRIM:-READ_TRIM]
        dm, m, nm = nearest(core, members, D_CAP) if members else (np.nan, None, 0)
        kcap = D_CAP if np.isnan(dm) else int(dm)
        do, o, no = nearest(core, others, kcap)
        rows.append((sample_id, i, len(seq), q, dm, m, nm, do, o, no))
    stats = dict(sample_id=sample_id, reads_in_file=n_file, reads_len_ok=n_len,
                 reads_used=len(idx), reads_flipped=n_flip)
    return rows, stats


COLS = ["sample_id", "ridx", "length", "q", "d_member", "member", "n_member_tied",
        "d_other", "other", "n_other_tied"]


def assign_wells(wells, reads_dir, haps, anchor, processes=48, cap=READ_CAP, seed=0):
    """wells: iterable of (sample_id, members tuple). Returns (reads_df, well_stats_df).

    Wells without a fastq are reported in the stats with zero reads.
    """
    jobs, stats = [], []
    for k, (sid, members) in enumerate(wells):
        p = cio.fastq_path(reads_dir, sid)
        if p is None:
            stats.append(dict(sample_id=sid, reads_in_file=0, reads_len_ok=0,
                              reads_used=0, reads_flipped=0))
            continue
        jobs.append((sid, str(p), tuple(members), seed + k, cap))
    # biggest files first so the pool does not end on one long job
    jobs.sort(key=lambda j: -__import__("os").path.getsize(j[1]))
    rows = []
    with Pool(processes, initializer=_init, initargs=(haps, anchor)) as pool:
        for r, st in pool.imap_unordered(_job, jobs, chunksize=1):
            rows.extend(r)
            stats.append(st)
    df = pd.DataFrame(rows, columns=COLS)
    for c in ("sample_id", "member", "other"):
        df[c] = df[c].astype("category")
    df = df.astype({"ridx": "int32", "length": "int16", "q": "float32",
                    "d_member": "float32", "d_other": "float32",
                    "n_member_tied": "int8", "n_other_tied": "int16"})
    return df, pd.DataFrame(stats)


# ---- calls derived from the facts above ---------------------------------------------------
#
# How far may a read sit from a reference and still be that strain? It depends on the
# read. On the deep plates, reads of each well's dominant member sit a median distance
# that tracks the read's expected error count E = length * 10^(-q/10) almost exactly
# (q 16-18: median 28 bp, E 29; q 22-24: median 10, E 7), and beyond that there is a
# quality-INDEPENDENT tail (~10-12% of q > 20 reads) of chimeras and sequence the DB
# does not hold. No single bp cut separates the two -- a pooled histogram has no valley,
# and the q < 14 reads (E ~ 70-110 bp) reach into the chimera tail on error alone. So:
#
#     a read is "near" a reference if  d <= NEAR_BASE + NEAR_MULT * E
#     reads with q < Q_MIN get no call at all ('low_quality')
#
# Calibrated in 02_qc/q02 (share of dominant-member reads beyond the cut by q bin).
NEAR_BASE = 12
NEAR_MULT = 2.0
Q_MIN = 16


def near_cut(length, q, base=NEAR_BASE, mult=NEAR_MULT):
    return base + mult * np.asarray(length, float) * 10 ** (-np.asarray(q, float) / 10)


def classify(reads, base=NEAR_BASE, mult=NEAR_MULT, q_min=Q_MIN):
    """Add `call`, `strain`, `d_best`, `near_cut`:

        member       nearest strain is an expected member (ties included), within the cut
        foreign      a NON-member is strictly closer, within the cut
        unexplained  nothing in the DB within the cut -- chimera, or an organism the DB
                     does not hold (contaminant, or a member whose well has drifted)
        low_quality  q < q_min: too error-rich to tell the above apart
    """
    dm = reads.d_member.to_numpy(dtype=float)
    do = reads.d_other.to_numpy(dtype=float)
    dm_ = np.where(np.isnan(dm), np.inf, dm)
    do_ = np.where(np.isnan(do), np.inf, do)
    cut = near_cut(reads.length, reads.q, base, mult)
    lowq = reads.q.to_numpy(float) < q_min
    is_member = ~lowq & (dm_ <= do_) & (dm_ <= cut)
    is_foreign = ~lowq & (do_ < dm_) & (do_ <= cut)
    call = np.where(lowq, "low_quality",
                    np.where(is_member, "member",
                             np.where(is_foreign, "foreign", "unexplained")))
    strain = np.where(is_member, reads.member.astype(object),
                      np.where(is_foreign, reads.other.astype(object), None))
    out = reads.copy()
    out["d_best"] = np.minimum(dm_, do_)
    out["near_cut"] = cut
    out["call"] = pd.Categorical(call, categories=CALLS)
    out["strain"] = strain
    return out


CALLS = ["member", "foreign", "unexplained", "low_quality"]
