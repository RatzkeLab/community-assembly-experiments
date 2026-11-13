"""Population-level identity: what sequence does each read group in a well actually have?

Nearest-reference read assignment answers "which strain in the DB is this read closest
to", not "is this the strain". A read sits ~10-15 bp from its own reference by error
alone, so a related organism 8 bp away is assigned to the reference just as happily.
The base-exact answer needs a consensus: reads assigned to strain X in a well are
pooled, spoa builds a consensus over the highest-qscore ones (deep_consensus, the same
recipe that built the reference), and the consensus is compared with X's reference:

    <= CONFIRM_BP   the well holds X's sequence (1-2 bp is het-site wobble: operon
                    copies split ~50/50 flip the majority base between wells)
    larger          the read group is a DIFFERENT organism that X's reference happens
                    to be nearest to -- drift of the source well, or contamination

Reads within reach of no reference ("unexplained") are clustered per well; populations
large enough get a consensus, which is tested for being a two-parent chimera and
compared with the reference. Pooled across wells they become numbered unknown OTUs.
"""
import collections
from multiprocessing import Pool

import edlib
import numpy as np
import pandas as pd

import deep_consensus as dc
from . import io as cio

MIN_CONS_READS = 10       # a read group gets a consensus with at least this many reads
CONFIRM_BP = 2            # consensus-to-reference bp that still counts as "same sequence"
UNK_CLUSTER_THR = 0.04    # greedy seed clustering of unexplained reads (normalized NW)
MIN_UNK_READS = 8         # an unexplained population needs this many reads ...
MIN_UNK_FRAC = 0.01       # ... and this share of the well's reads
OTU_MERGE_BP = 3          # unknown consensuses this close (core bp) are one OTU
SPLIT_STEP = 50           # chimera search: candidate breakpoints every SPLIT_STEP bp

_HAPS = _IUPAC = _ANCHOR = _ALL = None


def _init(haps, iupac, anchor):
    global _HAPS, _IUPAC, _ANCHOR, _ALL
    _HAPS, _IUPAC, _ANCHOR = haps, iupac, anchor
    _ALL = [(s, h) for s in sorted(haps) for h in haps[s]]


def _hw(q, t, k=-1):
    return edlib.align(q, t, mode="HW", task="distance", k=k)["editDistance"]


def nearest_ref(seq, exclude=(), cap=400):
    """Nearest strain to a CONSENSUS: HW search over every haplotype, then exact core bp
    (IUPAC-aware, end-jitter-proof) against that strain. Returns (strain, bp)."""
    core = seq[dc.END_TRIM:len(seq) - dc.END_TRIM]
    best, who = None, None
    for s, h in _ALL:
        if s in exclude:
            continue
        d = _hw(core, h, cap if best is None else best)
        if d >= 0 and (best is None or d < best):
            best, who = d, s
    if who is None:
        return None, np.nan
    return who, strain_bp(seq, who)


def strain_bp(seq, strain):
    """Core bp from a consensus to a strain: min over its IUPAC consensus and haplotypes."""
    cands = list(_HAPS.get(strain, []))
    if strain in _IUPAC:
        cands.append(_IUPAC[strain])
    return min(dc.core_bp(seq, r) for r in cands) if cands else np.nan


def chimera_split(seq):
    """Best two-parent explanation of a consensus: for breakpoints every SPLIT_STEP bp,
    prefix and suffix each go to their nearest haplotype (HW). Returns
    (total_bp, prefix_strain, prefix_bp, suffix_strain, suffix_bp, breakpoint)."""
    core = seq[dc.END_TRIM:len(seq) - dc.END_TRIM]
    best = None

    def nearest_part(part):
        b, w = None, None
        for s, h in _ALL:
            d = _hw(part, h, -1 if b is None else b)
            if d >= 0 and (b is None or d < b):
                b, w = d, s
        return b, w

    for k in range(200, len(core) - 199, SPLIT_STEP):
        pa = nearest_part(core[:k])
        pb = nearest_part(core[k:])
        tot = pa[0] + pb[0]
        if best is None or tot < best[0]:
            best = (tot, pa[1], pa[0], pb[1], pb[0], k)
    return best if best else (np.nan, None, np.nan, None, np.nan, np.nan)


def _job(args):
    """groups: {(call, strain): (n_reads_total, [ridx of the best-qscore reads])};
    unexplained: [ridx] (a subsample is fine); cut: reads an unexplained cluster needs."""
    sid, path, groups, unexplained, cut = args
    reads = cio.read_fastq(path)
    want = set(i for _, v in groups.values() for i in v) | set(unexplained)
    ori = {i: (dc.orient(reads[i][0], _ANCHOR), reads[i][1]) for i in want}
    rows, unk = [], []
    for (call, strain), (n_total, idxs) in groups.items():
        if n_total < MIN_CONS_READS:
            continue
        cons = dc.top_q_consensus([ori[i] for i in idxs])
        bp_self = strain_bp(cons, strain)
        near, near_bp = nearest_ref(cons, exclude={strain}, cap=max(int(bp_self) + 5, 10)
                                    if np.isfinite(bp_self) else 400)
        rows.append(dict(sample_id=sid, call=call, strain=strain, n_reads=n_total,
                         bp_to_ref=bp_self, nearest_other=near, nearest_other_bp=near_bp,
                         consensus=cons))
    if len(unexplained) >= MIN_UNK_READS:
        seqs = [ori[i][0] for i in unexplained]
        comps = dc.greedy_clusters(seqs, UNK_CLUSTER_THR)
        for c in comps:
            if len(c) < cut:
                break
            cons = dc.top_q_consensus([ori[unexplained[j]] for j in c])
            near, near_bp = nearest_ref(cons)
            ch = chimera_split(cons)
            unk.append(dict(sample_id=sid, n_reads=len(c), nearest=near, nearest_bp=near_bp,
                            split_bp=ch[0], parent_a=ch[1], parent_a_bp=ch[2],
                            parent_b=ch[3], parent_b_bp=ch[4], breakpoint=ch[5],
                            consensus=cons))
    return rows, unk


def well_populations(jobs, haps, iupac, anchor, processes=48):
    """jobs: [(sample_id, path, {(call, strain): (n, [ridx])}, [unexplained ridx], cut)]."""
    jobs = sorted(jobs, key=lambda j: -(len(j[2]) * 50 + len(j[3])))
    rows, unk = [], []
    with Pool(processes, initializer=_init, initargs=(haps, iupac, anchor)) as pool:
        for r, u in pool.imap_unordered(_job, jobs, chunksize=1):
            rows.extend(r)
            unk.extend(u)
    # pool order depends on which worker finishes first; fix it so OTU numbering is stable
    rows = pd.DataFrame(rows).sort_values(["sample_id", "call", "strain"]).reset_index(drop=True)
    unk = pd.DataFrame(unk)
    if len(unk):
        unk = unk.sort_values(["sample_id", "n_reads", "consensus"],
                              ascending=[True, False, True]).reset_index(drop=True)
    return rows, unk


def merge_otus(unk):
    """Pool per-well unexplained consensuses into OTUs: greedy, largest first, a
    consensus joins the first OTU whose seed is within OTU_MERGE_BP core bp."""
    if unk.empty:
        return unk.assign(otu=pd.Series(dtype=str)), pd.DataFrame()
    order = unk.sort_values(["n_reads", "sample_id", "consensus"],
                            ascending=[False, True, True], kind="stable").index
    seeds, label = [], {}
    for i in order:
        c = unk.at[i, "consensus"]
        for k, s in enumerate(seeds):
            if abs(len(s) - len(c)) <= 30 and dc.core_bp(c, s) <= OTU_MERGE_BP:
                label[i] = k
                break
        else:
            label[i] = len(seeds)
            seeds.append(c)
    unk = unk.copy()
    unk["otu"] = [f"U{label[i] + 1:03d}" for i in unk.index]
    otus = (unk.groupby("otu")
            .agg(n_wells=("sample_id", "nunique"), n_reads=("n_reads", "sum"),
                 nearest=("nearest", lambda x: collections.Counter(x).most_common(1)[0][0]),
                 nearest_bp=("nearest_bp", "median"), split_bp=("split_bp", "median"))
            .reset_index())
    otus["seed"] = [seeds[int(o[1:]) - 1] for o in otus.otu]
    return unk, otus


# ---- pass 2: count every unexplained read against the OTU consensuses ----------------------

_OTU = None


def _init_otu(otu_seqs, anchor):
    global _OTU, _ANCHOR
    _OTU, _ANCHOR = otu_seqs, anchor


def _otu_job(args):
    sid, path, ridx, cuts = args
    reads = cio.read_fastq(path)
    out = collections.Counter()
    for i, cut in zip(ridx, cuts):
        s = dc.orient(reads[i][0], _ANCHOR)[25:-25]
        best, who = None, None
        for name, seed in _OTU:
            d = _hw(s, seed, cut if best is None else best)
            if d >= 0 and (best is None or d < best):
                best, who = d, name
        out[who or "none"] += 1
    return sid, out


def count_otus(jobs, otus, anchor, processes=48):
    """jobs: [(sample_id, path, [unexplained ridx], [per-read cut])] -> long table of read counts
    per (well, OTU); reads near no OTU count as 'none' (chimeras, singletons)."""
    otu_seqs = list(zip(otus.otu, otus.seed))
    rows = []
    with Pool(processes, initializer=_init_otu, initargs=(otu_seqs, anchor)) as pool:
        for sid, cnt in pool.imap_unordered(_otu_job, jobs, chunksize=4):
            rows += [dict(sample_id=sid, otu=k, reads=v) for k, v in cnt.items()]
    return pd.DataFrame(rows)
