"""Per-well composition tables, detection calls, and replicate distances with a
sequencing-noise null.

Depth matters for every number here: plates 6 and 9 of 20261026 have ~100 reads per well
against ~2000 on plates 5 and 10. A member at 1% is expected to show 1 read at 100 --
absent by chance 37% of the time -- and two perfect replicates sampled at 100 reads differ
by a Bray-Curtis of ~0.1 from counting alone. So every absence carries the depth it was
called at, and every replicate distance is reported next to the distance that pure
multinomial sampling would produce at the same depths.
"""
import numpy as np
import pandas as pd

PRESENT_FRAC = 0.01       # member present at >= 1% of the well's reads
TRACE_MIN_READS = 1       # any read below that = trace
PSEUDO = 0.5              # pseudocount (reads) for log-ratio work
CALLS = ["member", "foreign", "unexplained", "low_quality"]


def twin_matrix(dist_csv, max_bp):
    """Strain x strain boolean: haplotype-aware 16S distance <= max_bp (diagonal True).
    At <= 4 bp, read cross-talk reaches 24-37% (p90; 20261021 d02), so a read 'from' a
    strain that close to a member is not evidence of anything foreign."""
    d = pd.read_csv(dist_csv, index_col=0)
    return d <= max_bp


def member_table(lay_seq, reads_cls, stats):
    """One row per (sequenced well, expected member), zeros included.

    lay_seq: layout rows of the sequenced plates (members_list column).
    reads_cls: classified reads (assign.classify). stats: q01 per-well read stats.
    """
    used = stats.set_index("sample_id").reads_used
    cnt = (reads_cls[reads_cls.call == "member"]
           .groupby(["sample_id", "strain"], observed=True).size())
    calls = (reads_cls.groupby(["sample_id", "call"], observed=True).size()
             .unstack(fill_value=0).reindex(columns=CALLS, fill_value=0))
    rows = []
    for r in lay_seq.itertuples():
        n = int(used.get(r.sample_id, 0))
        lowq = int(calls.low_quality.get(r.sample_id, 0)) if r.sample_id in calls.index else 0
        nc = n - lowq                                  # reads that got a call
        nm = int(calls.member.get(r.sample_id, 0)) if r.sample_id in calls.index else 0
        for m in r.members_list:
            k = int(cnt.get((r.sample_id, m), 0))
            rows.append(dict(sample_id=r.sample_id, plate=r.plate,
                             culture_plate=r.culture_plate, timepoint=r.timepoint,
                             well=r.well, community_id=r.community_id,
                             replicate=r.replicate, size=r.size, strain=m, reads=k,
                             reads_used=n, reads_called=nc, member_reads=nm,
                             frac_all=k / nc if nc else np.nan,
                             frac_members=k / nm if nm else np.nan))
    t = pd.DataFrame(rows)
    t["status"] = np.select(
        [t.reads_called == 0, t.frac_all >= PRESENT_FRAC, t.reads >= TRACE_MIN_READS],
        ["no_data", "present", "trace"], "absent")
    # rule of three: with n reads and 0 seen, the 95% upper bound on the true share is
    # ~3/n; an absence is only informative about a 1%-member when 3/n < 1%.
    t["detect_limit_95"] = np.where(t.reads_called > 0, 3.0 / t.reads_called.clip(lower=1), np.nan)
    t["absence_informative"] = t.detect_limit_95 < PRESENT_FRAC
    return t


def well_summary(lay_seq, reads_cls, stats):
    calls = (reads_cls.groupby(["sample_id", "call"], observed=True).size()
             .unstack(fill_value=0).reindex(columns=CALLS, fill_value=0))
    s = stats.set_index("sample_id")
    out = lay_seq.set_index("sample_id")[["plate", "culture_plate", "timepoint", "well",
                                          "row", "col", "community_id", "replicate",
                                          "well_type", "size"]].copy()
    out = out.join(s[["reads_in_file", "reads_len_ok", "reads_used"]]).join(calls)
    out[CALLS] = out[CALLS].fillna(0)
    out["reads_called"] = out.reads_used - out.low_quality
    out["low_quality_frac"] = out.low_quality / out.reads_used.where(out.reads_used > 0)
    for c in CALLS[:3]:          # shares of reads that got a call
        out[f"{c}_frac"] = out[c] / out.reads_called.where(out.reads_called > 0)
    return out.reset_index()


# ---- distances ------------------------------------------------------------------------------

def bray_curtis(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    s = p.sum() + q.sum()
    return float(np.abs(p - q).sum() / s) if s > 0 else np.nan


def clr(counts, pseudo=PSEUDO):
    x = np.log(np.asarray(counts, float) + pseudo)
    return x - x.mean(axis=-1, keepdims=True)


def aitchison(c1, c2, pseudo=PSEUDO):
    return float(np.linalg.norm(clr(c1, pseudo) - clr(c2, pseudo)))


def jsd(p, q):
    p = np.asarray(p, float); q = np.asarray(q, float)
    p, q = p / p.sum(), q / q.sum()
    m = 0.5 * (p + q)
    def kl(a, b):
        nz = a > 0
        return float((a[nz] * np.log2(a[nz] / b[nz])).sum())
    return float(np.sqrt(0.5 * kl(p, m) + 0.5 * kl(q, m)))


def null_bray_curtis(p, n1, n2, rng, n_sim=200):
    """Bray-Curtis between two multinomial samples of the SAME composition p at depths
    n1, n2 (compositions, so the depths themselves do not enter the distance)."""
    p = np.asarray(p, float)
    p = p / p.sum()
    a = rng.multinomial(int(n1), p, size=n_sim) / max(n1, 1)
    b = rng.multinomial(int(n2), p, size=n_sim) / max(n2, 1)
    return float(np.mean(np.abs(a - b).sum(axis=1) / 2))


def rarefy(counts, n, rng):
    """Subsample a count vector to n reads without replacement."""
    counts = np.asarray(counts, int)
    if counts.sum() <= n:
        return counts.copy()
    pool = np.repeat(np.arange(len(counts)), counts)
    pick = rng.choice(pool, n, replace=False)
    return np.bincount(pick, minlength=len(counts))


def replicate_pairs(mt, min_member_reads=30, seed=0, n_sim=200):
    """All pairs of wells holding the same community at the same time point, within a
    barcode plate ('within_plate') and across the two passage plates ('between_plates').

    mt: member_table. Distances use member reads only (foreign / unexplained reads are a
    QC question, not a composition one). Each pair carries the Bray-Curtis two perfect
    replicates would show from sampling alone at those depths, using the pair's pooled
    composition as the truth.
    """
    rng = np.random.default_rng(seed)
    ok = mt[mt.member_reads >= min_member_reads]
    wide = {sid: g.set_index("strain").reads for sid, g in ok.groupby("sample_id")}
    meta = ok.drop_duplicates("sample_id").set_index("sample_id")
    rows = []
    for (cid, tp), g in meta.groupby(["community_id", "timepoint"]):
        ids = list(g.index)
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                ca, cb = wide[a], wide[b].reindex(wide[a].index).fillna(0)
                na, nb = ca.sum(), cb.sum()
                pa, pb = ca / na, cb / nb
                pooled = (pa + pb) / 2
                kind = ("within_plate" if meta.at[a, "plate"] == meta.at[b, "plate"]
                        else "between_plates")
                rows.append(dict(community_id=cid, timepoint=tp, kind=kind,
                                 a=a, b=b, plate_a=meta.at[a, "plate"],
                                 plate_b=meta.at[b, "plate"], size=meta.at[a, "size"],
                                 reads_a=int(na), reads_b=int(nb),
                                 bray_curtis=bray_curtis(pa, pb),
                                 aitchison=aitchison(ca.to_numpy(), cb.to_numpy()),
                                 jsd=jsd(pa, pb),
                                 dominant_same=bool(pa.idxmax() == pb.idxmax()),
                                 null_bray_curtis=null_bray_curtis(pooled, na, nb, rng, n_sim)))
    out = pd.DataFrame(rows)
    if len(out):
        out["excess_bray_curtis"] = out.bray_curtis - out.null_bray_curtis
    return out
