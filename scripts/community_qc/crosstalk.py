"""Demultiplexing cross-talk, measured on wells that were never sequenced.

A read is barcoded at both ends. If one end is misread into another barcode of the set,
the read lands in the well with the misread end's barcode and the OTHER end's correct
one -- so a well receives stray reads in proportion to how many reads the wells SHARING
ONE of its barcodes produced:

    stray(w) = a * F(w) + b * R(w) + c

F(w) = reads in sequenced wells with w's forward barcode (w excluded), R(w) the same for
the reverse barcode, c a flat floor for everything else. Barcode plates that were
never PCR'd have stray reads only, so they fit a, b, c directly; the same rates then give
every sequenced well its expected stray-read count AND their composition (the strains of
the wells it shares barcodes with). A foreign strain seen in a well can then be compared
with what cross-talk alone would put there.

The mechanism is checked, not assumed: `validate_sources` asks whether the reads in
never-sequenced wells actually match the members of their barcode-sharing wells more
often than the members of random sequenced wells.
"""
import numpy as np
import pandas as pd
from scipy import optimize


def sharing_matrices(sheet, sequenced):
    """sheet: sample_id, fwd, rev (all wells). sequenced: boolean Series by sample_id.
    Returns (ids, seq_ids, SF, SR): SF[i, j] = 1 if well i and SEQUENCED well j share the
    forward barcode and i != j; SR likewise for reverse."""
    ids = sheet.sample_id.to_numpy()
    seq_ids = ids[sequenced.reindex(ids).fillna(False).to_numpy(bool)]
    s = sheet.set_index("sample_id")
    f_all, r_all = s.loc[ids, "fwd"].to_numpy(), s.loc[ids, "rev"].to_numpy()
    f_seq, r_seq = s.loc[seq_ids, "fwd"].to_numpy(), s.loc[seq_ids, "rev"].to_numpy()
    SF = (f_all[:, None] == f_seq[None, :]).astype(np.float32)
    SR = (r_all[:, None] == r_seq[None, :]).astype(np.float32)
    self_ = ids[:, None] == seq_ids[None, :]
    SF[self_] = 0
    SR[self_] = 0
    return ids, seq_ids, SF, SR


def fit_rates(y, F, R):
    """Poisson MLE of y ~ a*F + b*R + c with a, b, c >= 0."""
    y, F, R = (np.asarray(v, float) for v in (y, F, R))
    sF, sR = max(F.mean(), 1e-9), max(R.mean(), 1e-9)     # fit on unit scale
    Fs, Rs = F / sF, R / sR

    def nll(p):
        mu = np.maximum(p[0] * Fs + p[1] * Rs + p[2], 1e-9)
        return float((mu - y * np.log(mu)).sum())

    x0 = np.full(3, max(y.mean() / 3, 1e-3))
    res = optimize.minimize(nll, x0, method="L-BFGS-B",
                            bounds=[(0, None), (0, None), (0, None)])
    a, b, c = res.x[0] / sF, res.x[1] / sR, res.x[2]
    mu = a * F + b * R + c
    # deviance-based fit summary + dispersion (Pearson chi2 / df)
    dispersion = float(((y - mu) ** 2 / np.maximum(mu, 1e-9)).sum() / max(len(y) - 3, 1))
    r = np.corrcoef(mu, y)[0, 1] if len(y) > 2 else np.nan
    return dict(a_fwd_share=a, b_rev_share=b, c_floor=c, n_wells=len(y),
                pearson_r=r, dispersion=dispersion, converged=bool(res.success))


def expected_stray(ids, seq_ids, SF, SR, depth, rates):
    """Expected stray reads per well (all wells), from the sequenced wells' depths."""
    d = depth.reindex(seq_ids).fillna(0).to_numpy(float)
    F, R = SF @ d, SR @ d
    mu = rates["a_fwd_share"] * F + rates["b_rev_share"] * R + rates["c_floor"]
    return pd.DataFrame({"sample_id": ids, "F_share_depth": F, "R_share_depth": R,
                         "expected_stray_reads": mu})


def expected_stray_composition(ids, seq_ids, SF, SR, counts, rates):
    """Expected stray reads per (well, strain).

    counts: DataFrame (sequenced sample_id x strain) of reads per strain, ALREADY scaled
    to each well's full read count (not the capped subset). The flat floor c carries no
    composition and is left out (it is spread over every strain in the run)."""
    N = counts.reindex(seq_ids).fillna(0)
    M = rates["a_fwd_share"] * SF + rates["b_rev_share"] * SR
    X = M @ N.to_numpy(float)
    return pd.DataFrame(X, index=ids, columns=N.columns)


def validate_sources(stray_counts, ids, seq_ids, SF, SR, member_matrix, n_perm=200, seed=0):
    """Do reads in never-sequenced wells come from their barcode-sharing wells?

    stray_counts:  DataFrame (sample_id x strain) of reads with a nearest strain, for the
                   wells being tested (never-sequenced ones).
    member_matrix: DataFrame (sequenced sample_id x strain), True where the strain -- or a
                   16S twin of it, which a read cannot tell apart -- is a member.

    Returns (observed share of reads whose strain is a member of a barcode-sharing well,
    permutation null shares). The null shuffles WHICH sequenced wells each tested well
    shares a barcode with, keeping how many.
    """
    strains = member_matrix.columns
    C = stray_counts.reindex(columns=strains, fill_value=0)
    pos = pd.Series(np.arange(len(ids)), index=ids)
    rows = pos.reindex(C.index).to_numpy()
    share = ((SF + SR) > 0)[rows].astype(np.float32)
    Mem = member_matrix.reindex(seq_ids).fillna(False).to_numpy(np.float32)
    Cv = C.to_numpy(float)
    total = Cv.sum()

    def explained(sh):
        return float((Cv * ((sh @ Mem) > 0)).sum() / max(total, 1))

    obs = explained(share)
    rng = np.random.default_rng(seed)
    null = np.array([explained(rng.permuted(share, axis=1)) for _ in range(n_perm)])
    return obs, null
