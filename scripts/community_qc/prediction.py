"""Does the 24 h state of a passage well predict the same well later on?

Each passage well is sampled at 24 h and again after daily dilution (5 d), so every
member has an abundance at both times in the SAME culture. Models work in within-well
log-ratio space (abundances only mean anything relative to the rest of the well):

    x_iw = log(reads at 24h + pseudo), centred within well w   (CLR)
    y_iw = same at the later time point

    persistence     y = x                       nothing changes
    strain          y = beta_i                  only WHO is in the well matters (beta =
                                                strain i's typical later log-abundance,
                                                learned on other communities)
    24h + strain    y = gamma * x + beta_i      24 h abundance, discounted, plus strain
    sister ceiling  y = mean of the other replicate wells' observed later state -- not a
                    model from 24 h, the reproducibility the later state allows at all
    sister 24h      the 24h + strain model fed the SISTER replicates' mean 24 h state
                    instead of the well's own: if own-24h beats sister-24h, a well's
                    own early deviations carry forward (history matters); if equal, the
                    later state is set by the community, not by where the well was at 24 h

beta and gamma are fitted with every replicate of a community held out together
(leave-community-out folds), so no model sees the community it is scored on.
"""
import numpy as np
import pandas as pd
from scipy import stats

from .composition import PSEUDO, PRESENT_FRAC, bray_curtis


def paired_members(mt, t0="24h", t1="5d"):
    """One row per (culture plate, well, member) with reads at both time points."""
    keys = ["culture_plate", "well", "community_id", "replicate", "size", "strain"]
    a = mt[mt.timepoint == t0][keys + ["sample_id", "reads", "reads_used", "member_reads"]]
    b = mt[mt.timepoint == t1][keys + ["sample_id", "reads", "reads_used", "member_reads"]]
    return a.merge(b, on=keys, suffixes=("_0", "_1"))


def _clr_within(df, col, pseudo=PSEUDO):
    v = np.log(df[col].to_numpy(float) + pseudo)
    g = df.groupby("wid")[col].transform("size").to_numpy()
    m = pd.Series(v).groupby(df.wid.to_numpy()).transform("mean").to_numpy()
    return v - m, g


def _design(df, strains):
    """Within-well centred strain indicator matrix."""
    S = len(strains)
    pos = {s: i for i, s in enumerate(strains)}
    D = np.zeros((len(df), S))
    D[np.arange(len(df)), df.strain.map(pos).to_numpy()] = 1.0
    Dm = pd.DataFrame(D).groupby(df.wid.to_numpy()).transform("mean").to_numpy()
    return D - Dm


def _fit(x, y, Dc, use_x=True, use_strain=True, ridge=1.0):
    cols = []
    if use_x:
        cols.append(x[:, None])
    if use_strain:
        cols.append(Dc)
    X = np.hstack(cols)
    pen = np.full(X.shape[1], ridge)
    if use_x:
        pen[0] = 1e-6                      # do not shrink gamma
    A = X.T @ X + np.diag(pen)
    coef = np.linalg.solve(A, X.T @ y)
    gamma = coef[0] if use_x else 0.0
    beta = coef[1:] if (use_x and use_strain) else (coef if use_strain else np.zeros(Dc.shape[1]))
    return gamma, beta


def _softmax_within(score, wid):
    s = pd.Series(score)
    e = np.exp(s - s.groupby(wid).transform("max"))
    return (e / e.groupby(wid).transform("sum")).to_numpy()


def cross_validate(pm, n_folds=10, ridge=1.0, seed=0, min_reads_0=30, min_reads_1=30):
    """Leave-community-out predictions for every model. Returns (per-member predictions,
    fitted gamma per fold)."""
    d = pm[(pm.member_reads_0 >= min_reads_0) & (pm.member_reads_1 >= min_reads_1)].copy()
    d["wid"] = d.culture_plate.astype(str) + "_" + d.well
    d = d.sort_values(["wid", "strain"]).reset_index(drop=True)
    d["x"], _ = _clr_within(d, "reads_0")
    d["y"], _ = _clr_within(d, "reads_1")
    d["p0"] = d.reads_0 / d.groupby("wid").reads_0.transform("sum")
    d["p1"] = d.reads_1 / d.groupby("wid").reads_1.transform("sum")

    # sister replicates: same community on the same culture plate, other wells
    grp = d.groupby(["culture_plate", "community_id", "strain"])
    for col in ("p0", "p1"):
        tot = grp[col].transform("sum")
        n = grp[col].transform("size")
        d[f"sis_{col}"] = np.where(n > 1, (tot - d[col]) / (n - 1).clip(lower=1), np.nan)
    d["n_sisters"] = grp.p1.transform("size") - 1
    xs = np.log(d.sis_p0.fillna(0) * d.groupby("wid").reads_0.transform("sum") + PSEUDO)
    d["x_sis"] = xs - xs.groupby(d.wid).transform("mean")

    strains = sorted(d.strain.unique())
    Dc = _design(d, strains)
    comms = np.array(sorted(d.community_id.unique()))
    rng = np.random.default_rng(seed)
    fold_of = dict(zip(comms, rng.permutation(len(comms)) % n_folds))
    d["fold"] = d.community_id.map(fold_of)

    for name in ("strain", "x_strain", "sister24_strain"):
        d[f"score_{name}"] = np.nan
    gammas = []
    for f in range(n_folds):
        tr, te = (d.fold != f).to_numpy(), (d.fold == f).to_numpy()
        if not te.any():
            continue
        x, y = d.x.to_numpy(), d.y.to_numpy()
        _, b_s = _fit(x[tr], y[tr], Dc[tr], use_x=False, ridge=ridge)
        g, b_xs = _fit(x[tr], y[tr], Dc[tr], use_x=True, ridge=ridge)
        gammas.append(dict(fold=f, gamma=g, n_train_members=int(tr.sum())))
        d.loc[te, "score_strain"] = Dc[te] @ b_s
        d.loc[te, "score_x_strain"] = g * x[te] + Dc[te] @ b_xs
        d.loc[te, "score_sister24_strain"] = g * d.x_sis.to_numpy()[te] + Dc[te] @ b_xs

    wid = d.wid.to_numpy()
    d["pred_persistence"] = d.p0
    d["pred_strain"] = _softmax_within(d.score_strain.to_numpy(), wid)
    d["pred_x_strain"] = _softmax_within(d.score_x_strain.to_numpy(), wid)
    d["pred_sister24_strain"] = np.where(d.n_sisters > 0,
                                         _softmax_within(d.score_sister24_strain.to_numpy(), wid),
                                         np.nan)
    sp = d.sis_p1 / d.groupby("wid").sis_p1.transform("sum")
    d["pred_sister_ceiling"] = sp
    return d, pd.DataFrame(gammas)


MODELS = ["persistence", "strain", "x_strain", "sister24_strain", "sister_ceiling"]


def score_wells(d, models=MODELS, seed=0, n_sim=200):
    """Per held-out well: Bray-Curtis to the observed later state, Spearman across
    members, dominant member right, plus the Bray-Curtis a PERFECT prediction would
    still show from sampling the later state at its depth (noise floor)."""
    rng = np.random.default_rng(seed)
    rows = []
    for wid, g in d.groupby("wid"):
        p1 = g.p1.to_numpy()
        n1 = int(g.reads_1.sum())
        sim = rng.multinomial(n1, p1, size=n_sim) / max(n1, 1)
        floor = float(np.mean(np.abs(sim - p1).sum(axis=1) / 2))
        base = dict(wid=wid, culture_plate=g.culture_plate.iat[0],
                    community_id=g.community_id.iat[0], size=g["size"].iat[0],
                    reads_0=int(g.reads_0.sum()), reads_1=n1, noise_floor_bc=floor)
        for m in models:
            p = g[f"pred_{m}"].to_numpy()
            if np.isnan(p).any():
                continue
            rho = stats.spearmanr(p, p1).statistic if len(g) >= 3 else np.nan
            rows.append(dict(base, model=m, bray_curtis=bray_curtis(p, p1),
                             spearman=rho,
                             dominant_correct=bool(np.argmax(p) == np.argmax(p1))))
    return pd.DataFrame(rows)


def survival_auc(d, models=MODELS, present=PRESENT_FRAC):
    """AUC of each model's predicted abundance for 'member still >= 1% later'."""
    from sklearn.metrics import roc_auc_score
    out = []
    lab = (d.p1 >= present).to_numpy()
    for m in models:
        p = d[f"pred_{m}"].to_numpy()
        ok = ~np.isnan(p)
        if ok.sum() > 10 and 0 < lab[ok].mean() < 1:
            out.append(dict(model=m, auc=roc_auc_score(lab[ok], p[ok]), n=int(ok.sum()),
                            share_surviving=float(lab[ok].mean())))
    return pd.DataFrame(out)
