"""Step 4: the tables the notebooks read.

  q04_member_table.csv       one row per (sequenced well, expected member): reads, share,
                             present / trace / absent, and whether an absence is
                             informative at that depth
  q04_well_summary.csv       one row per sequenced well: depth, member / foreign /
                             unexplained shares (unexplained split into chimera / unknown
                             OTU / unclustered), expected cross-talk
  q04_strain_detection.csv   per strain and barcode plate: in how many of its wells it
                             was seen, and whether the reads are its sequence
  q04_foreign.csv            one row per (well, foreign entity): a known non-member strain
                             or a non-chimeric unknown OTU, with reads, share, expected
                             cross-talk reads and the excess over it
  q04_foreign_association.csv for every foreign entity seen in >= 3 wells: is it tied to
                             one expected member (that member's source well holds it ->
                             drift / source contamination) or not (spread / cross-well)?
  q04_neighbour_test.csv     do foreign strains at 5 d come from the neighbouring wells?
"""
import json

import numpy as np
import pandas as pd
from scipy import stats as sst

import config
from community_qc import assign, composition as comp, io as cio

FOREIGN_MIN_READS = 3
FOREIGN_MIN_FRAC = 0.01     # a foreign entity "is in" a well at >= 1% and >= 3 reads
EXCESS_FOLD = 3             # ... and >= 3x what demux cross-talk explains


def neighbours(row_i, col_i):
    return [(row_i + dr, col_i + dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1)
            if (dr or dc)]


def main():
    th = json.load(open(config.OUT / "q02_thresholds.json"))
    twin_bp = th["TWIN_BP"]
    ceiling = th["stray_strain_ceiling"]["p99"]
    lay = cio.load_layout(config.LAYOUT_CSV)
    lay_seq = lay[lay.plate.isin(config.SEQUENCED_PLATES)].copy()
    reads = pd.read_parquet(config.OUT / "q01_read_assignments.parquet")
    reads = reads[reads.sample_id.isin(set(lay_seq.sample_id))]
    cls = assign.classify(reads)
    cls["sample_id"] = cls.sample_id.astype(str)
    stats = pd.read_csv(config.OUT / "q01_well_read_stats.csv")
    stats = stats[stats.sample_id.isin(set(lay_seq.sample_id))]
    grp = pd.read_csv(config.OUT / "q03_group_consensus.csv")
    otus = pd.read_csv(config.OUT / "q03_otus.csv")
    otu_counts = pd.read_csv(config.OUT / "q03_otu_counts.csv")
    xt_wells = pd.read_csv(config.OUT / "q02_crosstalk_wells.csv").set_index("sample_id")
    xt_long = pd.read_csv(config.OUT / "q02_expected_stray_long.csv")
    dist = pd.read_csv(config.DISTANCE_MATRIX_CSV, index_col=0)

    # ---- member table ---------------------------------------------------------------------
    mt = comp.member_table(lay_seq[lay_seq.well_type == "community"], cls, stats)
    g = grp[grp.call == "member"][["sample_id", "strain", "bp_to_ref", "nearest_other",
                                    "nearest_other_bp"]]
    mt = mt.merge(g, on=["sample_id", "strain"], how="left")
    mt.to_csv(config.OUT / "q04_member_table.csv", index=False)

    # ---- well summary ---------------------------------------------------------------------
    ws = comp.well_summary(lay_seq, cls, stats)
    chim = set(otus.otu[otus.chimera])
    oc = otu_counts.assign(kind=np.where(otu_counts.otu == "none", "unclustered",
                                         np.where(otu_counts.otu.isin(chim), "chimera",
                                                  "unknown_otu")))
    oc = oc.pivot_table(index="sample_id", columns="kind", values="reads", aggfunc="sum",
                        fill_value=0)
    for k in ("chimera", "unknown_otu", "unclustered"):
        ws[f"unexplained_{k}"] = ws.sample_id.map(oc[k] if k in oc else {}).fillna(0)
        ws[f"unexplained_{k}_frac"] = ws[f"unexplained_{k}"] / ws.reads_called.where(ws.reads_called > 0)
    ws["expected_stray_frac"] = ws.sample_id.map(xt_wells.expected_stray_frac)
    ws.to_csv(config.OUT / "q04_well_summary.csv", index=False)

    # ---- foreign entities per well --------------------------------------------------------
    members_of = lay_seq.set_index("sample_id").members_list.to_dict()
    used = stats.set_index("sample_id").reads_used
    f_strain = (cls[cls.call == "foreign"].groupby(["sample_id", "strain"], observed=True)
                .size().rename("reads").reset_index().rename(columns={"strain": "entity"}))
    f_strain["kind"] = "strain"
    f_otu = otu_counts[~otu_counts.otu.isin(chim | {"none"})].rename(columns={"otu": "entity"})
    f_otu["kind"] = "unknown_otu"
    fo = pd.concat([f_strain, f_otu], ignore_index=True)
    called_n = ws.set_index("sample_id").reads_called
    fo["reads_called"] = fo.sample_id.map(called_n)
    fo["frac"] = fo.reads / fo.reads_called
    exp = xt_long.set_index(["sample_id", "strain"]).expected_stray_reads
    # q02's expectation is in length-filtered reads of the receiving well; express it in
    # the reads that got a call, the denominator used here
    scale = called_n / stats.set_index("sample_id").reads_len_ok.where(lambda s: s > 0)
    fo["expected_stray_reads"] = [
        exp.get((s, e), 0.0) * scale.get(s, 1.0) if k == "strain" else 0.0
        for s, e, k in zip(fo.sample_id, fo.entity, fo.kind)]

    def nearest_member_bp(sid, ent, kind):
        ms = [m for m in members_of.get(sid, ()) if m in dist.index]
        if kind != "strain" or ent not in dist.index or not ms:
            return np.nan
        return float(dist.loc[ent, ms].min())
    fo["nearest_member_bp"] = [nearest_member_bp(s, e, k)
                               for s, e, k in zip(fo.sample_id, fo.entity, fo.kind)]
    fo["twin_of_member"] = fo.nearest_member_bp <= twin_bp
    fo["excess_reads"] = fo.reads - fo.expected_stray_reads
    fo["above_stray_ceiling"] = fo.reads > ceiling
    fo["called"] = ((fo.reads >= FOREIGN_MIN_READS) & (fo.frac >= FOREIGN_MIN_FRAC)
                    & (fo.reads >= EXCESS_FOLD * fo.expected_stray_reads)
                    & ~fo.twin_of_member)
    meta = lay_seq.set_index("sample_id")[["plate", "culture_plate", "timepoint", "well",
                                           "row_i", "col_i", "community_id", "well_type"]]
    fo = fo.join(meta, on="sample_id")
    fo.to_csv(config.OUT / "q04_foreign.csv", index=False)

    # ---- strain detection ---------------------------------------------------------------
    sd = (mt.groupby(["strain", "plate", "timepoint"])
          .agg(wells=("sample_id", "size"),
               present=("status", lambda s: (s == "present").sum()),
               trace=("status", lambda s: (s == "trace").sum()),
               absent=("status", lambda s: (s == "absent").sum()),
               absent_informative=("absence_informative",
                                   lambda s: int((s & (mt.loc[s.index, "status"] == "absent")).sum())),
               median_frac=("frac_all", "median"),
               consensus_wells=("bp_to_ref", lambda s: s.notna().sum()),
               consensus_confirmed=("bp_to_ref", lambda s: (s <= 2).sum()),
               consensus_median_bp=("bp_to_ref", "median"))
          .reset_index())
    pool = pd.read_csv(config.POOL_CSV).set_index("strain")
    sd = sd.join(pool[["tier", "status", "source_od"]].rename(columns={"status": "ref_status"}),
                 on="strain")
    sd.to_csv(config.OUT / "q04_strain_detection.csv", index=False)

    # ---- association: is a foreign entity tied to one expected member? ---------------------
    comm = lay_seq[lay_seq.well_type == "community"]
    rows = []
    called = fo[fo.called]
    for (ent, kind), g in called.groupby(["entity", "kind"]):
        wells_with = set(g.sample_id)
        if len(wells_with) < 3:
            continue
        for tp_plates, label in (((5, 6), "24h"), ((9, 10), "5d")):
            pool_w = comm[comm.plate.isin(tp_plates)]
            has = pool_w.sample_id.isin(wells_with).to_numpy()
            if has.sum() == 0:
                continue
            best = None
            for strain in set(m for ms in pool_w.members_list for m in ms):
                memb = pool_w.members_list.apply(lambda ms: strain in ms).to_numpy()
                a = int((has & memb).sum()); b = int((has & ~memb).sum())
                c = int((~has & memb).sum()); d = int((~has & ~memb).sum())
                if a == 0:
                    continue
                orr, p = sst.fisher_exact([[a, b], [c, d]], alternative="greater")
                cand = dict(entity=ent, kind=kind, timepoint=label, wells=int(has.sum()),
                            best_member=strain, with_member=a, without_member=b,
                            member_wells=a + c, coverage=a / (a + c), specificity=a / (a + b),
                            odds_ratio=orr, p=p)
                if best is None or p < best["p"]:
                    best = cand
            if best:
                nm = len(set(m for ms in pool_w.members_list for m in ms))
                best["p_bonferroni"] = min(1.0, best["p"] * nm)
                rows.append(best)
    assoc = pd.DataFrame(rows)
    if len(assoc):
        assoc["verdict"] = np.select(
            [(assoc.p_bonferroni < 0.01) & (assoc.specificity >= 0.8),
             (assoc.p_bonferroni < 0.01)],
            ["tracks one member (source well holds it)", "enriched with one member"],
            "not tied to a member (spread)")
    assoc.to_csv(config.OUT / "q04_foreign_association.csv", index=False)

    # ---- where could each foreign event have come from? ------------------------------------
    used_on = {}                                      # strain -> culture plates it was put on
    for r in lay[lay.well_type == "community"].itertuples():
        for m in r.members_list:
            used_on.setdefault(m, set()).add(r.culture_plate)
    tracked = set()
    if len(assoc):
        tracked = set(assoc.entity[assoc.verdict.str.startswith("tracks")])

    def source(r):
        if r.kind != "strain":
            return "tracks_member" if r.entity in tracked else "unknown_otu"
        if r.twin_of_member:
            return "twin_of_member"
        if r.entity in tracked:
            return "tracks_member"
        plates = used_on.get(r.entity, set())
        if r.culture_plate in plates:
            return "same_culture_plate"
        if plates:
            return "other_culture_plate"
        return "not_in_experiment"
    fo["source_class"] = [source(r) for r in fo.itertuples()]

    # d5 events: was the entity already in the same culture well at 24 h?
    tp0 = fo[fo.timepoint == "24h"].set_index(["culture_plate", "well", "entity"]).reads
    depth0 = ws[ws.timepoint == "24h"].set_index(["culture_plate", "well"]).reads_called
    fo["reads_same_well_24h"] = [tp0.get((c, w, e), 0) if t == "5d" else np.nan
                                 for c, w, e, t in zip(fo.culture_plate, fo.well, fo.entity,
                                                       fo.timepoint)]
    fo["depth_same_well_24h"] = [depth0.get((c, w), np.nan) if t == "5d" else np.nan
                                 for c, w, t in zip(fo.culture_plate, fo.well, fo.timepoint)]
    fo.to_csv(config.OUT / "q04_foreign.csv", index=False)
    called = fo[fo.called]

    # ---- neighbour test: does a foreign strain sit in an ADJACENT well's community? --------
    # Only events whose strain is on that culture plate can come from a neighbour, so only
    # those are tested; the null keeps each event's strain and draws its position at random
    # among the plate's wells. The 24 h plates are the control: no passaging yet.
    pos = lay_seq[lay_seq.well_type.isin(["community", "blank"])]
    rows = []
    rng = np.random.default_rng(0)
    for plate in config.SEQUENCED_PLATES:
        p = pos[pos.plate == plate]
        at = {(r.row_i, r.col_i): set(r.members_list) for r in p.itertuples()}
        ev = called[(called.plate == plate) & (called.source_class == "same_culture_plate")]
        if ev.empty:
            continue
        cells = list(at)
        twins_of = {e: {m for m in dist.index if dist.at[m, e] <= twin_bp}
                    for e in ev.entity.unique() if e in dist.index}

        def hit(rc, strain):
            tw = twins_of.get(strain, {strain})
            return any(at.get(nb, set()) & tw for nb in neighbours(*rc))

        obs = np.mean([hit((r.row_i, r.col_i), r.entity) for r in ev.itertuples()])
        null = []
        for _ in range(1000):
            rc = [cells[i] for i in rng.choice(len(cells), len(ev), replace=True)]
            null.append(np.mean([hit(c, s) for c, s in zip(rc, ev.entity)]))
        null = np.array(null)
        rows.append(dict(plate=plate, timepoint=p.timepoint.iat[0],
                         culture_plate=p.culture_plate.iat[0], events=len(ev),
                         share_in_neighbour=obs, null_mean=null.mean(),
                         null_p95=np.quantile(null, .95),
                         enrichment=obs / null.mean() if null.mean() else np.nan,
                         p=(np.sum(null >= obs) + 1) / (len(null) + 1)))
    pd.DataFrame(rows).to_csv(config.OUT / "q04_neighbour_test.csv", index=False)

    # ---- per-well flags: one row per sequenced well, for downstream exclusion -------------
    fl = ws[["sample_id", "plate", "culture_plate", "timepoint", "well", "community_id",
             "replicate", "well_type", "size", "reads_called", "member_frac",
             "foreign_frac", "unexplained_frac"]].copy()
    by = called.groupby(["sample_id", "source_class"]).frac.sum().unstack(fill_value=0)
    for c in ("not_in_experiment", "other_culture_plate", "same_culture_plate",
              "tracks_member", "unknown_otu"):
        fl[f"foreign_{c}"] = fl.sample_id.map(by[c] if c in by else {}).fillna(0)
    absent = mt[(mt.status == "absent") & mt.absence_informative].groupby("sample_id").strain.apply(
        lambda x: " ".join(sorted(x)))
    fl["members_absent"] = fl.sample_id.map(absent).fillna("")
    fl["flag_low_depth"] = fl.reads_called < 100
    fl["flag_external_dna"] = fl.foreign_not_in_experiment >= 0.10
    fl["flag_cross_well"] = (fl.foreign_same_culture_plate + fl.foreign_other_culture_plate) >= 0.05
    fl["flag_source_co_occupant"] = (fl.foreign_tracks_member + fl.foreign_unknown_otu) >= 0.05
    fl["flag_blank_with_reads"] = (fl.well_type == "blank") & (fl.reads_called >= 20)
    flags = [c for c in fl.columns if c.startswith("flag_")]
    fl["n_flags"] = fl[flags].sum(axis=1)
    fl.to_csv(config.OUT / "q04_well_flags.csv", index=False)

    # ---- console summary -------------------------------------------------------------------
    print(f"member table: {len(mt):,} (well, member) rows")
    for p in config.SEQUENCED_PLATES:
        m = mt[mt.plate == p]
        print(f"  plate {p:>2} ({m.timepoint.iat[0]}): present {np.mean(m.status == 'present'):.1%}, "
              f"trace {np.mean(m.status == 'trace'):.1%}, absent {np.mean(m.status == 'absent'):.1%}"
              f" (informative absences {np.mean((m.status == 'absent') & m.absence_informative):.1%})")
    print(f"foreign entities called: {int(fo.called.sum())} (well, entity) events in "
          f"{fo[fo.called].sample_id.nunique()} wells")
    print(called.groupby(["plate", "source_class"]).size().unstack(fill_value=0).to_string())
    print(pd.read_csv(config.OUT / "q04_neighbour_test.csv").round(3).to_string(index=False))


if __name__ == "__main__":
    main()
