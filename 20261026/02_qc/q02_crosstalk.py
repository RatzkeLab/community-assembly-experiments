"""Step 2: calibrate the read call, and measure demultiplexing cross-talk on the barcode
plates that were never sequenced.

1. The read call (community_qc.assign.classify) is quality-aware: a read is "near" a
   reference if d <= 12 + 2 * E, E = length * 10^(-q/10) its expected error count;
   q < 16 reads get no call. Calibration table: for reads of each deep well's dominant
   member, the share beyond the cut per q bin -- flat across q if the cut is fair to
   good and bad reads alike.
2. Stray-read rates a, b, c (community_qc.crosstalk) are fitted on the never-sequenced
   wells using LENGTH-FILTERED reads. The unfiltered counts are dominated by a ~1260 bp
   off-target product whose second end matched a barcode at edit distance exactly 6
   with no primer (minibar tag h-(6,-1)) -- an artefact of demultiplexing at -e 6 that
   the 1300-1600 bp read filter removes before any analysis sees it.
3. Validation: do stray reads in never-sequenced wells carry the strains of the wells
   they share a barcode with, more than chance?
4. An empirical ceiling: the most reads ANY single strain reached in a never-sequenced
   well. A foreign strain below that in a sequenced well is within what demultiplexing
   alone produces.

Outputs
  q02_thresholds.json            read-call parameters, stray ceilings
  q02_call_calibration.csv       (1) share beyond the cut by q bin
  q02_crosstalk_fit.csv          (2)+(3) rates, fit diagnostics, validation
  q02_crosstalk_wells.csv        every sheet well: depths, expected stray reads
  q02_expected_stray_long.csv    sequenced wells: expected stray reads per strain (>= 0.05)
  q02_unsequenced_strain_counts.csv  (4) reads per (never-sequenced well, strain)
"""
import json

import numpy as np
import pandas as pd

import config
from community_qc import assign, crosstalk as xt, io as cio
from community_qc.composition import twin_matrix

TWIN_BP = 4          # <= 4 bp: read cross-talk p90 24-37% (20261021 d02) -> not separable
SOURCE_MIN_FRAC = 0.005
Q_BINS = [0, 14, 16, 18, 20, 22, 24, 26, 45]


def main():
    reads = pd.read_parquet(config.OUT / "q01_read_assignments.parquet")
    reads["sample_id"] = reads.sample_id.astype(str)
    stats = pd.read_csv(config.OUT / "q01_well_read_stats.csv")
    sheet = cio.load_sheet(config.MINIBAR_SHEET)
    seq_mask = sheet.plate.isin(config.SEQUENCED_PLATES)

    # ---- 1. read-call calibration ------------------------------------------------------------
    deep = set(sheet.sample_id[sheet.plate.isin([5, 10])])
    r = reads[reads.sample_id.isin(deep)]
    near_m = r[r.d_member <= r.d_other.fillna(np.inf)]
    top = near_m.groupby(["sample_id", "member"], observed=True).size().rename("n").reset_index()
    top["f"] = top.n / top.sample_id.map(r.groupby("sample_id").size())
    dom = r.merge(top[top.f >= 0.3][["sample_id", "member"]], on=["sample_id", "member"])
    dom = dom.assign(E=dom.length * 10 ** (-dom.q / 10),
                     beyond=dom.d_member > assign.near_cut(dom.length, dom.q),
                     qbin=pd.cut(dom.q, Q_BINS))
    cal = (dom.groupby("qbin", observed=True)
           .agg(reads=("q", "size"), median_d=("d_member", "median"), median_E=("E", "median"),
                p90_d=("d_member", lambda x: x.quantile(.9)), share_beyond_cut=("beyond", "mean"))
           .reset_index())
    cal["share_of_all_reads"] = (reads[reads.sample_id.isin(deep)]
                                 .groupby(pd.cut(reads.q, Q_BINS), observed=True).size()
                                 .reindex(cal.qbin).to_numpy() / len(r))
    cal.to_csv(config.OUT / "q02_call_calibration.csv", index=False)
    print("read call: near if d <= 12 + 2*E; dominant-member reads beyond the cut, by q:")
    print(cal.round(3).to_string(index=False))

    cls = assign.classify(reads)

    # ---- 2. stray-read rates from never-sequenced wells (length-filtered reads) ------------
    depth = stats.set_index("sample_id").reads_len_ok.fillna(0)
    raw = stats.set_index("sample_id").reads_in_file.fillna(0)
    sequenced = pd.Series(seq_mask.to_numpy(), index=sheet.sample_id)
    ids, seq_ids, SF, SR = xt.sharing_matrices(sheet, sequenced)
    d_seq = depth.reindex(seq_ids).fillna(0).to_numpy(float)
    F, R = SF @ d_seq, SR @ d_seq
    unseq = ~sequenced.reindex(ids).to_numpy()
    y = depth.reindex(ids).fillna(0).to_numpy(float)
    rates = xt.fit_rates(y[unseq], F[unseq], R[unseq])
    total_seq, total_unseq = d_seq.sum(), y[unseq].sum()
    raw_unseq = raw.reindex(ids[unseq]).sum()
    print(f"\nnever-sequenced wells: {raw_unseq:,.0f} reads as demultiplexed, {total_unseq:,.0f} "
          f"after the 1300-1600 bp filter = {total_unseq / (total_seq + total_unseq):.2%} of "
          f"length-filtered reads; median {np.median(y[unseq]):.0f}, 99th pct "
          f"{np.quantile(y[unseq], .99):.0f} per well")
    print(f"stray-read rates: fwd-share a={rates['a_fwd_share']:.2e}, rev-share "
          f"b={rates['b_rev_share']:.2e} per read of a sharing well, floor c="
          f"{rates['c_floor']:.2f} reads/well; r(pred, obs)={rates['pearson_r']:.2f}, "
          f"dispersion {rates['dispersion']:.1f}")

    wells = xt.expected_stray(ids, seq_ids, SF, SR, depth, rates)
    wells = wells.merge(sheet[["sample_id", "plate", "well"]], on="sample_id")
    wells["sequenced"] = wells.plate.isin(config.SEQUENCED_PLATES)
    wells["reads_len_ok"] = wells.sample_id.map(depth).fillna(0)
    wells["reads_in_file"] = wells.sample_id.map(raw).fillna(0)
    wells["expected_stray_frac"] = (wells.expected_stray_reads
                                    / wells.reads_len_ok.where(wells.reads_len_ok > 0))
    wells.to_csv(config.OUT / "q02_crosstalk_wells.csv", index=False)

    # ---- expected stray composition for sequenced wells ------------------------------------
    called = cls[cls.call.isin(["member", "foreign"])]
    cnt = called.groupby(["sample_id", "strain"], observed=True).size().unstack(fill_value=0)
    used = stats.set_index("sample_id").reads_used
    scale = (depth / used.where(used > 0)).fillna(0)
    cnt_seq = cnt.reindex(seq_ids).fillna(0).mul(scale.reindex(seq_ids).fillna(0), axis=0)
    exp_comp = xt.expected_stray_composition(ids, seq_ids, SF, SR, cnt_seq, rates)
    sub = exp_comp.loc[exp_comp.index.isin(seq_ids)]
    sub.index.name, sub.columns.name = "sample_id", "strain"
    long = sub.stack().rename("expected_stray_reads").reset_index()
    long = long[long.expected_stray_reads >= 0.05]
    long.to_csv(config.OUT / "q02_expected_stray_long.csv", index=False)

    # ---- 3. validation ------------------------------------------------------------------------
    twins = twin_matrix(config.DISTANCE_MATRIX_CSV, TWIN_BP)
    frac = cnt_seq.div(cnt_seq.sum(axis=1).where(lambda s: s > 0), axis=0).fillna(0)
    present = (frac >= SOURCE_MIN_FRAC).astype(float)
    strains = sorted(set(twins.columns) | set(present.columns))
    Tm = twins.reindex(index=strains, columns=strains, fill_value=False).to_numpy(float)
    np.fill_diagonal(Tm, 1.0)
    present = present.reindex(columns=strains, fill_value=0)
    member_matrix = pd.DataFrame(present.to_numpy() @ Tm > 0, index=present.index, columns=strains)
    stray = cls[cls.sample_id.isin(set(ids[unseq])) & (cls.call == "foreign")]
    stray_counts = stray.groupby(["sample_id", "strain"], observed=True).size().unstack(fill_value=0)
    obs, null = xt.validate_sources(stray_counts, ids, seq_ids, SF, SR, member_matrix)
    print(f"stray reads whose strain is in a barcode-sharing sequenced well: {obs:.1%} "
          f"(random sharing partners: {null.mean():.1%}, 95th pct {np.quantile(null, .95):.1%}; "
          f"n = {int(stray_counts.values.sum()):,} reads)")

    # ---- 4. empirical per-strain stray ceiling ---------------------------------------------
    sc = stray_counts.stack().rename("reads").reset_index()
    sc = sc[sc.reads > 0]
    sc.to_csv(config.OUT / "q02_unsequenced_strain_counts.csv", index=False)
    per_well_max = stray_counts.max(axis=1).reindex(ids[unseq]).fillna(0)
    ceiling = dict(p99=float(np.quantile(per_well_max, .99)),
                   p999=float(np.quantile(per_well_max, .999)), max=float(per_well_max.max()))
    print(f"most reads any one strain reached in a never-sequenced well: 99th pct "
          f"{ceiling['p99']:.0f}, 99.9th {ceiling['p999']:.0f}, max {ceiling['max']:.0f}")

    with open(config.OUT / "q02_thresholds.json", "w") as fh:
        json.dump(dict(NEAR_BASE=assign.NEAR_BASE, NEAR_MULT=assign.NEAR_MULT,
                       Q_MIN=assign.Q_MIN, TWIN_BP=TWIN_BP,
                       stray_strain_ceiling=ceiling), fh, indent=1)

    fit = pd.DataFrame([dict(**rates, stray_reads_unsequenced=float(total_unseq),
                             stray_reads_unsequenced_unfiltered=float(raw_unseq),
                             reads_sequenced=float(total_seq),
                             validation_explained=obs, validation_null_mean=float(null.mean()),
                             validation_null_p95=float(np.quantile(null, .95)),
                             validation_reads=int(stray_counts.values.sum()),
                             **{f"strain_ceiling_{k}": v for k, v in ceiling.items()})])
    fit.to_csv(config.OUT / "q02_crosstalk_fit.csv", index=False)

    sw = wells[wells.sequenced]
    print("expected stray-read share of sequenced wells, median [90th pct] by barcode plate:")
    for p, g in sw.groupby("plate"):
        print(f"  plate {p:>2}: {g.expected_stray_frac.median():.2%} "
              f"[{g.expected_stray_frac.quantile(.9):.2%}]")


if __name__ == "__main__":
    main()
