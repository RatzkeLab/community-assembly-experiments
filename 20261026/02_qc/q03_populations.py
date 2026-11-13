"""Step 3: base-exact identity of every read group in every sequenced well, and the
populations that match nothing in the reference (community_qc.populations).

  a. member / foreign groups with >= 10 reads -> spoa consensus over the best-qscore
     reads -> bp to the strain's reference (IUPAC consensus or any haplotype)
  b. unexplained reads -> per-well greedy clusters -> consensus of each population with
     >= 8 reads and >= 1% of the well -> nearest reference + two-parent chimera test
  c. unexplained consensuses pooled across wells into OTUs (<= 3 bp apart)
  d. every unexplained read counted against the OTU consensuses

Outputs
  q03_group_consensus.csv        (a) one row per (well, call, strain) group
  q03_unexplained_populations.csv (b)+(c) one row per (well, population), with OTU
  q03_otus.csv / q03_otus.fasta  (c) the OTUs, their seed consensus, nearest reference
  q03_otu_counts.csv             (d) reads per (well, OTU); 'none' = near no OTU
"""
import numpy as np
import pandas as pd

import config
from community_qc import assign, io as cio, populations as pop

TOP_Q = 100              # reads handed to the consensus per group (spoa uses the top 50)
UNEXPLAINED_SUB = 400    # unexplained reads clustered per well (all are counted in step d)


def main():
    reads = pd.read_parquet(config.OUT / "q01_read_assignments.parquet")
    stats = pd.read_csv(config.OUT / "q01_well_read_stats.csv").set_index("sample_id")
    lay = cio.load_layout(config.LAYOUT_CSV)
    seq_ids = set(lay.sample_id[lay.plate.isin(config.SEQUENCED_PLATES)])
    cls = assign.classify(reads[reads.sample_id.isin(seq_ids)])
    cls["sample_id"] = cls.sample_id.astype(str)

    haps = cio.load_haplotypes(config.HAPLOTYPE_FASTA)
    iupac = {k.split()[0]: v for k, v in cio.load_fasta(config.IUPAC_FASTA).items()}
    plain = cio.load_fasta(config.PLAIN_FASTA)
    anchor = next(s for k, s in plain.items() if k.split()[0] == config.ORIENT_ANCHOR_STRAIN)

    rng = np.random.default_rng(0)
    jobs = []
    for sid, g in cls.groupby("sample_id"):
        path = cio.fastq_path(config.READS_DIR, sid)
        if path is None:
            continue
        groups = {}
        for (call, strain), gg in g[g.call.isin(["member", "foreign"])].groupby(
                ["call", "strain"], observed=True):
            if len(gg) >= pop.MIN_CONS_READS:
                top = gg.nlargest(TOP_Q, "q").ridx.tolist()
                groups[(call, strain)] = (len(gg), top)
        unx = g.ridx[g.call == "unexplained"].to_numpy()
        n_unx = len(unx)
        if n_unx > UNEXPLAINED_SUB:
            unx = rng.choice(unx, UNEXPLAINED_SUB, replace=False)
        # cluster-size cut, expressed in the subsample: >= 8 reads and >= 1% of the well
        n_used = int(stats.at[sid, "reads_used"])
        cut = max(pop.MIN_UNK_READS,
                  pop.MIN_UNK_FRAC * n_used * (len(unx) / n_unx if n_unx else 1))
        jobs.append((sid, str(path), groups, [int(i) for i in unx], cut))
    print(f"{len(jobs)} wells, {sum(len(j[2]) for j in jobs):,} read groups, "
          f"{sum(len(j[3]) for j in jobs):,} unexplained reads clustered")

    grp, unk = pop.well_populations(jobs, haps, iupac, anchor, processes=config.PROCESSES)
    grp.drop(columns="consensus").to_csv(config.OUT / "q03_group_consensus.csv", index=False)
    print(f"group consensus: {len(grp):,} groups; within {pop.CONFIRM_BP} bp of the "
          f"reference: {(grp.bp_to_ref <= pop.CONFIRM_BP).mean():.1%}")

    unk, otus = pop.merge_otus(unk)
    if len(otus):
        tw = pd.read_csv(config.DISTANCE_MATRIX_CSV, index_col=0)
        otus = otus.merge(unk.groupby("otu").agg(
            parent_a=("parent_a", lambda x: x.mode().iat[0] if x.notna().any() else None),
            parent_b=("parent_b", lambda x: x.mode().iat[0] if x.notna().any() else None)),
            on="otu")

        def parents_bp(r):
            if r.parent_a in tw.index and r.parent_b in tw.index:
                return tw.at[r.parent_a, r.parent_b]
            return np.nan
        otus["parents_bp"] = otus.apply(parents_bp, axis=1)
        # a chimera: two different organisms explain it far better than any one does
        otus["chimera"] = ((otus.split_bp <= np.maximum(8, 0.3 * otus.nearest_bp))
                           & (otus.parents_bp > 20))
    unk.drop(columns="consensus").to_csv(config.OUT / "q03_unexplained_populations.csv",
                                         index=False)
    otus.drop(columns="seed").to_csv(config.OUT / "q03_otus.csv", index=False)
    with open(config.OUT / "q03_otus.fasta", "w") as fh:
        for r in otus.itertuples():
            fh.write(f">{r.otu} wells={r.n_wells} reads={r.n_reads} nearest={r.nearest} "
                     f"nearest_bp={r.nearest_bp:.0f} chimera={r.chimera}\n{r.seed}\n")
    print(f"unexplained populations: {len(unk):,} in {unk.sample_id.nunique()} wells -> "
          f"{len(otus)} OTUs ({int(otus.chimera.sum()) if len(otus) else 0} chimeric)")

    # ---- d. count every unexplained read against the OTUs -----------------------------------
    if len(otus):
        cjobs = []
        for sid, g in cls[cls.call == "unexplained"].groupby("sample_id"):
            cjobs.append((sid, str(cio.fastq_path(config.READS_DIR, sid)),
                          g.ridx.tolist(), g.near_cut.round().astype(int).tolist()))
        counts = pop.count_otus(cjobs, otus, anchor, processes=config.PROCESSES)
        counts.to_csv(config.OUT / "q03_otu_counts.csv", index=False)
        tot = counts.groupby(counts.otu == "none").reads.sum()
        print(f"unexplained reads near an OTU: {tot.get(False, 0):,}; near none: "
              f"{tot.get(True, 0):,}")


if __name__ == "__main__":
    main()
