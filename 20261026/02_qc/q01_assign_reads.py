"""Step 1: every read of every well on the sheet -> nearest expected member / nearest other
strain (community_qc.assign). The only slow step (~10 min on 56 cores).

All 4004 sheet wells are processed, not only the four sequenced plates: the 2772 wells
that were never PCR'd (barcode plates 1-4, 7, 8, 11, 12 and the decoys) hold only
mis-assigned reads, and step 2 uses them to calibrate demultiplexing cross-talk. Those
wells are searched with NO expected members (open nearest-strain search), because
nothing was supposed to be in them.

Outputs
  q01_read_assignments.parquet   one row per read (see assign.COLS)
  q01_well_read_stats.csv        reads in file / length-filtered / used, per well
  q01_demux_health.csv           minibar match classes and the edit distances it used
"""
import time

import pandas as pd

import config
from community_qc import assign
from community_qc import io as cio


def main():
    t0 = time.time()
    lay = cio.load_layout(config.LAYOUT_CSV).set_index("sample_id")
    sheet = cio.load_sheet(config.MINIBAR_SHEET)
    haps = cio.load_haplotypes(config.HAPLOTYPE_FASTA)
    plain = cio.load_fasta(config.PLAIN_FASTA)
    anchor = next(s for k, s in plain.items() if k.split()[0] == config.ORIENT_ANCHOR_STRAIN)

    wells = []
    for sid, plate in zip(sheet.sample_id, sheet.plate):
        members = lay.at[sid, "members_list"] if (plate in config.SEQUENCED_PLATES
                                                  and sid in lay.index) else ()
        wells.append((sid, members))
    missing = {m for _, ms in wells for m in ms if m not in haps}
    print(f"{len(wells)} sheet wells; {len(haps)} strains / "
          f"{sum(map(len, haps.values()))} haplotypes in the reference")
    if missing:
        print(f"  WARNING: {len(missing)} expected members have no reference: {sorted(missing)}")

    reads, stats = assign.assign_wells(wells, config.READS_DIR, haps, anchor,
                                       processes=config.PROCESSES)
    reads.to_parquet(config.OUT / "q01_read_assignments.parquet", index=False)
    stats = stats.merge(sheet[["sample_id", "plate", "well"]], on="sample_id", how="right")
    stats.to_csv(config.OUT / "q01_well_read_stats.csv", index=False)

    tally, settings = cio.demux_match_classes(config.DEMUX_LOG_DIR)
    health = pd.DataFrame([dict(**tally, edit_distances_used=str(settings))])
    health.to_csv(config.OUT / "q01_demux_health.csv", index=False)

    print(f"{len(reads):,} reads assigned in {time.time() - t0:.0f} s; "
          f"{stats.reads_flipped.sum():,} needed flipping")
    print(f"minibar used (barcode, primer) edit distance {settings}; match classes "
          f"HH {tally.get('HH', 0):,} / Hh {tally.get('Hh', 0):,} / hh {tally.get('hh', 0):,}")


if __name__ == "__main__":
    main()
