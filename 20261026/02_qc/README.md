# 20261026 — QC of the first sequencing run (20261112 flowcell)

Sequenced: passage plates 5 and 6 at 24 h and 5 d = **barcode plates 5, 6, 9, 10**.
Plates 6 and 9 came out ~20× shallower (weak PCR). The 10 d samples were not
sequenced (contaminated).

| notebook | question |
|---|---|
| `01_strain_identity_contamination.ipynb` | did the intended strains land, base-exact? what else is in the wells, and where did it come from? |
| `02_technical_replicates.ipynb` | how reproducible are replicate wells within a plate, and between plates 5 and 6? |
| `03_24h_predicts_5d.ipynb` | is there coverage to ask whether 24 h predicts 5 d, and does it? |

Each notebook opens with a summary; every number in it is reproduced by a cell below.
Notebooks only read `outputs/` and save figures to `figures/`.

## Running

    conda activate karl_seq_analysis
    python run_all.py                 # ~20 min: q01 (~10 min) ... q05, then executes the notebooks
    python run_all.py --from 2        # reuse q01's read assignments
    python run_all.py --no-notebooks

Paths and run facts live in `config.py`. To point this at a re-demultiplexed run (or at
plates 1–4 later), change `DEMUX_DIR` / `SEQUENCED_PLATES` there. Shared code is in
`../../scripts/community_qc/` (assign, crosstalk, populations, composition,
prediction, plate_od, plots), reusable for the next community run.

| step | does | main outputs |
|---|---|---|
| `q01_assign_reads.py` | every read of all 4004 sheet wells: distance to the nearest expected member and to the nearest other strain (haplotype DB) | `q01_read_assignments.parquet`, `q01_well_read_stats.csv`, `q01_demux_health.csv` |
| `q02_crosstalk.py` | calibrates the read call; fits demux cross-talk on the never-PCR'd plates and checks the mechanism | `q02_call_calibration.csv`, `q02_crosstalk_*.csv`, `q02_thresholds.json` |
| `q03_populations.py` | spoa consensus per (well, strain) read group vs the reference; unexplained reads → per-well populations → OTUs, with chimera test | `q03_group_consensus.csv`, `q03_otus.csv/.fasta`, `q03_otu_counts.csv` |
| `q04_well_tables.py` | member table, well summary, foreign events with source class, association test, neighbour test, per-well flags | `q04_*.csv` (**`q04_well_flags.csv` = exclusion list**) |
| `q05_plate_reader.py` | daily OD600 of all six plates, d1–d11 | `q05_od_daily.csv` |

## Decisions worth knowing before changing anything

- **Ties go to the expected member.** 254/378 reference strains share a byte-identical
  16S with another strain; an open nearest-reference search splits a member's reads
  with its twin and calls half of them contamination (M19 / M15 is the first case you
  hit). Cost: contamination *by* a member's 16S twin is invisible. Foreign strains ≤ 4 bp
  from a member are never called (`twin_of_member`).
- **The read call is quality-aware:** near a reference if d ≤ 12 + 2·E,
  E = length × 10^(−q/10); q < 16 gets no call. Genuine reads' distance tracks E; a
  pooled distance histogram has no valley because chimeras / foreign sequence form a
  quality-independent tail (`q02_call_calibration.csv`). A fixed cut either drops good
  low-q reads or counts an unknown organism 45 bp away as a member.
- **Cross-talk uses length-filtered reads.** At -e 6, most reads in never-PCR'd wells are
  a ~1260 bp off-target product whose second end matched a barcode at edit distance
  exactly 6 with no primer (minibar tag `h-(6,-1)`); the 1300–1600 bp filter removes it.
- **Source class of a foreign read** (`q04_foreign.csv`): `tracks_member` (appears only
  with one member → that member's source well holds it), `same_culture_plate`,
  `other_culture_plate`, `not_in_experiment` (never inoculated → entered outside the
  cultures), `unknown_otu`. Recurring chimeras are separated out first (`q03_otus.csv`).
- **The neighbour test has a control:** the 24 h plates (no passaging yet) show no
  enrichment; the 5 d plates do.

## Follow-ups

1. **Re-run PCR for plates 6 and 9** with fresh reagents and a no-template control: both
   weak, and both carry the same contaminating-DNA profile (strains never inoculated here).
2. **Re-demultiplex at `-e 4`**: `01_setup/demultiplex_config.yaml` uses the key
   `barcode_edit_dist`, which the pipeline never reads, so its default `-e 6` applied.
   The correct key is `barcode_edit_distance` (as in `01_setup/outputs/demultiplex_config_20261026.yaml`).
3. Decide how to treat source-well co-occupants (M19 + U001/E14, F17 + F7, F14 + F2,
   G2 + H8, N15 + N23/O3): part of the member, or flag those communities.
4. K11, I8, B17 never establish: drop them from those communities' expected membership
   before modelling.
