"""Paths and run facts for the 20261026 QC (first sequencing run, 2026-11-12).

What was sequenced: passage plates 5 and 6 at 24 h and at 5 d only, i.e. BARCODE plates
5, 6, 9, 10 (barcode plate N = layout `plate`; the culture plate is `culture_plate`).
Every other barcode plate on the sheet (1-4, 7, 8, 11, 12) and the decoys (99) were never
PCR'd, so any read landing in them was mis-assigned by the demultiplexer: they are the
run's built-in negative control and are kept in every step.

Plates 6 and 9 (plate 6 at 24 h, plate 5 at 5 d) came out at ~1/20 the depth of 5 and 10
(median ~100 vs ~2000 reads per well) -- weak PCR, possibly re-run.
"""
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
EXP_DIR = BASE.parent
SEQ_ROOT = Path("/home/rl/scripts/karl/sequencing_experiments")

# shared code: this experiment family's library + the pairwise library it builds on
sys.path.insert(0, str(SEQ_ROOT / "community_assembly_experiments" / "scripts"))
sys.path.insert(0, str(SEQ_ROOT / "pairwise_interaction_experiments" / "scripts"))

EXPERIMENT = "20261026"

# --- design -----------------------------------------------------------------------------
SETUP_OUT = EXP_DIR / "01_setup" / "outputs"
LAYOUT_CSV = SETUP_OUT / "layout.csv"
COMMUNITIES_CSV = SETUP_OUT / "communities.csv"
POOL_CSV = SETUP_OUT / "pool.csv"
MINIBAR_SHEET = SETUP_OUT / "minibar_sheet_20261026.tsv"

# --- sequencing -------------------------------------------------------------------------
DEMUX_DIR = Path("/home/rl/data/interim/karl/demultiplexing/20261112_demux")
READS_DIR = DEMUX_DIR / "unflipped"
DEMUX_LOG_DIR = DEMUX_DIR / "logs" / "demux"
DEMUX_SUMMARY = DEMUX_DIR / "summary" / "demux_summary.tsv"
# The run was demultiplexed with `barcode_edit_dist: 6` in
# 01_setup/demultiplex_config.yaml -- the misspelled key the pipeline never reads, so its
# default -e 6 applied (the setup's outputs/demultiplex_config_20261026.yaml asks for 4).
# On 20260907's equally dense sheet, -e 6 measured ~5% false assignment vs ~1.2% at -e 4.
DEMUX_EDIT_DISTANCE = 6

SEQUENCED_PLATES = [5, 6, 9, 10]
UNSEQUENCED_PLATES = [1, 2, 3, 4, 7, 8, 11, 12, 99]     # 99 = decoy barcode pairs
LOW_DEPTH_PLATES = [6, 9]                                # weak PCR this run

# --- reference: base-exact 16S from 20260907, one entry per operon haplotype --------------
REFDB_DIR = (SEQ_ROOT / "pairwise_interaction_experiments" / "analyses" / "20261021")
HAPLOTYPE_FASTA = REFDB_DIR / "01_reference_db" / "outputs" / "reference_db_20260907_haplotypes.fasta"
PLAIN_FASTA = REFDB_DIR / "01_reference_db" / "outputs" / "reference_db_20260907.fasta"
IUPAC_FASTA = REFDB_DIR / "01_reference_db" / "outputs" / "reference_db_20260907_iupac.fasta"
STRAIN_REFERENCE_CSV = REFDB_DIR / "01_reference_db" / "outputs" / "s01_strain_reference.csv"
DISTANCE_MATRIX_CSV = REFDB_DIR / "02_distinguishability" / "outputs" / "d01_distance_matrix_bp.csv"
ORIENT_ANCHOR_STRAIN = "A1"

# --- plate reader: daily OD of all six culture plates ------------------------------------
PLATE_READER_DIR = Path("/home/rl/data/raw/karl/plate_reader_csvs/data_ascii/Karl_2026/Karl_20261028_OD")
INOCULATION_DATE = "2026-10-29"     # 'plate N d1' reads are dated 30/10

# --- outputs ------------------------------------------------------------------------------
OUT = BASE / "outputs"
FIG = BASE / "figures"
OUT.mkdir(exist_ok=True)
FIG.mkdir(exist_ok=True)

PROCESSES = 56
