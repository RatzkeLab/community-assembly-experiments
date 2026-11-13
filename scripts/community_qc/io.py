"""Inputs: layout, minibar sheet, reads, reference haplotypes."""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd
from Bio import SeqIO

ROWS16 = "ABCDEFGHIJKLMNOP"
_QS = re.compile(r"qs:f:([0-9.]+)")


def load_layout(layout_csv):
    """layout.csv with `members` as a tuple and helper columns.

    `plate` is the BARCODE plate (what the sample ID says); `culture_plate` and
    `timepoint` say which physical culture and when it was sampled.
    """
    lay = pd.read_csv(layout_csv)
    lay["members_list"] = lay.members.fillna("").str.split().apply(tuple)
    lay["row_i"] = lay.row.map(ROWS16.index)
    lay["col_i"] = lay.col.astype(int) - 1
    return lay


def load_sheet(sheet_tsv):
    """minibar sheet -> sample_id, plate, well, fwd, rev (barcode sequences).

    Includes the decoy rows (Plate99_*) that are in no layout. Barcodes are kept as
    sequences so decoys and real wells share one key without an index lookup.
    """
    sh = pd.read_csv(sheet_tsv, sep="\t")
    out = pd.DataFrame({
        "sample_id": sh.SampleID,
        "plate": sh.SampleID.str.slice(5, 7).astype(int),
        "well": sh.SampleID.str.split("_").str[1],
        "fwd": sh.FwIndex.str.upper(),
        "rev": sh.RvIndex.str.upper(),
    })
    return out


def fastq_path(reads_dir, sample_id):
    for ext in (".fastq", ".fastq.gz"):
        p = Path(reads_dir) / f"{sample_id}{ext}"
        if p.exists():
            return p
    return None


def read_fastq(path):
    """[(seq, qscore)] in file order. Light 4-line parser; qscore from dorado's qs:f: tag
    (mean phred if absent)."""
    out = []
    if path is None or not Path(path).exists() or Path(path).stat().st_size == 0:
        return out
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        while True:
            h = fh.readline()
            if not h:
                break
            s = fh.readline().strip().upper()
            fh.readline()
            qual = fh.readline().strip()
            m = _QS.search(h)
            if m:
                q = float(m.group(1))
            else:
                q = float(np.mean([ord(c) - 33 for c in qual])) if qual else np.nan
            out.append((s, q))
    return out


def load_haplotypes(fasta):
    """{strain: [haplotype sequences]} from the `<strain>_h<k>` FASTA."""
    haps = {}
    for r in SeqIO.parse(str(fasta), "fasta"):
        haps.setdefault(r.id.rsplit("_h", 1)[0], []).append(str(r.seq).upper())
    return haps


def load_fasta(fasta):
    return {r.id: str(r.seq).upper() for r in SeqIO.parse(str(fasta), "fasta")}


def demux_match_classes(log_dir):
    """Sum minibar's per-chunk match-class tallies (`seqs: H n HH n Hh n hh n ...`).

    HH = barcode AND primer found at both ends; hh = barcode-only fallback. A healthy run
    is ~99% HH among assigned reads (20260630's broken sheet was 99.99% hh). Also returns
    the edit distances minibar actually used, which is the only reliable record of them.
    """
    tot, settings = {}, set()
    for f in sorted(Path(log_dir).glob("*.log")):
        for ln in f.read_text().splitlines():
            if "seqs:" in ln:
                toks = ln.split("seqs:")[1].split("(")[0].split()
                for k, v in zip(toks[::2], toks[1::2]):
                    tot[k] = tot.get(k, 0) + int(v)
            m = re.search(r"Index edit dist (\d+), Primer edit dist (\d+)", ln)
            if m:
                settings.add((int(m.group(1)), int(m.group(2))))
    return tot, sorted(settings)
