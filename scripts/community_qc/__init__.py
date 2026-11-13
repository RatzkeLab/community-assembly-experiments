"""QC and analysis library for the community experiments (16S nanopore readout).

Nothing here holds experiment-specific constants or writes a file on import; every
function takes the paths / tables it needs. Each experiment's `02_qc/config.py` supplies
them.

    io           layout, minibar sheet, fastq reading, reference haplotypes
    assign       per-read nearest-strain assignment, expected members first
    crosstalk    demultiplexing cross-talk, calibrated on never-sequenced barcode plates
    populations  per-well base-exact consensus of each read group; unexplained populations
    composition  per-well counts -> composition tables, detection calls, replicate metrics
    plate_od     daily plate-reader OD series
    prediction   24 h -> later time point models (leave-community-out)
"""
