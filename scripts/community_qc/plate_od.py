"""Daily plate-reader OD of the culture plates.

The 20261026 folder holds one ODFull (61-wavelength) read per culture plate per day,
labelled in ID1 as 'plate N dK', plus preculture / source-plate reads and a set of
'plate contamination check N' reads. Day is taken from the header DATE (run start), not
from the label: two plate-3/4 reads on 09/11 are labelled d10 although they are the day
after the d10 reads (the other four plates call that day d11). Labels are kept so the
mismatch stays visible.
"""
import datetime as _dt
import re
from pathlib import Path

import numpy as np
import pandas as pd

import plate_reader as pr      # pairwise_interaction_experiments/scripts

_LABEL = re.compile(r"plate\s*(\d+)\s*(?:d(\d+)|(24)h)", re.I)     # 'plate 3 24h' = d1
_CHECK = re.compile(r"contamination check\s*(\d+)", re.I)


def index_folder(folder, inoculation_date):
    t0 = _dt.date.fromisoformat(inoculation_date)
    rows = []
    for p in sorted(Path(folder).glob("*.csv")):
        h = pr.parse_header(p)
        d = _dt.datetime.strptime(h["date"], "%d/%m/%Y").date()
        id1 = (h["id1"] or "").strip().strip('"')
        m, c = _LABEL.search(id1), _CHECK.search(id1)
        rows.append(dict(file=p.name, path=p, date=d, time=h["time"], id1=id1,
                         culture_plate=int(m.group(1)) if m else np.nan,
                         label_day=(int(m.group(2)) if m.group(2) else 1) if m else np.nan,
                         day=(d - t0).days,
                         kind="daily" if m else ("contamination_check" if c else "other"),
                         check_no=int(c.group(1)) if c else np.nan))
    idx = pd.DataFrame(rows)
    idx["label_mismatch"] = (idx.kind == "daily") & (idx.label_day != idx.day)
    return idx


def daily_od(folder, inoculation_date, wavelength=600):
    """Long table: culture_plate, day, well, od (at `wavelength`), plus the file index.
    If a plate has two reads on one day the later one is kept and flagged."""
    idx = index_folder(folder, inoculation_date)
    daily = idx[idx.kind == "daily"].sort_values(["culture_plate", "day", "time"])
    daily["n_reads_that_day"] = daily.groupby(["culture_plate", "day"]).file.transform("size")
    keep = daily.drop_duplicates(["culture_plate", "day"], keep="last")
    frames = []
    for r in keep.itertuples():
        grid, wl = pr.read_plate(r.path, wavelength)
        t = pr.to_long(grid, int(r.culture_plate))
        t = t.rename(columns={"dest_plate": "culture_plate", "dest_well": "well"})
        t["day"] = r.day
        t["file"] = r.file
        frames.append(t)
    return pd.concat(frames, ignore_index=True), idx


def check_plates(folder, inoculation_date, wavelength=600):
    idx = index_folder(folder, inoculation_date)
    frames = []
    for r in idx[idx.kind == "contamination_check"].itertuples():
        grid, wl = pr.read_plate(r.path, wavelength)
        t = pr.to_long(grid, int(r.check_no)).rename(columns={"dest_plate": "check_no",
                                                              "dest_well": "well"})
        t["file"] = r.file
        frames.append(t)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
