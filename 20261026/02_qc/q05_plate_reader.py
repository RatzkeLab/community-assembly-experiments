"""Step 5: daily OD600 of all six culture plates, joined to the layout.

Not needed for the sequencing QC itself, but it answers two questions sequencing cannot:
did a well GROW (an absent member in a well that never grew is a different failure from
one that was outcompeted), and WHEN did the medium-only blank wells start growing -- the
cheapest contamination time course there is.

Outputs
  q05_plate_reader_index.csv   every file: label, run date, day since inoculation
  q05_od_daily.csv             culture_plate, day, well, od600 + layout columns
"""
import pandas as pd

import config
from community_qc import io as cio, plate_od


def main():
    od, idx = plate_od.daily_od(config.PLATE_READER_DIR, config.INOCULATION_DATE)
    idx.drop(columns="path").to_csv(config.OUT / "q05_plate_reader_index.csv", index=False)
    lay = cio.load_layout(config.LAYOUT_CSV)
    phys = (lay.drop_duplicates(["culture_plate", "well"])
            [["culture_plate", "well", "community_id", "replicate", "well_type", "size",
              "members", "row", "col"]])
    od = od.merge(phys, on=["culture_plate", "well"], how="left")
    od["well_type"] = od.well_type.fillna("outer_ring")
    od.to_csv(config.OUT / "q05_od_daily.csv", index=False)
    mism = idx[idx.label_mismatch]
    print(f"{len(idx)} plate-reader files; {int((idx.kind == 'daily').sum())} daily reads of "
          f"plates {sorted(idx.culture_plate.dropna().astype(int).unique())}, days "
          f"{int(od.day.min())}-{int(od.day.max())}")
    if len(mism):
        print("label/date mismatches (date wins):")
        print(mism[["file", "id1", "day"]].to_string(index=False))
    b = od[od.well_type == "blank"].pivot_table(index=["culture_plate", "well"],
                                                 columns="day", values="od")
    print("blank wells, OD600 by day:")
    print(b.round(2).to_string())


if __name__ == "__main__":
    main()
