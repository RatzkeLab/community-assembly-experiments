"""20261026 part 2: barcode plates 13-16 = main plates 1-4 at day 5.

    python run_setup_part02.py            # refuses to overwrite outputs_part02/
    python run_setup_part02.py --force    # rebuild (deterministic: identical picklists)

Reads run_setup.py's outputs/ (left untouched, md5-checked against its manifest) and writes
outputs_part02/:
  primer_shooting/echo_primers_20261026_plate_13..16.csv   Echo primer picklists
  layout.csv                  outputs/layout.csv + plates 13-16 (rows 1-12 unchanged)
  minibar_sheet_20261026_part02.tsv, demultiplex_config_20261026_part02.yaml
                              the part-1 sheet + plates 13-16 (decoys kept): use for any run
                              that carries plates 13-16
  part02_barcodes.csv         where each new well's barcode combination comes from
  summary.txt, validation.txt, manifest.json

Barcodes. Part 1 kept every combination disjoint from 20261021's (identical layouts make a
cross-experiment swap invisible). Four more plates cannot all be disjoint: after the 4,004
combinations of part 1 and the 2,772 of 20261021, 625 valid ones are left for 1,232 wells.
They are therefore taken, in this order:
  1. combinations never on any sheet;
  2. 20261021's decoy combinations (on its sheet, never in a physical well: no DNA exists);
  3. combinations of 20261021 wells, only where that 20261021 well held a strain the new
     well never receives (or was a blank), so a stray 20261021 read still shows up as a
     foreign strain rather than passing as the new well's own.
No combination of part 1 (its decoys included) is reused.
"""
import argparse
import datetime
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KARL = Path("/home/rl/scripts/karl")
SEQ = KARL / "sequencing_experiments"
sys.path.insert(0, str(SEQ / "pairwise_interaction_experiments" / "scripts"))

import numpy as np
import pandas as pd
import yaml

from experiment_setup import barcodes as B, echo as E, plates, validate as V

# barcode plate: (culture plate, time point). Same layout as the culture plate's 24 h plate.
BARCODE_PLATES = {13: (1, "5d"), 14: (2, "5d"), 15: (3, "5d"), 16: (4, "5d")}
PART = 2                                   # rng = default_rng([seed, PART])

P1 = HERE / "outputs"
OUT = HERE / "outputs_part02"
REF_OUT = HERE.parent / "00_obsolete" / "20261021_failed_run" / "01_setup" / "outputs"
BC_COLS = ["fwd_idx", "rev_idx", "fwd_source_well", "rev_source_well"]


def kp(rel):
    """setup_config.yaml paths predate the move under sequencing_experiments/."""
    for base in (KARL, SEQ):
        if (base / rel).exists():
            return base / rel
    raise FileNotFoundError(rel)


def md5(path):
    return hashlib.md5(open(path, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(HERE / "setup_config.yaml"))
    exp = str(cfg["experiment"])
    ref_exp = str(cfg["copy_from"]["experiment"])
    if OUT.exists():
        if not args.force:
            sys.exit(f"{OUT.name}/ exists; pass --force to rebuild it")
        shutil.rmtree(OUT)
    OUT.mkdir()
    rng = np.random.default_rng([cfg["seed"], PART])
    m1 = json.load(open(P1 / "manifest.json"))
    vol = cfg["primers"]["volume_nL"]

    # ---- part 1 and 20261021 -------------------------------------------------------------
    lay1 = pd.read_csv(P1 / "layout.csv", float_precision="round_trip")
    sheet1 = pd.read_csv(P1 / f"minibar_sheet_{exp}.tsv", sep="\t")
    ref_lay = pd.read_csv(REF_OUT / "layout.csv")
    ref_sheet = pd.read_csv(REF_OUT / f"minibar_sheet_{ref_exp}.tsv", sep="\t")
    table = B.barcode_table(kp(cfg["inputs"]["primer_specs"]))
    failed = B.failed_barcodes(kp(cfg["inputs"]["failed_barcodes_fwd"]),
                               kp(cfg["inputs"]["failed_barcodes_rev"]))
    fwd_w, rev_w = B.primer_source_wells()
    decoy = int(cfg["decoy_barcode_plate"])

    used1 = B.combos_in_sheet(sheet1, table)
    plate_of = lambda sid: int(sid.split("_")[0][5:])
    assigned1 = [(plate_of(s), s.split("_")[1], i, j) for s, (i, j) in zip(sheet1.SampleID, used1)]
    ref_combo = dict(zip(B.combos_in_sheet(ref_sheet, table), ref_sheet.SampleID))
    ref_well = ref_lay.assign(m=ref_lay.members.fillna("").str.split().map(frozenset)) \
                      .set_index("sample_id")

    # ---- plates 13-16 --------------------------------------------------------------------
    new = []
    for bp, (cp, t) in sorted(BARCODE_PLATES.items()):
        c = lay1[(lay1.plate == cp) & (lay1.culture_plate == cp) & lay1.inoculated].copy()
        if len(c) != len(plates.inner_wells()):
            sys.exit(f"culture plate {cp}: {len(c)} wells in outputs/layout.csv")
        c["plate"], c["timepoint"], c["inoculated"] = bp, t, False
        c["sample_id"] = [f"Plate{bp:02d}_{w}" for w in c.well]
        new.append(c)
    new = pd.concat(new, ignore_index=True)
    taken = set(used1)
    if taken & set(map(tuple, new[["plate", "well"]].values)):
        sys.exit("a new barcode plate number is already on the part-1 sheet")

    # ---- barcodes ------------------------------------------------------------------------
    valid = [(i, j) for i in range(96) for j in range(96)
             if i != j and i not in failed["fwd"] and j not in failed["rev"]]
    ref_decoy = {k for k, s in ref_combo.items() if plate_of(s) == decoy}
    tier1 = [k for k in valid if k not in taken and k not in ref_combo]
    tier2 = [k for k in valid if k not in taken and k in ref_decoy]
    tier3 = [k for k in valid if k not in taken and k in ref_combo and k not in ref_decoy]
    free = [tier1[k] for k in rng.permutation(len(tier1))] + \
           [tier2[k] for k in rng.permutation(len(tier2))]
    pool3 = [tier3[k] for k in rng.permutation(len(tier3))]
    members = new.members.fillna("").str.split().map(frozenset).tolist()
    order = rng.permutation(len(new))
    combo, source = [None] * len(new), [None] * len(new)
    t1 = set(tier1)
    for n, w in enumerate(order):
        if n < len(free):
            combo[w] = free[n]
            source[w] = "unused" if free[n] in t1 else f"{ref_exp} decoy"
            continue
        for k, cand in enumerate(pool3):
            if safe_reuse(ref_well.loc[ref_combo[cand]], members[w]):
                combo[w] = pool3.pop(k)
                source[w] = f"{ref_exp} well"
                break
        else:
            sys.exit("ran out of reusable 20261021 combinations")
    new["fwd_idx"] = [i for i, _ in combo]
    new["rev_idx"] = [j for _, j in combo]
    new["fwd_source_well"] = [fwd_w[i] for i in new.fwd_idx]
    new["rev_source_well"] = [rev_w[j] for j in new.rev_idx]
    assigned2 = list(zip(new.plate, new.well, new.fwd_idx, new.rev_idx))

    prov = new[["sample_id", "culture_plate", "timepoint", "well", "community_id",
                "fwd_idx", "rev_idx"]].copy()
    prov["combination_source"] = source
    prov[f"{ref_exp}_sample"] = [ref_combo.get(k, "") for k in combo]
    prov[f"{ref_exp}_community"] = [ref_well.community_id.get(s, "") for s in prov[f"{ref_exp}_sample"]]
    prov.to_csv(OUT / "part02_barcodes.csv", index=False)

    # ---- outputs -------------------------------------------------------------------------
    lay = pd.concat([lay1, new[lay1.columns]], ignore_index=True)
    lay.to_csv(OUT / "layout.csv", index=False)
    sheet2 = B.minibar_sheet(assigned2, table)
    is_decoy = sheet1.SampleID.str.startswith(f"Plate{decoy}_")
    sheet = pd.concat([sheet1[~is_decoy], sheet2, sheet1[is_decoy]], ignore_index=True)
    sheet_path = OUT / f"minibar_sheet_{exp}_part02.tsv"
    sheet.to_csv(sheet_path, sep="\t", index=False)
    demux = B.demux_config(sheet_path, f"/home/rl/data/interim/karl/demultiplexing/{exp}_part02_demux/",
                           barcode_edit_distance=cfg["demux"]["barcode_edit_distance"])
    with open(OUT / f"demultiplex_config_{exp}_part02.yaml", "w") as fh:
        fh.write(f"# Barcode plates 1-{max(BARCODE_PLATES)} of {exp}. Plate{decoy}_* rows are decoys: "
                 "combinations put in no\n# well; reads assigned to them measure mis-assignment. "
                 "Plates not on the flowcell\n# are negative controls.\n")
        yaml.safe_dump(demux, fh, sort_keys=False)
    pdir = OUT / "primer_shooting"; pdir.mkdir()
    prim = E.primer_transfers(assigned2, vol, fwd_w, rev_w)
    for p, g in prim.groupby("Destination Plate Name"):
        g.to_csv(pdir / f"echo_primers_{exp}_plate_{int(p):02d}.csv", index=False)

    # ---- validation ----------------------------------------------------------------------
    checks = {
        "inputs": check_inputs(cfg, m1, exp, ref_exp),
        "layout": check_layout(pd.read_csv(OUT / "layout.csv", float_precision="round_trip"),
                               lay1, new),
        "barcodes": V.check_barcodes(assigned1 + assigned2, failed)
                    + check_reuse(prov, ref_combo, ref_well, new, used1, ref_exp),
        "primer picklists": check_picklists(pdir, exp, new, sheet2, table, vol, fwd_w, rev_w),
        "sheet": check_sheet(pd.read_csv(sheet_path, sep="\t"), sheet1, sheet2),
    }
    mb_errs, mb_info = V.check_minibar(sheet, cfg["demux"]["barcode_edit_distance"])
    checks["minibar"] = mb_errs
    report = summarize(exp, ref_exp, new, prov, lay, ref_lay, tier1, tier2, tier3, vol, mb_info, decoy)
    vlines = [f"{k}: {'PASS' if not v else 'FAIL'}" + "".join(f"\n    {e}" for e in v[:20])
              for k, v in checks.items()]
    (OUT / "validation.txt").write_text("\n".join(vlines) + "\n")
    (OUT / "summary.txt").write_text(report)
    manifest = dict(
        experiment=exp, part=PART, created=datetime.datetime.now().isoformat(timespec="seconds"),
        barcode_plates={str(k): list(v) for k, v in BARCODE_PLATES.items()},
        seed=[cfg["seed"], PART], script_md5=md5(__file__),
        built_on={**{f"outputs/{f}": md5(P1 / f) for f in
                     ("layout.csv", f"minibar_sheet_{exp}.tsv", "manifest.json")},
                  f"{ref_exp}/layout.csv": md5(REF_OUT / "layout.csv"),
                  f"{ref_exp}/minibar_sheet_{ref_exp}.tsv": md5(REF_OUT / f"minibar_sheet_{ref_exp}.tsv"),
                  **{k: md5(kp(cfg["inputs"][k])) for k in
                     ("primer_specs", "failed_barcodes_fwd", "failed_barcodes_rev")}},
        outputs={str(p.relative_to(OUT)): md5(p) for p in sorted(OUT.rglob("*")) if p.is_file()})
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    print(report)
    print("\n".join(vlines))
    if any(checks.values()):
        sys.exit(f"VALIDATION FAILED -- see {OUT.name}/validation.txt")


def safe_reuse(ref_row, new_members):
    """A 20261021 well's combination may go to a new well if that 20261021 well was a blank
    or held a strain the new well never receives."""
    return ref_row.well_type == "blank" or not ref_row.m <= new_members


def check_inputs(cfg, m1, exp, ref_exp):
    """Part 1's outputs, 20261021's outputs and the barcode inputs are the files part 1 was
    built from / wrote."""
    errs = []
    for f in ["layout.csv", f"minibar_sheet_{exp}.tsv"] + \
             [f"primer_shooting/echo_primers_{exp}_plate_{p:02d}.csv" for p in range(1, 13)]:
        if md5(P1 / f) != m1["outputs"][f]:
            errs.append(f"outputs/{f} differs from part 1's manifest")
    for f in ("layout.csv", f"minibar_sheet_{ref_exp}.tsv"):
        if md5(REF_OUT / f) != m1["copied_from"][f]:
            errs.append(f"{ref_exp} {f} differs from what part 1 was built on")
    for k in ("primer_specs", "failed_barcodes_fwd", "failed_barcodes_rev"):
        if md5(kp(cfg["inputs"][k])) != m1["inputs"][k]:
            errs.append(f"input {k} changed since part 1")
    return errs


def check_layout(lay, lay1, new):
    """Rows 1-12 are part 1's; plates 13-16 hold their culture plate's wells exactly."""
    errs = V.check_copy(lay[lay.plate.isin(lay1.plate.unique())], lay1, list(lay1.columns),
                        "plates 1-12")
    keep = [c for c in lay1.columns if c not in BC_COLS + ["plate", "timepoint", "inoculated",
                                                           "sample_id"]]
    for bp, (cp, t) in BARCODE_PLATES.items():
        a = new[new.plate == bp]
        errs += V.check_copy(a, lay1[(lay1.plate == cp) & lay1.inoculated], keep,
                             f"plate {bp} vs culture plate {cp}")
        if set(a.timepoint) != {t} or a.inoculated.any() or (a.culture_plate != cp).any():
            errs.append(f"plate {bp}: culture_plate / timepoint / inoculated wrong")
    if lay.sample_id.duplicated().any():
        errs.append("duplicate sample_id")
    if not (OUT / "layout.csv").read_bytes().startswith((P1 / "layout.csv").read_bytes()):
        errs.append("layout.csv does not start with outputs/layout.csv byte for byte")
    return errs


def check_sheet(sheet, sheet1, sheet2):
    """The written sheet = part 1's rows unchanged + the new plates' rows."""
    errs = V.check_copy(sheet[sheet.SampleID.isin(sheet1.SampleID)], sheet1, list(sheet1.columns),
                        "part-1 rows of the sheet")
    errs += V.check_copy(sheet[sheet.SampleID.isin(sheet2.SampleID)], sheet2, list(sheet2.columns),
                         "new rows of the sheet")
    if len(sheet) != len(sheet1) + len(sheet2):
        errs.append(f"sheet has {len(sheet)} rows, expected {len(sheet1) + len(sheet2)}")
    return errs


def check_reuse(prov, ref_combo, ref_well, new, used1, ref_exp):
    errs = []
    if set(zip(new.fwd_idx, new.rev_idx)) & set(used1):
        errs.append("a part-1 combination (decoys included) is reused")
    reused = prov[prov.combination_source == f"{ref_exp} well"]
    nm = new.set_index("sample_id").members.fillna("").str.split().map(frozenset)
    bad = [s for s, r in zip(reused.sample_id, reused[f"{ref_exp}_sample"])
           if not safe_reuse(ref_well.loc[r], nm[s])]
    if bad:
        errs.append(f"{len(bad)} wells reuse a {ref_exp} combination whose well's strains they all hold")
    for src, want in (("unused", False), (f"{ref_exp} decoy", True), (f"{ref_exp} well", True)):
        k = prov[prov.combination_source == src]
        if (k[f"{ref_exp}_sample"].ne("") != want).any():
            errs.append(f"combination source '{src}' mislabelled")
    return errs


def check_picklists(pdir, exp, new, sheet2, table, vol, fwd_w, rev_w):
    """Each picklist puts exactly the layout's two oligos in every well, and those oligos'
    barcodes are the sheet's."""
    errs = []
    fbc = dict(zip(table[table.side == "fwd"].source_well, table[table.side == "fwd"].barcode))
    rbc = dict(zip(table[table.side == "rev"].source_well, table[table.side == "rev"].barcode))
    sh = sheet2.set_index("SampleID")
    for bp in BARCODE_PLATES:
        f = pdir / f"echo_primers_{exp}_plate_{bp:02d}.csv"
        g = pd.read_csv(f)
        if list(g.columns) != ["Source Well", "Destination Plate Name", "Destination Well",
                               "Transfer Volume"]:
            errs.append(f"{f.name}: columns")
        if (g["Transfer Volume"] != vol).any() or (g["Destination Plate Name"] != bp).any():
            errs.append(f"{f.name}: volume or destination plate name")
        a = new[new.plate == bp].set_index("well")
        per = g.groupby("Destination Well")["Source Well"].apply(list)
        if set(per.index) != set(a.index) or set(a.index) != set(plates.inner_wells()):
            errs.append(f"{f.name}: wells differ from the layout")
            continue
        for w, src in per.items():
            fw = [s for s in src if s in set(fwd_w)]
            rv = [s for s in src if s in set(rev_w)]
            if (len(src), fw, rv) != (2, [a.at[w, "fwd_source_well"]], [a.at[w, "rev_source_well"]]):
                errs.append(f"{f.name} {w}: oligos {src} differ from the layout")
            elif (fbc[fw[0]], rbc[rv[0]]) != (sh.at[f"Plate{bp:02d}_{w}", "FwIndex"],
                                             sh.at[f"Plate{bp:02d}_{w}", "RvIndex"]):
                errs.append(f"{f.name} {w}: oligo barcodes differ from the sheet")
    return errs


def draw_uL(*frames, vol):
    """uL drawn per oligo source well by the physical wells of the given layouts."""
    s = pd.concat([pd.concat([d.fwd_source_well, d.rev_source_well]) for d in frames])
    return s.value_counts() * vol / 1000


def summarize(exp, ref_exp, new, prov, lay, ref_lay, tier1, tier2, tier3, vol, mb_info, decoy):
    S = [f"experiment {exp} part 2: barcode plates "
         + ", ".join(f"{b} = plate {p} @ {t}" for b, (p, t) in sorted(BARCODE_PLATES.items())),
         f"  {len(new)} wells ({(new.well_type == 'community').sum()} community, "
         f"{(new.well_type == 'blank').sum()} blank), layout = culture plate's 24 h plate well for well",
         f"\nbarcode combinations for the new wells (part 1's {len(lay) - len(new) + 308:,}, "
         f"decoys incl., never reused):"]
    vc = prov.combination_source.value_counts()
    S.append(f"  never on any sheet:          {vc.get('unused', 0):4d} (of {len(tier1)} left)")
    S.append(f"  {ref_exp} decoys (no DNA):     {vc.get(f'{ref_exp} decoy', 0):4d} (of {len(tier2)})")
    S.append(f"  {ref_exp} wells, safe reuse:   {vc.get(f'{ref_exp} well', 0):4d} (of {len(tier3)}; "
             f"each {ref_exp} well held a strain the new well lacks, or was a blank)")
    r = prov[prov.combination_source == f"{ref_exp} well"]
    S.append(f"    their {ref_exp} plates: "
             + str(r[f"{ref_exp}_sample"].str[5:7].astype(int).value_counts().sort_index().to_dict()))
    left = len(tier3) - len(r)
    S.append(f"  left for further plates: {left:,}, all {ref_exp} well combinations")
    phys = lay[lay.plate != decoy]
    p13 = draw_uL(new, vol=vol)
    p_all = draw_uL(phys, vol=vol)
    with_ref = draw_uL(phys, ref_lay, vol=vol)
    S.append(f"\nprimer drawn per oligo source well ({vol} nL per oligo per well):")
    S.append(f"  plates 13-16:          median {p13.median():.1f} uL, max {p13.max():.1f} uL")
    S.append(f"  plates 1-16:           median {p_all.median():.1f} uL, max {p_all.max():.1f} uL "
             f"({p_all.idxmax()})")
    S.append(f"  + {ref_exp} plates 1-8: median {with_ref.median():.1f} uL, max {with_ref.max():.1f} uL "
             f"({with_ref.idxmax()})  [if all shot from one primer plate]")
    S.append(f"minibar (whole sheet, {len(phys) + 308:,} rows): {mb_info}")
    return "\n".join(S) + "\n"


if __name__ == "__main__":
    main()
