"""Shared matplotlib style + plate-map helper for the QC notebooks.

Colour is assigned by the job it does:
  categorical (identity)  CAT, fixed order, never cycled; > 8 groups fold into 'other'
  sequential (magnitude)  SEQ, one hue light -> dark
  diverging (sign)        DIV, blue <-> red through a neutral grey
  status                  STATUS, reserved for ok / warn / bad, always with a label
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQ_STEPS = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
SEQ = LinearSegmentedColormap.from_list("seq_blue", SEQ_STEPS)
DIV = LinearSegmentedColormap.from_list("div_blue_red",
                                        ["#184f95", "#6da7ec", "#f0efec", "#ec835a", "#d03b3b"])
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
ROWS16 = "ABCDEFGHIJKLMNOP"


def setup():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "regular", "axes.labelsize": 9.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2, "lines.markersize": 5,
        "axes.prop_cycle": mpl.cycler(color=CAT),
        "font.family": "sans-serif", "figure.dpi": 110,
    })


def plate_map(ax, values, wells, cmap=SEQ, vmin=None, vmax=None, title="", label="",
              mask_wells=None, annotate=None, norm=None, rows=range(1, 15), cols=range(1, 23)):
    """Inner 14 x 22 wells (B-O, 2-23) of a 384 plate. `values` aligned with `wells`.
    mask_wells: wells drawn as hatched grey (no data). annotate: {well: short text}."""
    grid = np.full((16, 24), np.nan)
    for w, v in zip(wells, values):
        grid[ROWS16.index(w[0]), int(w[1:]) - 1] = v
    sub = grid[rows.start:rows.stop, cols.start:cols.stop]
    im = ax.imshow(sub, cmap=cmap, vmin=vmin, vmax=vmax, norm=norm, aspect="equal",
                   interpolation="nearest")
    ax.set_xticks(range(sub.shape[1]), [str(c + 1) for c in cols], fontsize=6.5)
    ax.set_yticks(range(sub.shape[0]), [ROWS16[r] for r in rows], fontsize=6.5)
    ax.tick_params(length=0)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    if mask_wells:
        for w in mask_wells:
            r, c = ROWS16.index(w[0]) - rows.start, int(w[1:]) - 1 - cols.start
            ax.add_patch(plt.Rectangle((c - .5, r - .5), 1, 1, fill=False, hatch="////",
                                       edgecolor=AXIS, lw=0))
    if annotate:
        for w, txt in annotate.items():
            r, c = ROWS16.index(w[0]) - rows.start, int(w[1:]) - 1 - cols.start
            ax.text(c, r, txt, ha="center", va="center", fontsize=6, color=INK)
    ax.set_title(title, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label(label, color=INK2, fontsize=8.5)
    cb.outline.set_visible(False)
    return im


def label_end(ax, x, y, text, color=INK2, dx=4):
    """Direct label at a line's end (text in ink, never the series colour)."""
    ax.annotate(text, (x, y), xytext=(dx, 0), textcoords="offset points",
                va="center", fontsize=8.5, color=color)
