"""Shared figure style: light background, large type, one validated palette.

Categorical colors are the first three slots of a palette validated for
colour-vision deficiencies (all pairs); ε values, which are ordered, use a
single-hue blue ramp.  Every series also gets its own marker or line style,
so identity never relies on colour alone.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8984"
GRID = "#e6e5e0"
SURFACE = "#ffffff"

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

METHODS = ("pSGLD", "MMD", "Sinkhorn")
METHOD_COLOR = {"pSGLD": BLUE, "MMD": ORANGE, "Sinkhorn": AQUA}
METHOD_MARKER = {"pSGLD": "o", "MMD": "s", "Sinkhorn": "^"}
METHOD_LABEL = {"pSGLD": "pSGLD", "MMD": "MMD regularizer", "Sinkhorn": "Sinkhorn regularizer"}

CLASS_COLOR = (ORANGE, BLUE)            # class 0, class 1
PRIOR_COLOR = MUTED
TIMEAVG_COLOR = BLUE
FINAL_COLOR = ORANGE

DIM_LABELS = (r"$a_0^{(0)}$", r"$a_0^{(1)}$", r"$a_1^{(0)}$", r"$a_1^{(1)}$", r"$a_2$")

_RAMP = ("#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b")
_PRIOR_C1, _PRIOR_C2 = 0.05, 0.5

RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titlesize": 12.5,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 8,
    "axes.labelsize": 11.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "grid.linestyle": "-",
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 10.5,
    "legend.fontsize": 10,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
    "lines.markersize": 6,
    "font.size": 11,
    "mathtext.fontset": "dejavusans",
}


def eps_colors(eps_values) -> list[str]:
    """Ordered single-hue colors for a list of ε values (light = small ε)."""
    n = len(eps_values)
    idx = np.round(np.linspace(0, len(_RAMP) - 1, n)).astype(int) if n > 1 else [2]
    return [_RAMP[i] for i in idx]


def eps_label(eps) -> str:
    return rf"$\varepsilon = {float(eps):g}$"


def subplots(nrows=1, ncols=1, width=11.0, height=3.8, **kw):
    """Figure and axes with the shared style and constrained layout."""
    plt.rcParams.update(RC)
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height),
                             layout="constrained", squeeze=False, **kw)
    return fig, axes


def _readable_log_ticks(fig):
    """On log axes spanning less than ~2.5 decades, label 1-2-5 ticks as plain numbers."""
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
    for ax in fig.get_axes():
        for axis, lim in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            if axis.get_scale() != "log" or not isinstance(axis.get_major_locator(), LogLocator):
                continue
            lo, hi = sorted(lim)
            if lo <= 0 or np.log10(hi / lo) > 2.5:
                continue
            axis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())


def save(fig, out_dir, name) -> Path:
    _readable_log_ticks(fig)
    out = Path(out_dir) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    # bbox "tight": nothing is clipped even where the fonts of the machine differ
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return out


def smooth(x, w=15):
    from scipy.ndimage import uniform_filter1d
    x = np.asarray(x, dtype=float)
    return uniform_filter1d(x, size=min(w, len(x)), mode="nearest") if len(x) else x


def prior_density_1d(theta):
    """Normalized marginal of ν^∞ ∝ exp(−0.05 a⁴ − 0.5 a²) (independent coordinates)."""
    fine = np.linspace(-8, 8, 4001)
    z = np.exp(-(_PRIOR_C1 * fine ** 4 + _PRIOR_C2 * fine ** 2)).sum() * (fine[1] - fine[0])
    return np.exp(-(_PRIOR_C1 * theta ** 4 + _PRIOR_C2 * theta ** 2)) / z


def kde(samples, grid):
    from scipy.stats import gaussian_kde
    samples = np.asarray(samples, dtype=float).ravel()
    if samples.size < 3 or np.ptp(samples) == 0:
        return np.zeros_like(grid)
    return gaussian_kde(samples, bw_method="scott")(grid)


def note(ax, text, loc="lower right"):
    """Small secondary-ink annotation inside the axes."""
    x, ha = (0.98, "right") if "right" in loc else (0.02, "left")
    y, va = (0.03, "bottom") if "lower" in loc else (0.97, "top")
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, color=INK_2, fontsize=10)


def class_scatter(ax, X, y, s=10, alpha=0.75, labels=("class 0", "class 1")):
    for k in (0, 1):
        m = np.asarray(y) == k
        ax.scatter(X[m, 0], X[m, 1], s=s, color=CLASS_COLOR[k], alpha=alpha,
                   linewidths=0, label=labels[k] if labels else None)


def boundary(ax, gx, gy, P, color=INK, lw=1.8, alpha=1.0, shade=True):
    """Decision probability shading (light) and the 0.5 level set."""
    if shade:
        from matplotlib.colors import LinearSegmentedColormap
        cmap = LinearSegmentedColormap.from_list("pp", ["#fbe1d6", SURFACE, "#d6e6f8"])
        ax.contourf(gx, gy, P, levels=np.linspace(0, 1, 11), cmap=cmap, alpha=0.9)
    ax.contour(gx, gy, P, levels=[0.5], colors=[color], linewidths=lw, alpha=alpha)
