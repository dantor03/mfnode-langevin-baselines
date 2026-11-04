"""Figures of experiments A–K, drawn from the arrays each experiment saves.

Every function takes the dict of arrays written by the experiment
(``figures/data/<X>.npz``) and an output directory, and returns the paths of
the PNG files it wrote.  Nothing here imports torch, so the figures can be
redrawn, restyled and tested without training anything.
"""

from __future__ import annotations

import numpy as np
from matplotlib.lines import Line2D

from .style import (
    AQUA, BLUE, CLASS_COLOR, DIM_LABELS, FINAL_COLOR, INK, INK_2, METHOD_COLOR,
    METHOD_LABEL, METHOD_MARKER, METHODS, MUTED, ORANGE, PRIOR_COLOR, SURFACE, TIMEAVG_COLOR,
    boundary, class_scatter, eps_colors, eps_label, kde, note, prior_density_1d, save,
    smooth, subplots,
)

EPS_PAIR_COLOR = {0.0: ORANGE, 0.01: BLUE}   # ε = 0 versus ε > 0 comparisons


def _f(x) -> float:
    return float(np.asarray(x))


def _square_limits(ax, pts, pad=0.4, q=(1, 99)):
    pts = np.asarray(pts).reshape(-1, 2)
    lo, hi = np.percentile(pts, q[0], axis=0), np.percentile(pts, q[1], axis=0)
    c, half = (lo + hi) / 2, (hi - lo).max() / 2 + pad
    ax.set_xlim(c[0] - half, c[0] + half)
    ax.set_ylim(c[1] - half, c[1] + half)
    ax.set_aspect("equal")


def _eps_pair_color(eps):
    return EPS_PAIR_COLOR.get(round(float(eps), 6), BLUE if eps > 0 else ORANGE)


# ── A: transport of the features ───────────────────────────────────────────────

def plot_A(d, out_dir):
    fig, ax = subplots(1, 4, width=13, height=3.95)
    ax = ax[0]
    y = d["y"]
    T, t_mid = _f(d["T"]), _f(d["t_mid"])
    for a, key, title in ((ax[0], "x_t0", r"$\gamma_0$: input data"),
                          (ax[1], "x_tmid", rf"$\gamma_{{T/2}}$ at $t = {t_mid:g}$"),
                          (ax[2], "x_T", rf"$\gamma_T$ at $t = T = {T:g}$")):
        class_scatter(a, d[key], y, labels=None if key != "x_t0" else ("class 0", "class 1"))
        a.set_title(title)
        a.set_xlabel("$x_1$")
        _square_limits(a, d[key])
    ax[0].set_ylabel("$x_2$")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside upper right", ncols=2,
               markerscale=1.6)

    for path, lab in zip(d["paths"], d["path_labels"]):
        ax[2].plot(path[:, 0], path[:, 1], color=CLASS_COLOR[int(lab)], lw=0.8, alpha=0.45)
    _square_limits(ax[2], np.concatenate([d["x_T"], d["paths"].reshape(-1, 2)]))
    w, b = d["w"], _f(d["b"])
    if abs(w[1]) > 1e-12:
        xs = np.array(ax[2].get_xlim())
        ax[2].plot(xs, -(w[0] * xs + b) / w[1], color=INK, lw=1.4, ls="--")
    note(ax[2], "dashed: linear read-out\nlines: 40 trajectories", "lower right")

    boundary(ax[3], d["grid_x"], d["grid_y"], d["grid_p"])
    class_scatter(ax[3], d["x_t0"], y, s=7, alpha=0.6, labels=None)
    ax[3].set_title(f"Boundary, accuracy {_f(d['acc']):.3f}")
    ax[3].set_xlabel("$x_1$")
    _square_limits(ax[3], d["x_t0"])
    fig.suptitle(rf"A · Features transported by the mean-field ODE "
                 rf"($\varepsilon = {_f(d['eps']):g}$, $M = {int(d['M'])}$ neurons)",
                 x=0.01, ha="left", fontsize=13)
    return [save(fig, out_dir, "A_feature_evolution.png")]


# ── B: effect of ε ─────────────────────────────────────────────────────────────

def plot_B(d, out_dir):
    eps = d["eps"]
    cols = eps_colors(eps)
    fig, ax = subplots(1, 3, width=13, height=3.8)
    ax = ax[0]
    for k, e in enumerate(eps):
        ax[0].semilogy(d["loss"][k], color=cols[k], label=eps_label(e))
        ax[1].plot(smooth(d["acc"][k]), color=cols[k])
        ax[2].semilogy(np.maximum(d["reg"][k], 1e-12), color=cols[k])
    ax[0].set(title="Total cost $J$", xlabel="epoch", ylabel="$J$ (log scale)")
    ax[1].set(title="Training accuracy (moving average)", xlabel="epoch", ylabel="accuracy",
              ylim=(0.45, 1.02))
    ax[2].set(title="Energy of the prior per parameter", xlabel="epoch",
              ylabel=r"$\mathcal{E}/N_{\mathrm{params}}$ (log scale)")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="outside right center")
    fig.suptitle("B · Convergence of pSGLD on make_moons for five values of ε",
                 x=0.01, ha="left", fontsize=13)
    out1 = save(fig, out_dir, "B1_convergence_curves.png")

    n = len(eps)
    fig, ax = subplots(1, n, width=13, height=4.0, sharex=True, sharey=True)
    ax = ax[0]
    speed = np.hypot(d["U"], d["V"])
    scale = speed.max() * 18 if speed.max() > 0 else 1.0
    for k, e in enumerate(eps):
        boundary(ax[k], d["grid_x"], d["grid_y"], d["grid_p"][k])
        class_scatter(ax[k], d["X"], d["y"], s=5, alpha=0.5, labels=None)
        ax[k].quiver(d["qx"], d["qy"], d["U"][k], d["V"][k], color=INK_2, scale=scale,
                     width=0.006, alpha=0.8)
        ax[k].set_title(f"{eps_label(e)}\naccuracy {_f(d['final_acc'][k]):.3f}", fontsize=11.5)
        ax[k].set_aspect("equal")
        ax[k].set_xlabel("$x_1$")
    ax[0].set_ylabel("$x_2$")
    ax[0].set_xlim(d["grid_x"].min(), d["grid_x"].max())
    ax[0].set_ylim(d["grid_y"].min(), d["grid_y"].max())
    fig.suptitle(r"B · Decision boundaries (black) and velocity field $F(x, t = 0.5)$ "
                 "(arrows, common scale)", x=0.01, ha="left", fontsize=13)
    out2 = save(fig, out_dir, "B2_decision_boundaries.png")
    return [out1, out2]


# ── C: Polyak–Łojasiewicz inequality ───────────────────────────────────────────

def plot_C(d, out_dir):
    eps, mu = d["eps"], d["mu"]
    cols = eps_colors(eps)
    pos = mu[mu > 0]
    mu_min = float(pos.min()) if pos.size else 0.0
    fig, ax = subplots(1, 3, width=13, height=3.9)
    ax = ax[0]
    for k, e in enumerate(eps):
        excess = d["loss"][k] - d["J_star"][k]
        gn2 = d["grad_norm2"][k]
        ok = (excess > 5e-5) & (gn2 > 1e-10)
        ax[0].scatter(excess[ok], gn2[ok], s=6, color=cols[k], alpha=0.35, linewidths=0)
        ax[1].semilogy(np.maximum(excess, 1e-10), color=cols[k],
                       label=rf"{eps_label(e)}, $\hat\mu = {mu[k]:.3g}$")
        pl = d["pl_ratio"][k]
        idx = np.nonzero(np.isfinite(pl) & (pl > 0))[0]
        ax[2].semilogy(idx, pl[idx], color=cols[k], lw=1.4)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    xl = np.logspace(-5, 0, 100)
    if mu_min > 0:
        ax[0].plot(xl, 2 * mu_min * xl, color=INK, lw=1.6)
        ax[2].axhline(mu_min, color=INK, lw=1.6)
    ax[0].set(title=r"PL inequality: $\|\nabla J\|^2$ against $J - J^*$",
              xlabel=r"$J(\theta) - J^*$", ylabel=r"$\|\nabla J(\theta)\|^2$")
    ax[1].set(title="Excess cost (SGD, constant step)", xlabel="epoch",
              ylabel=r"$J(\theta_s) - J^*$")
    ax[2].set(title=r"PL ratio $\|\nabla J\|^2 / (2(J - J^*))$", xlabel="epoch",
              ylabel="ratio (log scale)")
    h, lab = ax[1].get_legend_handles_labels()
    if mu_min > 0:
        h.append(Line2D([], [], color=INK, lw=1.6))
        lab.append(rf"black: $\hat\mu_{{\min}} = {mu_min:.3g}$" "\n"
                   r"(left: $2\hat\mu_{\min}(J - J^*)$)")
    fig.legend(h, lab, loc="outside right center")
    fig.suptitle(r"C · Empirical PL constant $\hat\mu$ (10th percentile of the ratio) "
                 "along gradient descent", x=0.01, ha="left", fontsize=13)
    return [save(fig, out_dir, "C_pl_verification.png")]


# ── D: genericity ──────────────────────────────────────────────────────────────

_D_TITLES = {"d1": "D1 · data fixed, initialization varies",
             "d2": "D2 · initialization fixed, data vary",
             "d3": "D3 · both vary"}


def plot_D(d, out_dir):
    eps = d["eps"]
    fig, ax = subplots(2, 3, width=13, height=7.4)
    for j, p in enumerate(("d1", "d2", "d3")):
        a = ax[0, j]
        for k, e in enumerate(eps):
            L = d[f"loss_{p}"][k]
            col = _eps_pair_color(e)
            med = np.median(L, axis=0)
            a.fill_between(np.arange(L.shape[1]), L.min(axis=0), L.max(axis=0),
                           color=col, alpha=0.15, linewidth=0)
            js = d[f"J_star_{p}"][k]
            a.semilogy(med, color=col,
                       label=rf"{eps_label(e)}: $J^* = {js.mean():.2g} \pm {js.std():.1g}$")
        a.set(title=_D_TITLES[p], xlabel="epoch")
        a.legend(loc="upper right", fontsize=9.5)
        if j == 0:
            a.set_ylabel("$J$: median and range over seeds")

        b = ax[1, j]
        X, Y = d[f"X_{p}"], d[f"y_{p}"]
        for s in range(X.shape[0]):
            class_scatter(b, X[s], Y[s], s=4, alpha=0.1, labels=None)
        acc = d[f"acc_{p}"][-1]
        good = np.nonzero(acc >= 0.95)[0]
        for s in good:
            boundary(b, d[f"grid_x_{p}"], d[f"grid_y_{p}"], d[f"P_{p}"][s],
                     color=BLUE, lw=1.2, alpha=0.7, shade=False)
        b.set_title(rf"Boundaries for {eps_label(eps[-1])} "
                    f"({len(good)} of {len(acc)} runs)", fontsize=11.5)
        b.set_xlabel("$x_1$")
        b.set_aspect("equal")
        b.set_xlim(d[f"grid_x_{p}"].min(), d[f"grid_x_{p}"].max())
        b.set_ylim(d[f"grid_y_{p}"].min(), d[f"grid_y_{p}"].max())
        if j == 0:
            b.set_ylabel("$x_2$")
    note(ax[1, 2], "runs with accuracy ≥ 0.95", "lower right")
    fig.suptitle(f"D · Is the minimizer generic? {d['loss_d1'].shape[1]} seeds per protocol",
                 x=0.01, ha="left", fontsize=13)
    return [save(fig, out_dir, "D_genericity.png")]


# ── E: robustness of the neuron importances ────────────────────────────────────

def plot_E(d, out_dir):
    fig, ax = subplots(1, 2, width=11, height=3.8, sharey=True)
    ax = ax[0]
    for a, key, title in ((ax[0], "imp_e1", f"E1 · data fixed (seed {int(d['data_seed'])}), "
                                             "initialization varies"),
                          (ax[1], "imp_e2", f"E2 · initialization fixed (seed "
                                             f"{int(d['init_seed'])}), data vary")):
        imp = d[key]
        r = np.arange(1, imp.shape[1] + 1)
        for row in imp:
            a.plot(r, row, color=BLUE, lw=0.9, alpha=0.3)
        a.plot(r, imp.mean(axis=0), color=INK, lw=2.0)
        a.set(title=title, xlabel="neuron rank")
        a.set_yscale("log")
    ax[0].set_ylabel(r"importance $\|a_0^m\|_2$ (log scale)")
    ax[0].legend(handles=[Line2D([], [], color=BLUE, lw=1, alpha=0.6, label="one seed"),
                          Line2D([], [], color=INK, lw=2, label="mean over seeds")],
                 loc="upper right")
    fig.suptitle(rf"E · Sorted neuron importances across {d['imp_e1'].shape[0]} seeds "
                 rf"($\varepsilon = {_f(d['eps']):g}$, pSGLD)",
                 x=0.01, ha="left", fontsize=13)
    return [save(fig, out_dir, "E_parameter_robustness.png")]


# ── F: make_circles and the SO(2) symmetry ─────────────────────────────────────

def plot_F(d, out_dir):
    fig, ax = subplots(1, 3, width=13, height=3.9)
    ax = ax[0]
    for key, col, lab in (("loss_f1", ORANGE, "F1 · data vary"),
                          ("loss_f2", BLUE, "F2 · initialization varies")):
        L = d[key]
        for row in L:
            ax[0].plot(row, color=col, lw=0.6, alpha=0.2)
        tail = L[:, -50:].mean(axis=1)
        ax[0].plot(L.mean(axis=0), color=col,
                   label=rf"{lab}: ${tail.mean():.3g} \pm {tail.std():.1g}$")
    ax[0].set_yscale("log")
    ax[0].set(title=f"Total cost $J$ ({d['loss_f1'].shape[0]} runs each)", xlabel="epoch",
              ylabel="$J$ (log scale)")
    ax[0].legend(loc="upper right", title="mean of the last 50 epochs", fontsize=9.5,
                 title_fontsize=9.5)

    a1 = d["a1_f2"].reshape(-1, 2)
    imp = np.linalg.norm(d["a0_f2"].reshape(-1, 2), axis=1)
    order = np.argsort(imp)
    sc = ax[1].scatter(a1[order, 0], a1[order, 1], c=imp[order], cmap="Blues", s=14,
                       vmin=0, linewidths=0)
    fig.colorbar(sc, ax=ax[1], label=r"$\|a_0^m\|_2$", shrink=0.85)
    lim = np.percentile(np.abs(a1), 99) * 1.1
    ax[1].set(xlim=(-lim, lim), ylim=(-lim, lim), title=r"Input weights $a_1^m$ (F2, all runs)",
              xlabel=r"$a_1^{(0)}$", ylabel=r"$a_1^{(1)}$")
    ax[1].set_aspect("equal")

    bins = np.linspace(-np.pi, np.pi, 25)
    for key0, key1, col, lab in (("a0_f1", "a1_f1", ORANGE, "F1"), ("a0_f2", "a1_f2", BLUE, "F2")):
        v = d[key1].reshape(-1, 2)
        w = np.linalg.norm(d[key0].reshape(-1, 2), axis=1)
        ax[2].hist(np.arctan2(v[:, 1], v[:, 0]), bins=bins, weights=w, density=True,
                   histtype="step", color=col, lw=2, label=lab)
    ax[2].axhline(1 / (2 * np.pi), color=MUTED, lw=1.4, label="uniform")
    ax[2].set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi],
                     [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    ax[2].set(title=r"Direction of $a_1^m$, weighted by $\|a_0^m\|_2$",
              xlabel="angle", ylabel="density")
    ax[2].set_ylim(0, 1.35 * ax[2].get_ylim()[1])
    ax[2].legend(loc="upper left", ncols=3)
    fig.suptitle(rf"F · make_circles: does $\nu^*$ inherit the rotational symmetry of the data? "
                 rf"($\varepsilon = {_f(d['eps']):g}$)", x=0.01, ha="left", fontsize=13)
    return [save(fig, out_dir, "F_circles_parameter_distribution.png")]


# ── G: J*_N → J*_∞ ─────────────────────────────────────────────────────────────

def plot_G(d, out_dir):
    N, eps = d["N"].astype(float), d["eps"]
    fig, ax = subplots(1, 2, width=11, height=3.9)
    ax = ax[0]
    mask = N >= 100
    for k, e in enumerate(eps):
        col = _eps_pair_color(e)
        te, tr = d["bce_test"][k], d["bce_train"][k]
        ax[0].plot(N, te.mean(1), color=col, marker="o", label=f"{eps_label(e)}, test")
        ax[0].fill_between(N, te.mean(1) - te.std(1), te.mean(1) + te.std(1),
                           color=col, alpha=0.15, linewidth=0)
        ax[0].plot(N, tr.mean(1), color=col, marker="o", ls="--", mfc="white",
                   label=f"{eps_label(e)}, train")
        gap = te.mean(1) - tr.mean(1)
        ok = mask & (gap > 0)
        ax[1].scatter(N[ok], gap[ok], color=col, s=36, zorder=3)
        if ok.sum() >= 2:
            slope, icpt = np.polyfit(np.log(N[ok]), np.log(gap[ok]), 1)
            nf = np.geomspace(N[ok].min(), N[ok].max(), 50)
            ax[1].plot(nf, np.exp(icpt) * nf ** slope, color=col,
                       label=rf"{eps_label(e)}: gap $\propto N^{{{slope:.2f}}}$")
    if mask.any():
        ref_n = np.geomspace(N[mask].min(), N[mask].max(), 50)
        g0 = d["bce_test"][0].mean(1) - d["bce_train"][0].mean(1)
        start = g0[mask][0] if g0[mask][0] > 0 else 1e-2
        ax[1].plot(ref_n, start * (ref_n / ref_n[0]) ** -0.5, color=MUTED, lw=1.4,
                   label=r"$N^{-1/2}$ reference")
    for a in ax:
        a.set_xscale("log")
        a.set_xticks(N, [str(int(n)) for n in N])
        a.minorticks_off()
        a.set_xlabel("training set size $N$")
    ax[1].set_xticks(N[mask], [str(int(n)) for n in N[mask]])
    ax[1].set_yscale("log")
    ax[0].set(title="Cross-entropy on train and on a 10,000-point test set", ylabel="BCE")
    ax[1].set(title="Generalization gap (test − train), $N \\geq 100$", ylabel="gap (log scale)")
    ax[0].legend(loc="upper right", fontsize=9.5)
    ax[1].legend(loc="lower left", fontsize=9.5)
    fig.suptitle(rf"G · Convergence of the empirical problem as $N \to \infty$ "
                 rf"(mean ± sd over {d['bce_test'].shape[2]} seeds)",
                 x=0.01, ha="left", fontsize=13)
    return [save(fig, out_dir, "G_convergence_problem.png")]


# ── H: does pSGLD sample ν*? ───────────────────────────────────────────────────

def plot_H(d, out_dir):
    import matplotlib.pyplot as plt
    from .style import RC
    plt.rcParams.update(RC)
    fig = plt.figure(figsize=(13, 7.4), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.05])
    marg = top.subplots(1, 5, sharey=True)
    ax_mmd, ax_w1 = bottom.subplots(1, 2, width_ratios=[2, 3])

    grid = np.linspace(-3.5, 3.5, 600)
    prior = prior_density_1d(grid)
    dens = [kde(d["timeavg"][:, j], grid) for j in range(5)]
    ymax = max(prior.max(), *(x.max() for x in dens))
    for j, a in enumerate(marg):
        a.plot(grid, prior, color=PRIOR_COLOR, lw=1.6, ls="--", label=r"prior $\nu^\infty$")
        a.fill_between(grid, dens[j], color=TIMEAVG_COLOR, alpha=0.15, linewidth=0)
        a.plot(grid, dens[j], color=TIMEAVG_COLOR, label=r"time average $\hat\nu^*$")
        a.plot(d["final"][:, j], np.full(len(d["final"]), -0.05 * ymax), "|",
               color=FINAL_COLOR, ms=9, mew=1.2, label="final particles")
        a.set_title(DIM_LABELS[j])
        a.set_xlim(-3, 3)
    marg[0].set_ylim(-0.1 * ymax, 1.1 * ymax)
    marg[0].set_ylabel("density")
    top.legend(*marg[0].get_legend_handles_labels(), loc="outside upper right", ncols=3)
    top.suptitle(rf"H · pSGLD particles against their time average and the prior "
                 rf"($\varepsilon = {_f(d['eps']):g}$, $M = {d['final'].shape[0]}$)",
                 x=0.01, ha="left", fontsize=13)

    names = [r"final particles vs $\hat\nu^*$", r"$\hat\nu^*$ vs prior",
             "final particles vs prior"]
    colors = [FINAL_COLOR, TIMEAVG_COLOR, AQUA]
    vals = np.asarray(d["mmd"], dtype=float)
    ypos = np.arange(3)[::-1]
    ax_mmd.barh(ypos, np.maximum(vals, 0), color=colors, height=0.55)
    ax_mmd.set_yticks(ypos, names)
    ax_mmd.grid(axis="y", visible=False)
    span = max(np.abs(vals).max(), 1e-12)
    for yy, v in zip(ypos, vals):
        txt = f"{v:.2g}" if v > 0.02 * span else f"≈ 0 ({v:.1e})"
        ax_mmd.text(max(v, 0) + 0.02 * span, yy, txt, va="center", color=INK, fontsize=10.5)
    ax_mmd.set_xlim(0, 1.35 * span)
    ax_mmd.set(title="MMD² between the three distributions", xlabel="MMD²")

    x = np.arange(5)
    w = 0.27
    for k in range(3):
        ax_w1.bar(x + (k - 1) * w, d["w1"][k], w * 0.92, color=colors[k], label=names[k])
    ax_w1.set_xticks(x, DIM_LABELS)
    ax_w1.grid(axis="x", visible=False)
    ax_w1.set_ylim(0, 1.3 * float(np.max(d["w1"])))
    ax_w1.set(title="Wasserstein-1 distance per coordinate", ylabel="$W_1$")
    ax_w1.legend(loc="upper left", ncols=3, fontsize=9.5)
    return [save(fig, out_dir, "H_psgld_diagnostic.png")]


# ── I and J: explicit regularizers versus pSGLD ────────────────────────────────

def _w1_bars(a, w1, names, mmd, title):
    x = np.arange(5)
    w = 0.8 / len(names)
    for k, m in enumerate(names):
        lab = "pSGLD final particles (sampling floor)" if m == "pSGLD" else METHOD_LABEL[m]
        a.bar(x + (k - (len(names) - 1) / 2) * w, w1[k], w * 0.92, color=METHOD_COLOR[m],
              label=rf"{lab}: MMD² {float(mmd[k]):.1e}")
    a.set_xticks(x, DIM_LABELS)
    a.grid(axis="x", visible=False)
    a.set(title=title, ylabel="$W_1$")
    a.legend(loc="upper left", fontsize=9.5)
    return float(np.max(w1))


def _curves(a, curves, ylabel, title, log=True, w=20):
    n = curves.shape[1]
    for k, m in enumerate(METHODS):
        a.plot(smooth(curves[k], w), color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
               markevery=max(n // 7, 1), label=METHOD_LABEL[m])
    if log:
        a.set_yscale("log")
    a.set(title=title, xlabel="epoch", ylabel=ylabel)


def plot_I(d, out_dir):
    fig, ax = subplots(2, 2, width=12, height=7.6)
    _curves(ax[0, 0], np.maximum(d["bce"], 1e-8), "BCE (log scale)", "Training cross-entropy")
    ax[0, 0].legend(loc="upper right")

    grid = np.linspace(-3, 3, 500)
    a = ax[0, 1]
    a.plot(grid, prior_density_1d(grid), color=PRIOR_COLOR, ls="--", lw=1.6,
           label=r"prior $\nu^\infty$")
    ref = kde(d["ref"][:, 0], grid)
    a.fill_between(grid, ref, color=BLUE, alpha=0.15, linewidth=0)
    a.plot(grid, ref, color=BLUE, label=r"pSGLD time average $\hat\nu^*$")
    for m, ls in (("MMD", "-"), ("Sinkhorn", "--")):
        k = METHODS.index(m)
        a.plot(grid, kde(d["final"][k][:, 0], grid), color=METHOD_COLOR[m], ls=ls,
               label=f"{METHOD_LABEL[m]}, trained particles")
    a.set(title=r"Marginal of $a_0^{(0)}$", xlabel=r"$a_0^{(0)}$", ylabel="density")
    a.set_ylim(0, 1.55 * a.get_ylim()[1])
    a.legend(loc="upper left", fontsize=9.5, ncols=2)

    t1 = _w1_bars(ax[1, 0], d["w1_final"], METHODS, d["mmd_final"],
                  r"Trained particles: $W_1$ to $\hat\nu^*$ of pSGLD")
    t2 = _w1_bars(ax[1, 1], d["w1"], ("MMD", "Sinkhorn"), d["mmd"],
                  rf"After {int(d['n_snapshot_steps'])} more pSGLD steps from each solution")
    top = 1.5 * max(t1, t2, 1e-6)
    for a in (ax[1, 0], ax[1, 1]):
        a.set_ylim(0, top)
    fig.suptitle(rf"I · Can an explicit MMD or Sinkhorn penalty replace the Langevin noise? "
                 rf"(make_moons, $\varepsilon = {_f(d['eps']):g}$)", x=0.01, ha="left",
                 fontsize=13)
    return [save(fig, out_dir, "I_regularizer_comparison.png")]


def plot_J(d, out_dir):
    fig, ax = subplots(1, 3, width=13, height=3.9)
    ax = ax[0]
    _curves(ax[0], d["mse_curve"], "MSE (log scale)", "Training MSE (★: final test MSE)")
    n = d["mse_curve"].shape[1]
    for k, m in enumerate(METHODS):
        ax[0].scatter([n - 1], [d["mse_te"][k]], marker="*", s=160, color=METHOD_COLOR[m],
                      edgecolors="white", linewidths=0.8, zorder=4)
    ax[0].legend(loc="upper right")

    x = np.arange(3)
    w = 0.36
    for k, m in enumerate(METHODS):
        ax[1].bar(x[k] - w / 2, d["r2_tr"][k], w * 0.92, color=METHOD_COLOR[m], alpha=0.45)
        ax[1].bar(x[k] + w / 2, d["r2_te"][k], w * 0.92, color=METHOD_COLOR[m])
        ax[1].text(x[k] + w / 2, d["r2_te"][k], f"{d['r2_te'][k]:.3f}", ha="center",
                   va="bottom", fontsize=10, color=INK)
    ax[1].set_xticks(x, [METHOD_LABEL[m].replace(" regularizer", "") for m in METHODS])
    ax[1].grid(axis="x", visible=False)
    ax[1].set(title="$R^2$: train (light) and test (solid)", ylabel="$R^2$")
    lo = min(0.0, float(np.min([d["r2_tr"], d["r2_te"]])))
    ax[1].set_ylim(lo, max(float(np.max([d["r2_tr"], d["r2_te"]])) * 1.2, 0.05))

    y = d["y_te"]
    lims = [float(y.min()) - 0.3, float(y.max()) + 0.3]
    ax[2].plot(lims, lims, color=MUTED, lw=1.2)
    for k, m in enumerate(METHODS):
        ax[2].scatter(y, d["pred_te"][k], s=9, alpha=0.4, color=METHOD_COLOR[m],
                      marker=METHOD_MARKER[m], linewidths=0, label=METHOD_LABEL[m])
    ax[2].set(xlim=lims, ylim=lims, title="Test set: prediction against target",
              xlabel="target (standardized)", ylabel="prediction")
    ax[2].set_aspect("equal")
    leg = ax[2].legend(loc="upper left", fontsize=9.5, markerscale=1.8)
    for h in getattr(leg, "legend_handles", None) or leg.legendHandles:
        h.set_alpha(1)
    fig.suptitle(rf"J · California Housing projected to $\mathbb{{R}}^2$ by PCA "
                 rf"({_f(d['explained_var']):.0%} of the variance), "
                 rf"$\varepsilon = {_f(d['eps']):g}$", x=0.01, ha="left", fontsize=13)
    return [save(fig, out_dir, "J_california_regression.png")]


# ── K: ε sweep on California Housing ───────────────────────────────────────────

def plot_K(d, out_dir):
    eg = d["eps_grid"].astype(float)
    vals = d["r2_te"]
    fig, ax = subplots(1, 3, width=13, height=4.1)
    ax = ax[0]

    floor = -1.0       # R² below this is a failed fit: drawn off scale, value printed
    finite = vals[np.isfinite(vals)]
    on_scale = finite[finite >= floor]
    off = vals < floor
    # off-scale values sit on a line of their own, clearly below every plotted one
    lo = (float(on_scale.min()) if on_scale.size else floor) - (0.08 if off.any() else 0.03)
    hi = float(finite.max()) + 0.04
    for k, m in enumerate(METHODS):
        v = vals[k]
        ax[0].plot(eg, np.where(off[k], lo, v), color=METHOD_COLOR[m], marker=METHOD_MARKER[m],
                   label=METHOD_LABEL[m])
        for n_off, j in enumerate(np.nonzero(off[k])[0]):
            ax[0].plot(eg[j], lo, marker="v", ms=11, color=METHOD_COLOR[m], mec=SURFACE, mew=1.5)
            # printed right under its marker, in the space kept free below the curves
            ax[0].annotate(rf"$R^2 = {v[j]:.1f}$ at $\varepsilon = {eg[j]:g}$",
                           (eg[j], lo), xytext=(6, -12 - 13 * (k + n_off)),
                           textcoords="offset points", ha="right", va="top",
                           fontsize=9.5, color=METHOD_COLOR[m])
    ax[0].set_xscale("log")
    ax[0].set_ylim(lo - (0.12 if off.any() else 0.03), hi)
    title = r"Test $R^2$ against $\varepsilon$"
    if off.any():
        title += r" ($\blacktriangledown$: off scale)"
    ax[0].set(title=title, xlabel=r"$\varepsilon$", ylabel="$R^2$ (test)")
    ax[0].legend(loc="lower left", fontsize=9.5)

    for k, m in enumerate(METHODS):
        ax[1].plot(eg, np.maximum(d["reg_frac"][k], 1e-8), color=METHOD_COLOR[m],
                   marker=METHOD_MARKER[m])
    for level, txt in ((1.0, "regularizer = MSE"), (0.1, "regularizer = 10% of MSE")):
        ax[1].axhline(level, color=MUTED, lw=1.0)
        ax[1].text(eg.min(), level * 1.15, txt, color=INK_2, fontsize=9.5)
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set(title=r"Weight of the regularizer: $\varepsilon \cdot$reg / MSE",
              xlabel=r"$\varepsilon$", ylabel="ratio (log scale)")

    eps_orig = _f(d["eps_orig"])
    i0 = int(np.argmin(np.abs(eg - eps_orig)))
    ish = int(np.argmin(np.abs(eg - _f(d["best_shared"]))))
    ypos = np.arange(3)[::-1]
    a = ax[2]
    for k, m in enumerate(METHODS):
        ib = int(np.argmax(vals[k]))
        yy = ypos[k]
        xs = [vals[k][i0], vals[k][ish], vals[k][ib]]
        a.plot([min(xs), max(xs)], [yy, yy], color=MUTED, lw=1.2, zorder=1)
        a.scatter(vals[k][i0], yy, s=110, facecolors="white", edgecolors=INK_2, lw=1.4,
                  zorder=3)
        a.scatter(vals[k][ish], yy, s=40, marker="s", color=INK_2, zorder=4)
        a.scatter(vals[k][ib], yy, s=70, marker="D", color=METHOD_COLOR[m], zorder=5)
        a.annotate(rf"best $\varepsilon = {eg[ib]:g}$", (vals[k][ib], yy), xytext=(0, 10),
                   textcoords="offset points", ha="center", fontsize=9.5, color=INK_2)
    a.set_yticks(ypos, [METHOD_LABEL[m].replace(" regularizer", "") for m in METHODS])
    a.set_ylim(-0.6, 2.6)
    x_all = np.concatenate([vals[:, [i0, ish]].ravel(), vals.max(axis=1)])
    pad = max(0.15 * np.ptp(x_all), 0.01)
    a.set_xlim(x_all.min() - pad, x_all.max() + pad)
    a.grid(axis="y", visible=False)
    a.set(title="Test $R^2$ for three choices of ε", xlabel="$R^2$ (test)")
    handles = [
        Line2D([], [], ls="", marker="o", mfc="white", mec=INK_2, ms=9,
               label=rf"$\varepsilon = {eps_orig:g}$ (as in J)"),
        Line2D([], [], ls="", marker="s", color=INK_2, ms=6,
               label=rf"best common $\varepsilon = {eg[ish]:g}$"),
        Line2D([], [], ls="", marker="D", color=INK_2, ms=7, label="best ε per method")]
    fig.legend(handles=handles, loc="outside lower right", ncols=3, fontsize=9.5)
    fig.suptitle(rf"K · ε sweep on California Housing "
                 rf"({_f(d['explained_var']):.0%} of the variance in ℝ²)",
                 x=0.01, ha="left", fontsize=13)
    return [save(fig, out_dir, "K_epsilon_sweep.png")]


PLOTS = {"A": plot_A, "B": plot_B, "C": plot_C, "D": plot_D, "E": plot_E, "F": plot_F,
         "G": plot_G, "H": plot_H, "I": plot_I, "J": plot_J, "K": plot_K}
