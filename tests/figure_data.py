"""Synthetic data with the same keys and shapes that experiments A–K save.

Used by ``test_figures.py`` to draw every figure without torch or training.
The numbers are made up but have realistic ranges, so a rendered figure can
also be inspected by eye.
"""

import numpy as np

E = 120          # epochs
M = 64           # particles


def _moons(n=300, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(0, np.pi, n)
    y = np.arange(n) % 2
    X = np.where(y[:, None] == 0, np.c_[np.cos(t), np.sin(t)],
                 np.c_[1 - np.cos(t), 0.5 - np.sin(t)])
    X = X + 0.12 * rng.standard_normal((n, 2))
    return (X - X.mean(0)) / X.std(0), y


def _grid(X, n=60, pad=0.5):
    gx = np.linspace(X[:, 0].min() - pad, X[:, 0].max() + pad, n)
    gy = np.linspace(X[:, 1].min() - pad, X[:, 1].max() + pad, n)
    xx, yy = np.meshgrid(gx, gy)
    P = 1 / (1 + np.exp(-4 * (yy - 0.6 * np.sin(1.6 * xx))))
    return gx, gy, P


def _decay(rng, start, end, rate, noise=0.05, n=E):
    t = np.arange(n)
    return (end + (start - end) * np.exp(-rate * t)) * np.exp(noise * rng.standard_normal(n))


def _particles(rng, n, scale=1.0):
    return rng.standard_normal((n, 5)) * scale * np.array([1.4, 1.1, 0.9, 1.2, 0.8])


def make(name, seed=0):
    rng = np.random.default_rng(seed)
    X, y = _moons()
    gx, gy, P = _grid(X)
    if name == "A":
        xT = X + np.c_[np.zeros(len(X)), 1.5 * (y - 0.5)]
        xm = (X + xT) / 2
        idx = np.r_[np.nonzero(y == 0)[0][:20], np.nonzero(y == 1)[0][:20]]
        paths = np.stack([X[idx] + s * (xT[idx] - X[idx]) for s in np.linspace(0, 1, 11)], 1)
        return dict(y=y, T=1.0, t_mid=0.5, x_t0=X, x_tmid=xm, x_T=xT, paths=paths,
                    path_labels=y[idx], w=np.array([0.0, 1.0]), b=0.0, grid_x=gx, grid_y=gy,
                    grid_p=P, acc=0.998, eps=0.01, M=M)
    if name == "B":
        eps = np.array([0.0, 0.001, 0.01, 0.1, 0.5])
        k = len(eps)
        v = np.linspace(-2.5, 2.5, 16)
        qx, qy = np.meshgrid(v, v)
        return dict(eps=eps,
                    loss=np.stack([_decay(rng, 0.7, 0.01 + e, 0.04) for e in eps]),
                    acc=np.stack([np.clip(1 - 0.5 * np.exp(-0.05 * np.arange(E))
                                          + 0.01 * rng.standard_normal(E), 0, 1)
                                  for _ in eps]),
                    reg=np.stack([_decay(rng, 0.2, 0.05 / (1 + 10 * e), 0.02) for e in eps]),
                    grid_x=gx, grid_y=gy, grid_p=np.stack([P] * k), X=X, y=y, qx=qx, qy=qy,
                    U=np.stack([np.sin(qy) / (1 + e) for e in eps]),
                    V=np.stack([np.cos(qx) / (1 + e) for e in eps]),
                    final_acc=np.array([1.0, 1.0, 0.998, 0.99, 0.95]))
    if name == "C":
        eps = np.array([0.0, 0.001, 0.01, 0.1, 0.5])
        loss = np.stack([_decay(rng, 0.7, 0.02 + e / 5, 0.03, noise=0.0) for e in eps])
        J_star = loss.min(1)
        excess = loss - J_star[:, None]
        gn2 = 2 * 0.02 * excess * np.exp(0.5 * rng.standard_normal(loss.shape)) + 1e-9
        pl = np.where(excess > 1e-9, gn2 / (2 * np.maximum(excess, 1e-10)), np.nan)
        return dict(eps=eps, mu=np.array([0.02, 0.018, 0.015, 0.009, 0.004]), loss=loss,
                    J_star=J_star, grad_norm2=gn2, pl_ratio=pl,
                    acc_final=np.ones(len(eps)))
    if name == "D":
        S = 4
        out = dict(eps=np.array([0.0, 0.01]))
        for p in ("d1", "d2", "d3"):
            out[f"loss_{p}"] = np.stack([[_decay(rng, 0.7, 0.01 + e, 0.04) for _ in range(S)]
                                         for e in (0.0, 0.01)])
            out[f"J_star_{p}"] = out[f"loss_{p}"].min(-1)
            out[f"mu_{p}"] = rng.uniform(0.001, 0.01, (2, S))
            out[f"acc_{p}"] = np.array([[1.0, 0.99, 0.97, 0.9]] * 2)
            Xs = np.stack([_moons(seed=s)[0] for s in range(S)])
            out[f"X_{p}"] = Xs
            out[f"y_{p}"] = np.stack([_moons(seed=s)[1] for s in range(S)])
            g = _grid(Xs.reshape(-1, 2))
            out[f"grid_x_{p}"], out[f"grid_y_{p}"] = g[0], g[1]
            out[f"P_{p}"] = np.stack([g[2]] * S)
        return out
    if name == "E":
        S = 5
        base = np.sort(np.abs(rng.standard_normal(M)))[::-1] + 0.05
        return dict(eps=0.01, data_seed=42, init_seed=2,
                    imp_e1=np.stack([base * rng.uniform(0.8, 1.2, M) for _ in range(S)]),
                    imp_e2=np.stack([base * rng.uniform(0.8, 1.2, M) for _ in range(S)]))
    if name == "F":
        S = 5
        out = dict(eps=0.01)
        for p in ("f1", "f2"):
            out[f"loss_{p}"] = np.stack([_decay(rng, 0.7, 0.05, 0.03) for _ in range(S)])
            ang = rng.uniform(-np.pi, np.pi, (S, M))
            r = rng.gamma(4, 0.3, (S, M))
            out[f"a1_{p}"] = np.stack([r * np.cos(ang), r * np.sin(ang)], -1)
            out[f"a0_{p}"] = rng.standard_normal((S, M, 2))
            out[f"acc_{p}"] = np.full(S, 0.99)
        return out
    if name == "G":
        N = np.array([50, 100, 200, 400, 800, 1600, 3200])
        S = 5
        out = dict(N=N, eps=np.array([0.0, 0.01]))
        tr, te = [], []
        for e in (0.0, 0.01):
            gap = 0.6 * (N / 50.0) ** -0.55 * (1 - 10 * e)
            base = 0.02 + 5 * e
            tr.append(base + 0.01 * rng.random((len(N), S)))
            te.append(base + gap[:, None] + 0.02 * rng.random((len(N), S)))
        out.update(bce_train=np.array(tr), bce_test=np.array(te),
                   acc_test=np.full((2, len(N), S), 0.99),
                   J_star_train=np.array(tr) * 0.9)
        return out
    if name == "H":
        final = _particles(rng, M)
        timeavg = _particles(rng, 6400)
        return dict(eps=0.01, final=final, timeavg=timeavg,
                    mmd=np.array([-0.0012, 0.21, 0.19]),
                    w1=np.abs(rng.normal(0.2, 0.1, (3, 5))) * np.array([[0.4], [1], [1]]),
                    n_snapshot_steps=2000)
    if name == "I":
        bce = np.stack([_decay(rng, 0.7, f, 0.04) for f in (1e-3, 4e-3, 2e-3)])
        return dict(eps=0.01, bce=bce, reg=bce * 0.1,
                    final=np.stack([_particles(rng, M, s) for s in (1.0, 0.7, 0.8)]),
                    ref=_particles(rng, 6400), w1_final=np.abs(rng.normal(0.25, 0.1, (3, 5))),
                    mmd_final=np.array([0.004, 0.05, 0.03]),
                    w1=np.abs(rng.normal(0.1, 0.04, (2, 5))), mmd=np.array([0.002, 0.003]),
                    n_snapshot_steps=1000)
    if name == "J":
        n_te = 200
        y_te = rng.standard_normal(n_te)
        r2 = np.array([0.55, 0.52, 0.5])
        pred = np.stack([np.sqrt(r) * y_te + np.sqrt(1 - r) * 0.6 * rng.standard_normal(n_te)
                         for r in r2])
        return dict(eps=0.01, explained_var=0.48,
                    mse_curve=np.stack([_decay(rng, 1.2, 1 - r, 0.05) for r in r2]),
                    mse_tr=1 - r2 - 0.02, mse_te=1 - r2, r2_tr=r2 + 0.02, r2_te=r2,
                    pred_te=pred, y_te=y_te, w1=np.ones((2, 5)) * 0.1, mmd=np.zeros(2))
    if name == "K":
        eg = np.array([0.001, 0.01, 0.1, 0.5, 1.0, 5.0])
        r2 = np.array([[0.55, 0.56, 0.57, 0.52, 0.45, 0.2],
                       [0.54, 0.55, 0.50, 0.30, -0.2, -1.8],
                       [0.53, 0.55, 0.56, 0.50, 0.40, -0.6]])
        frac = np.array([eg * c for c in (0.3, 0.5, 0.4)])
        return dict(eps_grid=eg, r2_te=r2, r2_tr=r2 + 0.02, mse_te=1 - r2, reg_frac=frac,
                    best_shared=0.01, eps_orig=0.01, explained_var=0.48)
    raise KeyError(name)
