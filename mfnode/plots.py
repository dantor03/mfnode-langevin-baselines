"""Evaluate a trained model on grids, for the figures (the drawing is in ``plotting``)."""

import numpy as np
import torch

from .config import DEVICE


def decision_grid(model, X_np=None, pad=0.5, n=200, bounds=None):
    """
    P(y = 1 | x) of the full model (ODE + linear read-out) on a regular grid of ℝ².

    The 0.5 level set is the decision boundary in the original space.  It
    reflects the joint action of the ODE, which maps X_0 to X_T, and of the
    linear classifier W·X_T + b on the transported features.

    Args:
        model  : trained MeanFieldResNet (classification)
        X_np   : data used to choose the window (ignored if ``bounds`` is given)
        pad    : margin added around the data
        n      : grid points per axis
        bounds : optional (xmin, xmax, ymin, ymax)

    Returns:
        gx (n,), gy (n,), P (n, n) with P[i, j] evaluated at (gx[j], gy[i]).
    """
    if bounds is None:
        xmin, ymin = X_np.min(axis=0) - pad
        xmax, ymax = X_np.max(axis=0) + pad
    else:
        xmin, xmax, ymin, ymax = bounds
    gx, gy = np.linspace(xmin, xmax, n), np.linspace(ymin, ymax, n)
    xx, yy = np.meshgrid(gx, gy)
    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()].astype(np.float32), device=DEVICE)
    model.eval()
    with torch.no_grad():
        P = torch.sigmoid(model(grid)).cpu().numpy().reshape(xx.shape)
    return gx, gy, P


def velocity_grid(model, t=0.5, lim=2.5, n=16):
    """The velocity field F(x, t) on an n × n grid of [−lim, lim]²."""
    v = np.linspace(-lim, lim, n)
    qx, qy = np.meshgrid(v, v)
    grid = torch.tensor(np.c_[qx.ravel(), qy.ravel()].astype(np.float32), device=DEVICE)
    model.eval()
    with torch.no_grad():
        vel = model.velocity(t, grid).cpu().numpy()
    return qx, qy, vel[:, 0].reshape(qx.shape), vel[:, 1].reshape(qx.shape)
