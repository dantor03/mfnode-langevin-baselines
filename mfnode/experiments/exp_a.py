"""
=============================================================================
exp_a.py — Experimento A: Neural ODE + evolución de features γ_t
=============================================================================
"""

import numpy as np
import torch

from ..config import SEED, DEVICE
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train, mu_pl_estimate
from ..plots import decision_grid
from ..plotting import publish


# =============================================================================
# EXPERIMENTO A
#   Neural ODE con regularización entrópica + evolución de features γ_t
# =============================================================================
def experiment_A(n_epochs: int = 800):
    """
    Neural ODE con regularización entrópica + evolución de features γ_t.

    OPTIMIZADOR:
        Usa Adam + cosine annealing (modo por defecto de train()).  Al ser un
        experimento de demostración (no de muestreo de ν*), no se necesita SGLD.

    OBJETIVO MATEMÁTICO:
        Mostrar empíricamente que la ODE de campo medio puede transformar una
        distribución γ_0 (make_moons, no separable linealmente) en γ_T que sí
        lo es.  Esto es la "separabilidad emergente" del paper: la red aprende
        un flujo F(x,t) que reorganiza las features sin etiquetar explícitamente
        qué puntos mueve a dónde.

    CONEXIÓN CON EL PAPER:
        • γ_t = distribución push-forward de γ_0 bajo la ODE (ec. 1.5):
              γ_t = (ϕ_t)_# γ_0   donde ϕ_t es el flujo del campo F
        • La ecuación de continuidad (ec. 1.3) describe cómo evoluciona γ_t:
              ∂_t γ_t + div_x(F(x,t) γ_t) = 0
          Esta PDE garantiza que la "masa" (densidad de datos) se conserva
          y se transporta según el campo F — no se crean ni destruyen puntos.
        • X_T es linealmente separable por el clasificador W·X_T + b porque
          el campo F ha "alineado" las dos clases durante el transporte.

    FIGURA (A_feature_evolution.png, datos en figures/data/A.npz), 1×4:
        γ_0  |  γ_{T/2}  |  γ_T con 40 trayectorias completas y la recta del
        clasificador lineal  |  frontera de decisión P(y=1|x) = 0.5 en ℝ².

    RESULTADO ESPERADO:
        • γ_0 tiene forma de lunas entrelazadas (no separable)
        • γ_T muestra las dos clases separadas (linealmente separables)
        • La separación es gradual y suave: no hay "saltos" bruscos
        • acc ≈ 1.0 con ε=0.01, M=64, T=1.0
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO A  —  Feature evolution γ_t")
    print("=" * 62)

    X, y, X_np, y_np = get_moons()

    print("  Entrenando modelo base  ε=0.01, M=64, T=1.0, n_steps=10 …")
    torch.manual_seed(SEED)
    model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
    hist  = train(model, X, y, epsilon=0.01, n_epochs=n_epochs, verbose=True)
    acc   = hist['accuracy'][-1]
    mu    = mu_pl_estimate(hist)
    print(f"  → J* = {hist['J_star']:.5f} | acc = {acc:.4f} | μ_PL = {mu:.4f}")

    # ── Obtener trayectoria γ_t ──────────────────────────────────────────────
    model.eval()
    with torch.no_grad():
        _, traj = model.integrate(X, return_trajectory=True)

    # ── Datos de la figura ───────────────────────────────────────────────────
    # γ_t en t = 0, T/2 y T; 40 trayectorias completas (20 por clase); la recta
    # del clasificador lineal sobre X_T; P(y = 1 | x) en una rejilla de ℝ².
    states = np.stack([xt.cpu().numpy() for _, xt in traj])      # (n_t+1, N, 2)
    n_t    = len(traj) - 1
    rng    = np.random.default_rng(SEED)
    idx    = np.concatenate([rng.choice(np.where(y_np == k)[0], size=20, replace=False)
                             for k in (0, 1)])
    gx, gy, P = decision_grid(model, X_np)
    with torch.no_grad():
        w = model.classifier.weight.cpu().numpy().ravel()
        b = model.classifier.bias.cpu().numpy()[0]
    publish('A', dict(
        y=y_np, T=model.T, t_mid=traj[n_t // 2][0],
        x_t0=states[0], x_tmid=states[n_t // 2], x_T=states[-1],
        paths=states[:, idx].transpose(1, 0, 2), path_labels=y_np[idx],
        w=w, b=b, grid_x=gx, grid_y=gy, grid_p=P, acc=acc, mu=mu, eps=0.01, M=64,
        loss=hist['loss'], loss_term=hist['loss_term'], loss_reg=hist['loss_reg'],
    ))
    return model, hist
