"""
=============================================================================
exp_b.py — Experimento B: Efecto del parámetro de regularización entrópica ε
=============================================================================
"""

import numpy as np
import torch

from ..config import SEED, DEVICE
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train, mu_pl_estimate
from ..plots import decision_grid, velocity_grid
from ..plotting import publish


# =============================================================================
# EXPERIMENTO B
#   Efecto del parámetro de regularización entrópica ε
# =============================================================================
def experiment_B(epsilons=None, n_epochs: int = 700):
    """
    Efecto del parámetro de regularización entrópica ε.

    Todos los modelos se inicializan con la MISMA semilla aleatoria (torch.manual_seed
    antes de cada creación), de forma que la única diferencia entre ellos es ε.
    Esto aísla el efecto de la regularización de posibles diferencias por inicialización.

    EFECTO TEÓRICO DE ε (papel sec. 1.1 y 1.3):
        ε = 0 : Sin regularización.  La optimización es libre pero sin garantías
                teóricas.  En datos simples puede funcionar bien, pero en general
                puede quedar atrapada en mínimos locales.

        ε > 0 : La regularización entrópica fuerza ν_t hacia el prior ν^∞.
                • Da lugar a la FORMA DE GIBBS del control óptimo (ec. 1.9):
                      ν_t*(a) ∝ exp(-ℓ(a) - (1/ε) ∫_A b(x,a)·∇u_t dγ_t)
                  donde u_t es la función de valor (solución de la ecuación HJB).
                  Interpretación: el control óptimo concentra ν_t* alrededor de
                  los parámetros que minimizan L(x,y) pero "penalizados" por ℓ(a).
                • Garantiza desigualdad log-Sobolev → condición PL → convergencia exp.
                • No necesita ser grande: ε arbitrariamente pequeño da garantías.

        ε grande: Mayor sesgo hacia el prior → parámetros más concentrados cerca
                  de 0 → fronteras más suaves.  Mayor J* (el óptimo global es
                  "peor" en clasificación pura porque la regularización penaliza).

    TRADE-OFF EMPÍRICO ε vs μ_PL:
        Observamos empíricamente que μ̂_PL es MENOR para ε grande.  Esto no
        contradice el paper: la teoría garantiza μ > 0 para todo ε > 0, pero
        no dice que μ crezca con ε.  El trade-off es:
          • ε grande → mejor regularización, J* más alto, μ̂ posiblemente menor
          • ε pequeño → menor regularización, J* más bajo, μ̂ posiblemente mayor
          • ε = 0 → sin garantía, pero en datos fáciles funciona bien

    OPTIMIZADOR:
        Usa SGLD (use_sgld=True): SGD + cosine annealing + ruido de Langevin
        √(2·η·ε)·ξ.  Implementa la dinámica de Gibbs ν_t* ∝ exp(−J(θ)/ε),
        por lo que el ruido es coherente con el rol teórico de ε como temperatura.
        Con ε=0 el ruido es exactamente cero (se recupera SGD puro).

    FIGURAS GENERADAS (datos en figures/data/B.npz):
        B1 — 1×3: J (log), accuracy (media móvil) y energía del prior por
             parámetro, una curva por ε (azules de claro a oscuro).
             Esperado: todas las ε convergen a ~100% acc; J* crece con ε
        B2 — 1×n_eps: frontera de decisión con P(y=1|x) sombreada y campo
             F(x, t=0.5) con escala común de flechas, en [−2.5, 2.5]².
             Esperado: fronteras más suaves para ε mayor; campo más uniforme.
    """
    if epsilons is None:
        epsilons = [0.0, 0.001, 0.01, 0.1, 0.5]

    print("\n" + "=" * 62)
    print("EXPERIMENTO B  —  Efecto del parámetro ε")
    print("=" * 62)

    X, y, X_np, y_np = get_moons()
    results = {}

    for eps in epsilons:
        print(f"\n  ε = {eps} ─────────────────────────────────────────")
        # Semilla fija → misma inicialización para todos los ε.
        # Así la única diferencia entre modelos es ε (temperatura Langevin
        # y coeficiente del prior energético).
        torch.manual_seed(SEED)
        model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
        hist  = train(model, X, y, epsilon=eps, n_epochs=n_epochs,
                      verbose=False, use_sgld=True)
        mu    = mu_pl_estimate(hist)
        results[eps] = {'model': model, 'hist': hist, 'mu': mu}
        print(f"    J*={hist['J_star']:.5f} | μ_PL={mu:.5f} "
              f"| acc={hist['accuracy'][-1]:.4f}")

    # ── Datos de las figuras B1 y B2 ─────────────────────────────────────────
    # B2 usa la misma ventana [−2.5, 2.5]² para la frontera y para F(x, 0.5).
    hists = [results[eps]['hist'] for eps in epsilons]
    probs, U, V = [], [], []
    for eps in epsilons:
        m = results[eps]['model']
        gx, gy, P = decision_grid(m, bounds=(-2.5, 2.5, -2.5, 2.5))
        qx, qy, u, v = velocity_grid(m, t=0.5, lim=2.5, n=16)
        probs.append(P)
        U.append(u)
        V.append(v)
    publish('B', dict(
        eps=np.array(epsilons, dtype=float),
        loss=[h['loss'] for h in hists], acc=[h['accuracy'] for h in hists],
        reg=[h['loss_reg'] for h in hists],
        J_star=[h['J_star'] for h in hists], mu=[results[e]['mu'] for e in epsilons],
        final_acc=[h['accuracy'][-1] for h in hists],
        X=X_np, y=y_np, grid_x=gx, grid_y=gy, grid_p=np.stack(probs),
        qx=qx, qy=qy, U=np.stack(U), V=np.stack(V),
    ))
    return results
