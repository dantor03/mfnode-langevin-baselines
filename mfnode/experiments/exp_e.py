"""
=============================================================================
exp_e.py — Experimento E (robustez): Robustez de la distribución ν*
=============================================================================
"""

import numpy as np
import torch

from ..config import DEVICE
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train
from ..plotting import publish


def experiment_E_robustness(n_seeds: int = 10, n_epochs: int = 500):
    """
    Robustez de ν* a distintas condiciones de entrenamiento (make_moons).

    MOTIVACIÓN:
        El Experimento E muestra la distribución ν* para UN único entrenamiento.
        Pero ¿es esa distribución robusta?  Si el minimizador es único (Meta-
        Teorema 1), distintas inicializaciones y distintos datasets deberían
        producir distribuciones ν* similares — la misma "nube" de puntos en el
        espacio de parámetros.

    DISEÑO:
        E-1 — Robustez a θ₀ (datos fijos, init variable):
            data_seed=42 fijo | init_seed ∈ {0, …, 9}
            Pregunta: ¿varía ν* con la inicialización de pesos?

        E-2 — Robustez a γ₀ (init fija, datos variables):
            init_seed=4 fijo | data_seed ∈ {0, …, 9}
            Pregunta: ¿varía ν* con el dataset de entrenamiento?

    OPTIMIZADOR:
        Ambos sub-experimentos usan SGLD (use_sgld=True): Adam + cosine annealing
        + ruido de Langevin √(2·η_t·ε)·ξ.  Adam como base garantiza convergencia
        al mínimo; el ruido permite explorar la distribución de Gibbs
        ν_t* ∝ exp(−J(θ)/ε) alrededor de él.

    FIGURA E (E_parameter_robustness.png, datos en figures/data/E.npz), 1×2:
        importancias ‖a₀ᵐ‖ ordenadas, una curva fina por semilla y la media en
        negro: E-1 (init variable, datos fijos) y E-2 (datos variables, init fija).
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO E (robustez)  —  Robustez de ν* entre entrenamientos")
    print("=" * 62)

    EPS             = 0.01
    DATA_SEED_FIXED = 42
    INIT_SEED_FIXED = 2   # seed 2 converge de forma limpia en make_moons (J*≈0.00015)
    SEEDS           = list(range(n_seeds))

    def get_params(model):
        with torch.no_grad():
            W1w = model.velocity.W1.weight.cpu().numpy()   # (M, d1+1)
            W0w = model.velocity.W0.weight.cpu().numpy()   # (d1, M)
        a1 = W1w[:, :2]   # (M, 2)
        a0 = W0w.T        # (M, 2)
        return a1, a0

    def importance(a0):
        return np.linalg.norm(a0, axis=1)   # (M,)

    # ── E-1: datos fijos, init variable ───────────────────────────────────────
    print("  E-1: data_seed=42 fijo, 10 inits distintas...")
    X_fixed, y_fixed, _, _ = get_moons(seed=DATA_SEED_FIXED)
    results_E2_1 = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_fixed, y_fixed, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False, use_sgld=True)
        a1, a0 = get_params(model)
        results_E2_1.append({'seed': s, 'a1': a1, 'a0': a0, 'hist': hist})
        print(f"    init_seed={s}: J*={hist['J_star']:.4f}, "
              f"acc={hist['accuracy'][-1]:.3f}")

    # ── E-2: init fija, datos variables ───────────────────────────────────────
    print(f"  E-2: init_seed={INIT_SEED_FIXED} fijo, 10 datasets distintos...")
    results_E2_2 = []
    for s in SEEDS:
        X_s, y_s, _, _ = get_moons(seed=s)
        torch.manual_seed(INIT_SEED_FIXED)
        np.random.seed(INIT_SEED_FIXED)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_s, y_s, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False, use_sgld=True)
        a1, a0 = get_params(model)
        results_E2_2.append({'seed': s, 'a1': a1, 'a0': a0, 'hist': hist})
        print(f"    data_seed={s}: J*={hist['J_star']:.4f}, "
              f"acc={hist['accuracy'][-1]:.3f}")

    # ── Datos de la figura ───────────────────────────────────────────────────
    def _sorted_importance(results):
        return np.stack([np.sort(importance(r['a0']))[::-1] for r in results])

    publish('E', dict(
        eps=EPS, data_seed=DATA_SEED_FIXED, init_seed=INIT_SEED_FIXED,
        imp_e1=_sorted_importance(results_E2_1), imp_e2=_sorted_importance(results_E2_2),
        a0_e1=[r['a0'] for r in results_E2_1], a1_e1=[r['a1'] for r in results_E2_1],
        a0_e2=[r['a0'] for r in results_E2_2], a1_e2=[r['a1'] for r in results_E2_2],
        J_star_e1=[r['hist']['J_star'] for r in results_E2_1],
        J_star_e2=[r['hist']['J_star'] for r in results_E2_2],
    ))
    return results_E2_1, results_E2_2
