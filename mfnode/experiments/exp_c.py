"""
=============================================================================
exp_c.py — Experimento C: Verificación empírica de la condición PL
=============================================================================

NOTA SOBRE EL OPTIMIZADOR:
    Este experimento usa SGD con lr constante (sin cosine annealing).
    Motivación: la condición PL es una propiedad GEOMÉTRICA del paisaje de J,
    no del optimizador.  Con Adam + cosine annealing el lr → 0 al final, lo que
    aplana J(θ^s) − J* artificialmente y contamina la estimación de J* y de μ̂.
    Con SGD + lr constante, J* es el verdadero mínimo alcanzado por el gradient
    flow, y el ratio ‖∇J‖²/(2(J−J*)) refleja fielmente la geometría de J.
"""

import numpy as np
import torch

from ..config import SEED, DEVICE
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train, mu_pl_estimate
from ..plotting import publish


# =============================================================================
# EXPERIMENTO C
#   Verificación empírica de la desigualdad Polyak-Łojasiewicz
# =============================================================================
def experiment_C(epsilons=None, n_epochs: int = 700):
    """
    Verificación empírica de la desigualdad Polyak-Łojasiewicz.

    Entrena con SGD + lr constante (use_sgd=True) para una medición limpia
    de la condición PL sin artefactos del cosine annealing.

    META-TEOREMA 2 (paper, sec. 1.4):
        Para condiciones iniciales γ_0 en un conjunto abierto denso 𝒪 y ε > 0,
        la función objetivo J satisface la desigualdad PL LOCAL cerca del
        minimizador estable:
            I(γ_0, ν) ≥ c · (J(γ_0, ν) − J(γ_0, ν*))   con c > 0
        donde I es la "información de Fisher" (análogo de ‖∇J‖² en espacios de medidas).

        En la aproximación de M parámetros finitos, esto se traduce en:
            ‖∇J(θ)‖² ≥ 2μ · (J(θ) − J*)

        COROLARIO: gradient descent con step size η converge como:
            J(θ_k) − J* ≤ (1 − 2ημ)^k · (J(θ_0) − J*)  →  decay exponencial

    PANELES GENERADOS (C_pl_verification.png, datos en figures/data/C.npz):
        C1 — Log-log: ‖∇J‖² frente a J−J*, un punto por época, con la recta
             2μ̂_min(J−J*).  Si PL se cumple, los puntos quedan por encima.
        C2 — Semilog: J−J* por época; con PL el decaimiento es exponencial.
        C3 — Ratio PL ‖∇J‖²/(2(J−J*)) por época, con la recta μ̂_min.
        El μ̂ de cada ε (percentil 10 del ratio) aparece en la leyenda.  La
        teoría garantiza μ > 0 para todo ε > 0, no que μ crezca con ε: con ε
        grande J* sube y el ratio puede bajar.

    Args:
        epsilons  : lista de valores de ε a comparar
        n_epochs  : número de épocas de entrenamiento (SGD, lr constante)
    """
    if epsilons is None:
        epsilons = [0.0, 0.001, 0.01, 0.1, 0.5]

    print("\n" + "=" * 62)
    print("EXPERIMENTO C  —  Verificación PL  (SGD + lr constante)")
    print("=" * 62)

    X, y, _, _ = get_moons()

    # ── Entrenamiento con SGD + lr constante ─────────────────────────────────
    results_eps = {}
    for eps in epsilons:
        print(f"  ε={eps} ...")
        torch.manual_seed(SEED)
        model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
        hist  = train(model, X, y, epsilon=eps,
                      n_epochs=n_epochs, verbose=False, use_sgd=True)
        mu    = mu_pl_estimate(hist)
        results_eps[eps] = {'model': model, 'hist': hist, 'mu': mu}
        print(f"    J*={hist['J_star']:.5f} | μ̂={mu:.5f} "
              f"| acc={hist['accuracy'][-1]:.4f}")

    # ── Datos de la figura ───────────────────────────────────────────────────
    hists = [results_eps[e]['hist'] for e in epsilons]
    publish('C', dict(
        eps=np.array(epsilons, dtype=float),
        mu=[results_eps[e]['mu'] for e in epsilons],
        loss=[h['loss'] for h in hists], J_star=[h['J_star'] for h in hists],
        grad_norm2=[h['grad_norm2'] for h in hists],
        pl_ratio=np.array([h['pl_ratio'] for h in hists], dtype=float),
        acc_final=[h['accuracy'][-1] for h in hists],
    ))

    # ── Tabla resumen ─────────────────────────────────────────────────────────
    print("\n  ┌─────────┬────────────┬────────────┬────────────┐")
    print(  "  │    ε    │ μ_PL (P10) │  J* final  │  Acc final │")
    print(  "  ├─────────┼────────────┼────────────┼────────────┤")
    for eps, res in results_eps.items():
        acc = res['hist']['accuracy'][-1]
        print(f"  │ {eps:>7.3f} │ {res['mu']:>10.5f} │ "
              f"{res['hist']['J_star']:>10.5f} │ {acc:>10.4f} │")
    print(  "  └─────────┴────────────┴────────────┴────────────┘")
    print("\n  Interpretación:")
    print("  • Optimizador: SGD + lr constante.  J* = mínimo genuino del gradient")
    print("    flow, sin artefactos de cosine annealing.")
    print("  • μ̂ > 0 para ε > 0  →  Meta-Teorema 2 verificado empíricamente.")
    print("  • ε = 0 puede mostrar μ̂ > 0 en datos fáciles, pero sin garantía teórica.")
    print("  • μ̂ no es necesariamente monótono en ε: la teoría garantiza μ > 0")
    print("    para todo ε > 0, no que μ crezca con ε.")
    print("  • LIMITACIÓN DE IMPLEMENTACIÓN: la regularización usa solo el término")
    print("    de energía E_{ν}[ℓ(a)] = (1/N_p)Σ(0.05θ⁴+0.5θ²), no la KL completa.")
    print("    Es una 'penalización supercoerciva de energía', no entropía verdadera.")
    print("  • RESTRICCIÓN TEMPORAL: a₀ᵐ y a₁ᵐ son constantes en t; solo a₂ᵐ(t)")
    print("    varía linealmente.  El espacio de controles es un subconjunto estricto")
    print("    del control de campo medio continuo analizado en el paper.")

    return results_eps
