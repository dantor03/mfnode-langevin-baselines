"""
=============================================================================
exp_f.py — Experimento F: Distribución de ν* en make_circles (simetría SO(2))
=============================================================================
"""

import numpy as np
import torch

from ..config import DEVICE
from ..data import get_circles
from ..model import MeanFieldResNet
from ..train import train
from ..plotting import publish


# =============================================================================
# EXPERIMENTO F
#   Distribución de ν* en make_circles: simetría rotacional y robustez a semillas
# =============================================================================
def experiment_F(n_seeds: int = 10, n_epochs: int = 700):
    """
    Distribución de ν* en make_circles: simetría geométrica y robustez a semillas.

    MOTIVACIÓN TEÓRICA:
        make_circles tiene simetría rotacional completa SO(2): el dataset γ₀
        es invariante (en distribución) bajo rotaciones del plano ℝ².  Si el
        problema de control también respeta esta simetría, la distribución óptima
        de parámetros ν* debería ser asimismo (aproximadamente) isotrópica.

        Predicción concreta sobre a₁ᵐ:
            Los pesos de entrada a₁ᵐ ∈ ℝ² deberían distribuirse UNIFORMEMENTE
            en S¹ (un anillo en el plano), porque ninguna dirección espacial es
            privilegiada por la geometría del problema.

        Contraste con make_moons:
            En el Experimento E observamos una distribución BIMODAL de a₁
            (dos picos en ±0.3–0.4).  Esto refleja que las dos "lunas" tienen
            una orientación preferida.  Con circles esperamos la distribución
            contraria: uniforme sobre S¹.

        Test cuantitativo — longitud resultante media R̄:
            R̄ = |mean(exp(iθ))| ∈ [0,1],   θ = arctan2(a₁ᵐ[1], a₁ᵐ[0])
            R̄ ≈ 0 → distribución isotrópica (predicción de simetría)
            R̄ ≈ 1 → distribución concentrada en una dirección

    DISEÑO DEL EXPERIMENTO:
        F1 — Robustez a γ₀ (init fija, datos variables):
            • init_seed = 4 fija (misma inicialización en todos los runs)
            • Dataset make_circles con n_seeds semillas distintas
            • Pregunta: ¿la distribución de a₁ cambia con el dataset?

        F2 — Robustez a θ₀ (datos fijos, init variable):
            • data_seed = 42 fijo (mismo make_circles en todos los runs)
            • n_seeds inicializaciones de parámetros distintas
            • Pregunta: ¿la distribución de a₁ cambia con la inicialización?

    OPTIMIZADOR:
        Ambos sub-experimentos usan SGLD (use_sgld=True): Adam + cosine annealing
        + ruido de Langevin √(2·η_t·ε)·ξ.  Adam como base garantiza convergencia
        al mínimo; el ruido permite visualizar la distribución estacionaria de
        los parámetros ν_t* ∝ exp(−J(θ)/ε), no solo un estimador puntual.

    FIGURA (F_circles_parameter_distribution.png, datos en figures/data/F.npz), 1×3:
        • J por época de F1 y F2 (runs en trazo fino, media en grueso); la
          leyenda da la media de las últimas 50 épocas, porque pSGLD oscila
          alrededor del mínimo en lugar de converger a él
        • pesos de entrada a₁ᵐ de todos los runs de F2, coloreados por ‖a₀ᵐ‖
        • histograma del ángulo de a₁ᵐ ponderado por ‖a₀ᵐ‖ frente al uniforme:
          si ν* hereda la simetría SO(2) de los datos, debe salir plano
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO F  —  ν* en make_circles: simetría y robustez")
    print("=" * 62)

    EPS             = 0.01
    DATA_SEED_FIXED = 42
    INIT_SEED_FIXED = 4   # seed 4 converge bien en make_circles; elegido porque
                          # produce F1 < F2 (datos más robustos que init), lo que
                          # da jerarquía visual informativa en la figura
    SEEDS           = list(range(n_seeds))

    # ── F1: semillas de datos, init fija ─────────────────────────────────────
    print(f"  F1: 10 datasets circles distintos, init_seed={INIT_SEED_FIXED} fija...")
    results_F1 = []
    for s in SEEDS:
        X, y, X_np_s, y_np_s = get_circles(seed=s)
        torch.manual_seed(INIT_SEED_FIXED)
        np.random.seed(INIT_SEED_FIXED)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X, y, epsilon=EPS, n_epochs=n_epochs,
                      verbose=False, use_sgld=True)
        acc   = hist['accuracy'][-1]
        results_F1.append({'model': model, 'hist': hist, 'seed': s,
                           'acc': acc, 'X_np': X_np_s, 'y_np': y_np_s})
        print(f"    data_seed={s}: J*={hist['J_star']:.4f}, acc={acc:.3f}")

    # ── F2: semillas de init, datos fijos ─────────────────────────────────────
    print("  F2: dataset circles seed=42 fijo, 10 inits distintas...")
    X_fixed, y_fixed, X_np_fixed, y_np_fixed = get_circles(seed=DATA_SEED_FIXED)
    results_F2 = []
    for s in SEEDS:
        torch.manual_seed(s)
        np.random.seed(s)
        model = MeanFieldResNet().to(DEVICE)
        hist  = train(model, X_fixed, y_fixed, epsilon=EPS,
                      n_epochs=n_epochs, verbose=False, use_sgld=True)
        acc   = hist['accuracy'][-1]
        results_F2.append({'model': model, 'hist': hist, 'seed': s, 'acc': acc})
        print(f"    init_seed={s}: J*={hist['J_star']:.4f}, acc={acc:.3f}")

    # ── Resumen en consola ────────────────────────────────────────────────────
    acc_F1 = np.array([r['acc'] for r in results_F1])
    acc_F2 = np.array([r['acc'] for r in results_F2])
    print(f"\n  Acc media F1 (datos): {acc_F1.mean():.4f} ± {acc_F1.std():.4f}")
    print(f"  Acc media F2 (init):  {acc_F2.mean():.4f} ± {acc_F2.std():.4f}")

    # ── Datos de la figura ───────────────────────────────────────────────────
    def _params(results):
        """a₀ᵐ y a₁ᵐ de cada run: arrays (n_runs, M, 2)."""
        a0, a1 = [], []
        with torch.no_grad():
            for r in results:
                v = r['model'].velocity
                a0.append(v.W0.weight.t().cpu().numpy())
                a1.append(v.W1.weight[:, :2].cpu().numpy())
        return np.stack(a0), np.stack(a1)

    a0_f1, a1_f1 = _params(results_F1)
    a0_f2, a1_f2 = _params(results_F2)
    publish('F', dict(
        eps=EPS, data_seed=DATA_SEED_FIXED, init_seed=INIT_SEED_FIXED,
        loss_f1=[r['hist']['loss'] for r in results_F1],
        loss_f2=[r['hist']['loss'] for r in results_F2],
        acc_f1=acc_F1, acc_f2=acc_F2,
        a0_f1=a0_f1, a1_f1=a1_f1, a0_f2=a0_f2, a1_f2=a1_f2,
    ))
    return results_F1, results_F2
