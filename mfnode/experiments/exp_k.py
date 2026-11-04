"""
=============================================================================
exp_k.py — Experimento K: Barrido de ε para regresión (California Housing)
=============================================================================

MOTIVACIÓN:
    El experimento J usó ε=0.01 (igual que clasificación), pero la escala de
    la loss cambia drásticamente:

        Clasificación (exp_i, época 600):
            BCE = 0.0004    ε·reg = 0.01 × 0.133 = 0.00133
            → regularizador = 333% del BCE

        Regresión (exp_j, época 600):
            MSE = 0.7178    ε·reg = 0.01 × 0.176 = 0.00176
            → regularizador = 0.25% del MSE

    Con ε=0.01, el regularizador es prácticamente invisible en regresión.
    La inferioridad de pSGLD puede ser un artefacto del valor de ε, no
    una propiedad fundamental del método.

PROTOCOLO:
    Para cada ε en una rejilla logarítmica {0.001, 0.01, 0.1, 0.5, 1, 5}:
        1. Generar prior_samples de ν^∞ ∝ exp(-ℓ(a))  (una sola vez, fijo)
        2. Entrenar pSGLD, MMD-reg y Sinkhorn-reg en California Housing
        3. Registrar MSE_train, MSE_test, R²_train, R²_test
        4. Calcular fracción de regularización: ε·reg / MSE_train

FIGURA (K_epsilon_sweep.png, datos en figures/data/K.npz), 1×3:
    K1 — R²_test frente a ε (log); los valores fuera de escala se imprimen
    K2 — peso del regularizador ε·reg / MSE frente a ε
    K3 — R²_test con ε = 0.01 (el de J), con el mejor ε común y con el mejor
         ε de cada método.  R²_train y los MSE quedan en K.npz.
"""


import numpy as np
import torch

from ..config import DEVICE
from ..data import get_california_regression
from ..model import MeanFieldResNet
from ..train import train
from ..metrics import sample_prior_langevin
from ..plotting import publish

def _r2(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(float(ss_tot), 1e-8))


def experiment_K(epsilon_grid=None, n_epochs=500, n_prior=1_000,
                 sinkhorn_blur=0.05, n_train=800, n_test=400):
    """
    Barrido de ε para encontrar el régimen óptimo en regresión.

    Args:
        epsilon_grid  : lista de valores de ε a probar
        n_epochs      : épocas de entrenamiento por run (reducidas para velocidad)
        n_prior       : muestras del prior por ε (1000 es suficiente para el barrido)
        sinkhorn_blur : regularización del OT Sinkhorn
        n_train       : muestras de entrenamiento
        n_test        : muestras de test
    """
    if epsilon_grid is None:
        epsilon_grid = [0.001, 0.01, 0.1, 0.5, 1.0, 5.0]

    print("\n" + "=" * 62)
    print("EXPERIMENTO K  —  Barrido de ε (California Housing)")
    print("=" * 62)
    print(f"  ε ∈ {epsilon_grid}")
    print(f"  {n_epochs} epochs  |  train={n_train}  test={n_test}")

    # ── 1. Datos ────────────────────────────────────────────────────────────
    ds   = get_california_regression(n_train=n_train, n_test=n_test)
    X_tr = ds['X_train']
    y_tr = ds['y_train']
    X_te = ds['X_test']
    y_te = ds['y_test']
    y_tr_np = y_tr.cpu().numpy()
    y_te_np = y_te.cpu().numpy()
    print(f"  Varianza explicada PCA(2): {ds['explained_var']:.1%}")

    # ── 2. Precomputar prior samples (una sola vez — ν^∞ no depende de ε) ───
    print(f"\n  Generando {n_prior} muestras del prior ν^∞...")
    prior_samples = sample_prior_langevin(n_prior, dim=5)
    print(f"  → {prior_samples.shape}")

    # ── 3. Barrido ──────────────────────────────────────────────────────────
    # results[método][ε] = dict con métricas
    results = {name: {} for name in ['pSGLD', 'MMD', 'Sinkhorn']}

    for eps in epsilon_grid:
        print(f"\n  ── ε = {eps} " + "─" * 40)
        prior_s = prior_samples

        configs = [
            ('pSGLD',    dict(use_sgld=True)),
            ('MMD',      dict(use_mmd=True,     prior_samples=prior_s)),
            ('Sinkhorn', dict(use_sinkhorn=True, prior_samples=prior_s,
                              sinkhorn_blur=sinkhorn_blur)),
        ]

        for name, kwargs in configs:
            torch.manual_seed(42)
            np.random.seed(42)
            m = MeanFieldResNet(task='regression').to(DEVICE)
            h = train(m, X_tr, y_tr, epsilon=eps, n_epochs=n_epochs,
                      verbose=False, **kwargs)

            m.eval()
            with torch.no_grad():
                p_tr = m(X_tr).cpu().numpy()
                p_te = m(X_te).cpu().numpy()

            mse_tr = float(np.mean((p_tr - y_tr_np) ** 2))
            mse_te = float(np.mean((p_te - y_te_np) ** 2))
            r2_tr  = _r2(y_tr_np, p_tr)
            r2_te  = _r2(y_te_np, p_te)
            # Fracción de regularización: ε·reg / MSE_train al final
            reg_final = np.array(h['loss_reg'])[-1]
            reg_frac  = float(eps * reg_final / max(mse_tr, 1e-8))

            results[name][eps] = dict(
                mse_tr=mse_tr, mse_te=mse_te,
                r2_tr=r2_tr,   r2_te=r2_te,
                reg_frac=reg_frac,
                reg_final=reg_final,
            )
            print(f"    [{name:<10}]  MSE={mse_te:.4f}  R²_te={r2_te:.3f}"
                  f"  reg_frac={reg_frac:.3f}")

    # ── 4. Resumen ──────────────────────────────────────────────────────────
    print("\n  R²_test por método y ε:")
    header = f"  {'ε':<7}" + "".join(f"{'pSGLD':>10}{'MMD':>10}{'Sinkhorn':>10}")
    print(header)
    for eps in epsilon_grid:
        row = f"  {eps:<7}"
        for name in ['pSGLD', 'MMD', 'Sinkhorn']:
            row += f"{results[name][eps]['r2_te']:>10.3f}"
        print(row)

    # ε óptimo por método (máximo R²_test)
    best_eps = {}
    for name in ['pSGLD', 'MMD', 'Sinkhorn']:
        best_eps[name] = max(epsilon_grid,
                             key=lambda e: results[name][e]['r2_te'])
        print(f"  ε óptimo {name}: {best_eps[name]}"
              f"  →  R²_test = {results[name][best_eps[name]]['r2_te']:.3f}")

    # ε óptimo compartido (maximiza el mínimo entre métodos)
    best_shared = max(epsilon_grid,
                      key=lambda e: min(results[n][e]['r2_te']
                                        for n in ['pSGLD', 'MMD', 'Sinkhorn']))
    print(f"\n  ε compartido óptimo: {best_shared}")
    for name in ['pSGLD', 'MMD', 'Sinkhorn']:
        r = results[name][best_shared]
        print(f"    {name:<10}  R²_te={r['r2_te']:.3f}  MSE_te={r['mse_te']:.4f}"
              f"  reg_frac={r['reg_frac']:.3f}")

    # ── 5. Datos de la figura ──────────────────────────────────────────────
    names = ['pSGLD', 'MMD', 'Sinkhorn']

    def _grid(key):
        return [[results[n][e][key] for e in epsilon_grid] for n in names]

    publish('K', dict(
        eps_grid=np.array(epsilon_grid, dtype=float), n_epochs=n_epochs,
        r2_te=_grid('r2_te'), r2_tr=_grid('r2_tr'),
        mse_te=_grid('mse_te'), mse_tr=_grid('mse_tr'),
        reg_frac=_grid('reg_frac'), reg_final=_grid('reg_final'),
        best_eps=[best_eps[n] for n in names], best_shared=best_shared, eps_orig=0.01,
        explained_var=ds['explained_var'],
    ))

    return {
        'results':      results,
        'best_eps':     best_eps,
        'best_shared':  best_shared,
        'epsilon_grid': epsilon_grid,
        'dataset':      ds,
    }
