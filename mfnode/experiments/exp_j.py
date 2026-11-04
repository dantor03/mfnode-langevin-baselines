"""
=============================================================================
exp_j.py — Experimento J: Regresión en California Housing (datos reales)
=============================================================================

PREGUNTA:
    ¿Funcionan pSGLD, MMD-reg y Sinkhorn-reg igual de bien en una tarea de
    regresión con datos reales que en clasificación sintética?

    El setup es idéntico al exp_i salvo:
        • Dataset: California Housing proyectado a ℝ² via PCA (8 features → 2)
        • Loss: MSE en lugar de BCE
        • Métrica de calidad: R² (train y test) en lugar de accuracy
        • Se separa train/test para medir generalización

PROTOCOLO:
    1. Cargar California Housing, proyectar a ℝ², estandarizar target.
    2. Precomputar muestras del prior ν^∞ (mismas que exp_i — el prior no
       depende de los datos, solo de la dimensión d=5 del espacio A).
    3. Entrenar 3 modelos (pSGLD, MMD-reg, Sinkhorn-reg) con MSE loss.
    4. Evaluar MSE y R² en train y test para cada método.
    5. Recoger ν̂* por método y comparar vs referencia pSGLD (W₁, MMD²).

FIGURA (J_california_regression.png, datos en figures/data/J.npz), 1×3:
    J1 — MSE de entrenamiento por época (★: MSE de test final)
    J2 — R² de train y de test de cada método
    J3 — predicción frente a valor real en el conjunto de test
    W₁ y MMD² de ν̂* frente a pSGLD se imprimen y se guardan en J.npz.
"""


import numpy as np
import torch

from ..config import DEVICE
from ..data import get_california_regression
from ..model import MeanFieldResNet
from ..train import train
from ..metrics import (
    mmd_rbf,
    wasserstein_1d,
    sample_prior_langevin,
    collect_psgld_snapshots,
)
from ..plotting import publish

DIM_LABELS = [r'$a_0^{(0)}$', r'$a_0^{(1)}$',
              r'$a_1^{(0)}$', r'$a_1^{(1)}$', r'$a_2$']

N_SNAPSHOT_STEPS = 1_000
THIN             = 10


def _r2_numpy(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return float(1.0 - ss_res / max(ss_tot, 1e-8))


def experiment_J(n_epochs=700, epsilon=0.01, n_prior=2_000,
                 sinkhorn_blur=0.05, n_train=800, n_test=400,
                 n_snapshot_steps=N_SNAPSHOT_STEPS, thin=THIN):
    """
    Compara pSGLD, MMD-reg y Sinkhorn-reg en regresión con California Housing.

    Args:
        n_epochs      : épocas de entrenamiento
        epsilon       : coeficiente de regularización
        n_prior       : muestras del prior ν^∞ para MMD/Sinkhorn
        sinkhorn_blur : regularización del OT
        n_train       : muestras de entrenamiento
        n_test        : muestras de test
        n_snapshot_steps, thin : pasos de pSGLD tras el entrenamiento para ν̂*
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO J  —  Regresión: California Housing")
    print("=" * 62)
    print(f"  ε={epsilon}  |  {n_epochs} epochs  |  "
          f"train={n_train}  test={n_test}")

    # ── 1. Datos ────────────────────────────────────────────────────────────
    ds = get_california_regression(n_train=n_train, n_test=n_test)
    X_tr, y_tr = ds['X_train'], ds['y_train']
    X_te, y_te = ds['X_test'],  ds['y_test']
    print(f"  Varianza explicada PCA(2): {ds['explained_var']:.1%}")

    # ── 2. Prior samples ────────────────────────────────────────────────────
    print(f"\n  Generando {n_prior} muestras del prior ν^∞...")
    prior_samples = sample_prior_langevin(n_prior, dim=5)

    # ── 3. Entrenamiento ────────────────────────────────────────────────────
    histories = {}
    models    = {}
    test_metrics = {}

    configs = [
        ('pSGLD',    dict(use_sgld=True)),
        ('MMD',      dict(use_mmd=True,     prior_samples=prior_samples)),
        ('Sinkhorn', dict(use_sinkhorn=True, prior_samples=prior_samples,
                          sinkhorn_blur=sinkhorn_blur)),
    ]

    for name, kwargs in configs:
        print(f"\n  [{name}] Entrenando {n_epochs} epochs (MSE + regresión)...")
        torch.manual_seed(42)
        np.random.seed(42)
        m = MeanFieldResNet(task='regression').to(DEVICE)
        h = train(m, X_tr, y_tr, epsilon=epsilon, n_epochs=n_epochs,
                  verbose=True, **kwargs)
        histories[name] = h
        models[name]    = m

        # Métricas test al final del entrenamiento
        m.eval()
        with torch.no_grad():
            pred_tr = m(X_tr).cpu().numpy()
            pred_te = m(X_te).cpu().numpy()
        y_tr_np = y_tr.cpu().numpy()
        y_te_np = y_te.cpu().numpy()
        mse_tr  = float(np.mean((pred_tr - y_tr_np) ** 2))
        mse_te  = float(np.mean((pred_te - y_te_np) ** 2))
        r2_tr   = _r2_numpy(y_tr_np, pred_tr)
        r2_te   = _r2_numpy(y_te_np, pred_te)
        test_metrics[name] = dict(mse_tr=mse_tr, mse_te=mse_te,
                                   r2_tr=r2_tr,   r2_te=r2_te,
                                   pred_te=pred_te, y_te=y_te_np)
        print(f"    MSE train={mse_tr:.4f}  test={mse_te:.4f}  "
              f"R² train={r2_tr:.3f}  test={r2_te:.3f}")

    # ── 4. ν̂* por método ────────────────────────────────────────────────────
    timeavg = {}
    for name, m in models.items():
        print(f"\n  [{name}] Recogiendo ν̂* ({n_snapshot_steps} pasos)...")
        snaps = collect_psgld_snapshots(
            m, X_tr, y_tr, epsilon,
            n_steps=n_snapshot_steps, thin=thin
        )
        timeavg[name] = snaps.reshape(-1, 5)

    # ── 5. Métricas de distribución ─────────────────────────────────────────
    ref = timeavg['pSGLD']
    w1_mmd  = [wasserstein_1d(timeavg['MMD'][:, d],      ref[:, d]) for d in range(5)]
    w1_sink = [wasserstein_1d(timeavg['Sinkhorn'][:, d], ref[:, d]) for d in range(5)]
    N_sub   = 500
    mmd_mmd_psgld  = mmd_rbf(timeavg['MMD'][:N_sub],      ref[:N_sub])
    mmd_sink_psgld = mmd_rbf(timeavg['Sinkhorn'][:N_sub], ref[:N_sub])

    print("\n  Comparación ν̂* vs referencia pSGLD:")
    print(f"  MMD²(MMD, pSGLD)      = {mmd_mmd_psgld:.4e}")
    print(f"  MMD²(Sinkhorn, pSGLD) = {mmd_sink_psgld:.4e}")
    print(f"\n  {'método':<10} {'MSE_tr':>8} {'MSE_te':>8} {'R²_tr':>7} {'R²_te':>7}")
    for name in ['pSGLD', 'MMD', 'Sinkhorn']:
        m = test_metrics[name]
        print(f"  {name:<10} {m['mse_tr']:>8.4f} {m['mse_te']:>8.4f} "
              f"{m['r2_tr']:>7.3f} {m['r2_te']:>7.3f}")

    # ── 6. Datos de la figura ──────────────────────────────────────────────
    names = ['pSGLD', 'MMD', 'Sinkhorn']

    def _metric(key):
        return [test_metrics[n][key] for n in names]

    publish('J', dict(
        eps=epsilon, n_epochs=n_epochs, n_snapshot_steps=n_snapshot_steps, thin=thin,
        explained_var=ds['explained_var'],
        mse_curve=[histories[n]['loss_term'] for n in names],
        r2_curve=[histories[n]['accuracy'] for n in names],
        mse_tr=_metric('mse_tr'), mse_te=_metric('mse_te'),
        r2_tr=_metric('r2_tr'), r2_te=_metric('r2_te'),
        pred_te=np.stack(_metric('pred_te')), y_te=test_metrics['pSGLD']['y_te'],
        w1=[w1_mmd, w1_sink], mmd=[mmd_mmd_psgld, mmd_sink_psgld],
    ))

    return {
        'histories':       histories,
        'models':          models,
        'test_metrics':    test_metrics,
        'timeavg':         timeavg,
        'w1_mmd':          w1_mmd,
        'w1_sinkhorn':     w1_sink,
        'mmd_mmd_psgld':   mmd_mmd_psgld,
        'mmd_sink_psgld':  mmd_sink_psgld,
        'dataset':         ds,
    }
