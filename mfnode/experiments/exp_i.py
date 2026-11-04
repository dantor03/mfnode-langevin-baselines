"""
=============================================================================
exp_i.py — Experimento I: MMD y Sinkhorn como regularizadores de entrenamiento
=============================================================================

MOTIVACIÓN:
    El experimento H mostró que ν* ≠ ν^∞ (MMD ≈ 0.22): los datos desplazan
    significativamente la distribución óptima respecto al prior.
    pSGLD implementa esto implícitamente via ruido Langevin.

    Este experimento pregunta: ¿se puede reemplazar el ruido Langevin por una
    penalización explícita de distancia (MMD o Sinkhorn) entre las partículas
    actuales y el prior?

    Loss original (pSGLD implícito):
        J = BCE + ε·E_ν[ℓ(a)]   +   ruido Langevin

    Loss nuevo (explícito, sin ruido):
        J_MMD      = BCE + ε·MMD²(ν_N, ν^∞_samples)
        J_Sinkhorn = BCE + ε·S_blur(ν_N, ν^∞_samples)

PROTOCOLO:
    1. Precomputar N_prior=2000 muestras de ν^∞ por ULA (una sola vez).
    2. Entrenar 3 modelos (misma semilla, mismo dataset):
         (a) pSGLD   — baseline, sin prior_samples
         (b) MMD-reg — Adam puro, regularizador MMD²
         (c) Sinkhorn-reg — Adam puro, regularizador Sinkhorn
    3. Para cada modelo, recoger ν̂* (pSGLD post-convergencia 1000 pasos).
    4. Comparar con la ν̂* del baseline pSGLD como referencia.

FIGURA (I_regularizer_comparison.png, datos en figures/data/I.npz), 2×2:
    I1 — BCE de entrenamiento de los tres métodos
    I2 — marginal de a₀^(0): prior, ν̂*_pSGLD y las partículas entrenadas con
         MMD y con Sinkhorn
    I3 — W₁ por coordenada entre las partículas ENTRENADAS de cada método y
         ν̂*_pSGLD; las partículas finales del propio pSGLD dan el suelo que
         cabe esperar con solo M muestras
    I4 — W₁ tras los pasos adicionales de pSGLD del punto 3 (la comparación
         original).  Esos pasos acercan cualquier solución a la estacionaria de
         pSGLD, así que I4 mide sobre todo cuánto queda del punto de partida;
         I3 compara las soluciones de cada regularizador tal cual salen.
"""


import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from ..config import DEVICE
from ..model import MeanFieldResNet
from ..train import train
from ..metrics import (
    extract_particles,
    mmd_rbf,
    wasserstein_1d,
    sample_prior_langevin,
    collect_psgld_snapshots,
)
from ..plotting import publish

DIM_LABELS = [r'$a_0^{(0)}$', r'$a_0^{(1)}$',
              r'$a_1^{(0)}$', r'$a_1^{(1)}$', r'$a_2$']

N_SNAPSHOT_STEPS = 1_000   # pasos de pSGLD tras el entrenamiento para estimar ν̂*
THIN             = 10


def experiment_I(n_epochs=700, epsilon=0.01, n_prior=2_000,
                 sinkhorn_blur=0.05, n_snapshot_steps=N_SNAPSHOT_STEPS, thin=THIN):
    """
    Compara pSGLD, MMD-reg y Sinkhorn-reg como métodos de regularización.

    Args:
        n_epochs      : épocas de entrenamiento para los 3 métodos
        epsilon       : temperatura / coeficiente de regularización
        n_prior       : muestras de ν^∞ precomputadas para MMD/Sinkhorn
        sinkhorn_blur : regularización del OT en Sinkhorn
        n_snapshot_steps, thin : pasos de pSGLD tras el entrenamiento para ν̂*
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO I  —  MMD y Sinkhorn como regularizadores")
    print("=" * 62)
    print(f"  ε={epsilon}  |  {n_epochs} epochs  |  "
          f"{n_prior} prior samples  |  blur={sinkhorn_blur}")

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    X_np, y_np = make_moons(n_samples=500, noise=0.12, random_state=42)
    scaler = StandardScaler().fit(X_np)
    X_np   = scaler.transform(X_np).astype(np.float32)
    y_np   = y_np.astype(np.float32)
    X = torch.tensor(X_np, device=DEVICE)
    y = torch.tensor(y_np, device=DEVICE)

    # ── 2. Prior samples (una sola vez) ────────────────────────────────────
    print(f"\n  Generando {n_prior} muestras del prior ν^∞...")
    prior_samples = sample_prior_langevin(n_prior, dim=5)
    print(f"  → {prior_samples.shape}")

    # ── 3. Entrenamiento de los 3 métodos ───────────────────────────────────
    histories = {}
    models    = {}

    configs = [
        ('pSGLD',    dict(use_sgld=True)),
        ('MMD',      dict(use_mmd=True,      prior_samples=prior_samples)),
        ('Sinkhorn', dict(use_sinkhorn=True,  prior_samples=prior_samples,
                          sinkhorn_blur=sinkhorn_blur)),
    ]

    for name, kwargs in configs:
        print(f"\n  [{name}] Entrenando {n_epochs} epochs...")
        torch.manual_seed(42)
        np.random.seed(42)
        m = MeanFieldResNet().to(DEVICE)
        h = train(m, X, y, epsilon=epsilon, n_epochs=n_epochs,
                  verbose=True, **kwargs)
        histories[name] = h
        models[name]    = m

    # ── 4. ν̂* por método (snapshots pSGLD post-convergencia) ────────────────
    timeavg = {}
    for name, m in models.items():
        print(f"\n  [{name}] Recogiendo ν̂* ({n_snapshot_steps} pasos)...")
        snaps = collect_psgld_snapshots(
            m, X, y, epsilon,
            n_steps=n_snapshot_steps, thin=thin
        )
        timeavg[name] = snaps.reshape(-1, 5)
        print(f"    → {timeavg[name].shape[0]} muestras")

    # ── 5. Métricas: W₁ de cada ν̂* vs ν̂*_pSGLD ──────────────────────────────
    ref = timeavg['pSGLD']
    print("\n  W₁ de ν̂*_método vs ν̂*_pSGLD (referencia):")
    print(f"  {'dim':<8} {'MMD':>8} {'Sinkhorn':>10}")
    w1_mmd  = [wasserstein_1d(timeavg['MMD'][:, d],      ref[:, d]) for d in range(5)]
    w1_sink = [wasserstein_1d(timeavg['Sinkhorn'][:, d], ref[:, d]) for d in range(5)]
    for d in range(5):
        print(f"  {DIM_LABELS[d]:<8} {w1_mmd[d]:>8.4f} {w1_sink[d]:>10.4f}")

    # MMD global entre métodos
    N_sub = 500
    mmd_mmd_vs_psgld  = mmd_rbf(
        timeavg['MMD'][:N_sub],      ref[:N_sub])
    mmd_sink_vs_psgld = mmd_rbf(
        timeavg['Sinkhorn'][:N_sub], ref[:N_sub])
    print(f"\n  MMD²(ν̂*_MMD, ν̂*_pSGLD)      = {mmd_mmd_vs_psgld:.4e}")
    print(f"  MMD²(ν̂*_Sinkhorn, ν̂*_pSGLD) = {mmd_sink_vs_psgld:.4e}")

    # Partículas entrenadas de cada método, ANTES de continuar con pSGLD, frente
    # a ν̂*_pSGLD.  Las partículas finales del propio pSGLD dan el suelo que cabe
    # esperar con solo M muestras.
    names = ['pSGLD', 'MMD', 'Sinkhorn']
    final = {n: extract_particles(models[n]) for n in names}
    w1_final  = [[wasserstein_1d(final[n][:, d], ref[:, d]) for d in range(5)] for n in names]
    mmd_final = [mmd_rbf(final[n], ref[:N_sub]) for n in names]
    print("\n  W₁ de las partículas entrenadas vs ν̂*_pSGLD:")
    for n, w in zip(names, w1_final):
        print(f"    {n:<9} " + "  ".join(f"{v:.4f}" for v in w))

    # ── 6. Datos de la figura ──────────────────────────────────────────────
    publish('I', dict(
        eps=epsilon, n_epochs=n_epochs, n_snapshot_steps=n_snapshot_steps, thin=thin,
        bce=[histories[n]['loss_term'] for n in names],
        reg=[histories[n]['loss_reg'] for n in names],
        acc=[histories[n]['accuracy'] for n in names],
        final=np.stack([final[n].numpy() for n in names]),
        ref=ref, timeavg=np.stack([timeavg[n].numpy() for n in names]),
        w1_final=w1_final, mmd_final=mmd_final,
        w1=[w1_mmd, w1_sink], mmd=[mmd_mmd_vs_psgld, mmd_sink_vs_psgld],
    ))

    return {
        'histories':        histories,
        'models':           models,
        'timeavg':          timeavg,
        'w1_mmd':           w1_mmd,
        'w1_sinkhorn':      w1_sink,
        'mmd_mmd_psgld':    mmd_mmd_vs_psgld,
        'mmd_sink_psgld':   mmd_sink_vs_psgld,
    }
