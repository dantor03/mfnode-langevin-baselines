"""
=============================================================================
exp_h.py — Experimento H: Diagnóstico de Convergencia pSGLD → ν*
=============================================================================

MOTIVACIÓN:
    pSGLD tiene distribución estacionaria ν* ∝ exp(−J(θ)/ε).  Pero no podemos
    verificarlo directamente con KL(ν_N || ν^∞), ya que ν_N es discreta y la KL
    sería infinita.  En su lugar usamos dos métricas bien definidas para
    distribuciones de soporte mixto:
        • MMD²  (Maximum Mean Discrepancy con kernel Gaussiano)
        • W₁    (Wasserstein-1 exacta, por dimensión)

PREGUNTA:
    ¿Muestrea realmente pSGLD de ν*?
    ¿Cuánto difiere ν* del prior ν^∞ (efecto de los datos)?

PROTOCOLO:
    1. Entrenar un modelo con pSGLD (ε=0.01, 800 epochs).
    2. Post-entrenamiento: continuar pSGLD 2000 pasos con lr fijo → snapshots
       de partículas cada `thin` pasos.  El pool de todos los snapshots aproxima
       la distribución tiempo-promedio ν̂* ≈ ν* (teorema ergódico).
    3. Generar muestras de ν^∞ ∝ exp(−ℓ(a)) por ULA (Langevin MCMC sin datos).
    4. Calcular:
         MMD²(ν_N_final, ν̂*)    — partículas finales ≈ estacionario?
         MMD²(ν̂*,        ν^∞)   — ν* ≠ prior (datos hacen algo)?
         MMD²(ν_N_final, ν^∞)   — partículas vs prior (directamente)
         W₁ por dimensión        — resolución fina por componente de a

FIGURA (H_psgld_diagnostic.png, datos en figures/data/H.npz):
    Fila 1 — marginales de las 5 coordenadas: prior ν^∞ analítico (discontinua),
             KDE de ν̂* y las M partículas finales (marcas).
    Fila 2 — MMD² entre las tres distribuciones y W₁ por coordenada, con el
             mismo color para cada comparación.
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

# ── Etiquetas para las 5 dimensiones del espacio A ──────────────────────────
DIM_LABELS = [
    r'$a_0^{(0)}$',   # W0.weight[0, m]  — componente 0 de a₀^m
    r'$a_0^{(1)}$',   # W0.weight[1, m]  — componente 1 de a₀^m
    r'$a_1^{(0)}$',   # W1.weight[m, 0]  — componente 0 de a₁^m
    r'$a_1^{(1)}$',   # W1.weight[m, 1]  — componente 1 de a₁^m
    r'$a_2$',         # W1.bias[m]        — umbral (bias) de neurona m
]

def experiment_H(n_epochs=800, n_snapshot_steps=2_000, thin=10,
                 n_prior_samples=8_000, epsilon=0.01):
    """
    Diagnóstico de calidad del muestreo pSGLD mediante MMD y W₁.

    Args:
        n_epochs          : épocas de entrenamiento pSGLD
        n_snapshot_steps  : pasos adicionales para construir ν̂*
        thin              : sub-muestreo de snapshots (1 de cada `thin`)
        n_prior_samples   : muestras de ν^∞ a generar por Langevin
        epsilon           : temperatura (ε)
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO H  —  Diagnóstico: pSGLD → ν*")
    print("=" * 62)
    print(f"  ε={epsilon}  |  {n_epochs} epochs train  "
          f"|  {n_snapshot_steps} pasos snapshots (thin={thin})")

    # ── 1. Dataset ─────────────────────────────────────────────────────────
    X_np, y_np = make_moons(n_samples=500, noise=0.12, random_state=42)
    scaler = StandardScaler().fit(X_np)
    X_np   = scaler.transform(X_np).astype(np.float32)
    y_np   = y_np.astype(np.float32)
    X = torch.tensor(X_np, device=DEVICE)
    y = torch.tensor(y_np, device=DEVICE)

    # ── 2. Entrenamiento pSGLD ──────────────────────────────────────────────
    torch.manual_seed(42)
    np.random.seed(42)
    model = MeanFieldResNet().to(DEVICE)
    print("\n  [1/4] Entrenando con pSGLD...")
    train(model, X, y, epsilon=epsilon, n_epochs=n_epochs,
          verbose=True, use_sgld=True)

    # ── 3. Partículas finales ν_N ──────────────────────────────────────────
    particles_final = extract_particles(model)   # (64, 5)
    print(f"\n  [2/4] Recogiendo snapshots pSGLD "
          f"({n_snapshot_steps} pasos, thin={thin})...")
    snapshots = collect_psgld_snapshots(
        model, X, y, epsilon,
        n_steps=n_snapshot_steps, thin=thin
    )   # (n_snapshot_steps//thin, 64, 5)
    particles_timeavg = snapshots.reshape(-1, 5)   # (n_snaps*64, 5)
    print(f"     → {particles_timeavg.shape[0]} muestras de ν̂*")

    # ── 4. Prior ν^∞ por Langevin MCMC ────────────────────────────────────
    print(f"\n  [3/4] Generando {n_prior_samples} muestras del prior ν^∞...")
    prior_samples = sample_prior_langevin(
        n_prior_samples, dim=5
    )   # (n_prior_samples, 5)

    # ── 5. Métricas ────────────────────────────────────────────────────────
    print("\n  [4/4] Calculando MMD² y W₁...")
    # Sub-muestra para que MMD sea O(n²) manejable
    N_mmd = min(600, particles_timeavg.shape[0])
    idx_t = torch.randperm(particles_timeavg.shape[0])[:N_mmd]
    idx_p = torch.randperm(prior_samples.shape[0])[:N_mmd]
    pt    = particles_timeavg[idx_t]
    ps    = prior_samples[idx_p]
    pf    = particles_final       # solo 64, no es necesario sub-muestrear

    mmd_fn_ta = mmd_rbf(pf, pt)
    mmd_ta_pr = mmd_rbf(pt, ps)
    mmd_fn_pr = mmd_rbf(pf, ps)

    print(f"\n  MMD²(ν_N_final, ν̂*)   = {mmd_fn_ta:.4e}"
          f"  ← pequeño si ν_N ≈ ν*")
    print(f"  MMD²(ν̂*,        ν^∞)  = {mmd_ta_pr:.4e}"
          f"  ← grande si datos desplazan ν*")
    print(f"  MMD²(ν_N_final, ν^∞)  = {mmd_fn_pr:.4e}")

    # W₁ por dimensión
    w1_fn_ta = [wasserstein_1d(pf[:, d], particles_timeavg[:, d]) for d in range(5)]
    w1_ta_pr = [wasserstein_1d(particles_timeavg[:, d], prior_samples[:, d])
                for d in range(5)]
    w1_fn_pr = [wasserstein_1d(pf[:, d], prior_samples[:, d]) for d in range(5)]

    print("\n  W₁ por dimensión:")
    print(f"  {'dim':<6} {'ν_N vs ν̂*':>12} {'ν̂* vs ν^∞':>12} {'ν_N vs ν^∞':>12}")
    for d in range(5):
        print(f"  {DIM_LABELS[d]:<6} "
              f"{w1_fn_ta[d]:>12.4f} {w1_ta_pr[d]:>12.4f} {w1_fn_pr[d]:>12.4f}")

    # ── 6. Datos de la figura ──────────────────────────────────────────────
    # Orden de las comparaciones: [ν_N vs ν̂*, ν̂* vs ν^∞, ν_N vs ν^∞]
    publish('H', dict(
        eps=epsilon, n_epochs=n_epochs, n_snapshot_steps=n_snapshot_steps, thin=thin,
        final=pf, timeavg=particles_timeavg, prior=prior_samples,
        mmd=[mmd_fn_ta, mmd_ta_pr, mmd_fn_pr],
        w1=[w1_fn_ta, w1_ta_pr, w1_fn_pr],
    ))

    return {
        'particles_final':    particles_final,
        'particles_timeavg':  particles_timeavg,
        'prior_samples':      prior_samples,
        'mmd_final_timeavg':  mmd_fn_ta,
        'mmd_timeavg_prior':  mmd_ta_pr,
        'mmd_final_prior':    mmd_fn_pr,
        'w1_final_timeavg':   w1_fn_ta,
        'w1_timeavg_prior':   w1_ta_pr,
        'w1_final_prior':     w1_fn_pr,
    }
