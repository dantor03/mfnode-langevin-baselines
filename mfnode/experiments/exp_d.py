"""
=============================================================================
exp_d.py — Experimento D: Genericidad del minimizador (Meta-Teorema 1)
=============================================================================
"""

import numpy as np
import torch

from ..config import DEVICE
from ..data import get_moons
from ..model import MeanFieldResNet
from ..train import train, mu_pl_estimate
from ..plots import decision_grid
from ..plotting import publish


# =============================================================================
# EXPERIMENTO D
#   Genericidad: robustez a semillas de inicialización y de datos
# =============================================================================
def experiment_D(n_seeds: int = 10, n_epochs: int = 500):
    """
    Genericidad del minimizador: robustez a semillas de inicialización y de datos.

    META-TEOREMA 1 (paper, sec. 1.3):
        Para un conjunto abierto y denso 𝒪 de condiciones iniciales γ₀, el
        problema de control tiene un único minimizador ESTABLE.  La palabra
        "genéricamente" significa que casi toda distribución inicial de datos
        produce un paisaje de pérdida con un único mínimo profundo.

    CONEXIÓN CON LA CONDICIÓN PL (Meta-Teorema 2):
        Si la condición PL se cumple con μ > 0, gradient descent NO puede
        quedar atrapado en mínimos locales distintos: la desigualdad garantiza
        que desde cualquier punto del espacio de parámetros hay "cuesta abajo"
        hacia el mínimo global.  Esto implica robustez a la inicialización.

    DISEÑO DEL EXPERIMENTO:
        Dos sub-experimentos para separar dos fuentes de variabilidad:

        D1 — Robustez a la INICIALIZACIÓN (γ₀ fija, semilla de ν₀ variable):
            • Dataset fijo: data_seed = 42  (misma γ₀ en todos los runs)
            • Parámetros inicializados con n_seeds semillas distintas
            • Para ε ∈ {0, 0.01}: ¿convergen al mismo J*?
            • Predicción del paper: con ε > 0 la varianza de J* debe ser baja
              (unicidad del minimizador garantizada por PL)

        D2 — Robustez a γ₀ (inicialización fija, semilla de datos variable):
            • init_seed = 4  (semilla que converge bien en D1)
            • Dataset generado con n_seeds semillas distintas → n_seeds γ₀ distintas
            • Para ε ∈ {0, 0.01}: ¿converge siempre?
            • Predicción del paper: genericidad → casi toda γ₀ admite minimizador
              estable; las fronteras de decisión deben ser cualitativamente similares

        D3 — Variabilidad conjunta (ambas semillas varían simultáneamente):
            • Para la semilla s ∈ {0,...,n_seeds-1}: data_seed=s E init_seed=s
            • Ni el dataset ni los parámetros iniciales se repiten entre runs
            • Es el escenario más realista: en la práctica no se controla ninguna
              de las dos fuentes de aleatoriedad
            • Predicción: mayor dispersión que D1 y D2 por separado; con ε > 0
              la banda debe ser más estrecha que con ε = 0
            • Las fronteras de D3 combinan variabilidad geométrica (γ₀) y
              variabilidad de paisaje (θ₀) — aun así deben ser topológicamente
              similares si el Meta-Teorema 1 se cumple

    FIGURA (D_genericity.png, datos en figures/data/D.npz), 2×3:
        Fila 1: J por época (mediana y rango sobre semillas) con ε = 0 y
                ε = 0.01, una columna por protocolo D1, D2, D3; J* en la leyenda.
        Fila 2: fronteras de decisión superpuestas con ε = 0.01 de los runs
                con accuracy ≥ 0.95, sobre los datos de todos los runs.

    RESULTADO ESPERADO:
        • D3 debe tener la banda más ancha de los tres sub-experimentos
          (combina ambas fuentes de variabilidad)
        • Con ε=0.01 la banda de D3 debe ser más estrecha que con ε=0
        • Las fronteras de D3 son cualitativamente similares aunque provienen
          de puntos de partida totalmente distintos en (γ₀, θ₀)
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO D  —  Genericidad: robustez a semillas")
    print("=" * 62)

    SEEDS = list(range(n_seeds))
    EPS_COMPARE = [0.0, 0.01]
    DATA_SEED_FIXED = 42
    INIT_SEED_FIXED = 2   # seed 2 da el J* más bajo en D1 (J*≈0.00015) → D2 muestra
                          # variabilidad real de γ₀ sin contaminación por
                          # una mala inicialización

    # ── D1: γ₀ fija, inicialización variable ─────────────────────────────────
    print(f"\n  D1 — γ₀ fija (data_seed={DATA_SEED_FIXED}), "
          f"{n_seeds} seeds de inicialización")
    X, y, X_np, y_np = get_moons(seed=DATA_SEED_FIXED)

    d1_results = {}   # d1_results[eps] = lista de dicts {hist, model, mu}
    for eps in EPS_COMPARE:
        print(f"    ε={eps}:")
        d1_results[eps] = []
        for s in SEEDS:
            # Fijamos la semilla de inicialización para reproducibilidad
            torch.manual_seed(s)
            np.random.seed(s)
            model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
            hist  = train(model, X, y, epsilon=eps,
                          n_epochs=n_epochs, verbose=False)
            mu    = mu_pl_estimate(hist)
            print(f"      init_seed={s}: J*={hist['J_star']:.5f} | "
                  f"acc={hist['accuracy'][-1]:.3f} | μ̂={mu:.4f}")
            d1_results[eps].append({'hist': hist, 'model': model, 'mu': mu})

    # ── D2: inicialización fija, γ₀ variable ─────────────────────────────────
    print(f"\n  D2 — init fija (init_seed={INIT_SEED_FIXED}), "
          f"{n_seeds} seeds de datos (γ₀ distintas)")

    d2_results = {}   # d2_results[eps] = lista de dicts {hist, model, mu, X_np, y_np}
    for eps in EPS_COMPARE:
        print(f"    ε={eps}:")
        d2_results[eps] = []
        for s in SEEDS:
            Xs, ys, Xs_np, ys_np = get_moons(seed=s)
            # Misma inicialización de parámetros en todos los runs de D2
            torch.manual_seed(INIT_SEED_FIXED)
            np.random.seed(INIT_SEED_FIXED)
            model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
            hist  = train(model, Xs, ys, epsilon=eps,
                          n_epochs=n_epochs, verbose=False)
            mu    = mu_pl_estimate(hist)
            print(f"      data_seed={s}: J*={hist['J_star']:.5f} | "
                  f"acc={hist['accuracy'][-1]:.3f} | μ̂={mu:.4f}")
            d2_results[eps].append({
                'hist': hist, 'model': model, 'mu': mu,
                'X_np': Xs_np, 'y_np': ys_np
            })

    # ── D3: ambas semillas varían simultáneamente ─────────────────────────────
    print(f"\n  D3 — ambas semillas varían (data_seed=s, init_seed=s), "
          f"{n_seeds} pares distintos")

    d3_results = {}   # d3_results[eps] = lista de dicts {hist, model, mu, X_np, y_np}
    for eps in EPS_COMPARE:
        print(f"    ε={eps}:")
        d3_results[eps] = []
        for s in SEEDS:
            Xs, ys, Xs_np, ys_np = get_moons(seed=s)
            torch.manual_seed(s)
            np.random.seed(s)
            model = MeanFieldResNet(d1=2, M=64, T=1.0, n_steps=10).to(DEVICE)
            hist  = train(model, Xs, ys, epsilon=eps,
                          n_epochs=n_epochs, verbose=False)
            mu    = mu_pl_estimate(hist)
            print(f"      seed={s}: J*={hist['J_star']:.5f} | "
                  f"acc={hist['accuracy'][-1]:.3f} | μ̂={mu:.4f}")
            d3_results[eps].append({
                'hist': hist, 'model': model, 'mu': mu,
                'X_np': Xs_np, 'y_np': ys_np
            })

    # ── Resumen estadístico ───────────────────────────────────────────────────
    print("\n  RESUMEN — Robustez a init seeds (D1):")
    for eps in EPS_COMPARE:
        jstars = [r['hist']['J_star'] for r in d1_results[eps]]
        mus    = [r['mu']             for r in d1_results[eps]]
        print(f"    ε={eps:5.3f}: J* = {np.mean(jstars):.5f} ± {np.std(jstars):.5f} | "
              f"μ̂ = {np.mean(mus):.4f} ± {np.std(mus):.4f}")

    print("\n  RESUMEN — Robustez a data seeds (D2):")
    for eps in EPS_COMPARE:
        jstars = [r['hist']['J_star'] for r in d2_results[eps]]
        mus    = [r['mu']             for r in d2_results[eps]]
        print(f"    ε={eps:5.3f}: J* = {np.mean(jstars):.5f} ± {np.std(jstars):.5f} | "
              f"μ̂ = {np.mean(mus):.4f} ± {np.std(mus):.4f}")

    print("\n  RESUMEN — Variabilidad conjunta (D3):")
    for eps in EPS_COMPARE:
        jstars = [r['hist']['J_star'] for r in d3_results[eps]]
        mus    = [r['mu']             for r in d3_results[eps]]
        print(f"    ε={eps:5.3f}: J* = {np.mean(jstars):.5f} ± {np.std(jstars):.5f} | "
              f"μ̂ = {np.mean(mus):.4f} ± {np.std(mus):.4f}")

    # ── Datos de la figura ───────────────────────────────────────────────────
    # P(y = 1 | x) de todos los runs con ε = 0.01 en una rejilla común por
    # protocolo; la figura dibuja solo las fronteras de los runs con acc ≥ 0.95.
    eps_b = EPS_COMPARE[-1]
    data = {'eps': np.array(EPS_COMPARE, dtype=float)}
    for p, res in (('d1', d1_results), ('d2', d2_results), ('d3', d3_results)):
        Xs = np.stack([r.get('X_np', X_np) for r in res[eps_b]])
        ys = np.stack([r.get('y_np', y_np) for r in res[eps_b]])
        lo = Xs.reshape(-1, 2).min(axis=0) - 0.15
        hi = Xs.reshape(-1, 2).max(axis=0) + 0.15
        probs = []
        for r in res[eps_b]:
            gx, gy, P = decision_grid(r['model'], bounds=(lo[0], hi[0], lo[1], hi[1]), n=150)
            probs.append(P)
        data.update({
            f'loss_{p}':   [[r['hist']['loss'] for r in res[e]] for e in EPS_COMPARE],
            f'J_star_{p}': [[r['hist']['J_star'] for r in res[e]] for e in EPS_COMPARE],
            f'mu_{p}':     [[r['mu'] for r in res[e]] for e in EPS_COMPARE],
            f'acc_{p}':    [[r['hist']['accuracy'][-1] for r in res[e]] for e in EPS_COMPARE],
            f'X_{p}': Xs, f'y_{p}': ys, f'grid_x_{p}': gx, f'grid_y_{p}': gy,
            f'P_{p}': np.stack(probs).astype(np.float32),
        })
    publish('D', data)

    print("\n  INTERPRETACIÓN:")
    print("  • D3 combina ambas fuentes de variabilidad: esperar banda más ancha.")
    print("  • Con ε=0.01 la banda de D3 debe ser más estrecha que con ε=0.")
    print("  • Las fronteras de D3 deben ser topológicamente similares (Meta-Teorema 1).")

    return d1_results, d2_results, d3_results
