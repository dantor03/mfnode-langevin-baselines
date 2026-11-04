"""
=============================================================================
exp_g.py — Experimento G: Convergence Problem (N → ∞)
=============================================================================

Daudin & Delarue (2025) dejan explícitamente fuera del análisis el caso de
medidas iniciales empíricas, declarando que "la convergencia cuando N → ∞
se abordará en un trabajo futuro".  Este experimento da la primera evidencia
empírica de esa convergencia.

PREGUNTA CENTRAL:
    A medida que el tamaño del dataset de entrenamiento N crece,
    ¿converge el minimizador J*_N y la pérdida generalizada BCE_test(N)
    hacia un valor asintótico J*_∞?  ¿Acelera ε > 0 esa convergencia?

PROTOCOLO:
    - Oráculo de test: N_test = 10.000 puntos (seed=999, fijo).
      Aproxima la medida continua γ₀.  El StandardScaler se ajusta sobre
      este oráculo y se aplica también a los datos de entrenamiento, para
      que todos los runs vivan en el mismo espacio de features.

    - Entrenamiento: N ∈ {50, 100, 200, 400, 800, 1600, 3200}.
      Para cada N y cada semilla, se muestrean N puntos frescos de
      make_moons (misma distribución, seed = semilla del run).

    - ε ∈ {0.0, 0.01}.  Con ε=0 y use_sgld=True el ruido es exactamente
      cero (= Adam estándar).  Con ε=0.01 se usa pSGLD precondicionado.

    - n_seeds runs independientes por combinación (N, ε).

MÉTRICAS:
    - BCE_train : entropía cruzada sobre los N puntos de entrenamiento
                 (evaluada sin ruido al final del entrenamiento).
    - J*_train  : mínimo de la pérdida total J = BCE + ε·E durante el
                 entrenamiento (lo que el optimizador realmente minimizó).
    - BCE_test  : entropía cruzada sobre el oráculo de 10.000 puntos.
                 NO incluye el término de regularización: mide la calidad
                 predictiva pura del modelo aprendido.
    - acc_train / acc_test : accuracy sobre entrenamiento y oráculo.
    - Gap       : BCE_test − BCE_train (gap de generalización).

FIGURAS (G_convergence_problem.png, datos en figures/data/G.npz):
    G1 — BCE_train (discontinua) y BCE_test (continua) frente a N, media ± sd
         sobre semillas.  Se espera que ambas converjan y que el gap se cierre.
    G2 — Gap BCE_test − BCE_train frente a N ≥ 100 en log-log, con el ajuste
         gap ∝ N^α de cada ε y la referencia N^{-1/2}.
"""

import os

import numpy as np
import torch
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

from ..config import DEVICE, OUTPUT_DIR
from ..model import MeanFieldResNet
from ..train import train
from ..plotting import publish


def _save_results_G(results, epsilons, n_values, path, n_epochs):
    """Serializa el dict results a .npz para poder reanudar o regenerar la figura."""
    save_dict = {'EPSILONS': np.array(epsilons), 'N_VALUES': np.array(n_values),
                 'n_epochs': np.array(n_epochs)}
    for ei, eps in enumerate(epsilons):
        for ni, N in enumerate(n_values):
            r = results[eps][N]
            for metric in ('BCE_train', 'J_star_train', 'BCE_test', 'acc_train', 'acc_test'):
                save_dict[f'{metric}_{ei}_{ni}'] = np.array(r[metric])
    np.savez(path, **save_dict)


def _load_results_G(path, epsilons, n_values, n_epochs):
    """Carga results desde .npz.  Devuelve None si no existe o es de otro n_epochs."""
    if not os.path.exists(path):
        return None
    data = np.load(path)
    if 'n_epochs' in data and int(data['n_epochs']) != n_epochs:
        print(f"  ✗ {path} es de {int(data['n_epochs'])} épocas, no de {n_epochs}: se ignora")
        return None
    results = {
        eps: {N: {'BCE_train': [], 'J_star_train': [], 'BCE_test': [],
                  'acc_train': [], 'acc_test': []}
              for N in n_values}
        for eps in epsilons
    }
    for ei, eps in enumerate(epsilons):
        for ni, N in enumerate(n_values):
            for metric in ('BCE_train', 'J_star_train', 'BCE_test', 'acc_train', 'acc_test'):
                key = f'{metric}_{ei}_{ni}'
                if key in data:
                    results[eps][N][metric] = data[key].tolist()
    print(f"  ✓ Resultados parciales cargados desde {path}")
    return results


def experiment_G(n_seeds: int = 5, n_epochs: int = 700):
    """
    Convergence Problem: estudio empírico de J*_N → J*_∞ cuando N → ∞.

    Args:
        n_seeds  : número de semillas por combinación (N, ε)
        n_epochs : épocas de entrenamiento por run
    """
    print("\n" + "=" * 62)
    print("EXPERIMENTO G  —  Convergence Problem: N → ∞")
    print("=" * 62)

    N_VALUES = [50, 100, 200, 400, 800, 1600, 3200]
    EPSILONS = [0.0, 0.01]
    N_TEST   = 10_000
    SEEDS    = list(range(n_seeds))
    RESULTS_FILE = os.path.join(OUTPUT_DIR, 'G_results.npz')

    # ── Oráculo de test ───────────────────────────────────────────────────────
    # Se usa seed=999 para evitar solapamiento con las semillas de entrenamiento.
    # El scaler se ajusta sobre el oráculo y se aplica a TODOS los runs, así
    # todos viven en el mismo espacio de features.
    X_oracle_raw, y_oracle_raw = make_moons(
        n_samples=N_TEST, noise=0.12, random_state=999
    )
    oracle_scaler = StandardScaler().fit(X_oracle_raw)
    X_oracle = oracle_scaler.transform(X_oracle_raw).astype(np.float32)
    y_oracle = y_oracle_raw.astype(np.float32)
    X_test   = torch.tensor(X_oracle, device=DEVICE)
    y_test   = torch.tensor(y_oracle, device=DEVICE)
    print(f"  Oráculo test: {N_TEST} puntos (seed=999, scaler fijo)")

    # ── Carga de resultados previos (reanudación) ─────────────────────────────
    # Si existe G_results.npz de una ejecución anterior (o parcial en Colab),
    # se cargan los resultados ya calculados y se saltan esas combinaciones.
    results = _load_results_G(RESULTS_FILE, EPSILONS, N_VALUES, n_epochs)
    if results is None:
        results = {
            eps: {
                N: {'BCE_train': [], 'J_star_train': [], 'BCE_test': [],
                    'acc_train': [], 'acc_test': []}
                for N in N_VALUES
            }
            for eps in EPSILONS
        }

    # ── Bucle principal ───────────────────────────────────────────────────────
    for eps in EPSILONS:
        print(f"\n  ε = {eps} ──────────────────────────────────────────────")
        for N in N_VALUES:
            # Saltar si ya tenemos suficientes semillas para esta combinación
            already = len(results[eps][N]['BCE_test'])
            if already >= len(SEEDS):
                r  = results[eps][N]
                bt = np.array(r['BCE_train'])
                te = np.array(r['BCE_test'])
                print(f"    N={N:4d}: ya calculado ({already} seeds) — "
                      f"BCE_test={te.mean():.4f}±{te.std():.4f}  "
                      f"gap={(te - bt).mean():.4f}")
                continue
            for s in SEEDS[already:]:
                # Dataset de entrenamiento: N puntos frescos, escalados con el
                # scaler del oráculo para consistencia entre runs.
                X_raw, y_raw = make_moons(
                    n_samples=N, noise=0.12, random_state=s
                )
                X_np = oracle_scaler.transform(X_raw).astype(np.float32)
                y_np = y_raw.astype(np.float32)
                X    = torch.tensor(X_np, device=DEVICE)
                y    = torch.tensor(y_np, device=DEVICE)

                torch.manual_seed(s)
                np.random.seed(s)
                model = MeanFieldResNet().to(DEVICE)
                hist  = train(model, X, y, epsilon=eps,
                              n_epochs=n_epochs, verbose=False,
                              use_sgld=True)

                # ── Métricas de entrenamiento ─────────────────────────────
                J_star_train = hist['J_star']
                BCE_train    = hist['loss_term'][-1]   # BCE final (sin reg.)
                acc_train    = hist['accuracy'][-1]

                # ── Métricas sobre el oráculo ─────────────────────────────
                model.eval()
                with torch.no_grad():
                    # compute_loss con eps=0 → solo BCE (sin regularización)
                    _, BCE_test_t, _ = model.compute_loss(X_test, y_test, 0.0)
                    BCE_test = BCE_test_t  # ya es float (loss_term es scalar)
                    acc_test = (
                        (model(X_test) > 0).float() == y_test
                    ).float().mean().item()

                results[eps][N]['BCE_train'].append(BCE_train)
                results[eps][N]['J_star_train'].append(J_star_train)
                results[eps][N]['BCE_test'].append(BCE_test)
                results[eps][N]['acc_train'].append(acc_train)
                results[eps][N]['acc_test'].append(acc_test)

            # Resumen por N + guardado incremental
            r  = results[eps][N]
            bt = np.array(r['BCE_train'])
            te = np.array(r['BCE_test'])
            print(f"    N={N:4d}: BCE_train={bt.mean():.4f}±{bt.std():.4f}  "
                  f"BCE_test={te.mean():.4f}±{te.std():.4f}  "
                  f"gap={(te - bt).mean():.4f}  "
                  f"acc_test={np.mean(r['acc_test']):.3f}")
            # Guardado incremental: si Colab se corta, los resultados hasta aquí
            # están en G_results.npz y se pueden reanudar en la siguiente sesión.
            _save_results_G(results, EPSILONS, N_VALUES, RESULTS_FILE, n_epochs)
            print("      → checkpoint guardado en G_results.npz")

    # ── Datos de la figura ───────────────────────────────────────────────────
    def _metric(name):
        """(n_eps, n_N, n_seeds); un caché con más semillas se recorta a n_seeds."""
        return np.array([[results[e][N][name][:n_seeds] for N in N_VALUES]
                         for e in EPSILONS])

    data = dict(N=np.array(N_VALUES), eps=np.array(EPSILONS, dtype=float),
                n_epochs=n_epochs, n_test=N_TEST,
                bce_train=_metric('BCE_train'), bce_test=_metric('BCE_test'),
                J_star_train=_metric('J_star_train'),
                acc_train=_metric('acc_train'), acc_test=_metric('acc_test'))

    # Tasa del gap: ajuste lineal log(gap) = α·log(N) + c con N ≥ 100
    Ns   = np.array(N_VALUES, dtype=float)
    mask = Ns >= 100
    print("\n  Tasas de convergencia estimadas (gap ∝ N^α, ajuste N≥100):")
    for k, eps in enumerate(EPSILONS):
        gap = data['bce_test'][k].mean(1) - data['bce_train'][k].mean(1)
        alpha, _ = np.polyfit(np.log(Ns[mask]), np.log(np.maximum(gap[mask], 1e-6)), 1)
        print(f"    ε={eps}: α = {alpha:.3f}  (gap ∝ N^{{{alpha:.2f}}})")

    publish('G', data)
    return results
