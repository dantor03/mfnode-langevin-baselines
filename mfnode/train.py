import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .metrics import (
    particles_live, compute_mmd_sigma,
    mmd_loss_train, sinkhorn_loss,
)


# =============================================================================
# § 4  TRAINING LOOP
# =============================================================================
def train(model, X, y, epsilon,
          lr: float = 0.01, n_epochs: int = 800, verbose: bool = True,
          use_sgd: bool = False, use_sgld: bool = False,
          use_mmd: bool = False, use_sinkhorn: bool = False,
          prior_samples=None, mmd_sigma: float = None,
          sinkhorn_blur: float = 0.05):
    """
    Training loop with four optimizer modes.

    - Default (use_sgd=False, use_sgld=False):
        Adam + cosine annealing.  Fast convergence; lr → 0 at the end can
        artificially flatten the curves (a scheduler artefact).

    - use_sgd=True:
        plain SGD with constant lr and NO noise: a clean deterministic gradient
        flow.  Used for exp_c (PL verification), where J* must be the actual
        minimum without artefacts.

    - use_sgld=True:
        pSGLD, preconditioned stochastic gradient Langevin dynamics
        (Li et al., 2016).  Each step is an Euler–Maruyama step of the
        preconditioned Langevin SDE:
            θ ← Adam(θ, ∇J) + √(2 η_t ε M_t) ξ,   ξ ~ N(0, I),
        where M_t[j] = min(1/(√(v̂_t[j]) + δ), 1) is Adam's preconditioner
        (bias-corrected second moment) for parameter j, capped at 1 to avoid
        exploding noise in flat directions (v̂_t ≈ 0).  The cap plays the role
        of the separate stabilization hyperparameter λ ≥ 1 of Li et al. (2016),
        instead of Adam's δ = 1e-8.
        The noise MUST be coupled to M_t: isotropic noise decouples the
        stochastic part from the preconditioner and the chain no longer has
        ν* ∝ exp(−J(θ)/ε) as stationary law.
        Adam is needed as the base optimizer: plain SGD does not converge
        because the BCE gradients are ~1e-5.  The L4 + L2 penalty acts as the
        prior ν^∞ ∝ exp(−ℓ(a)) and the noise supplies the entropy term −H(ν_t).
        With ε = 0 the noise is exactly zero (plain Adam).
        Used for exp_b, exp_e, exp_f (verification of ν*).

    - use_mmd=True / use_sinkhorn=True:
        deterministic Adam with the L4 + L2 penalty replaced by
        ε · MMD²(particles, prior) or ε · Sinkhorn(particles, prior).

    POLYAK–ŁOJASIEWICZ CONDITION (Meta-Theorem 2):
        ‖∇J(θ)‖² ≥ 2μ (J(θ) − J*)

    Metrics recorded at every epoch:
        loss       : J(θ) = terminal cost + ε · regularizer (evaluated WITHOUT noise)
        loss_term  : terminal cost alone
        loss_reg   : regularizer
        grad_norm2 : ‖∇J‖² before clipping, numerator of the PL ratio
        pl_ratio   : ‖∇J‖² / (2 (J − J*))
        accuracy   : fraction of correctly classified points (R² for regression)

    Recommended learning rate: lr = 0.01 in every mode.

    Args:
        model         : MeanFieldResNet to train
        X             : (N, d1) features (full batch)
        y             : (N,) labels
        epsilon       : ε, regularization weight and Langevin temperature
        lr            : initial learning rate
        n_epochs      : number of epochs
        verbose       : print progress every 200 epochs
        use_sgd       : SGD with constant lr, no noise (exp_c)
        use_sgld      : pSGLD, Adam + preconditioned Langevin noise (exp_b, e, f)
        use_mmd       : Adam + ε · MMD²(particles, prior_samples) instead of the
                        L4 + L2 penalty; deterministic; needs prior_samples
        use_sinkhorn  : Adam + ε · Sinkhorn divergence(particles, prior_samples);
                        like use_mmd but with entropic optimal transport, more
                        sensitive to the geometry of the parameter space;
                        needs prior_samples
        prior_samples : (N, d) precomputed samples of ν^∞ (sample_prior_langevin
                        in metrics.py), moved to the model's device
        mmd_sigma     : MMD kernel bandwidth (None: median heuristic, computed
                        once before training)
        sinkhorn_blur : entropic regularization of the Sinkhorn divergence

    Returns:
        hist : dict of per-epoch metric lists, plus 'J_star'.
    """
    # ── Validación de argumentos ──────────────────────────────────────────────
    if (use_mmd or use_sinkhorn) and prior_samples is None:
        raise ValueError("use_mmd and use_sinkhorn require prior_samples.")
    if sum([use_sgld, use_sgd, use_mmd, use_sinkhorn]) > 1:
        raise ValueError("Only one training mode can be active at a time.")

    # ── Optimizador y scheduler ───────────────────────────────────────────────
    if use_sgd:
        opt = optim.SGD(model.parameters(), lr=lr)
        sch = None
    else:
        # Adam + cosine annealing para todos los demás modos
        opt = optim.Adam(model.parameters(), lr=lr)
        sch = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=n_epochs)

    # ── Setup previo al bucle (MMD y Sinkhorn) ────────────────────────────────
    prior_dev   = None   # prior_samples en el dispositivo del modelo
    _sigma      = None   # ancho de banda MMD (fijo durante training)
    _kyy_mean   = None   # E[k(y,y')] precomputado (constante)
    _W_yy       = None   # OT_blur(Y,Y) precomputado (constante)

    if use_mmd or use_sinkhorn:
        prior_dev = prior_samples.to(X.device).float()

    if use_mmd:
        # Sigma: heurística de la mediana, calculada una vez
        _sigma = mmd_sigma if mmd_sigma is not None else compute_mmd_sigma(
            particles_live(model).detach(), prior_dev
        )
        # E[k(y,y')]: constante durante todo el entrenamiento
        with torch.no_grad():
            def _K(A, B, s):
                d = A.unsqueeze(1) - B.unsqueeze(0)
                return torch.exp(-(d ** 2).sum(-1) / (2.0 * s ** 2))
            Kyy = _K(prior_dev, prior_dev, _sigma)
            _kyy_mean = float(Kyy.sum() / (prior_dev.shape[0] ** 2))
        if verbose:
            print(f"    [MMD]  σ = {_sigma:.4f}  |  E[k(y,y')] = {_kyy_mean:.4f}")

    if use_sinkhorn:
        # W_yy: OT_blur(prior, prior), constante durante el entrenamiento
        with torch.no_grad():
            _W_yy = float(sinkhorn_loss(
                prior_dev, prior_dev, blur=sinkhorn_blur, n_iter=50
            ))
        if verbose:
            print(f"    [Sinkhorn]  blur = {sinkhorn_blur}  |  W_yy = {_W_yy:.4e}")

    hist = {'loss': [], 'loss_term': [], 'loss_reg': [],
            'grad_norm2': [], 'pl_ratio': [], 'accuracy': []}
    L_min = math.inf

    for ep in range(n_epochs):
        model.train()
        opt.zero_grad()

        # ── Cálculo de la pérdida según el modo activo ────────────────────────
        if use_mmd:
            # BCE (sin L4+L2) + ε·MMD²(partículas, prior)
            loss_bce, lt, _ = model.compute_loss(X, y, 0.0)
            pts     = particles_live(model)
            reg     = mmd_loss_train(pts, prior_dev, _sigma, _kyy_mean)
            loss    = loss_bce + epsilon * reg
            lr_val  = reg.item()
        elif use_sinkhorn:
            # BCE (sin L4+L2) + ε·Sinkhorn(partículas, prior)
            loss_bce, lt, _ = model.compute_loss(X, y, 0.0)
            pts     = particles_live(model)
            reg     = sinkhorn_loss(pts, prior_dev,
                                    blur=sinkhorn_blur, n_iter=50, W_yy=_W_yy)
            loss    = loss_bce + epsilon * reg
            lr_val  = reg.item()
        else:
            # pSGLD / Adam / SGD: BCE + ε·L4+L2
            loss, lt, lr_val = model.compute_loss(X, y, epsilon)

        loss.backward()

        # ‖∇J‖² se calcula DESPUÉS de backward() pero ANTES de clip y step.
        # Así medimos el gradiente "verdadero" antes de cualquier modificación,
        # que es lo que corresponde al numerador de la condición PL teórica.
        gn2 = sum(p.grad.pow(2).sum().item()
                  for p in model.parameters() if p.grad is not None)

        # Gradient clipping: evita explosiones en pasos con gradiente grande
        # (puede ocurrir en los primeros epochs cuando los params están lejos de J*)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        opt.step()

        # ── Ruido de Langevin precondicionado pSGLD (solo con use_sgld=True) ──
        # Implementa pSGLD (Li et al., 2016): el ruido está acoplado a la
        # matriz de precondicionamiento M_t de Adam, NO es isotrópico.
        #
        # La actualización correcta es:
        #   θ ← Adam(θ, ∇J) + √(2·η·ε·M_t)·ξ,   ξ ~ N(0, I)
        #
        # donde M_t[j] = 1 / (√(v̂_t[j]) + δ) es el precondicionador de Adam
        # para el parámetro j, con v̂_t[j] = v_t[j] / (1 - β₂ᵗ) el segundo
        # momento sesgado-corregido.
        #
        # Esto garantiza que la cadena de Markov converja a la distribución
        # de Gibbs ν* ∝ exp(−J(θ)/ε) con el precondicionador M_t.
        # Inyectar ruido isotrópico (σ=√(2ηε) igual para todos los parámetros)
        # desacopla el ruido de M_t y destruye la distribución estacionaria.
        #
        # Con ε=0 el ruido es exactamente cero (recupera Adam estándar).
        if use_sgld and epsilon > 0:
            current_lr = opt.param_groups[0]['lr']
            beta2      = opt.param_groups[0]['betas'][1]
            eps_adam   = opt.param_groups[0]['eps']
            with torch.no_grad():
                for p in model.parameters():
                    if p.grad is None:
                        continue
                    state = opt.state[p]
                    if len(state) == 0:
                        # Adam aún no inicializó estado (épocas muy tempranas):
                        # ruido isotrópico como fallback conservador.
                        noise_std = math.sqrt(2.0 * current_lr * epsilon)
                        p.add_(torch.randn_like(p) * noise_std)
                    else:
                        step = state['step']
                        if torch.is_tensor(step):
                            step = step.item()
                        v_t   = state['exp_avg_sq']
                        # Segundo momento sesgado-corregido: v̂_t = v_t/(1-β₂ᵗ)
                        bias_corr2 = 1.0 - beta2 ** step
                        v_hat = v_t / bias_corr2
                        # Precondicionador M_t[j] = 1/(√(v̂_t[j]) + δ)
                        # Clamp M a 1.0: en Li et al. 2016 el término de
                        # estabilización λ no es eps de Adam (1e-8) sino
                        # un hiperparámetro ≥ 1 que evita explosión de ruido
                        # en direcciones planas (v̂_t ≈ 0 → M ≈ 1/1e-8 → ∞).
                        # Con M ≤ 1, noise_std ≤ √(2·η·ε) = SGLD isotrópico:
                        # amplificamos ruido en direcciones curvas pero nunca
                        # más allá del nivel isotrópico.
                        M = 1.0 / (v_hat.sqrt() + eps_adam)
                        M = M.clamp(max=1.0)
                        # std del ruido: √(2·η·ε·M_t[j]) por componente
                        noise_std = (2.0 * current_lr * epsilon * M).sqrt()
                        p.add_(torch.randn_like(p) * noise_std)

        if sch is not None:
            sch.step()

        lv = loss.item()
        if lv < L_min:
            L_min = lv         # J* aproximado = mínimo observado hasta ahora
        # Ratio PL provisional (online): se recalculará con J* global al final
        excess = max(lv - L_min, 1e-10)
        pl = gn2 / (2.0 * excess) if excess > 1e-9 else float('nan')

        with torch.no_grad():
            pred = model(X)
            if getattr(model, 'task', 'classification') == 'regression':
                ss_res = ((pred - y) ** 2).sum()
                ss_tot = ((y - y.mean()) ** 2).sum().clamp(min=1e-8)
                acc = float(1.0 - ss_res / ss_tot)   # R²
            else:
                acc = ((pred > 0).float() == y).float().mean().item()

        hist['loss'].append(lv)
        hist['loss_term'].append(lt)
        hist['loss_reg'].append(lr_val)
        hist['grad_norm2'].append(gn2)
        hist['pl_ratio'].append(pl)
        hist['accuracy'].append(acc)

        if verbose and (ep + 1) % 200 == 0:
            print(f"    epoch {ep+1:4d} | J={lv:.4f} | BCE={lt:.4f} | "
                  f"reg={lr_val:.5f} | ‖∇J‖²={gn2:.3e} | acc={acc:.3f}")

    # Recalcular pl_ratio con J* GLOBAL (mínimo de toda la trayectoria de entrenamiento).
    # Esto es más justo que el J* online (que subestima J* en épocas tempranas):
    # el ratio ‖∇J‖²/(2(J−J*)) debe ser ≥ μ para todos los epochs cuando J* es fijo.
    J_star = min(hist['loss'])
    hist['J_star'] = J_star
    hist['pl_ratio'] = [
        gn2 / (2.0 * max(J - J_star, 1e-10))
        if J - J_star > 1e-9 else float('nan')
        for gn2, J in zip(hist['grad_norm2'], hist['loss'])
    ]
    return hist


def mu_pl_estimate(hist):
    """
    Conservative estimate of the PL constant μ: the 10th percentile of the PL ratio.

    The PL ratio at each epoch is ‖∇J‖²/(2(J − J*)).  If the PL inequality
    holds with constant μ, the ratio is ≥ μ at every epoch, so the natural
    estimator would be the minimum.  The minimum is very sensitive to numerical
    outliers (when J ≈ J* the denominator is tiny), so the 10th percentile is
    used instead: robust, and still conservative.

    Excluded values:
      • NaN (J − J* ≤ 1e-9, the model is already at J*)
      • values ≤ 0 (numerical anomalies)
      • values > 1e4 (gradient spikes not caught by clipping)
    """
    vals = [v for v in hist['pl_ratio']
            if not math.isnan(v) and 0 < v < 1e4]
    return np.percentile(vals, 10) if len(vals) > 10 else 0.0
