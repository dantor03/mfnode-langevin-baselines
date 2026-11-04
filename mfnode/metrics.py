"""
metrics.py: metrics to diagnose and regularize ν_N against the prior ν^∞.

Since ν_N = (1/M) Σ δ_{a^m} is discrete and the prior ν^∞ is continuous,
KL(ν_N || ν^∞) = +∞.  This module implements well-defined alternatives:

  DIAGNOSTICS (no computational graph):
    extract_particles        : the M parameter vectors, shape (M, 5)
    mmd_rbf                  : unbiased MMD² with a Gaussian kernel (median bandwidth)
    wasserstein_1d           : exact 1D Wasserstein-1 via scipy
    sample_prior_langevin    : samples of ν^∞ ∝ exp(-ℓ(a)) by ULA
    collect_psgld_snapshots  : continues pSGLD and records the particle trajectory

  TRAINING (differentiable with respect to the particles):
    particles_live           : particles WITH gradient (to be used inside a loss)
    compute_mmd_sigma        : median heuristic for the kernel bandwidth
    mmd_loss_train           : differentiable MMD²(particles, prior)
    sinkhorn_loss            : differentiable debiased Sinkhorn divergence (entropic OT)
"""

import copy
import math

import torch
import torch.nn as nn
import torch.optim as optim


# =============================================================================
# Particle extraction
# =============================================================================

def extract_particles(model):
    """
    Return the M per-neuron parameter vectors a^m = (a₀^m, a₁^m, a₂^m).

    Mapping from the layout of MeanFieldVelocity:
        W0.weight : (d1, M)    column m = a₀^m ∈ ℝ^{d1}
        W1.weight : (M, d1+1)  row m, first d1 columns = a₁^m ∈ ℝ^{d1}
        W1.bias   : (M,)       a₂^m ∈ ℝ (per-neuron bias)

    Note: W1.weight[:, d1] is the weight of the time input (an artefact of the
    time augmentation), not part of the space A of the paper.

    Returns:
        (M, 2*d1+1) CPU tensor with the coordinates of the M particles.
    """
    d1 = model.velocity.d1
    a0 = model.velocity.W0.weight.T.detach().cpu()           # (M, d1)
    a1 = model.velocity.W1.weight[:, :d1].detach().cpu()     # (M, d1)
    a2 = model.velocity.W1.bias.unsqueeze(1).detach().cpu()  # (M, 1)
    return torch.cat([a0, a1, a2], dim=1)                     # (M, 2*d1+1)


# =============================================================================
# MMD² with a Gaussian kernel
# =============================================================================

def mmd_rbf(X, Y, sigma=None):
    """
    Unbiased estimator of MMD² between samples X (N, d) and Y (M, d).

    Kernel:  k(x, y) = exp(−‖x−y‖² / (2σ²)).
    sigma=None uses the median heuristic on the pooled sample X ∪ Y.

    The unbiased estimator drops the diagonal terms of Kxx and Kyy:
        MMD² = E[k(x,x')] − 2 E[k(x,y)] + E[k(y,y')]
    with the normalization 1/(n(n−1)) instead of 1/n².

    Returns:
        float, the estimated MMD² (can be slightly negative for small n, m).
    """
    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    if not isinstance(Y, torch.Tensor):
        Y = torch.tensor(Y, dtype=torch.float32)
    X, Y = X.float().cpu(), Y.float().cpu()

    if sigma is None:
        Z     = torch.cat([X, Y], dim=0)
        diffs = Z.unsqueeze(1) - Z.unsqueeze(0)
        dists = (diffs ** 2).sum(-1).sqrt()
        pos   = dists[dists > 0]
        sigma = float(pos.median()) if len(pos) > 0 else 1.0
        sigma = max(sigma, 1e-8)

    def K(A, B):
        d = A.unsqueeze(1) - B.unsqueeze(0)           # (n, m, d)
        return torch.exp(-(d ** 2).sum(-1) / (2.0 * sigma ** 2))

    Kxx = K(X, X)
    Kxx.fill_diagonal_(0.0)
    Kyy = K(Y, Y)
    Kyy.fill_diagonal_(0.0)
    Kxy = K(X, Y)
    n, m = X.shape[0], Y.shape[0]

    return float(
        Kxx.sum() / (n * (n - 1))
        - 2.0 * Kxy.sum() / (n * m)
        + Kyy.sum() / (m * (m - 1))
    )


# =============================================================================
# Exact 1D Wasserstein-1
# =============================================================================

def wasserstein_1d(x, y):
    """
    Exact Wasserstein-1 distance between two univariate samples.

    W₁(P, Q) = ∫ |F_P(t) − F_Q(t)| dt with the empirical CDFs, computed by
    scipy.stats.wasserstein_distance in O(n log n).

    Args:
        x, y: 1D tensors or arrays.
    Returns:
        float, W₁.
    """
    from scipy.stats import wasserstein_distance
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy()
    if torch.is_tensor(y):
        y = y.detach().cpu().numpy()
    return float(wasserstein_distance(x.ravel(), y.ravel()))


# =============================================================================
# Sampling the prior ν^∞ with the unadjusted Langevin algorithm (ULA)
# =============================================================================

def sample_prior_langevin(n_samples, dim=5,
                           n_burnin=5_000, thin=20, step=0.01):
    """
    Draw samples of ν^∞ ∝ exp(−ℓ(a)) with the unadjusted Langevin algorithm (ULA).

    The prior ν^∞ is fixed and independent of ε: the temperature ε only scales
    the weight of the entropic term in the loss, not the shape of the prior.

    Potential:  ℓ(a) = c₁ Σⱼ aⱼ⁴ + c₂ Σⱼ aⱼ²    (c₁ = 0.05, c₂ = 0.5)
    Drift:      −∇ℓ(a) = −(4c₁ a³ + 2c₂ a)        (componentwise)

    ULA update (Euler–Maruyama step of the Langevin SDE):
        a ← a − η ∇ℓ(a) + √(2η) ξ,   ξ ~ N(0, I)

    The continuous-time SDE has stationary law exp(−ℓ(a)); ULA adds an O(η)
    bias, kept small with a small `step` and thinning.

    Args:
        n_samples : number of samples returned
        dim       : dimension of A (2*d1+1 = 5 for d1 = 2)
        n_burnin  : burn-in steps discarded
        thin      : keep one of every `thin` steps after burn-in
        step      : step size η; Lip(∇ℓ) ≈ 2c₂ = 1 near the origin, so step ≪ 2

    Returns:
        (n_samples, dim) CPU tensor.
    """
    c1, c2    = 0.05, 0.5
    a         = torch.randn(dim) * 0.3
    noise_std = math.sqrt(2.0 * step)
    total     = n_burnin + n_samples * thin
    samples   = []

    for i in range(total):
        grad = 4.0 * c1 * a ** 3 + 2.0 * c2 * a
        a    = a - step * grad + noise_std * torch.randn(dim)
        if i >= n_burnin and (i - n_burnin) % thin == 0:
            samples.append(a.clone())

    return torch.stack(samples)   # (n_samples, dim)


# =============================================================================
# pSGLD trajectory snapshots after training
# =============================================================================

def collect_psgld_snapshots(model, X, y, epsilon,
                             n_steps=2_000, thin=10, lr=1e-3):
    """
    Continue pSGLD from the current model and record snapshots of the M particles.

    Works on a deep copy: the model passed in is not modified.

    After training, the pSGLD chain is approximately stationary.  The pooled
    snapshots approximate the time-averaged distribution ν̂*, which by the
    ergodic theorem converges to the stationary law ν* ∝ exp(−J(θ)/ε).
    A fixed learning rate (no cosine annealing) keeps the temperature constant
    during sampling.

    Args:
        model    : trained MeanFieldResNet (not modified)
        X, y     : training data (needed to evaluate ∇J)
        epsilon  : pSGLD temperature (the ε used in training)
        n_steps  : number of additional pSGLD steps
        thin     : keep one of every `thin` steps
        lr       : fixed learning rate of the sampling phase

    Returns:
        (n_steps // thin, M, dim) CPU tensor, the particle trajectory.
    """
    m_copy = copy.deepcopy(model).to(X.device)
    m_copy.train()

    opt    = optim.Adam(m_copy.parameters(), lr=lr)
    beta2  = opt.param_groups[0]['betas'][1]
    eps_a  = opt.param_groups[0]['eps']
    snapshots = []

    for step_i in range(n_steps):
        opt.zero_grad()
        loss, _, _ = m_copy.compute_loss(X, y, epsilon)
        loss.backward()
        nn.utils.clip_grad_norm_(m_copy.parameters(), max_norm=5.0)
        opt.step()

        if epsilon > 0:
            with torch.no_grad():
                for p in m_copy.parameters():
                    if p.grad is None:
                        continue
                    st = opt.state[p]
                    if len(st) == 0:
                        p.add_(torch.randn_like(p) * math.sqrt(2.0 * lr * epsilon))
                    else:
                        s_val = st['step']
                        if torch.is_tensor(s_val):
                            s_val = s_val.item()
                        v_hat = st['exp_avg_sq'] / (1.0 - beta2 ** s_val)
                        M_t   = (1.0 / (v_hat.sqrt() + eps_a)).clamp(max=1.0)
                        p.add_(torch.randn_like(p) * (2.0 * lr * epsilon * M_t).sqrt())

        if (step_i + 1) % thin == 0:
            snapshots.append(extract_particles(m_copy))

    return torch.stack(snapshots)   # (n_steps // thin, M, dim)


# =============================================================================
# Differentiable training losses
# =============================================================================

def particles_live(model):
    """
    Return the M parameter vectors WITH the computational graph.

    Same as extract_particles() but without .detach(), so that gradients flow
    to W0, W1 and the bias of MeanFieldVelocity.  Use only inside the training
    loop, as part of the loss.

    Returns:
        (M, 2*d1+1) tensor on the model's device, with gradient.
    """
    d1 = model.velocity.d1
    a0 = model.velocity.W0.weight.T           # (M, d1) — CON grad
    a1 = model.velocity.W1.weight[:, :d1]     # (M, d1) — CON grad
    a2 = model.velocity.W1.bias.unsqueeze(1)  # (M, 1)  — CON grad
    return torch.cat([a0, a1, a2], dim=1)      # (M, 2*d1+1)


def compute_mmd_sigma(particles, prior_samples):
    """
    Median heuristic for the bandwidth σ of the Gaussian kernel.

    Median of all pairwise distances in the pooled sample (particles ∪ prior).
    It is computed once before training and kept fixed, so that the loss
    landscape does not change during training.

    Args:
        particles     : (M, d) tensor (any device)
        prior_samples : (N, d) CPU tensor
    Returns:
        float, σ.
    """
    X = particles.detach().cpu().float()
    Y = prior_samples.cpu().float()
    # Submuestrea Y para que el cálculo sea O((M+N_sub)²) manejable
    N_sub = min(Y.shape[0], max(X.shape[0] * 4, 200))
    Y_sub = Y[torch.randperm(Y.shape[0])[:N_sub]]
    Z     = torch.cat([X, Y_sub], dim=0)
    diffs = Z.unsqueeze(1) - Z.unsqueeze(0)
    dists = (diffs ** 2).sum(-1).sqrt()
    pos   = dists[dists > 0]
    sigma = float(pos.median()) if len(pos) > 0 else 1.0
    return max(sigma, 1e-8)


def mmd_loss_train(X, Y, sigma, kyy_mean=None):
    """
    Differentiable MMD² between the particles X and prior samples Y.

    Biased estimator (1/n² instead of 1/(n(n−1))): the bias is O(1/n) and does
    not change the direction of the gradient, and it is more stable than the
    unbiased one for small n = M.

    The term E[k(y,y')] = kyy_mean does not depend on X: it can be computed
    once before training and passed in.

    Args:
        X        : (M, d) particles, with gradient (output of particles_live)
        Y        : (N, d) prior samples, fixed, on the same device
        sigma    : kernel bandwidth (from compute_mmd_sigma)
        kyy_mean : precomputed E[k(y,y')] (None: computed here)

    Returns:
        Scalar tensor, MMD² (differentiable with respect to X).
    """
    def K(A, B):
        d = A.unsqueeze(1) - B.unsqueeze(0)      # (n, m, dim)
        return torch.exp(-(d ** 2).sum(-1) / (2.0 * sigma ** 2))

    n, m = X.shape[0], Y.shape[0]
    Kxx  = K(X, X)
    Kxy  = K(X, Y)

    kxx_mean = Kxx.sum() / (n * n)
    kxy_mean = Kxy.sum() / (n * m)

    if kyy_mean is None:
        with torch.no_grad():
            Kyy      = K(Y, Y)
            kyy_mean = float(Kyy.sum() / (m * m))

    return kxx_mean - 2.0 * kxy_mean + kyy_mean


def sinkhorn_loss(X, Y, blur=0.05, n_iter=50, W_yy=None):
    """
    Debiased Sinkhorn divergence (entropic OT) between the particles and the prior.

    S_blur(X, Y) = OT_blur(X, Y) − ½ OT_blur(X, X) − ½ OT_blur(Y, Y),

    where OT_blur(X, Y) is the transport cost ⟨P, C⟩ of the entropic optimal
    plan P with regularization `blur`.  Debiasing gives S_blur(X, X) = 0 and
    removes the shrinking bias of plain entropic OT.

    Implemented with log-domain Sinkhorn iterations (numerically stable), with
    the cost matrix rescaled to [0, 1] to avoid overflow.

    Gradients:
        ∂S/∂X = ∂OT_blur(X, Y)/∂X − ½ ∂OT_blur(X, X)/∂X;
    OT_blur(Y, Y) does not depend on X and can be precomputed (argument W_yy).

    Args:
        X      : (n, d) particles, with gradient
        Y      : (m, d) prior samples, fixed, same device
        blur   : entropic regularization (> 0; smaller is closer to exact OT but
                 less stable; 0.05–0.1 is a reasonable range)
        n_iter : Sinkhorn iterations (50 is usually enough for blur ≥ 0.05)
        W_yy   : precomputed OT_blur(Y, Y) (None: computed here)

    Returns:
        Scalar tensor, S_blur(X, Y) (differentiable with respect to X).
    """
    def _cost(A, B):
        diff = A.unsqueeze(1) - B.unsqueeze(0)   # (n, m, d)
        return (diff ** 2).sum(-1)                # (n, m)

    def _ot(C, blur, n_iter):
        """OT_blur by log-domain Sinkhorn; C is the (n, m) rescaled cost."""
        n, m  = C.shape
        log_K = -C / blur
        log_u = torch.zeros(n, device=C.device, dtype=C.dtype)
        log_v = torch.zeros(m, device=C.device, dtype=C.dtype)
        ln_n, ln_m = math.log(n), math.log(m)
        for _ in range(n_iter):
            log_u = -ln_n - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v = -ln_m - torch.logsumexp(log_K.T + log_u.unsqueeze(0), dim=1)
        log_P = log_u.unsqueeze(1) + log_K + log_v.unsqueeze(0)
        return (log_P.exp() * C).sum()

    C_xy    = _cost(X, Y)
    scale   = C_xy.detach().max().clamp(min=1e-8)   # normalizar en escala original
    W_xy    = _ot(C_xy / scale, blur, n_iter) * scale

    C_xx    = _cost(X, X)
    W_xx    = _ot(C_xx / scale, blur, n_iter) * scale

    if W_yy is None:
        with torch.no_grad():
            C_yy = _cost(Y.detach(), Y.detach())
            W_yy = float(_ot(C_yy / scale.detach(), blur, n_iter) * scale.detach())

    return W_xy - 0.5 * W_xx - 0.5 * W_yy
