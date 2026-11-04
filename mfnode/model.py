import torch
import torch.nn as nn


# =============================================================================
# § 2  PROTOTYPICAL VECTOR FIELD  b(x, a) = σ(a₁·x + a₂)·a₀
#
#  Paper, Example 1.1 (eq. 1.8):
#      A = ℝ^{d₁} × ℝ^{d₁} × ℝ,    a = (a₀, a₁, a₂)
#      b(x, a) = σ(a₁ · x + a₂) · a₀,     σ = tanh
#
#  Here d₁ = 2 (features in ℝ²), a₀ ∈ ℝ², a₁ ∈ ℝ², a₂ ∈ ℝ.
#  This is the "prototypical neuron": a₁·x + a₂ is a linear pre-activation,
#  σ the nonlinearity, and a₀ rescales the output in ℝ^{d₁}.
#
#  With M particles (ν_t approximated by M atoms aᵐ):
#      F(x, t) = ∫_A b(x, a) dν_t(a) ≈ (1/M) Σₘ b(x, aᵐ(t))
#              = (1/M) Σₘ σ(a₁ᵐ(t)·x + a₂ᵐ(t)) · a₀ᵐ(t)
#
#  This is a one-hidden-layer network with M neurons whose parameters should
#  vary freely in t.  The paper optimizes over arbitrary curves (ν_t)_{t∈[0,T]},
#  which with M particles means optimizing M paths aᵐ : [0, T] → A.
#
#  TIME AUGMENTATION:
#      In the implementation the weights (W₁, W₀) are STATIC and t is
#      concatenated to the input, so the effective field is
#          F(x, t) = W₀ tanh(W₁[:, :d₁] x + W₁[:, d₁] t + b₁).
#      This restricts the controls to a₀ᵐ and a₁ᵐ CONSTANT in t and
#      a₂ᵐ(t) = W₁[m, d₁]·t + b₁[m] linear in t.  F(x, t) does depend on t,
#      but the curve ν_t is restricted to this parametric family.
#
#  Matrix form:
#      W₁ ∈ ℝ^{M×(d₁+1)} stacks (a₁ᵐ, time column) as rows
#      W₀ ∈ ℝ^{M×d₁}     stacks a₀ᵐ as rows
#      h = σ([x, t] @ W₁ᵀ + b₁)   → (N, M)
#      F = h @ W₀ᵀ / M             → (N, d₁)   (the 1/M is absorbed in the init)
#
#  tanh satisfies the "discriminating property" (Sec. 1.2), which ensures
#  that b(x, ·) can approximate any continuous function.
#
#  ENTROPIC PENALTY (Assumption Regularity (i) and (ii)):
#      ε · E(ν_t | ν^∞)   with   ν^∞ ∝ exp(-ℓ(a)),   ℓ(a) = c₁|a|⁴ + c₂|a|²
#
#  Why a QUARTIC potential (c₁|a|⁴):
#      the super-coercivity condition requires ∇²ℓ(a) ≥ c(1 + |a|²) I, which
#      the quadratic term c₂|a|² alone does not satisfy but c₁|a|⁴ does.  This
#      faster-than-quadratic growth gives the log-Sobolev inequality for ν^∞,
#      which in turn implies the PL condition.
# =============================================================================
class MeanFieldVelocity(nn.Module):
    """
    Prototypical vector field of the paper (eq. 1.8) approximated with M particles.

    Implements F(x, t) = (1/M) Σₘ σ(a₁ᵐ(t)·x + a₂ᵐ(t)) a₀ᵐ(t), the effective
    field that moves the features through the control equation
        dX_t/dt = F(X_t, t).

    The time dependence a(t) is encoded by augmenting the input with the
    scalar t, i.e. parameters that vary smoothly with the network time, as in a
    Neural ODE with weights shared across layers.

    Attributes:
        d1  : dimension of the feature space (d₁ = 2 for make_moons)
        M   : number of particles approximating the control measure ν_t
        W1  : weights [d₁+1 → M], encode (a₁ᵐ, a₂ᵐ) for m = 1..M
        W0  : weights [M → d₁], encode a₀ᵐ for m = 1..M
    """

    def __init__(self, d1: int = 2, M: int = 64,
                 warm_start_neurons=None, warm_start_sigma: float = 0.0):
        super().__init__()
        self.d1, self.M = d1, M
        # W₁: cada fila m es el parámetro (a₁ᵐ ∈ ℝ^{d₁}, a₂ᵐ ∈ ℝ) de la neurona m
        # La columna extra de input es para el tiempo t (augmentación temporal)
        self.W1 = nn.Linear(d1 + 1, M, bias=True)
        # W₀: cada fila m es a₀ᵐ ∈ ℝ^{d₁} — escala la salida de cada neurona
        # Sin bias: la simetría b(x,a)=0 cuando a₀=0 debe preservarse
        self.W0 = nn.Linear(M, d1, bias=False)
        if warm_start_neurons is None:
            # Inicialización pequeña (std=0.1):
            #   • Evita que la ODE "explote" en los primeros pasos de entrenamiento
            #   • Coherente con la teoría: parámetros cerca del prior ν^∞ (que también
            #     concentra masa cerca del origen) al inicio del entrenamiento
            nn.init.normal_(self.W1.weight, std=0.3)
            nn.init.zeros_(self.W1.bias)
            nn.init.normal_(self.W0.weight, std=0.3)
        else:
            # Warm-start por neurona: el caller suministra M filas en R^{2 d1 + 1}
            # con layout [a0 (d1), a1 (d1), a2 (1)] (mismo orden que A_grid).
            # Usado para: muestreo categórico de ν*_T (Thm 1.4), muestreo
            # Gaussiano N(0, Cov[ν*_T]), o cualquier otra distribución diseñada.
    # NOTE: restriction of the control space with respect to the paper.
    #   The paper optimizes over arbitrary curves a^m : [0, T] → A, i.e. the
    #   control ν_t can be any continuous curve of measures.  This
    #   implementation restricts it to
    #       • a₀ᵐ ∈ ℝ²  constant in t  (fixed output weights)
    #       • a₁ᵐ ∈ ℝ²  constant in t  (fixed input weights)
    #       • a₂ᵐ(t) = W₁[m, 2]·t + b₁[m]  linear in t  (only the bias varies)
    #   a strict subset of the continuous mean-field controls.  The guarantees
    #   of the paper hold on the full space; here they are tested empirically
    #   on this subspace.
            # b(x,a)=tanh(a₁·x+a₂)·a₀, así que init "todas las neuronas en E[a]"
            # equivale a init en origen — usar este parámetro con muestras
            # individuales, no con una sola media broadcast.
            wn = torch.as_tensor(warm_start_neurons, dtype=torch.float32)
            assert wn.shape == (M, 2 * d1 + 1), (
                f"warm_start_neurons must have shape ({M}, {2*d1+1}), "
                f"got {tuple(wn.shape)}"
            )
            a0 = wn[:, :d1]               # (M, d1)
            a1 = wn[:, d1:2 * d1]         # (M, d1)
            a2 = wn[:, 2 * d1]            # (M,)
            sigma = warm_start_sigma
            with torch.no_grad():
                # W0.weight: (d1, M) — columna m es a₀ᵐ ⇒ W0[:, m] = wn[m, :d1]
                W0w = a0.t().contiguous()
                if sigma > 0:
                    W0w = W0w + sigma * torch.randn_like(W0w)
                self.W0.weight.copy_(W0w)
                # W1.weight: (M, d1+1) — fila m = [a1ᵐ, time-slope=0]
                # El "time-slope" W1[m, d1] no existe en el solver explícito
                # (a₂ constante en t), pero la red la usa para augmentación
                # temporal. Init a 0 ⇒ a₂(t) ≡ a₂ᵐ al inicio.
                W1w = torch.zeros(M, d1 + 1)
                W1w[:, :d1] = a1
                if sigma > 0:
                    W1w = W1w + sigma * torch.randn_like(W1w)
                self.W1.weight.copy_(W1w)
                # W1.bias: (M,) — parte constante de a₂ᵐ(t) ⇒ b1[m] = a₂ᵐ
                b1 = a2.clone()
                if sigma > 0:
                    b1 = b1 + sigma * torch.randn_like(b1)
                self.W1.bias.copy_(b1)

    def forward(self, t: float, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate the effective vector field F(x, t).

        Computes F(x, t) = (1/M) Σₘ σ(a₁ᵐ·x + a₂ᵐ) a₀ᵐ for N points at once.  This
        is the VELOCITY with which the ODE moves each feature x at time t.

        Args:
            t : normalized time in [0, 1], broadcast to the N points.
            x : (N, d1) feature component of the N points of γ_t; each particle of
                γ_t is a pair (x, y), but F only acts on x (the label is carried along)

        Returns:
            (N, d1) velocity dX/dt at each of the N points.
        """
        t_val = t.item() if torch.is_tensor(t) else float(t)
        # Construir input aumentado [x, t] ∈ ℝ^{d₁+1} para cada punto
        t_col = x.new_full((x.size(0), 1), t_val)           # (N, 1) — constante t
        # h = σ(W₁·[x,t]ᵀ + b₁) — activaciones de las M neuronas para cada punto
        h = torch.tanh(self.W1(torch.cat([x, t_col], dim=1)))   # (N, M)
        # F = W₀ᵀ·h — combinación lineal de las M salidas → velocidad en ℝ^{d₁}
        # El factor 1/M está absorbido en la escala de inicialización de W₀
        return self.W0(h)                                        # (N, d1)

    def entropic_penalty(self, c1: float = 0.05, c2: float = 0.5) -> torch.Tensor:
        """
        Approximation of the entropic penalty ε · E(ν_t | ν^∞).

        THEORY (paper):
            E(ν_t | ν^∞) = KL(ν_t || ν^∞) = ∫ log(dν_t/dν^∞) dν_t
                         = E_{ν_t}[ℓ(a)] − H(ν_t),
            with ℓ(a) = c₁|a|⁴ + c₂|a|² and H(ν_t) the differential entropy of ν_t.

        LIMITATION OF THE IMPLEMENTATION:
            The parameters are point estimates, so ν_t = (1/M) Σₘ δ_{θₘ} is a sum
            of Dirac masses.  Its differential entropy relative to a continuous
            prior is −∞, so the full KL is +∞.

            Only the ENERGY TERM is accessible:
                E_{ν_t}[ℓ(a)] ≈ (1/N_params) Σⱼ [c₁ θⱼ⁴ + c₂ θⱼ²],
            which is a polynomial (L4 + L2) weight decay.  It provides the energy
            of the prior ν^∞ ∝ exp(−ℓ(a)), but not the entropy term −H(ν_t).

            The full entropic regularization requires exploring the Gibbs measure
            ν_t* ∝ exp(−J(θ)/ε).  This is done with pSGLD (train.py, use_sgld=True):
                θ ← θ − η ∇J(θ) + √(2ηε) ξ,   ξ ~ N(0, I);
            the Langevin noise supplies the entropy term −H(ν_t) of the KL, while
            this energy term plays the role of the prior ν^∞.

        Hyperparameters (Assumption Regularity (i)):
            c₁ = 0.05, quartic term (super-coercivity, gives a log-Sobolev inequality)
            c₂ = 0.5,  quadratic term (basic convexity)
        The quartic term is ESSENTIAL: c₂|a|² alone gives quadratic convexity but
        not the log-Sobolev inequality that implies PL.

        Returns:
            Scalar, mean energy per parameter (approximation of the KL).
        """
        pen, n = torch.tensor(0.0, device=next(self.parameters()).device), 0
        for p in self.parameters():
            pen = pen + c1 * (p ** 4).sum() + c2 * (p ** 2).sum()
            n += p.numel()
        return pen / n


# =============================================================================
# § 3  MEAN-FIELD RESNET: Neural ODE in the feature space ℝ^{d₁}
#
#  Implements the optimal control system of the paper:
#
#      ┌─ Transport ODE (eqs. 1.3/1.5):
#      │   dX_t/dt = F(X_t, t) = ∫_A b(X_t, a) dν_t(a),   t ∈ [0, T]
#      │   X_0 = input ∈ ℝ^{d₁}
#      │   γ_t = law of X_t given (X_0, Y_0)  (eq. 1.5)
#      │
#      └─ Linear classifier (terminal cost):
#          logit = W · X_T + b  →  P(y = 1 | X_T) = σ(logit)
#          L(x, y) = BCE(W·x + b, y)
#
#  WHY THE ORIGINAL SPACE ℝ² (no embedding):
#      In the paper γ_t ∈ P(ℝ^{d₁} × ℝ^{d}) lives in the same space as the
#      data.  Embedding x ∈ ℝ² → h ∈ ℝ^H would run the ODE in ℝ^H, make γ_t
#      impossible to visualize and move away from the theoretical setting.
#
#  RK4 vs EULER:
#      The paper works with the continuous-time flow, whose most faithful
#      discretization here is RK4.
#        • Euler: local error O(dt²), global error O(dt)
#        • RK4:   local error O(dt⁵), global error O(dt⁴)
#      With dt = T/n_steps = 0.1 the global error is ~1e-4 for RK4 and ~1e-1
#      for Euler: the four evaluations per step (k1..k4) capture the
#      curvature of the flow without more steps.
# =============================================================================
class MeanFieldResNet(nn.Module):
    """
    Mean-field Neural ODE that transports γ_0 (e.g. make_moons) to a separable γ_T.

    The network has no fixed layers but a continuous parametrized flow:
        X_0 → [ODE: dX/dt = F(X, t)] → X_T → [linear] → logit → P(y = 1).

    Attributes:
        velocity   : MeanFieldVelocity, the vector field F(x, t)
        classifier : nn.Linear(d1, 1), linear read-out on X_T
        T          : time horizon of the ODE (default T = 1.0)
        n_steps    : number of RK4 steps on [0, T]
    """

    def __init__(self, d1: int = 2, M: int = 64, T: float = 1.0,
                 n_steps: int = 10, task: str = 'classification',
                 warm_start_neurons=None, warm_start_sigma: float = 0.0):
        super().__init__()
        self.velocity   = MeanFieldVelocity(
            d1=d1, M=M,
            warm_start_neurons=warm_start_neurons,
            warm_start_sigma=warm_start_sigma,
        )
        self.classifier = nn.Linear(d1, 1)
        self.T, self.n_steps, self.d1 = T, n_steps, d1
        self.task = task   # 'classification' (BCE) | 'regression' (MSE)

        # Inicialización pequeña del clasificador: para que al inicio
        # el logit sea ≈ 0 y la pérdida BCE empiece cerca de log(2) ≈ 0.693
        nn.init.normal_(self.classifier.weight, std=0.1)
        nn.init.zeros_(self.classifier.bias)

        n_p = sum(p.numel() for p in self.parameters())
        print(f"  MeanFieldResNet: d1={d1}, M={M}, T={T}, "
              f"n_steps={n_steps}, params={n_p:,}")

    # NOTA — Restricción del espacio de controles respecto al paper:
    #   El paper optimiza sobre trayectorias arbitrarias a^m : [0,T] → A,
    #   es decir, el control ν_t puede ser cualquier curva continua en el
    #   espacio de medidas.  Esta implementación restringe ese espacio a:
    #       • a₀ᵐ ∈ ℝ²  constante en t  (pesos de salida fijos)
    #       • a₁ᵐ ∈ ℝ²  constante en t  (pesos de entrada fijos)
    #       • a₂ᵐ(t) = W₁[m,2]·t + b₁[m]  lineal en t  (solo la bias varía)
    #   Esta familia paramétrica es un subconjunto estricto del control de
    #   campo medio continuo.  Las garantías teóricas del paper valen en el
    #   espacio completo; aquí se verifican empíricamente en este subespacio.

    # ── RK4 integrator ───────────────────────────────────────────────────────
    def _rk4(self, t_norm: float, x: torch.Tensor, dt: float) -> torch.Tensor:
        """
        One Runge–Kutta 4 step of dX/dt = F(X, t).

        RK4 evaluates the field at four intermediate points and combines them with
        weights [1, 2, 2, 1]/6, with local error O(dt⁵) (global O(dt⁴)) against
        O(dt²) (global O(dt)) for Euler:
            k1 = F(x,            t)
            k2 = F(x + dt/2·k1,  t + dt/2)
            k3 = F(x + dt/2·k2,  t + dt/2)
            k4 = F(x + dt·k3,    t + dt)
            x_new = x + (dt/6)(k1 + 2k2 + 2k3 + k4)

        Args:
            t_norm : normalized time at the start of the step, in [0, 1)
            x      : (N, d1) current positions of the N points
            dt     : normalized step size (= 1/n_steps)

        Returns:
            (N, d1) positions after the step.
        """
        k1 = self.velocity(t_norm,        x)
        k2 = self.velocity(t_norm + dt/2, x + dt/2 * k1)
        k3 = self.velocity(t_norm + dt/2, x + dt/2 * k2)
        k4 = self.velocity(t_norm + dt,   x + dt    * k3)
        return x + (dt / 6) * (k1 + 2*k2 + 2*k3 + k4)

    def integrate(self, x0: torch.Tensor,
                  return_trajectory: bool = False):
        """
        Integrate dX/dt = F(X, t) from t = 0 to t = T.

        Each step advances the network time by dt = T/n_steps.  X_T is the learned
        representation of the features, which the linear classifier separates.
        Time is normalized to [0, 1] internally, so that F(x, t_norm) does not
        depend on T.

        Args:
            x0                : (N, d1) initial features (the points of γ_0)
            return_trajectory : if True, also return the intermediate states of γ_t
                                (to visualize the evolution of the features)

        Returns:
            x_T (N, d1) if return_trajectory is False; otherwise (x_T, traj) with
            traj = [(t_real, x_t)] for t_real in {0, T/n, 2T/n, …, T}, the tensors
            in traj detached from the graph.
        """
        x  = x0.clone()
        dt = 1.0 / self.n_steps       # paso normalizado ∈ [0,1]
        traj = [(0.0, x.detach().clone())] if return_trajectory else None

        for i in range(self.n_steps):
            t_i = i * dt                              # tiempo normalizado al inicio del paso
            x   = self._rk4(t_i, x, dt)
            if return_trajectory:
                t_real = (i + 1) * self.T / self.n_steps   # tiempo real en [0, T]
                traj.append((t_real, x.detach().clone()))

        return (x, traj) if return_trajectory else x

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        """Integrate the ODE and apply the linear classifier to X_T (returns logits)."""
        x_T = self.integrate(x0)
        return self.classifier(x_T).squeeze(-1)   # (N,) — logits sin activar

    def compute_loss(self, x0, y, epsilon: float = 0.01):
        """
        Total cost J of the control problem (eq. 1.6 of the paper):

            J(γ_0, ν) = ∫ L(x, y) dγ_T(x, y)          terminal cost (BCE or MSE)
                      + ε ∫₀ᵀ E(ν_t | ν^∞) dt          entropic penalty

        The first term measures the fit; the second penalizes how far ν_t is from
        the prior ν^∞, which regularizes the parameters and, for ε > 0, gives the
        log-Sobolev → PL → exponential convergence chain.  With M particles the
        entropic term is approximated by the potential ℓ evaluated at the learned
        parameters (see entropic_penalty).

        Args:
            x0      : (N, d1) input features
            y       : (N,) labels in {0, 1} (or targets for regression)
            epsilon : entropic regularization coefficient (ε ≥ 0)

        Returns:
            loss_total : J (tensor with graph, for autograd)
            loss_term  : terminal cost (float)
            loss_reg   : entropic penalty (float)
        """
        pred      = self.forward(x0)
        if self.task == 'regression':
            loss_term = nn.MSELoss()(pred, y)
        else:
            loss_term = nn.BCEWithLogitsLoss()(pred, y)
        loss_reg  = self.velocity.entropic_penalty()
        return loss_term + epsilon * loss_reg, loss_term.item(), loss_reg.item()
