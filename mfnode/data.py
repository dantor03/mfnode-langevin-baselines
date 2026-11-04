import numpy as np
import torch
from sklearn.datasets import make_moons, make_circles
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import SEED, DEVICE


# =============================================================================
# § 1  DATA SETS
# =============================================================================
def get_moons(n=400, noise=0.12, seed=SEED):
    """
    make_moons as PyTorch tensors.

    In the setting of the paper this data set is γ_0, the JOINT empirical
    distribution of (features, labels) at network time t = 0:

        γ_0 = (1/N) Σᵢ δ_{(X₀ⁱ, Y₀ⁱ)}     on ℝ² × ℝ.

    The ODE moves only the features X₀ⁱ → X_tⁱ; the label Y₀ⁱ is carried
    along unchanged:

        γ_t = (1/N) Σᵢ δ_{(X_tⁱ, Y₀ⁱ)}.

    The figures show the x-marginal (γ_t)_x = (1/N) Σᵢ δ_{X_tⁱ} ∈ P(ℝ²),
    coloured by Y₀ⁱ.  Features are standardized (mean ≈ 0, std ≈ 1), in line
    with the bounded-data assumption of the paper.

    Returns:
        X    : (N, 2) tensor on DEVICE, standardized features
        y    : (N,) tensor on DEVICE, labels in {0, 1}
        X_np : (N, 2) NumPy array, for plotting
        y_np : (N,) NumPy array, for plotting
    """
    X_np, y_np = make_moons(n_samples=n, noise=noise, random_state=seed)
    X_np = StandardScaler().fit_transform(X_np).astype(np.float32)
    y_np = y_np.astype(np.float32)
    return (torch.tensor(X_np, device=DEVICE),
            torch.tensor(y_np, device=DEVICE),
            X_np, y_np)


def get_circles(n: int = 400, noise: float = 0.08, factor: float = 0.5,
                seed: int = SEED):
    """
    make_circles as PyTorch tensors.

    Like get_moons(), but the data γ₀ is invariant under rotations of ℝ².  This
    gives a theoretical prediction for the optimal parameter distribution ν*:
    if γ₀ is SO(2)-symmetric, ν* should be (approximately) isotropic too; in
    particular the input weights a₁ᵐ should spread uniformly over a ring in ℝ²
    instead of forming the two symmetric peaks seen on make_moons.

    Args:
        n      : total number of samples (half per class)
        noise  : standard deviation of the Gaussian noise added to each point
        factor : ratio of inner to outer radius, in (0, 1); with factor = 0.5,
                 class 0 is the outer circle and class 1 the inner one
        seed   : random seed

    Returns:
        X    : (N, 2) tensor on DEVICE, standardized features
        y    : (N,) tensor on DEVICE, labels in {0, 1}
        X_np : (N, 2) NumPy array, for plotting
        y_np : (N,) NumPy array, for plotting
    """
    X_np, y_np = make_circles(n_samples=n, noise=noise, factor=factor,
                              random_state=seed)
    X_np = StandardScaler().fit_transform(X_np).astype(np.float32)
    y_np = y_np.astype(np.float32)
    return (torch.tensor(X_np, device=DEVICE),
            torch.tensor(y_np, device=DEVICE),
            X_np, y_np)


def get_california_regression(n_train: int = 800, n_test: int = 400,
                               n_components: int = 2, seed: int = SEED):
    """
    California Housing projected to ℝ² by PCA, for regression.

    The original data set has 8 features (MedInc, HouseAge, AveRooms,
    AveBedrms, Population, AveOccup, Latitude, Longitude) and a continuous
    target (median house value in $100k, range ~0.15–5.0).

    Pipeline:
        1. subsample n_train + n_test rows (for training speed)
        2. standardize the features
        3. PCA(n_components) → ℝ² (the space where the ODE runs)
        4. standardize the target (mean 0, std 1, for a stable MSE)
        5. train/test split

    The variance explained by the projection is reported.  On California
    Housing the first two components mostly capture the geographic
    (Latitude/Longitude) and income (MedInc) signals, the most predictive of
    the price.

    Args:
        n_train      : training samples
        n_test       : test samples
        n_components : dimension of the PCA projection (= d1 of the model)
        seed         : random seed

    Returns:
        dict with
            X_train, y_train, X_test, y_test : tensors on DEVICE
            X_train_np, y_train_np, X_test_np, y_test_np : NumPy arrays
            pca           : fitted PCA (for plotting)
            scaler_y      : target scaler (to undo the standardization)
            explained_var : fraction of variance explained by the projection
    """
    from sklearn.datasets import fetch_california_housing

    data    = fetch_california_housing()
    X_full  = data.data.astype(np.float32)
    y_full  = data.target.astype(np.float32)

    rng  = np.random.RandomState(seed)
    idx  = rng.choice(len(X_full), size=n_train + n_test, replace=False)
    X_s, y_s = X_full[idx], y_full[idx]

    scaler_X  = StandardScaler().fit(X_s)
    X_scaled  = scaler_X.transform(X_s)
    pca       = PCA(n_components=n_components, random_state=seed).fit(X_scaled)
    X_2d      = pca.transform(X_scaled).astype(np.float32)

    scaler_y  = StandardScaler().fit(y_s.reshape(-1, 1))
    y_norm    = scaler_y.transform(y_s.reshape(-1, 1)).ravel().astype(np.float32)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_2d, y_norm,
        train_size=n_train, test_size=n_test,
        random_state=seed
    )

    explained = float(pca.explained_variance_ratio_.sum())
    print(f"  California Housing PCA({n_components}): "
          f"explained variance = {explained:.1%}  "
          f"| train={n_train}  test={n_test}")

    return {
        'X_train':    torch.tensor(X_tr, device=DEVICE),
        'y_train':    torch.tensor(y_tr, device=DEVICE),
        'X_test':     torch.tensor(X_te, device=DEVICE),
        'y_test':     torch.tensor(y_te, device=DEVICE),
        'X_train_np': X_tr,
        'y_train_np': y_tr,
        'X_test_np':  X_te,
        'y_test_np':  y_te,
        'pca':        pca,
        'scaler_y':   scaler_y,
        'explained_var': explained,
    }
