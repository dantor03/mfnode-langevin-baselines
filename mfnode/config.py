import os
import warnings

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # noqa: E402

warnings.filterwarnings('ignore')

# ── Reproducibility and device ──────────────────────────────────────────
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
elif torch.backends.mps.is_available():
    DEVICE = torch.device('mps')
else:
    DEVICE = torch.device('cpu')

from .paths import OUTPUT_DIR, DATA_DIR  # noqa: E402,F401

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"✓ Device: {DEVICE}")
print(f"✓ Figures → {OUTPUT_DIR}")
