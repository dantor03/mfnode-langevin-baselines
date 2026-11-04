import importlib.util
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Figures and data written by the tests go to a temporary folder, never to figures/.
os.environ.setdefault("MFNODE_OUTPUT_DIR", tempfile.mkdtemp(prefix="mfnode-tests-"))

# Without torch only the figure tests can run (they use saved or synthetic arrays).
if importlib.util.find_spec("torch") is None:
    collect_ignore = ["test_metrics.py", "test_model_train.py", "test_experiments.py"]
