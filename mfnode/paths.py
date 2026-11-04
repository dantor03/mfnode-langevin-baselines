"""Where figures and their data are written.  Imports nothing heavy (no torch).

``MFNODE_OUTPUT_DIR`` overrides the default ``figures/`` folder at the root
of the repository; the tests use it to write into a temporary directory.
"""

import os

_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

OUTPUT_DIR = os.path.abspath(os.environ.get('MFNODE_OUTPUT_DIR')
                             or os.path.join(_ROOT, 'figures'))
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
