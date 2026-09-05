"""Make the voxel_simulator package importable when the tests are run from this directory."""

from __future__ import annotations

import os
import sys

DATAGEN_ROOT = os.path.dirname(os.path.dirname(__file__))
if DATAGEN_ROOT not in sys.path:
    sys.path.insert(0, DATAGEN_ROOT)
