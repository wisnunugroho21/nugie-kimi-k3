"""Shared test configuration: run on CPU, repo root importable from anywhere.

x64 is deliberately NOT enabled — the model under test is a float32 program,
and flipping the global default would change dtypes inside it (e.g. RMSNorm's
gain init). The float64 reference oracle in test_rule.py is pure NumPy instead.
"""

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
