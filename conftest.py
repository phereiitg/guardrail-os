"""
Placing conftest.py at the project root makes pytest add the root to sys.path,
so tests can `import app...`. It also sets the working directory to the root so
the classifier/policy engine find config/ and model/ via their relative paths.
"""

import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.chdir(ROOT)
