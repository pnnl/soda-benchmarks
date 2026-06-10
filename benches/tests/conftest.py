import os
import sys

import pytest

# Ensure repo root is on sys.path so PolyBenchPyTorch package resolves.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


@pytest.fixture(scope="session")
def dataset():
    return os.environ.get("POLYBENCH_DATASET", "MINI")
