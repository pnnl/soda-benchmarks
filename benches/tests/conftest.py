import os

import pytest


@pytest.fixture(scope="session")
def dataset():
    return os.environ.get("POLYBENCH_DATASET", "MINI")
