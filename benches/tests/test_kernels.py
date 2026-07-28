"""Shape/correctness tests for all PolyBench PyTorch kernels.

Each test imports its kernel lazily (inside the function) so a broken kernel
only fails its own test.  No MLIR generation happens here — these are fast,
pure-PyTorch forward-pass checks.
"""

import importlib

# ---------------------------------------------------------------------------
# sb-cli package import smoke tests (Constitution Principle VII / T034)
# sb-cli generates files; it does not implement kernels, so no MLIR entries
# are needed in test_mlir.py. These tests verify the package is importable.
# ---------------------------------------------------------------------------


def test_sb_cli_imports() -> None:
    """sb_cli package and all submodules import without error."""
    importlib.import_module("sb_cli")
    importlib.import_module("sb_cli.flow")
    importlib.import_module("sb_cli.registry")
    importlib.import_module("sb_cli.init")
    importlib.import_module("sb_cli.fork")
    importlib.import_module("sb_cli.collect")
    importlib.import_module("sb_cli.templates")


# ---------------------------------------------------------------------------
# Linear-algebra kernels
# ---------------------------------------------------------------------------


def test_2mm(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.twomm.twomm"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.TwoMM(dims["ni"], dims["nj"], dims["nk"], dims["nl"])
    alpha, beta, A, B, C, D = k.init_array(
        dims["ni"], dims["nj"], dims["nk"], dims["nl"]
    )
    result = model(alpha, beta, A, B, C, D)
    assert result.shape == (dims["ni"], dims["nl"])


def test_3mm(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.threemm.threemm"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.ThreeMM(dims["ni"], dims["nj"], dims["nk"], dims["nl"], dims["nm"])
    A, B, C, D = k.init_array(
        dims["ni"], dims["nj"], dims["nk"], dims["nl"], dims["nm"]
    )
    result = model(A, B, C, D)
    assert result.shape == (dims["ni"], dims["nl"])


def test_atax(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.atax.atax"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.Atax(dims["m"], dims["n"])
    A, x = k.init_array(dims["m"], dims["n"])
    result = model(A, x)
    assert result.shape == (dims["n"],)


def test_bicg(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.bicg.bicg"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.Bicg(dims["m"], dims["n"])
    A, r, p = k.init_array(dims["m"], dims["n"])
    s, q = model(A, r, p)
    assert s.shape == (dims["m"],)
    assert q.shape == (dims["n"],)


def test_doitgen(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.doitgen.doitgen"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.Doitgen(dims["nr"], dims["nq"], dims["np"])
    A, C4 = k.init_array(dims["nr"], dims["nq"], dims["np"])
    result = model(A, C4)
    assert result.shape == (dims["nr"], dims["nq"], dims["np"])


def test_mvt(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.mvt.mvt"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.Mvt(dims["n"])
    A, x1, x2, y_1, y_2 = k.init_array(dims["n"])
    x1_out, x2_out = model(A, x1, x2, y_1, y_2)
    assert x1_out.shape == (dims["n"],)
    assert x2_out.shape == (dims["n"],)


# ---------------------------------------------------------------------------
# BLAS kernels
# ---------------------------------------------------------------------------


def test_gemm(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm"
    )
    dims = k.get_dataset_dimensions("gemm", dataset)
    model = k.Gemm(dims["ni"], dims["nj"], dims["nk"])
    alpha, beta, A, B, C = k.init_array(dims["ni"], dims["nj"], dims["nk"])
    out = model(alpha, beta, A, B, C)
    assert out.shape == (dims["ni"], dims["nj"])


def test_gemver(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.gemver.gemver"
    )
    dims = k.get_dataset_dimensions("gemver", dataset)
    model = k.Gemver(dims["n"])
    alpha, beta, A, u1, v1, u2, v2, x, y, z, w = k.init_array(dims["n"])
    out = model(alpha, beta, A, u1, v1, u2, v2, x, y, z)
    assert out.shape == (dims["n"],)


def test_gesummv(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.gesummv.gesummv"
    )
    dims = k.get_dataset_dimensions("gesummv", dataset)
    model = k.Gesummv(dims["n"])
    alpha, beta, A, B, x = k.init_array(dims["n"])
    out = model(alpha, beta, A, B, x)
    assert out.shape == (dims["n"],)


def test_symm(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.symm.symm"
    )
    dims = k.get_dataset_dimensions("symm", dataset)
    model = k.Symm(dims["m"], dims["n"])
    alpha, beta, A, B, C = k.init_array(dims["m"], dims["n"])
    out = model(alpha, A, B, beta, C)
    assert out.shape == (dims["m"], dims["n"])


def test_syr2k(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.syr2k.syr2k"
    )
    dims = k.get_dataset_dimensions("syr2k", dataset)
    model = k.Syr2k(dims["n"], dims["m"])
    alpha, beta, A, B, C = k.init_array(dims["n"], dims["m"])
    out = model(alpha, A, B, beta, C)
    assert out.shape == (dims["n"], dims["n"])


def test_syrk(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.syrk.syrk"
    )
    dims = k.get_dataset_dimensions("syrk", dataset)
    model = k.Syrk(dims["n"], dims["m"])
    alpha, beta, A, C = k.init_array(dims["n"], dims["m"])
    out = model(alpha, A, beta, C)
    assert out.shape == (dims["n"], dims["n"])


def test_trmm(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.trmm.trmm"
    )
    dims = k.get_dataset_dimensions("trmm", dataset)
    model = k.Trmm(dims["m"], dims["n"])
    alpha, A, B = k.init_array(dims["m"], dims["n"])
    out = model(alpha, A, B)
    assert out.shape == (dims["m"], dims["n"])
