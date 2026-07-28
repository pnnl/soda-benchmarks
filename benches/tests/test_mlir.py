"""MLIR generation tests for all PolyBench PyTorch kernels.

These tests call torchscript.compile() directly and assert that:
  - compilation succeeds without exceptions
  - the resulting MLIR text is non-empty and well-formed (starts with 'module')

No files are written to disk. Run selectively with:
    pixi run pytest -m slow
Skip with:
    pixi run pytest -m "not slow"
"""

import importlib

import pytest

pytest.importorskip(
    "torch_mlir",
    reason="torch-mlir not available; skipping MLIR generation tests",
)


def _compile(model, inputs, dialect="tosa"):
    """Compile model to MLIR and return the text. Raises on failure."""
    from torch_mlir import torchscript

    mlir_module = torchscript.compile(
        model, inputs, output_type=dialect, use_tracing=True
    )
    return str(mlir_module)


def _assert_valid_mlir(text):
    assert text.strip(), "MLIR output is empty"
    assert text.strip().startswith("module"), (
        f"MLIR output does not start with 'module':\n{text[:200]}"
    )


# ---------------------------------------------------------------------------
# Linear-algebra kernels
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_2mm_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.twomm.twomm"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.TwoMM(dims["ni"], dims["nj"], dims["nk"], dims["nl"])
    inputs = k.init_array(dims["ni"], dims["nj"], dims["nk"], dims["nl"])
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_3mm_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.threemm.threemm"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.ThreeMM(dims["ni"], dims["nj"], dims["nk"], dims["nl"], dims["nm"])
    inputs = k.init_array(dims["ni"], dims["nj"], dims["nk"], dims["nl"], dims["nm"])
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_atax_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.atax.atax"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.Atax(dims["m"], dims["n"])
    inputs = k.init_array(dims["m"], dims["n"])
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_bicg_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.bicg.bicg"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.Bicg(dims["m"], dims["n"])
    inputs = k.init_array(dims["m"], dims["n"])
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_doitgen_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.doitgen.doitgen"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.Doitgen(dims["nr"], dims["nq"], dims["np"])
    inputs = k.init_array(dims["nr"], dims["nq"], dims["np"])
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_mvt_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.kernels.mvt.mvt"
    )
    dims = k.get_dataset_dimensions(dataset)
    model = k.Mvt(dims["n"])
    inputs = k.init_array(dims["n"])
    _assert_valid_mlir(_compile(model, inputs))


# ---------------------------------------------------------------------------
# BLAS kernels
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_gemm_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm"
    )
    dims = k.get_dataset_dimensions("gemm", dataset)
    model = k.Gemm(dims["ni"], dims["nj"], dims["nk"])
    inputs = k.init_array(dims["ni"], dims["nj"], dims["nk"])
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_gemver_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.gemver.gemver"
    )
    dims = k.get_dataset_dimensions("gemver", dataset)
    model = k.Gemver(dims["n"])
    alpha, beta, A, u1, v1, u2, v2, x, y, z, w = k.init_array(dims["n"])
    inputs = (alpha, beta, A, u1, v1, u2, v2, x, y, z)  # w is output, not input
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_gesummv_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.gesummv.gesummv"
    )
    dims = k.get_dataset_dimensions("gesummv", dataset)
    model = k.Gesummv(dims["n"])
    inputs = k.init_array(dims["n"])
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_symm_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.symm.symm"
    )
    dims = k.get_dataset_dimensions("symm", dataset)
    model = k.Symm(dims["m"], dims["n"])
    alpha, beta, A, B, C = k.init_array(dims["m"], dims["n"])
    inputs = (alpha, A, B, beta, C)
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_syr2k_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.syr2k.syr2k"
    )
    dims = k.get_dataset_dimensions("syr2k", dataset)
    model = k.Syr2k(dims["n"], dims["m"])
    alpha, beta, A, B, C = k.init_array(dims["n"], dims["m"])
    inputs = (alpha, A, B, beta, C)
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_syrk_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.syrk.syrk"
    )
    dims = k.get_dataset_dimensions("syrk", dataset)
    model = k.Syrk(dims["n"], dims["m"])
    alpha, beta, A, C = k.init_array(dims["n"], dims["m"])
    inputs = (alpha, A, beta, C)
    _assert_valid_mlir(_compile(model, inputs))


@pytest.mark.slow
def test_trmm_mlir(dataset):
    k = importlib.import_module(
        "benches.PolyBenchPyTorch.linear_algebra.blas.trmm.trmm"
    )
    dims = k.get_dataset_dimensions("trmm", dataset)
    model = k.Trmm(dims["m"], dims["n"])
    inputs = k.init_array(dims["m"], dims["n"])
    _assert_valid_mlir(_compile(model, inputs))
