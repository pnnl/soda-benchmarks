#!/usr/bin/env python3
"""
Simple validation script to test all PolyBench PyTorch kernels.

Imports each kernel as a proper package and executes it with a configurable
dataset size to verify forward() works and MLIR generation succeeds.

Usage:
    python test_all_kernels.py          # runs all enabled tests
    POLYBENCH_DATASET=MINI python ...   # override dataset size
"""

import importlib
import os
import sys

# ---------------------------------------------------------------------------
# Ensure the repository root is on sys.path so package imports resolve.
# ---------------------------------------------------------------------------
repo_root = os.path.abspath(os.path.dirname(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

DATASET = os.environ.get("POLYBENCH_DATASET", "MINI")

# ---------------------------------------------------------------------------
# Kernel imports.
# Module names starting with a digit (twomm, threemm) cannot be used with the
# `import` statement directly, so importlib.import_module is used for all
# kernels for consistency. The __init__.py files in each subdirectory make
# these proper package imports — no file-path fallback needed.
# ---------------------------------------------------------------------------
k2mm     = importlib.import_module("PolyBenchPyTorch.linear_algebra.kernels.twomm.twomm")
k3mm     = importlib.import_module("PolyBenchPyTorch.linear_algebra.kernels.threemm.threemm")
katax    = importlib.import_module("PolyBenchPyTorch.linear_algebra.kernels.atax.atax")
kbicg    = importlib.import_module("PolyBenchPyTorch.linear_algebra.kernels.bicg.bicg")
kdoit    = importlib.import_module("PolyBenchPyTorch.linear_algebra.kernels.doitgen.doitgen")
kmvt     = importlib.import_module("PolyBenchPyTorch.linear_algebra.kernels.mvt.mvt")

kgemm    = importlib.import_module("PolyBenchPyTorch.linear_algebra.blas.gemm.gemm")
kgemver  = importlib.import_module("PolyBenchPyTorch.linear_algebra.blas.gemver.gemver")
kgesummv = importlib.import_module("PolyBenchPyTorch.linear_algebra.blas.gesummv.gesummv")
ksymm    = importlib.import_module("PolyBenchPyTorch.linear_algebra.blas.symm.symm")
ksyr2k   = importlib.import_module("PolyBenchPyTorch.linear_algebra.blas.syr2k.syr2k")
ksyrk    = importlib.import_module("PolyBenchPyTorch.linear_algebra.blas.syrk.syrk")
ktrmm    = importlib.import_module("PolyBenchPyTorch.linear_algebra.blas.trmm.trmm")


def save_mlir_output(kernel_name, model, *init_args):
    """Compile kernel to MLIR (TOSA dialect) and write to output/."""
    from torch_mlir import torchscript  # type: ignore
    mlir_module = torchscript.compile(model, init_args, output_type="tosa", use_tracing=True)
    output_dir = os.path.join(repo_root, "output")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{kernel_name}_tosa.mlir")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(str(mlir_module))
    print(f"MLIR written to {output_path}")


# ---------------------------------------------------------------------------
# Linear-algebra kernels
# ---------------------------------------------------------------------------

def test_2mm():
    dims = k2mm.get_dataset_dimensions(DATASET)
    model = k2mm.TwoMM(dims["ni"], dims["nj"], dims["nk"], dims["nl"])
    alpha, beta, A, B, C, D = k2mm.init_array(dims["ni"], dims["nj"], dims["nk"], dims["nl"])
    result = model(alpha, beta, A, B, C, D)
    assert result.shape == (dims["ni"], dims["nl"]), f"Unexpected shape: {result.shape}"
    print(f"✓ twomm: result shape {result.shape}, dtype {result.dtype}")
    save_mlir_output("twomm", model, alpha, beta, A, B, C, D)


def test_3mm():
    dims = k3mm.get_dataset_dimensions(DATASET)
    model = k3mm.ThreeMM(dims["ni"], dims["nj"], dims["nk"], dims["nl"], dims["nm"])
    A, B, C, D = k3mm.init_array(dims["ni"], dims["nj"], dims["nk"], dims["nl"], dims["nm"])
    result = model(A, B, C, D)
    assert result.shape == (dims["ni"], dims["nl"]), f"Unexpected shape: {result.shape}"
    print(f"✓ threemm: result shape {result.shape}, dtype {result.dtype}")
    save_mlir_output("threemm", model, A, B, C, D)


def test_atax():
    dims = katax.get_dataset_dimensions(DATASET)
    model = katax.Atax(dims["m"], dims["n"])
    A, x = katax.init_array(dims["m"], dims["n"])
    result = model(A, x)
    assert result.shape == (dims["n"],), f"Unexpected shape: {result.shape}"
    print(f"✓ atax: result shape {result.shape}, dtype {result.dtype}")
    save_mlir_output("atax", model, A, x)


def test_bicg():
    dims = kbicg.get_dataset_dimensions(DATASET)
    model = kbicg.Bicg(dims["m"], dims["n"])
    A, r, p = kbicg.init_array(dims["m"], dims["n"])
    s, q = model(A, r, p)
    assert s.shape == (dims["m"],), f"Unexpected s shape: {s.shape}"
    assert q.shape == (dims["n"],), f"Unexpected q shape: {q.shape}"
    print(f"✓ bicg: s shape {s.shape}, q shape {q.shape}, dtype {s.dtype}")
    save_mlir_output("bicg", model, A, r, p)


def test_doitgen():
    dims = kdoit.get_dataset_dimensions(DATASET)
    model = kdoit.Doitgen(dims["nr"], dims["nq"], dims["np"])
    A, C4 = kdoit.init_array(dims["nr"], dims["nq"], dims["np"])
    result = model(A, C4)
    assert result.shape == (dims["nr"], dims["nq"], dims["np"]), f"Unexpected shape: {result.shape}"
    print(f"✓ doitgen: result shape {result.shape}, dtype {result.dtype}")
    save_mlir_output("doitgen", model, A, C4)


def test_mvt():
    dims = kmvt.get_dataset_dimensions(DATASET)
    model = kmvt.Mvt(dims["n"])
    A, x1, x2, y_1, y_2 = kmvt.init_array(dims["n"])
    x1_out, x2_out = model(A, x1, x2, y_1, y_2)
    assert x1_out.shape == (dims["n"],), f"Unexpected x1_out shape: {x1_out.shape}"
    assert x2_out.shape == (dims["n"],), f"Unexpected x2_out shape: {x2_out.shape}"
    print(f"✓ mvt: x1_out shape {x1_out.shape}, x2_out shape {x2_out.shape}, dtype {x1_out.dtype}")
    save_mlir_output("mvt", model, A, x1, x2, y_1, y_2)


# ---------------------------------------------------------------------------
# BLAS kernels
# ---------------------------------------------------------------------------

def test_gemm():
    dims = kgemm.get_dataset_dimensions("gemm", DATASET)
    model = kgemm.Gemm(dims["ni"], dims["nj"], dims["nk"])
    alpha, beta, A, B, C = kgemm.init_array(dims["ni"], dims["nj"], dims["nk"])
    out = model(alpha, beta, A, B, C)
    assert out.shape == (dims["ni"], dims["nj"]), f"Unexpected shape: {out.shape}"
    print(f"✓ gemm: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("gemm", model, alpha, beta, A, B, C)


def test_gemver():
    dims = kgemver.get_dataset_dimensions("gemver", DATASET)
    model = kgemver.Gemver(dims["n"])
    alpha, beta, A, u1, v1, u2, v2, x, y, z, w = kgemver.init_array(dims["n"])
    out = model(alpha, beta, A, u1, v1, u2, v2, x, y, z)
    assert out.shape == (dims["n"],), f"Unexpected shape: {out.shape}"
    print(f"✓ gemver: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("gemver", model, alpha, beta, A, u1, v1, u2, v2, x, y, z)


def test_gesummv():
    dims = kgesummv.get_dataset_dimensions("gesummv", DATASET)
    model = kgesummv.Gesummv(dims["n"])
    alpha, beta, A, B, x = kgesummv.init_array(dims["n"])
    out = model(alpha, beta, A, B, x)
    assert out.shape == (dims["n"],), f"Unexpected shape: {out.shape}"
    print(f"✓ gesummv: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("gesummv", model, alpha, beta, A, B, x)


def test_symm():
    dims = ksymm.get_dataset_dimensions("symm", DATASET)
    model = ksymm.Symm(dims["m"], dims["n"])
    alpha, beta, A, B, C = ksymm.init_array(dims["m"], dims["n"])
    out = model(alpha, A, B, beta, C)
    assert out.shape == (dims["m"], dims["n"]), f"Unexpected shape: {out.shape}"
    print(f"✓ symm: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("symm", model, alpha, A, B, beta, C)


def test_syr2k():
    dims = ksyr2k.get_dataset_dimensions("syr2k", DATASET)
    model = ksyr2k.Syr2k(dims["n"], dims["m"])
    alpha, beta, A, B, C = ksyr2k.init_array(dims["n"], dims["m"])
    out = model(alpha, A, B, beta, C)
    assert out.shape == (dims["n"], dims["n"]), f"Unexpected shape: {out.shape}"
    print(f"✓ syr2k: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("syr2k", model, alpha, A, B, beta, C)


def test_syrk():
    dims = ksyrk.get_dataset_dimensions("syrk", DATASET)
    model = ksyrk.Syrk(dims["n"], dims["m"])
    alpha, beta, A, C = ksyrk.init_array(dims["n"], dims["m"])
    out = model(alpha, A, beta, C)
    assert out.shape == (dims["n"], dims["n"]), f"Unexpected shape: {out.shape}"
    print(f"✓ syrk: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("syrk", model, alpha, A, beta, C)


def test_trmm():
    dims = ktrmm.get_dataset_dimensions("trmm", DATASET)
    model = ktrmm.Trmm(dims["m"], dims["n"])
    alpha, A, B = ktrmm.init_array(dims["m"], dims["n"])
    out = model(alpha, A, B)
    assert out.shape == (dims["m"], dims["n"]), f"Unexpected shape: {out.shape}"
    print(f"✓ trmm: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("trmm", model, alpha, A, B)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main():
    print(f"Testing PolyBench PyTorch kernels (dataset={DATASET})...")
    print()

    tests = [
        ("linear-algebra kernels", [
            test_2mm, test_3mm, test_atax, test_bicg, test_doitgen, test_mvt,
        ]),
        ("BLAS kernels", [
            test_gemm, test_gemver, test_gesummv, test_symm,
            test_syr2k, test_syrk, test_trmm,
        ]),
    ]

    failed = []
    for group_name, group_tests in tests:
        print(f"Testing {group_name}...")
        for test_fn in group_tests:
            try:
                test_fn()
            except Exception as e:
                import traceback
                print(f"  ✗ {test_fn.__name__}: {e}")
                traceback.print_exc()
                failed.append(test_fn.__name__)
        print()

    if failed:
        print(f"✗ {len(failed)} test(s) failed: {', '.join(failed)}")
        return 1

    print("✓ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
