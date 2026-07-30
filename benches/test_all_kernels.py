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

import benches

# Generated MLIR goes to benches/output/
OUTPUT_ROOT = str(benches.ROOT)

DATASET = os.environ.get("POLYBENCH_DATASET", "MINI")

# ---------------------------------------------------------------------------
# Kernel imports.
# Module names starting with a digit (twomm, threemm) cannot be used with the
# `import` statement directly, so importlib.import_module is used for all
# kernels for consistency. The __init__.py files in each subdirectory make
# these proper package imports — no file-path fallback needed.
# ---------------------------------------------------------------------------
k2mm     = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.kernels.twomm.twomm")
k3mm     = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.kernels.threemm.threemm")
katax    = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.kernels.atax.atax")
kbicg    = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.kernels.bicg.bicg")
kdoit    = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.kernels.doitgen.doitgen")
kmvt     = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.kernels.mvt.mvt")

kgemm    = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.blas.gemm.gemm")
kgemver  = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.blas.gemver.gemver")
kgesummv = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.blas.gesummv.gesummv")
ksymm    = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.blas.symm.symm")
ksyr2k   = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.blas.syr2k.syr2k")
ksyrk    = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.blas.syrk.syrk")
ktrmm    = importlib.import_module("benches.PolyBenchPyTorch.linear_algebra.blas.trmm.trmm")


def save_mlir_output(kernel_name, model, inputs):
    """Compile kernel to MLIR (TOSA dialect) and write to output/."""
    from torch_mlir import torchscript  # type: ignore
    mlir_module = torchscript.compile(model, inputs, output_type="tosa", use_tracing=True)
    output_dir = os.path.join(OUTPUT_ROOT, "output")
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
    inputs = k2mm.init_array(dims["ni"], dims["nj"], dims["nk"], dims["nl"])
    result = model(*inputs)
    assert result.shape == (dims["ni"], dims["nl"]), f"Unexpected shape: {result.shape}"
    print(f"✓ twomm: result shape {result.shape}, dtype {result.dtype}")
    save_mlir_output("twomm", model, inputs)


def test_3mm():
    dims = k3mm.get_dataset_dimensions(DATASET)
    model = k3mm.ThreeMM(dims["ni"], dims["nj"], dims["nk"], dims["nl"], dims["nm"])
    inputs = k3mm.init_array(dims["ni"], dims["nj"], dims["nk"], dims["nl"], dims["nm"])
    result = model(*inputs)
    assert result.shape == (dims["ni"], dims["nl"]), f"Unexpected shape: {result.shape}"
    print(f"✓ threemm: result shape {result.shape}, dtype {result.dtype}")
    save_mlir_output("threemm", model, inputs)


def test_atax():
    dims = katax.get_dataset_dimensions(DATASET)
    model = katax.Atax(dims["m"], dims["n"])
    inputs = katax.init_array(dims["m"], dims["n"])
    result = model(*inputs)
    assert result.shape == (dims["n"],), f"Unexpected shape: {result.shape}"
    print(f"✓ atax: result shape {result.shape}, dtype {result.dtype}")
    save_mlir_output("atax", model, inputs)


def test_bicg():
    dims = kbicg.get_dataset_dimensions(DATASET)
    model = kbicg.Bicg(dims["m"], dims["n"])
    inputs = kbicg.init_array(dims["m"], dims["n"])
    s, q = model(*inputs)
    assert s.shape == (dims["m"],), f"Unexpected s shape: {s.shape}"
    assert q.shape == (dims["n"],), f"Unexpected q shape: {q.shape}"
    print(f"✓ bicg: s shape {s.shape}, q shape {q.shape}, dtype {s.dtype}")
    save_mlir_output("bicg", model, inputs)


def test_doitgen():
    dims = kdoit.get_dataset_dimensions(DATASET)
    model = kdoit.Doitgen(dims["nr"], dims["nq"], dims["np"])
    inputs = kdoit.init_array(dims["nr"], dims["nq"], dims["np"])
    result = model(*inputs)
    assert result.shape == (dims["nr"], dims["nq"], dims["np"]), f"Unexpected shape: {result.shape}"
    print(f"✓ doitgen: result shape {result.shape}, dtype {result.dtype}")
    save_mlir_output("doitgen", model, inputs)


def test_mvt():
    dims = kmvt.get_dataset_dimensions(DATASET)
    model = kmvt.Mvt(dims["n"])
    inputs = kmvt.init_array(dims["n"])
    x1_out, x2_out = model(*inputs)
    assert x1_out.shape == (dims["n"],), f"Unexpected x1_out shape: {x1_out.shape}"
    assert x2_out.shape == (dims["n"],), f"Unexpected x2_out shape: {x2_out.shape}"
    print(f"✓ mvt: x1_out shape {x1_out.shape}, x2_out shape {x2_out.shape}, dtype {x1_out.dtype}")
    save_mlir_output("mvt", model, inputs)


# ---------------------------------------------------------------------------
# BLAS kernels
# ---------------------------------------------------------------------------

def test_gemm():
    dims = kgemm.get_dataset_dimensions("gemm", DATASET)
    model = kgemm.Gemm(dims["ni"], dims["nj"], dims["nk"])
    inputs = kgemm.init_array(dims["ni"], dims["nj"], dims["nk"])
    out = model(*inputs)
    assert out.shape == (dims["ni"], dims["nj"]), f"Unexpected shape: {out.shape}"
    print(f"✓ gemm: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("gemm", model, inputs)


def test_gemver():
    dims = kgemver.get_dataset_dimensions("gemver", DATASET)
    model = kgemver.Gemver(dims["n"])
    inputs = kgemver.init_array(dims["n"])
    out = model(*inputs)
    assert out.shape == (dims["n"],), f"Unexpected shape: {out.shape}"
    print(f"✓ gemver: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("gemver", model, inputs)


def test_gesummv():
    dims = kgesummv.get_dataset_dimensions("gesummv", DATASET)
    model = kgesummv.Gesummv(dims["n"])
    inputs = kgesummv.init_array(dims["n"])
    out = model(*inputs)
    assert out.shape == (dims["n"],), f"Unexpected shape: {out.shape}"
    print(f"✓ gesummv: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("gesummv", model, inputs)


def test_symm():
    dims = ksymm.get_dataset_dimensions("symm", DATASET)
    model = ksymm.Symm(dims["m"], dims["n"])
    inputs = ksymm.init_array(dims["m"], dims["n"])
    out = model(*inputs)
    assert out.shape == (dims["m"], dims["n"]), f"Unexpected shape: {out.shape}"
    print(f"✓ symm: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("symm", model, inputs)


def test_syr2k():
    dims = ksyr2k.get_dataset_dimensions("syr2k", DATASET)
    model = ksyr2k.Syr2k(dims["n"], dims["m"])
    inputs = ksyr2k.init_array(dims["n"], dims["m"])
    out = model(*inputs)
    assert out.shape == (dims["n"], dims["n"]), f"Unexpected shape: {out.shape}"
    print(f"✓ syr2k: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("syr2k", model, inputs)


def test_syrk():
    dims = ksyrk.get_dataset_dimensions("syrk", DATASET)
    model = ksyrk.Syrk(dims["n"], dims["m"])
    inputs = ksyrk.init_array(dims["n"], dims["m"])
    out = model(*inputs)
    assert out.shape == (dims["n"], dims["n"]), f"Unexpected shape: {out.shape}"
    print(f"✓ syrk: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("syrk", model, inputs)


def test_trmm():
    dims = ktrmm.get_dataset_dimensions("trmm", DATASET)
    model = ktrmm.Trmm(dims["m"], dims["n"])
    inputs = ktrmm.init_array(dims["m"], dims["n"])
    out = model(*inputs)
    assert out.shape == (dims["m"], dims["n"]), f"Unexpected shape: {out.shape}"
    print(f"✓ trmm: result shape {out.shape}, dtype {out.dtype}")
    save_mlir_output("trmm", model, inputs)


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
