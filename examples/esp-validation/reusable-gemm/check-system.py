#!/usr/bin/env python3
"""Check the static invocation plan against CPU logs and physical RTL launches."""
import json
from pathlib import Path
import re
import sys

generated, logfile = map(Path, sys.argv[1:])
host = (generated / "host.mlir").read_text()
log = logfile.read_text()
assert "TEST PASSED" in log and "TEST FAILED" not in log
assert "** Fatal" not in log and "** Error" not in log
assert "** Failure: Program Completed!" in log
constants = {name: int(value) for name, value in re.findall(
    r"(%[\w.]+) = arith.constant (\d+) : i64", host)}
expected = []
for operands, instance, ordinal in re.findall(
        r"call @esp_gemm_invoke\(([^)]+)\) \{sodap.instance = (\d+) : i64, "
        r"sodap.invocation = (\d+)", host):
    args = [x.strip() for x in operands.split(",")]
    a, b, c, m, n, k = [constants[x] for x in args[1:7]]
    expected.append((int(ordinal), int(instance), m, n, k, a, b, c))
actual = [tuple(map(int, values)) for values in re.findall(
    r"REUSABLE_GEMM launch=(\d+) instance=(\d+) M=(\d+) N=(\d+) K=(\d+) "
    r"A=(\d+) B=(\d+) C=(\d+)", log)]
assert expected and actual == expected, (expected, actual)
count = len(expected)
instances = {row[1] for row in actual}
assert len(instances) == 1, instances
instance = instances.pop()
assert len(re.findall(r"REUSABLE_GEMM open instance=", log)) == 1
assert re.findall(r"REUSABLE_GEMM close instance=(\d+) invocations=(\d+)", log) == [
    (str(instance), str(count))]
assert re.findall(r"REUSABLE_GEMM complete=(\d+) instance=(\d+)", log) == [
    (str(i), str(instance)) for i in range(1, count + 1)]
for marker in ("START", "RESULT"):
    assert [int(x) for x in re.findall(
        rf"GEMM_BOUNDARY {marker} invocation=(\d+)", log)] == list(range(1, count + 1))
report = {"status": "PASS", "instance": instance, "hardware_invocations": count,
          "columns": ["invocation", "instance", "M", "N", "K", "A", "B", "C"],
          "invocations": actual}
logfile.with_name("reusable-validation.json").write_text(json.dumps(report, indent=2) + "\n")
print(f"PASS: {count} generated GEMM calls matched {count} physical launches on instance {instance}")
