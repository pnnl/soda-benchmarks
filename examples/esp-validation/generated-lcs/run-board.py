#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Run validated VC707 images with an already-recording remote UART terminal."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument("campaign", type=Path)
p.add_argument("results", type=Path)
p.add_argument("--kernels", nargs="+", required=True)
p.add_argument("--repeats", type=int, default=3)
p.add_argument("--uart-host", default="william@espdev")
p.add_argument("--uart-log", required=True)
p.add_argument("--fpga-host", default="espdev")
p.add_argument("--jtag-port", type=int, default=3131)
p.add_argument("--esplink-ip", default="128.59.22.75")
p.add_argument("--esplink-port", type=int, default=46313)
p.add_argument("--uart-timeout", type=float, default=45)
a = p.parse_args()
root, out = a.campaign.resolve(), a.results.resolve()
here = Path(__file__).resolve().parent
out.mkdir(parents=True, exist_ok=True)
reference = json.loads((root / "results.json").read_text())
assert a.repeats > 0
assert len(a.kernels) == len(set(a.kernels))
assert all(k in reference and reference[k]["ready_for_board"] for k in a.kernels)
env = dict(os.environ, XILINX_VIVADO="/opt/xilinx/Vivado/2023.2",
           PATH="/opt/xilinx/Vivado/2023.2/bin:/opt/riscv/bin:" + os.environ["PATH"],
           PYTHONDONTWRITEBYTECODE="1")
state_path = out / "board-results.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {
    "board": "vc707-03", "campaign": str(root), "uart_host": a.uart_host,
    "uart_log": a.uart_log, "jtag": f"{a.fpga_host}:{a.jtag_port}",
    "esplink": f"{a.esplink_ip}:{a.esplink_port}", "kernels": {}}


def now():
    return datetime.datetime.now().astimezone().isoformat()


def save():
    temporary = state_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(state_path)


def remote(command):
    return subprocess.check_output(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", a.uart_host, command],
        timeout=15)


def capture_size():
    return int(remote("wc -c < " + shlex.quote(a.uart_log)))


def command(args, log, timeout=180):
    with log.open("a") as stream:
        stream.write(now() + " " + shlex.join([str(x) for x in args]) + "\n")
        stream.flush()
        subprocess.run([str(x) for x in args], cwd=log.parent, env=env,
                       stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=True)


def check_uart(raw, name, mode):
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", raw.decode(errors="replace")).replace("\r", "")
    # Reset/loading can produce an earlier banner. Select the latest invocation.
    # The boot-loader/application UART handoff on the board can lose the
    # initial 'k'. Retain the exact benchmark/dataset/type checks and record it.
    banners = list(re.finditer(r"\b(?:kernel|ernel)=(\w+) dataset=MINI dtype=float32", text))
    if not banners:
        return None
    banner = banners[-1]
    assert banner[1] == name, f"Unexpected UART kernel: {banner[1]}"
    text = text[banner.start():]
    if "TEST FAILED" in text or "MISMATCH" in text:
        raise RuntimeError("Numerical validation failed; inspect UART log")
    numeric = re.search(r"max error = (\d+) e-9, tolerance = (\d+) e-9", text)
    profile = re.search(r"region\s+cycles\s+calls\s+mean\s+share\n-+\n(.*?)\n-+", text, re.S)
    if "TEST PASSED" not in text or not numeric or not profile:
        return None
    regions = {m[1]: {"cycles": int(m[2]), "calls": int(m[3])}
               for m in re.finditer(r"^(\w+)\s+(\d+)\s+(\d+)\s+", profile[1], re.M)}
    if ("accel" in regions) != (mode == "esp"):
        return None
    required = {"total", "kernel"}
    if mode == "esp":
        required |= {"alloc", "pack", "accel", "unpack"}
    assert required <= regions.keys(), f"Incomplete profiling: {regions}"
    assert all(regions[key]["calls"] == 1 for key in required)
    error, tolerance = map(int, numeric.groups())
    assert tolerance == reference[name]["offload"]["tolerance_e9"] and error <= tolerance
    return {"status": "PASS", "error_e9": error, "tolerance_e9": tolerance,
            "uart_banner_truncated": not banner[0].startswith("kernel="),
            "same_error_as_simulation": error == reference[name]["offload"]["error_e9"],
            "regions": regions}


capture_size()  # The user-owned recording must exist before any board operation.
for name in a.kernels:
    assert name not in state["kernels"], f"Results already exist for {name}; use a fresh results directory"
    folder = out / name
    folder.mkdir()
    state["kernels"][name] = entry = {"started": now(), "status": "PROGRAMMING", "runs": []}
    save()
    try:
        deploy = root / name / "deploy"
        for leaf, expected in reference[name]["artifacts"].items():
            assert hashlib.sha256((deploy / leaf).read_bytes()).hexdigest() == expected, leaf
        entry["artifacts"] = reference[name]["artifacts"]
        print(f"{name}: programming {a.fpga_host}:{a.jtag_port}", flush=True)
        command(["vivado", "-mode", "batch", "-nojournal", "-log", folder / "vivado.log",
                 "-source", here / "program-board.tcl", "-tclargs", a.fpga_host,
                 str(a.jtag_port), deploy / f"{name}.bit"], folder / "program.log", timeout=300)
        assert "LCS_FPGA_PROGRAMMED" in (folder / "program.log").read_text()
        entry["programming"] = "PASS"
        entry["status"] = "RUNNING"
        save()
        time.sleep(5)  # Same DDR calibration allowance as ESP's fpga-program.
        soc = root / name / "esp/socs/vc707"
        esplink = soc / "socgen/esp/esplink"
        for repeat in range(1, a.repeats + 1):
            for mode in ("cpu", "esp"):
                stem = f"{mode}-{repeat}"
                log = folder / (stem + ".loader.log")
                exe = deploy / (f"{name}-cpu.exe" if mode == "cpu" else f"{name}.exe")
                run = {"mode": mode, "repeat": repeat, "started": now(), "status": "LOADING"}
                entry["runs"].append(run)
                save()
                print(f"{name}: {mode} run {repeat}/{a.repeats}", flush=True)
                command(["make", "-C", soc, "soft", "esplink", f"TEST_PROGRAM={exe}",
                         f"ESPLINK_IP={a.esplink_ip}", f"ESPLINK_PORT={a.esplink_port}"], log)
                # ESP fpga-run's exact load/reset sequence, split to mark the
                # UART offset immediately before the final launch reset.
                command([esplink, "--reset"], log, timeout=30)
                command([esplink, "--brom", "-i", soc / "soft-build/ariane/prom.bin"], log, timeout=60)
                command([esplink, "--dram", "-i", soc / "soft-build/ariane/systest.bin"], log, timeout=60)
                time.sleep(1)
                offset = capture_size()
                run["uart_offset"] = offset
                command([esplink, "--reset"], log, timeout=30)
                run["status"] = "WAITING_UART"
                save()
                deadline = time.monotonic() + a.uart_timeout
                while True:
                    raw = remote(f"tail -c +{offset+1} -- " + shlex.quote(a.uart_log))
                    (folder / (stem + ".uart.log")).write_bytes(raw)
                    parsed = check_uart(raw, name, mode)
                    if parsed:
                        run.update(parsed, completed=now())
                        save()
                        print(f"{name}: {mode} run {repeat} PASS, kernel={parsed['regions']['kernel']['cycles']} cycles", flush=True)
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"{name}/{stem}: no complete passing UART report")
                    time.sleep(1)
        entry.update(status="PASS", completed=now())
        save()
    except Exception as error:
        entry.update(status="FAIL", error=str(error), stopped=now())
        save()
        raise
print("PASS: requested board images and applications validated", flush=True)
