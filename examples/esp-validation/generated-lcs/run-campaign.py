#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Validate generated kernels, build isolated VC707 images, retain status/logs."""
import argparse
import concurrent.futures
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import threading
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("campaign", type=Path)
p.add_argument("esp_source", type=Path)
p.add_argument("compiled_soc", type=Path)
p.add_argument("--kernels", nargs="+", default=["gemm", "syr2k", "trmm", "symm", "syrk", "gesummv", "doitgen"])
p.add_argument("--flow", choices=("baseline", "optimized", "transformed"), default="transformed")
p.add_argument("--sim-jobs", type=int, default=2)
p.add_argument("--fpga-jobs", type=int, default=2)
p.add_argument("--cpu-results", type=Path,
               help="Reuse passing CPU baselines with identical generated object and input data")
a = p.parse_args()
root, esp, template = (x.resolve() for x in (a.campaign, a.esp_source, a.compiled_soc))
here = Path(__file__).resolve().parent
lock = threading.Lock()
states = {name: {"stage": "queued"} for name in a.kernels}


def update(name, **values):
    with lock:
        states[name].update(values)
        states[name]["updated"] = datetime.datetime.now().astimezone().isoformat()
        temporary = root / "status.json.tmp"
        temporary.write_text(json.dumps(states, indent=2) + "\n")
        temporary.replace(root / "status.json")
        print(name, values, flush=True)


def run(command, log, *, env=None, timeout=7200):
    with log.open("w") as output:
        output.write("COMMAND " + json.dumps([str(c) for c in command]) + "\n")
        output.flush()
        subprocess.run([str(c) for c in command], stdout=output,
                       stderr=subprocess.STDOUT, env=env, check=True, timeout=timeout)


def fpga(name, folder, soc):
    update(name, fpga="RUNNING")
    try:
        run([here / "build-fpga.sh", soc,
             folder / "system-esp/full-system/simulation.log", folder / "fpga"],
            folder / "fpga-driver.log", timeout=18000)
        deploy = folder / "deploy"
        shutil.copy2(folder / "fpga/top.bit", deploy / f"{name}.bit")
        with (deploy / "SHA256SUMS").open("w") as output:
            for path in sorted(deploy.iterdir()):
                if path.is_file() and path.name != "SHA256SUMS":
                    output.write(hashlib.sha256(path.read_bytes()).hexdigest() + "  " + path.name + "\n")
        update(name, fpga="PASS", bitstream=str(deploy / f"{name}.bit"))
    except Exception as error:
        update(name, fpga="FAIL", fpga_error=str(error))


with concurrent.futures.ThreadPoolExecutor(max_workers=a.fpga_jobs) as synthesis:
    def kernel(name):
        folder = root / name
        folder.mkdir(exist_ok=True)
        generated = (root / "generated/experiments" / name / "output/esp" / a.flow).resolve()
        soc = folder / "esp/socs/vc707"
        update(name, stage="adapter", generated=str(generated), fpga="WAITING")
        try:
            if not soc.exists():
                run(["python3", here / "prepare-soc.py", generated, esp, template, folder / "esp"],
                    folder / "prepare.log")
            run([here / "run-adapter.sh", generated, folder / "adapter"],
                folder / "adapter.log", timeout=600)
            update(name, adapter="PASS", stage="system-esp")
            run([here / "run-system.sh", generated, esp, template, folder / "system-esp"],
                folder / "system-esp.log", timeout=7200)
            deploy = folder / "deploy"
            deploy.mkdir()
            shutil.copy2(folder / "system-esp/full-system/gemm_app.exe", deploy / f"{name}.exe")
            for file in ("prom.bin", "prom.srec"):
                shutil.copy2(template / "soft-build/ariane" / file, deploy / file)
            update(name, system_esp="PASS", stage="system-cpu")
            synthesis.submit(fpga, name, folder, soc)
            if a.cpu_results:
                previous = (a.cpu_results / name / "system-cpu").resolve()
                previous_sim = previous / "full-system"
                log = (previous_sim / "simulation.log").read_text()
                assert "TEST PASSED" in log and "Failure: Program Completed!" in log
                assert not any(token in log for token in
                               ("TEST FAILED", "** Fatal", "** Error", "GEMM_BOUNDARY START"))
                assert (generated / "cpu-baseline/06_kernel_riscv.o").read_bytes() == (previous_sim / "host.o").read_bytes()
                assert (generated / "cpu-baseline/esp-app/testdata.h").read_bytes() == (previous_sim / "testdata.h").read_bytes()
                original_manifest = json.loads((a.cpu_results / name / "esp/integration.json").read_text())
                assert Path(original_manifest["simulation_soc_template"]).resolve() == template
                (folder / "system-cpu").symlink_to(previous, target_is_directory=True)
                update(name, cpu_baseline_reused=str(previous))
            else:
                env = dict(os.environ, LCS_SYSTEM_MODE="cpu-baseline")
                run([here / "run-system.sh", generated, esp, template, folder / "system-cpu"],
                    folder / "system-cpu.log", env=env, timeout=7200)
            shutil.copy2(folder / "system-cpu/full-system/gemm_app.exe", deploy / f"{name}-cpu.exe")
            update(name, system_cpu="PASS", stage="simulation-complete")
        except Exception as error:
            update(name, stage="FAILED", error=str(error))

    with concurrent.futures.ThreadPoolExecutor(max_workers=a.sim_jobs) as simulation:
        list(simulation.map(kernel, a.kernels))
(root / "campaign.done").write_text(datetime.datetime.now().astimezone().isoformat() + "\n")
if any(state.get("system_esp") != "PASS" or state.get("system_cpu") != "PASS" or
       state.get("fpga") != "PASS" for state in states.values()):
    raise SystemExit(1)
